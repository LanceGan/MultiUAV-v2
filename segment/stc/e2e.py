"""Step-level shared-bandwidth environment for end-to-end MADRL baselines.

The environment exposes the local observation and continuous ``[v, theta]``
action used by the repository's MATD3 implementation. Tasks are discovered and
visited through motion; a task payload is generated on service and must be
uploaded before another task can be assigned to the same UAV. Concurrent
uploads in the same sector split the configured bandwidth equally.
"""

import math

import numpy as np

from common.config import ENERGY_PARAMS
from common.metrics import rotary_power


class SharedBandwidthE2EEnv:
    """Fixed-scenario multi-UAV inspection environment for DRL baselines."""

    def __init__(self, scenario, field, *, horizon_steps=1000,
                 safe_distance_m=30.0, depot_spacing_m=50.0):
        self.scenario = scenario
        self.field = field
        self.n_uav = int(scenario.n_uav)
        self.n_tasks = int(scenario.n_tasks)
        self.T = int(horizon_steps)
        self.dt = float(scenario.limits.delta_t_s)
        self.v_max = float(scenario.limits.v_max_mps)
        self.arrive_radius = float(scenario.limits.arrive_radius_m)
        self.eps_out = float(scenario.limits.eps_out)
        self.bandwidth_hz = float(scenario.limits.bandwidth_hz)
        self.safe_distance = float(safe_distance_m)
        self.depot_spacing = float(depot_spacing_m)
        self.task_positions = np.asarray(scenario.task_positions_m, dtype=float)
        self.task_data = np.asarray(scenario.task_data_bits, dtype=float)
        self.data_ref = max(float(self.task_data.max(initial=0.0)), 1.0)
        self.action_dim = 2
        self.local_obs_dim = 9 + 4 * self.n_tasks + 2 * (self.n_uav - 1)
        self.reset()

    @property
    def action_low(self):
        return np.asarray([0.0, 0.0], dtype=np.float32)

    @property
    def action_high(self):
        return np.asarray([self.v_max, 2.0 * math.pi], dtype=np.float32)

    def reset(self):
        depot = np.asarray(self.scenario.depot_m, dtype=float)
        self.t = 0
        self.positions = np.asarray([
            depot + np.asarray([i * self.depot_spacing, 0.0])
            for i in range(self.n_uav)], dtype=float)
        self.last_actions = np.zeros((self.n_uav, 2), dtype=float)
        self.visited = np.zeros(self.n_tasks, dtype=bool)
        self.assigned_uav = np.full(self.n_tasks, -1, dtype=int)
        self.buffers = np.zeros(self.n_uav, dtype=float)
        self.arrival_step = np.full(self.n_tasks, -1, dtype=int)
        self.upload_step = np.full(self.n_tasks, -1, dtype=int)
        self.energy_j = np.zeros(self.n_uav, dtype=float)
        self.outage_s = np.zeros(self.n_uav, dtype=float)
        self.collisions = 0
        self.out_of_bounds = 0
        self.trajectories = [[tuple(self.positions[i])] for i in range(self.n_uav)]
        self._last_samples = [None for _ in range(self.n_uav)]
        return self.observations()

    def observations(self):
        dep = np.asarray(self.scenario.depot_m, dtype=float)
        obs = []
        for i in range(self.n_uav):
            pos = self.positions[i]
            sample = self.field.sample(pos)
            self._last_samples[i] = sample
            best_eff = float(np.max(sample.a2g_efficiency))
            best_out = float(np.min(sample.g2a_outage))
            row = [
                (pos[0] - dep[0]) / max(self.scenario.length_m, 1.0),
                (pos[1] - dep[1]) / max(self.scenario.width_m, 1.0),
                self.last_actions[i, 0] / max(self.v_max, 1e-9),
                math.cos(float(self.last_actions[i, 1])),
                math.sin(float(self.last_actions[i, 1])),
                self.buffers[i] / self.data_ref,
                (self.T - self.t) / max(float(self.T), 1.0),
            ]
            row.extend([best_eff / 10.0, best_out])
            for k in range(self.n_tasks):
                delta = self.task_positions[k] - pos
                row.extend([
                    delta[0] / max(self.scenario.length_m, 1.0),
                    delta[1] / max(self.scenario.width_m, 1.0),
                    float(self.visited[k]),
                    self.task_data[k] / self.data_ref,
                ])
            for j in range(self.n_uav):
                if j == i:
                    continue
                delta = self.positions[j] - pos
                row.extend([
                    delta[0] / max(self.scenario.length_m, 1.0),
                    delta[1] / max(self.scenario.width_m, 1.0),
                ])
            obs.append(np.asarray(row, dtype=np.float32))
        return obs

    def _select_sector(self, sample):
        reliable = np.flatnonzero(sample.g2a_outage <= self.eps_out + 1e-12)
        if reliable.size:
            sector = int(reliable[np.argmax(sample.a2g_efficiency[reliable])])
        else:
            sector = int(np.argmin(sample.g2a_outage))
        return sector

    def _apply_shared_bandwidth(self, samples):
        sectors = np.full(self.n_uav, -1, dtype=int)
        active = self.buffers > 1e-9
        for i in range(self.n_uav):
            if active[i]:
                sectors[i] = self._select_sector(samples[i])
        uploaded = np.zeros(self.n_uav, dtype=float)
        for sector in sorted(set(int(s) for s in sectors[active])):
            members = np.flatnonzero(active & (sectors == sector))
            share = self.bandwidth_hz / max(len(members), 1)
            for i in members:
                eff = float(samples[i].a2g_efficiency[sector])
                available = max(0.0, eff * share * self.dt)
                take = min(float(self.buffers[i]), available)
                self.buffers[i] -= take
                uploaded[i] = take
        return uploaded, sectors

    def _service_tasks(self):
        newly = []
        candidates = []
        for k in range(self.n_tasks):
            if self.visited[k]:
                continue
            for i in range(self.n_uav):
                if self.buffers[i] > 1e-9:
                    continue
                dist = float(np.linalg.norm(self.positions[i] - self.task_positions[k]))
                if dist <= self.arrive_radius:
                    candidates.append((dist, k, i))
        served_uav = set()
        for dist, k, i in sorted(candidates):
            if self.visited[k] or i in served_uav:
                continue
            self.visited[k] = True
            self.assigned_uav[k] = i
            self.arrival_step[k] = self.t
            self.buffers[i] = float(self.task_data[k])
            served_uav.add(i)
            newly.append((i, k, dist))
        return newly

    def _nearest_unvisited_distance(self, i):
        unvisited = np.flatnonzero(~self.visited)
        if unvisited.size == 0:
            return 0.0
        return float(np.min(np.linalg.norm(
            self.task_positions[unvisited] - self.positions[i], axis=1)))

    def step(self, actions):
        actions = np.asarray(actions, dtype=float).reshape(self.n_uav, 2)
        self.t += 1
        rewards = np.zeros(self.n_uav, dtype=float)
        previous = self.positions.copy()
        previous_nearest = np.asarray(
            [self._nearest_unvisited_distance(i) for i in range(self.n_uav)])

        for i in range(self.n_uav):
            v = float(np.clip(actions[i, 0], 0.0, self.v_max))
            theta = float(actions[i, 1] % (2.0 * math.pi))
            self.last_actions[i] = [v, theta]
            new_pos = self.positions[i] + self.dt * v * np.asarray(
                [math.cos(theta), math.sin(theta)])
            clipped = np.clip(
                new_pos,
                [0.0, 0.0],
                [self.scenario.length_m, self.scenario.width_m])
            if np.linalg.norm(new_pos - clipped) > 1e-6:
                self.out_of_bounds += 1
                rewards[i] -= 1.0
            self.positions[i] = clipped
            self.trajectories[i].append(tuple(clipped))
            step_energy = rotary_power(v, ENERGY_PARAMS) * self.dt
            self.energy_j[i] += step_energy
            rewards[i] -= 1e-5 * step_energy
            rewards[i] -= 0.1

        for i in range(self.n_uav):
            for j in range(i + 1, self.n_uav):
                dist = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                if dist < self.safe_distance:
                    severity = (self.safe_distance - dist) / self.safe_distance
                    rewards[i] -= 2.0 * severity
                    rewards[j] -= 2.0 * severity
                    self.collisions += 1

        samples = [self.field.sample(pos) for pos in self.positions]
        uploaded, sectors = self._apply_shared_bandwidth(samples)
        for i, sample in enumerate(samples):
            sector = self._select_sector(sample)
            self.outage_s[i] += float(sample.g2a_outage[sector]) * self.dt
            rewards[i] += 5.0 * uploaded[i] / self.data_ref
            rewards[i] -= 2.0 * float(sample.g2a_outage[sector])

        newly = self._service_tasks()
        for i, _k, _dist in newly:
            rewards[i] += 100.0

        for k in np.flatnonzero(self.visited & (self.upload_step < 0)):
            i = int(self.assigned_uav[k])
            if self.buffers[i] <= 1e-9:
                self.upload_step[k] = self.t

        for i in range(self.n_uav):
            current = self._nearest_unvisited_distance(i)
            denom = max(self.v_max * self.dt, 1e-9)
            rewards[i] += 2.0 * (previous_nearest[i] - current) / denom
            # Pan-style absolute distance-to-target penalty. The progress term
            # supplies the gradient; this term prevents hovering in open space
            # from becoming reward-optimal.
            rewards[i] -= 0.2 * current / denom

        all_uploaded = bool(np.all(self.upload_step >= 0))
        terminal = bool(np.all(self.visited) and all_uploaded)
        truncated = (not terminal) and self.t >= self.T
        if terminal:
            rewards += 1000.0

        info = {
            "terminal": terminal,
            "truncated": truncated,
            "t": self.t,
            "visited": self.visited.copy(),
            "buffers": self.buffers.copy(),
            "uploaded_bits": uploaded,
            "sectors": sectors,
            "all_uploaded": all_uploaded,
            "collisions": self.collisions,
            "out_of_bounds": self.out_of_bounds,
        }
        return self.observations(), rewards.tolist(), [terminal] * self.n_uav, info

    def metrics(self):
        flight_s = self.t * self.dt
        outage_ratio = self.outage_s / max(flight_s, 1e-9)
        delays = [
            (self.upload_step[k] - self.arrival_step[k]) * self.dt
            for k in range(self.n_tasks)
            if self.arrival_step[k] >= 0 and self.upload_step[k] >= 0
        ]
        makespan_s = float(self.t * self.dt)
        total_delay_s = float(sum(delays))
        all_uploaded = bool(np.all(self.upload_step >= 0))
        feasible = bool(
            np.all(self.visited) and all_uploaded
            and np.all(outage_ratio <= self.eps_out + 1e-12)
            and self.collisions == 0)
        return {
            "feasible": feasible,
            "makespan_s": makespan_s,
            "total_delay_s": total_delay_s,
            "outage_ratio": outage_ratio.tolist(),
            "max_outage_ratio": float(outage_ratio.max(initial=0.0)),
            "nearest_unvisited_m": [
                self._nearest_unvisited_distance(i)
                for i in range(self.n_uav)],
            "visited_tasks": int(np.sum(self.visited)),
            "uploaded_tasks": int(np.sum(self.upload_step >= 0)),
            "collisions": int(self.collisions),
            "out_of_bounds": int(self.out_of_bounds),
            "energy_j": self.energy_j.tolist(),
            "steps": int(self.t),
        }


def apply_greedy_speed(env, actions):
    """Pan-style distance-aware speed taper.

    The nearest UAV to its closest unvisited task keeps its requested speed;
    farther UAVs are progressively slowed to avoid futile task races.
    """
    actions = np.asarray(actions, dtype=float).copy()
    distances = np.asarray(
        [max(env._nearest_unvisited_distance(i), 1e-9)
         for i in range(env.n_uav)])
    d_min = float(np.min(distances))
    factors = np.clip(d_min / distances, 0.25, 1.0)
    actions[:, 0] *= factors
    return actions


def apply_repulsion(env, actions):
    """Small potential-field correction away from very close neighboring UAVs."""
    actions = np.asarray(actions, dtype=float).copy()
    for i in range(env.n_uav):
        v = float(actions[i, 0])
        theta = float(actions[i, 1])
        vector = v * np.asarray([math.cos(theta), math.sin(theta)])
        for j in range(env.n_uav):
            if i == j:
                continue
            delta = env.positions[i] - env.positions[j]
            dist = float(np.linalg.norm(delta))
            if 1e-6 < dist < env.safe_distance * 2.0:
                vector += 20.0 * delta / (dist * dist)
        speed = float(np.linalg.norm(vector))
        if speed <= 1e-9:
            actions[i] = [0.0, theta]
        else:
            actions[i] = [
                float(np.clip(speed, 0.0, env.v_max)),
                float(math.atan2(vector[1], vector[0]) % (2.0 * math.pi)),
            ]
    return actions
