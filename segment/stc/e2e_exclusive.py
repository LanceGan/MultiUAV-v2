"""Fixed-route full-trajectory environment with exclusive sector association.

This module is intentionally separate from :mod:`segment.stc.e2e`.  The older
environment remains available for the historical shared-bandwidth experiments;
the class here implements the accepted experiment contract:

* routes are supplied as ``{uav_id: [task_id, ...]}`` and cannot be changed by
  the trajectory policy;
* every airborne UAV is associated with exactly one sector per slot;
* a sector serves at most one UAV per slot;
* task and terminal acceptance use a 10 m radius;
* the payload of the previous task must be fully uploaded before the UAV may
  enter the next task's acceptance region;
* the final payload must be uploaded before the UAV may enter the terminal
  acceptance region.

The environment uses SI units throughout.
"""

import math

import numpy as np

from common.config import ENERGY_PARAMS
from common.metrics import rotary_power


REWARD_SCALE = 0.01
TIME_PENALTY = 0.1
PROGRESS_REWARD_WEIGHT = 0.2
GUIDE_ALIGNMENT_REWARD = 0.05
UPLOAD_REWARD_WEIGHT = 5.0
OUTAGE_PENALTY_WEIGHT = 2.0
QOS_RISK_WEIGHT = 0.0
QOS_VIOLATION_PENALTY = 1000.0
SERVICE_REWARD = 100.0
TERMINAL_REWARD = 1000.0
EARLY_ARRIVAL_PENALTY = 100.0
BOUNDARY_PENALTY = 20.0
COLLISION_PENALTY = 20.0
INVALID_STALL_START_STEPS = 20
INVALID_STALL_PENALTY_WEIGHT = 0.1
INVALID_STALL_PENALTY_CAP = 2.0
MAP_GUIDE_HOLD_OFFSET_M = 5.0
MAP_GUIDE_ANGLES = 12


class ExclusiveSectorFixedRouteEnv:
    """Multi-UAV inspection environment with one sector per UAV per slot."""

    def __init__(
        self,
        scenario,
        field,
        routes,
        *,
        horizon_steps=2000,
        arrival_radius_m=10.0,
        safe_distance_m=30.0,
        depot_spacing_m=50.0,
        outage_probability_threshold=0.5,
        outage_max_s=20.0,
    ):
        self.scenario = scenario
        self.field = field
        self.n_uav = int(scenario.n_uav)
        self.n_tasks = int(scenario.n_tasks)
        self.T = int(horizon_steps)
        if self.T <= 0:
            raise ValueError("horizon_steps must be positive")

        self.dt = float(scenario.limits.delta_t_s)
        self.v_max = float(scenario.limits.v_max_mps)
        self.bandwidth_hz = float(scenario.limits.bandwidth_hz)
        self.eps_out = float(scenario.limits.eps_out)
        self.arrival_radius = float(arrival_radius_m)
        self.safe_distance = float(safe_distance_m)
        self.depot_spacing = float(depot_spacing_m)
        self.outage_probability_threshold = float(outage_probability_threshold)
        self.outage_max_s = float(outage_max_s)
        if not 0.0 < self.outage_probability_threshold <= 1.0:
            raise ValueError("outage_probability_threshold must be in (0, 1]")
        if self.outage_max_s <= 0.0:
            raise ValueError("outage_max_s must be positive")

        self.task_positions = np.asarray(scenario.task_positions_m, dtype=float)
        self.task_data = np.asarray(scenario.task_data_bits, dtype=float)
        self.depot = np.asarray(scenario.depot_m, dtype=float)
        self.terminal = np.asarray(scenario.terminal_m, dtype=float)
        self.data_ref = max(float(self.task_data.max(initial=0.0)), 1.0)

        self.routes = self._validate_routes(routes)
        self.action_dim = 2
        # Position, velocity, target, progress, buffer, time, radio, neighbors.
        # QoS observability adds current contiguous-outage ratio and the last
        # serving-sector outage probability to the base observation.
        self.local_obs_dim = 15 + 2 * (self.n_uav - 1)
        self.reset()

    @property
    def action_low(self):
        return np.asarray([0.0, -math.pi], dtype=np.float32)

    @property
    def action_high(self):
        return np.asarray([self.v_max, math.pi], dtype=np.float32)

    @staticmethod
    def _wrap_angle(angle):
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    def _validate_routes(self, routes):
        validated = {}
        flat = []
        for i in range(self.n_uav):
            route = tuple(int(t) for t in routes.get(i, ()))
            if any(t < 0 or t >= self.n_tasks for t in route):
                raise ValueError("route for UAV %d contains an invalid task id" % i)
            if len(route) != len(set(route)):
                raise ValueError("route for UAV %d contains duplicate tasks" % i)
            validated[i] = route
            flat.extend(route)
        if sorted(flat) != list(range(self.n_tasks)):
            raise ValueError("fixed routes must partition all tasks exactly once")
        return validated

    def reset(self):
        self.t = 0
        self.positions = np.asarray(
            [self.depot + np.asarray([i * self.depot_spacing, 0.0])
             for i in range(self.n_uav)],
            dtype=float,
        )
        self.last_actions = np.zeros((self.n_uav, 2), dtype=float)
        self.route_index = np.zeros(self.n_uav, dtype=int)
        self.done = np.asarray([len(self.routes[i]) == 0 for i in range(self.n_uav)])
        self.completion_step = np.full(self.n_uav, -1, dtype=int)

        self.buffers = np.zeros(self.n_uav, dtype=float)
        self.buffer_task = np.full(self.n_uav, -1, dtype=int)
        self.visited = np.zeros(self.n_tasks, dtype=bool)
        self.assigned_uav = np.full(self.n_tasks, -1, dtype=int)
        self.arrival_step = np.full(self.n_tasks, -1, dtype=int)
        self.upload_step = np.full(self.n_tasks, -1, dtype=int)

        self.energy_j = np.zeros(self.n_uav, dtype=float)
        self.outage_s = np.zeros(self.n_uav, dtype=float)
        self.current_contiguous_outage_s = np.zeros(self.n_uav, dtype=float)
        self.max_contiguous_outage_s = np.zeros(self.n_uav, dtype=float)
        self.qos_violation_penalized = np.zeros(self.n_uav, dtype=bool)
        self.airborne_steps = np.zeros(self.n_uav, dtype=int)
        self.collisions = 0
        self.out_of_bounds = 0
        self.early_service_arrivals = 0
        self.early_terminal_arrivals = 0
        self.sector_conflicts = 0
        self.safety_projections = 0
        self.last_safety_projections = 0
        self.upload_guard_projections = 0
        self.last_upload_guard_projections = 0
        self.stall_steps = np.zeros(self.n_uav, dtype=int)
        self.max_stall_steps = np.zeros(self.n_uav, dtype=int)

        self.sectors = np.full(self.n_uav, -1, dtype=int)
        self.last_efficiency = np.zeros(self.n_uav, dtype=float)
        self.last_outage = np.zeros(self.n_uav, dtype=float)
        self._guide_cache = {}
        self.reward_term_totals = self._empty_reward_terms()
        self.reward_term_last = self._empty_reward_terms()
        self.trajectories = [
            [tuple(self.positions[i])] for i in range(self.n_uav)
        ]
        return self.observations()

    def _empty_reward_terms(self):
        return {
            name: np.zeros(self.n_uav, dtype=float)
            for name in (
                "time", "progress", "guide_alignment", "boundary", "collision",
                "upload", "outage", "service", "terminal", "early_arrival",
                "invalid_stall", "qos_risk", "qos", "energy",
            )
        }

    def _add_reward_term(self, uav_id, name, value):
        value = float(value) * REWARD_SCALE
        self.reward_term_last[name][uav_id] += value
        self.reward_term_totals[name][uav_id] += value
        return value

    def _target_kind(self, i):
        if self.done[i]:
            return "done", None
        route = self.routes[i]
        index = int(self.route_index[i])
        if index < len(route):
            return "task", int(route[index])
        return "terminal", None

    def _target_position(self, i):
        kind, task = self._target_kind(i)
        if kind == "task":
            return self.task_positions[task]
        if kind == "terminal":
            return self.terminal
        return self.positions[i]

    @staticmethod
    def _segment_enters_disk(start, target, endpoint, radius):
        segment = np.asarray(endpoint, dtype=float) - np.asarray(start, dtype=float)
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-12:
            return False
        projection = float(np.dot(
            np.asarray(target, dtype=float) - np.asarray(start, dtype=float),
            segment,
        ) / length_sq)
        projection = float(np.clip(projection, 0.0, 1.0))
        closest = np.asarray(start, dtype=float) + projection * segment
        return float(np.linalg.norm(closest - np.asarray(target, dtype=float))) < radius

    def _guide_hold_point(self, i, target, cache_key):
        """MapGuide-style safe upload point outside the next service disk."""
        position_bin = (
            int(round(float(self.positions[i, 0]) / 50.0)),
            int(round(float(self.positions[i, 1]) / 50.0)),
        )
        cache_lookup = cache_key + position_bin
        if cache_lookup in self._guide_cache:
            return self._guide_cache[cache_lookup]
        radius = self.arrival_radius + MAP_GUIDE_HOLD_OFFSET_M
        candidates = []
        for k in range(MAP_GUIDE_ANGLES):
            angle = 2.0 * math.pi * k / MAP_GUIDE_ANGLES
            point = np.asarray(target, dtype=float) + radius * np.asarray(
                [math.cos(angle), math.sin(angle)]
            )
            point = np.clip(
                point,
                [0.0, 0.0],
                [self.scenario.length_m, self.scenario.width_m],
            )
            sample = self.field.sample(point)
            reliable_sectors = np.flatnonzero(
                sample.g2a_outage <= self.eps_out + 1e-12
            )
            if reliable_sectors.size:
                sector = int(reliable_sectors[
                    np.argmax(sample.a2g_efficiency[reliable_sectors])
                ])
            else:
                sector = int(np.argmin(sample.g2a_outage))
            outage = float(sample.g2a_outage[sector])
            efficiency = float(sample.a2g_efficiency[sector])
            reliable = outage <= self.eps_out + 1e-12
            candidate_key = (
                0 if reliable else 1,
                -efficiency if reliable else outage,
                outage if reliable else -efficiency,
                k,
            )
            candidates.append((candidate_key, point))
        safe = [
            item for item in candidates
            if not self._segment_enters_disk(
                self.positions[i], target, item[1], self.arrival_radius
            )
        ]
        if safe:
            _, point = min(safe, key=lambda item: item[0])
        else:
            direction = self.positions[i] - np.asarray(target, dtype=float)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-9:
                direction = np.asarray([1.0, 0.0])
                norm = 1.0
            point = np.asarray(target, dtype=float) + radius * direction / norm
            point = np.clip(
                point,
                [0.0, 0.0],
                [self.scenario.length_m, self.scenario.width_m],
            )
        self._guide_cache[cache_lookup] = point
        return point

    def guide_target(self, i):
        """Return the MapGuide target for UAV ``i``.

        With no pending payload, the guide is the actual task/terminal anchor.
        With a pending payload, it is a radio-aware point on a 15 m ring around
        the next anchor, keeping the UAV outside the 10 m service disk until the
        upload is complete.
        """
        kind, task = self._target_kind(i)
        if kind == "done":
            return self.positions[i]
        anchor = self.task_positions[task] if kind == "task" else self.terminal
        if self.buffers[i] <= 1e-9:
            return anchor
        return self._guide_hold_point(i, anchor, (kind, task, "upload"))

    def guide_action(self, i):
        """Deterministic MapGuide action in the environment's ``[v, theta]`` form."""
        guide = self.guide_target(i)
        delta = np.asarray(guide, dtype=float) - self.positions[i]
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            return np.asarray([0.0, 0.0], dtype=np.float32)
        speed = min(self.v_max, max(0.0, (distance - 1.0) / self.dt))
        # Heading is relative to the current MapGuide bearing.  Zero therefore
        # means "fly directly towards the guide point", matching the single-UAV
        # Variant-A action parameterisation.
        return np.asarray([speed, 0.0], dtype=np.float32)

    def project_safe_actions(self, actions, outage_threshold, *,
                             update_stats=True):
        """Replace actions that would enter a position without a reliable sector.

        This is a one-step look-ahead safety projection.  It uses the same
        MapGuide action as the teacher fallback and is not part of the learned
        reward or action space.

        ``update_stats=False`` is used to derive safe teacher targets without
        counting the same projection twice.  The executed action still uses the
        default ``update_stats=True`` call.
        """
        actions = np.asarray(actions, dtype=float).reshape(self.n_uav, 2).copy()
        threshold = float(outage_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("outage_threshold must be in [0, 1]")
        projections = 0
        upload_guards = 0
        for i in range(self.n_uav):
            if self.done[i]:
                continue
            guide = self.guide_target(i)
            bearing = math.atan2(
                guide[1] - self.positions[i, 1],
                guide[0] - self.positions[i, 0],
            )
            theta = self._wrap_angle(bearing + self._wrap_angle(actions[i, 1]))
            speed = float(np.clip(actions[i, 0], 0.0, self.v_max))
            intended = self.positions[i] + self.dt * speed * np.asarray(
                [math.cos(theta), math.sin(theta)]
            )
            clipped = np.clip(
                intended,
                [0.0, 0.0],
                [self.scenario.length_m, self.scenario.width_m],
            )
            guard_upload = False
            if self.buffers[i] > 1e-9 and self._target_kind(i)[0] != "done":
                _, task = self._target_kind(i)
                anchor = (
                    self.task_positions[task]
                    if task is not None else self.terminal
                )
                guard_upload = self._segment_enters_disk(
                    self.positions[i], anchor, clipped, self.arrival_radius
                )
            sample = self.field.sample(clipped)
            if (
                guard_upload
                or
                np.linalg.norm(intended - clipped) > 1e-9
                or not np.any(sample.g2a_outage <= threshold)
            ):
                actions[i] = self.guide_action(i)
                projections += 1
                if guard_upload:
                    upload_guards += 1
        if update_stats:
            self.last_safety_projections = projections
            self.safety_projections += projections
            self.last_upload_guard_projections = upload_guards
            self.upload_guard_projections += upload_guards
        return np.clip(actions, self.action_low, self.action_high)

    def observations(self):
        obs = []
        for i in range(self.n_uav):
            pos = self.positions[i]
            target = self._target_position(i)
            guide = self.guide_target(i)
            guide_delta = guide - pos
            sample = self.field.sample(pos)
            kind, _ = self._target_kind(i)
            route_len = max(len(self.routes[i]), 1)
            row = [
                (pos[0] - self.depot[0]) / max(self.scenario.length_m, 1.0),
                (pos[1] - self.depot[1]) / max(self.scenario.width_m, 1.0),
                self.last_actions[i, 0] / max(self.v_max, 1e-9),
                math.cos(float(self.last_actions[i, 1])),
                math.sin(float(self.last_actions[i, 1])),
                self.buffers[i] / self.data_ref,
                (self.T - self.t) / max(float(self.T), 1.0),
                guide_delta[0] / max(self.scenario.length_m, 1.0),
                guide_delta[1] / max(self.scenario.width_m, 1.0),
                1.0 if kind == "terminal" else 0.0,
                min(float(self.route_index[i]), route_len) / route_len,
                float(np.max(sample.a2g_efficiency)) / 10.0,
                float(np.min(sample.g2a_outage)),
                min(
                    float(self.current_contiguous_outage_s[i])
                    / max(self.outage_max_s, 1e-9),
                    1.0,
                ),
                float(self.last_outage[i]),
            ]
            for j in range(self.n_uav):
                if j == i:
                    continue
                other_delta = self.positions[j] - pos
                row.extend([
                    other_delta[0] / max(self.scenario.length_m, 1.0),
                    other_delta[1] / max(self.scenario.width_m, 1.0),
                ])
            obs.append(np.asarray(row, dtype=np.float32))
        return obs

    def _assign_exclusive_sectors(self, samples):
        """Assign distinct sectors by deterministic one-to-one matching.

        Reliable pairs (G2A outage within the route threshold) are preferred
        lexicographically, followed by higher A2G spectral efficiency.  Since
        the sector count is much larger than the UAV count, every active UAV
        receives a sector without waiting.
        """
        active = np.flatnonzero(~self.done)
        sectors = np.full(self.n_uav, -1, dtype=int)
        if active.size == 0:
            return sectors
        if self.field.n_sectors < active.size:
            raise ValueError("not enough sectors for exclusive assignment")

        pairs = []
        for i in active:
            sample = samples[int(i)]
            for m in range(self.field.n_sectors):
                outage = float(sample.g2a_outage[m])
                efficiency = float(sample.a2g_efficiency[m])
                reliable = outage <= self.eps_out + 1e-12
                key = (
                    0 if reliable else 1,
                    -efficiency if reliable else outage,
                    outage if reliable else -efficiency,
                    int(i),
                    int(m),
                )
                pairs.append((key, int(i), int(m)))

        used_uav = set()
        used_sector = set()
        for _key, i, m in sorted(pairs):
            if i in used_uav or m in used_sector:
                continue
            used_uav.add(i)
            used_sector.add(m)
            sectors[i] = m
            if len(used_uav) == active.size:
                break
        if len(used_uav) != active.size:
            raise ValueError("failed to assign one distinct sector per UAV")
        return sectors

    def _service_and_terminal(self):
        """Apply task/terminal acceptance after movement and upload."""
        rewards = np.zeros(self.n_uav, dtype=float)
        for i in range(self.n_uav):
            if self.done[i]:
                continue
            kind, task = self._target_kind(i)
            target = self.terminal if kind == "terminal" else self.task_positions[task]
            distance = float(np.linalg.norm(self.positions[i] - target))
            if distance > self.arrival_radius + 1e-12:
                continue

            if kind == "task":
                if self.buffers[i] > 1e-9:
                    self.early_service_arrivals += 1
                    rewards[i] += self._add_reward_term(
                        i, "early_arrival", -EARLY_ARRIVAL_PENALTY
                    )
                    continue
                self.visited[task] = True
                self.assigned_uav[task] = i
                self.arrival_step[task] = self.t
                self.buffers[i] = float(self.task_data[task])
                self.buffer_task[i] = task
                self.route_index[i] += 1
                rewards[i] += self._add_reward_term(i, "service", SERVICE_REWARD)
            else:
                if self.buffers[i] > 1e-9:
                    self.early_terminal_arrivals += 1
                    rewards[i] += self._add_reward_term(
                        i, "early_arrival", -EARLY_ARRIVAL_PENALTY
                    )
                    continue
                self.done[i] = True
                self.completion_step[i] = self.t
                rewards[i] += self._add_reward_term(i, "terminal", TERMINAL_REWARD)
        return rewards

    def step(self, actions):
        actions = np.asarray(actions, dtype=float).reshape(self.n_uav, 2)
        self.t += 1
        self.reward_term_last = self._empty_reward_terms()
        rewards = np.zeros(self.n_uav, dtype=float)
        previous_positions = self.positions.copy()
        previous_buffers = self.buffers.copy()
        guide_points = [self.guide_target(i) for i in range(self.n_uav)]
        previous_guide_distance = np.asarray([
            np.linalg.norm(guide_points[i] - previous_positions[i])
            for i in range(self.n_uav)
        ])

        for i in range(self.n_uav):
            if self.done[i]:
                continue
            rewards[i] += self._add_reward_term(i, "time", -TIME_PENALTY)
            speed = float(np.clip(actions[i, 0], 0.0, self.v_max))
            guide_delta = guide_points[i] - previous_positions[i]
            guide_bearing = math.atan2(guide_delta[1], guide_delta[0])
            relative_theta = self._wrap_angle(actions[i, 1])
            theta = self._wrap_angle(guide_bearing + relative_theta)
            self.last_actions[i] = [speed, theta]
            proposed = self.positions[i] + self.dt * speed * np.asarray(
                [math.cos(theta), math.sin(theta)]
            )
            clipped = np.clip(
                proposed,
                [0.0, 0.0],
                [self.scenario.length_m, self.scenario.width_m],
            )
            if np.linalg.norm(proposed - clipped) > 1e-9:
                self.out_of_bounds += 1
                rewards[i] += self._add_reward_term(
                    i, "boundary", -BOUNDARY_PENALTY
                )
            self.positions[i] = clipped
            self.trajectories[i].append(tuple(clipped))

            energy = rotary_power(speed, ENERGY_PARAMS) * self.dt
            self.energy_j[i] += energy
            rewards[i] += self._add_reward_term(i, "energy", -1e-5 * energy)

        active = np.flatnonzero(~self.done)
        self.airborne_steps[active] += 1

        max_step_distance = max(self.v_max * self.dt, 1e-9)
        for i in active:
            guide = guide_points[i]
            current_guide_distance = float(
                np.linalg.norm(guide - self.positions[i])
            )
            progress = (
                float(previous_guide_distance[i]) - current_guide_distance
            ) / max_step_distance
            rewards[i] += self._add_reward_term(
                i, "progress", PROGRESS_REWARD_WEIGHT * progress
            )

            move_vec = self.positions[i] - previous_positions[i]
            guide_vec = guide - previous_positions[i]
            move_norm = float(np.linalg.norm(move_vec))
            guide_norm = float(np.linalg.norm(guide_vec))
            if move_norm > 1e-9 and guide_norm > 1e-9:
                alignment = float(np.clip(
                    np.dot(move_vec, guide_vec) / (move_norm * guide_norm),
                    -1.0, 1.0,
                ))
                rewards[i] += self._add_reward_term(
                    i, "guide_alignment", GUIDE_ALIGNMENT_REWARD * alignment
                )

            kind, task = self._target_kind(i)
            anchor = (
                self.task_positions[task] if kind == "task"
                else self.terminal if kind == "terminal"
                else self.positions[i]
            )
            anchor_distance = float(np.linalg.norm(self.positions[i] - anchor))
            valid_upload_hold = bool(
                previous_buffers[i] > 1e-9
                and kind != "done"
                and self.arrival_radius + 0.25
                <= anchor_distance
                <= self.arrival_radius + MAP_GUIDE_HOLD_OFFSET_M + 2.0
            )
            if move_norm < 1.0 and not valid_upload_hold:
                self.stall_steps[i] += 1
                self.max_stall_steps[i] = max(
                    self.max_stall_steps[i], self.stall_steps[i]
                )
                if self.stall_steps[i] > INVALID_STALL_START_STEPS:
                    excess = self.stall_steps[i] - INVALID_STALL_START_STEPS
                    penalty = min(
                        INVALID_STALL_PENALTY_CAP,
                        INVALID_STALL_PENALTY_WEIGHT * excess,
                    )
                    rewards[i] += self._add_reward_term(
                        i, "invalid_stall", -penalty
                    )
            else:
                self.stall_steps[i] = 0

        for offset, i in enumerate(active):
            for j in active[offset + 1:]:
                distance = float(np.linalg.norm(
                    self.positions[i] - self.positions[j]
                ))
                if distance < self.safe_distance:
                    self.collisions += 1
                    severity = (self.safe_distance - distance) / self.safe_distance
                    penalty = COLLISION_PENALTY * severity
                    rewards[i] += self._add_reward_term(
                        i, "collision", -penalty
                    )
                    rewards[j] += self._add_reward_term(
                        j, "collision", -penalty
                    )

        samples = [self.field.sample(pos) for pos in self.positions]
        self.sectors = self._assign_exclusive_sectors(samples)
        if active.size:
            assigned = self.sectors[active]
            if len(set(int(m) for m in assigned)) != active.size:
                self.sector_conflicts += 1

        uploaded = np.zeros(self.n_uav, dtype=float)
        for i in active:
            sector = int(self.sectors[i])
            efficiency = float(samples[i].a2g_efficiency[sector])
            outage = float(samples[i].g2a_outage[sector])
            self.last_efficiency[i] = efficiency
            self.last_outage[i] = outage
            self.outage_s[i] += outage * self.dt
            if outage >= self.outage_probability_threshold:
                self.current_contiguous_outage_s[i] += self.dt
                self.max_contiguous_outage_s[i] = max(
                    self.max_contiguous_outage_s[i],
                    self.current_contiguous_outage_s[i],
                )
                ratio = (
                    self.current_contiguous_outage_s[i]
                    / max(self.outage_max_s, 1e-9)
                )
                if ratio > 0.5:
                    risk = min(1.0, (ratio - 0.5) / 0.5)
                    rewards[i] += self._add_reward_term(
                        i, "qos_risk", -QOS_RISK_WEIGHT * risk ** 2
                    )
                if (
                    self.max_contiguous_outage_s[i] > self.outage_max_s + 1e-12
                    and not self.qos_violation_penalized[i]
                ):
                    self.qos_violation_penalized[i] = True
                    rewards[i] += self._add_reward_term(
                        i, "qos", -QOS_VIOLATION_PENALTY
                    )
            else:
                self.current_contiguous_outage_s[i] = 0.0
            rewards[i] += self._add_reward_term(
                i, "outage", -OUTAGE_PENALTY_WEIGHT * outage
            )

            if self.buffers[i] <= 1e-9:
                continue
            available = max(0.0, efficiency * self.bandwidth_hz * self.dt)
            take = min(float(self.buffers[i]), available)
            self.buffers[i] -= take
            uploaded[i] = take
            rewards[i] += self._add_reward_term(
                i, "upload", UPLOAD_REWARD_WEIGHT * take / self.data_ref
            )
            if self.buffers[i] <= 1e-9:
                task = int(self.buffer_task[i])
                if task >= 0 and self.upload_step[task] < 0:
                    self.upload_step[task] = self.t
                self.buffer_task[i] = -1

        rewards += self._service_and_terminal()
        terminal = bool(np.all(self.done))
        truncated = (not terminal) and self.t >= self.T

        info = {
            "terminal": terminal,
            "truncated": truncated,
            "t": self.t,
            "visited": self.visited.copy(),
            "buffers": self.buffers.copy(),
            "uploaded_bits": uploaded,
            "sectors": self.sectors.copy(),
            "route_index": self.route_index.copy(),
            "done": self.done.copy(),
            "guide_points": np.asarray(guide_points, dtype=float),
            "stall_steps": self.stall_steps.copy(),
            "outage_now": self.last_outage.copy(),
            "current_contiguous_outage_s": (
                self.current_contiguous_outage_s.copy()
            ),
            "max_contiguous_outage_s": (
                self.max_contiguous_outage_s.copy()
            ),
            "collisions": int(self.collisions),
            "out_of_bounds": int(self.out_of_bounds),
            "early_service_arrivals": int(self.early_service_arrivals),
            "early_terminal_arrivals": int(self.early_terminal_arrivals),
            "sector_conflicts": int(self.sector_conflicts),
            "safety_projections": int(self.last_safety_projections),
            "upload_guard_projections": int(
                self.last_upload_guard_projections
            ),
            "reward_terms": {
                k: v.copy() for k, v in self.reward_term_last.items()
            },
        }
        return self.observations(), rewards.tolist(), terminal, info

    def metrics(self):
        flight_s = self.t * self.dt
        airborne_s = self.airborne_steps.astype(float) * self.dt
        outage_ratio = np.divide(
            self.outage_s,
            airborne_s,
            out=np.zeros_like(self.outage_s),
            where=airborne_s > 0.0,
        )
        delays = [
            (self.upload_step[k] - self.arrival_step[k]) * self.dt
            for k in range(self.n_tasks)
            if self.arrival_step[k] >= 0 and self.upload_step[k] >= 0
        ]
        unserved = int(np.sum(~self.visited))
        unfinished_uploads = int(np.sum(self.upload_step < 0))
        terminal_distances = [
            float(np.linalg.norm(self.positions[i] - self.terminal))
            for i in range(self.n_uav)
        ]

        failure_reasons = []
        if unserved:
            failure_reasons.append("unserved_tasks")
        if unfinished_uploads:
            failure_reasons.append("unfinished_uploads")
        if self.early_service_arrivals:
            failure_reasons.append("early_service_arrival")
        if self.early_terminal_arrivals:
            failure_reasons.append("early_terminal_arrival")
        if not bool(np.all(self.done)):
            failure_reasons.append("terminal_not_reached")
        if self.collisions:
            failure_reasons.append("collision")
        if self.out_of_bounds:
            failure_reasons.append("out_of_bounds")
        if np.any(
            self.max_contiguous_outage_s > self.outage_max_s + 1e-12
        ):
            failure_reasons.append("contiguous_outage_excess")
        if self.sector_conflicts:
            failure_reasons.append("sector_conflict")

        return {
            "feasible": not failure_reasons,
            "failure_reasons": failure_reasons,
            "makespan_s": float(flight_s),
            "total_delay_s": float(sum(delays)),
            "delay_count": len(delays),
            "visited_tasks": int(np.sum(self.visited)),
            "uploaded_tasks": int(np.sum(self.upload_step >= 0)),
            "unserved_tasks": unserved,
            "unfinished_uploads": unfinished_uploads,
            "outage_ratio": outage_ratio.tolist(),
            "max_outage_ratio": float(outage_ratio.max(initial=0.0)),
            "total_expected_outage_s": self.outage_s.tolist(),
            "max_contiguous_outage_s": self.max_contiguous_outage_s.tolist(),
            "outage_max_s": float(self.outage_max_s),
            "outage_probability_threshold": float(
                self.outage_probability_threshold
            ),
            "early_service_arrivals": int(self.early_service_arrivals),
            "early_terminal_arrivals": int(self.early_terminal_arrivals),
            "collisions": int(self.collisions),
            "out_of_bounds": int(self.out_of_bounds),
            "sector_conflicts": int(self.sector_conflicts),
            "safety_projections": int(self.safety_projections),
            "upload_guard_projections": int(self.upload_guard_projections),
            "terminal_distances_m": terminal_distances,
            "completion_step": self.completion_step.tolist(),
            "max_stall_steps": self.max_stall_steps.tolist(),
            "energy_j": self.energy_j.tolist(),
            "reward_term_totals": {
                k: v.tolist() for k, v in self.reward_term_totals.items()
            },
            "steps": int(self.t),
        }
