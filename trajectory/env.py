"""Layer-3 communication-aware multi-UAV inspection environment.

Gym-like environment consumed by the parameter-shared independent TD3 (Task 7)
and the rollout / evaluation harness (Task 8).

Design (faithful to the single-UAV ``World.py`` physics + the multi-UAV spec):

* State (per UAV ``i``)::
      [x_norm, y_norm,
       dx_target_norm, dy_target_norm,
       data_buffer_norm, remaining_time,
       (dx_other, dy_other) for every other UAV]

  with ``local_obs_dim = 6 + 2*(uav_num - 1)``. Positions are normalised by
  ``length`` / ``width`` (region spans ``[0, length] x [0, width]`` in 100 m
  units); ``data_buffer`` by the largest per-point data size; ``remaining_time``
  is ``(T - t)/T``.

* Action (per UAV): ``[v, theta]`` with ``v in [0, v_max]`` and
  ``theta in [0, 2*pi]``. Movement is ``x += v*cos(theta)*delta_t`` (and the
  symmetric ``y`` update); out-of-bounds moves are cancelled and penalised.

* Data offloading: at every step ``rate = B*log2(1 + 10**(SINR_A2G/10))`` with
  ``B = 1.0`` Mbps, and ``data_buffer -= rate*delta_t``. **Unit convention
  (mandatory): ``data_buffer`` is in Mbit and ``rate`` in Mbps, so
  ``tau = data_buffer/rate`` is in seconds — no 8x factor.** Data of point k is
  loaded when the UAV leaves the previous waypoint (or at ``reset`` for the
  first point) and offloaded while flying toward point k; at arrival, if it is
  still not fully offloaded and its *instantaneous* remaining transmission
  time ``data_buffer / rate > tau_max`` (rate evaluated at the current
  position), a ``P_trans`` penalty is applied.

* Energy: each step ``E += rotary_power(v, ENERGY_PARAMS) * delta_t``; reward
  term ``P_energy = lambda_E * rotary_power(v) * delta_t``.

* G2A outage: each step ``outage = getPointMiniOutage(...)``; reward term
  ``P_out = lambda_out * outage`` (enabled — the single-UAV code had this
  commented out).

* Collision: if the distance to another active UAV is ``< safe_distance*2``
  (``safe_distance = 0.1`` = 10 m in 100 m units by default), a ``P_collision``
  penalty is applied. UAVs are spawned at ``ini_loc + [i*depot_spacing, 0]`` so
  they do not overlap at the depot (``depot_spacing = 0.5`` = 50 m by default).

* Completion: each UAV visits its assigned sequence (injected via
  ``set_uav_traverse``); when every UAV has finished its targets, a team reward
  is issued and the episode terminates.

Interfaces used by Tasks 7-8::

    MultiUAVInspectionEnv(cfg, safe_distance=0.1, depot_spacing=0.5)
    reset() -> list[np.ndarray]
    step(actions) -> (obs_list, rewards, dones, info)
    get_obs(i)
    _offload(i, dt)
    _collision_penalty(i)
    _data_buffer          # list, len uav_num
    local_obs_dim, action_dim, UAVs
    set_uav_traverse(dict)

``step`` returns ``rewards`` as a per-UAV list and ``dones`` as a per-UAV list
of *individual* completion flags; the episode-level terminal flag lives in
``info["terminal"]`` (use that for replay bootstrapping in Task 7).
"""

import math
import os

import numpy as np

from common.config import ENERGY_PARAMS, get_data_sizes
from common.metrics import rotary_power
from radio_map import radio_map_A2G, radio_map_G2A
from radio_map.entity import UAV

# ---------------------------------------------------------------------------
# Reward / penalty magnitudes (positive constants; penalties are subtracted).
# ---------------------------------------------------------------------------
R_ARRIVE = 100.0            # reaching an assigned target point
R_COMPLETE = 200.0          # team reward when all UAVs finish all targets
P_TRANS = 10.0              # per-point delay-QoS violation (tau_k > tau_max)
P_COLLISION = 10.0          # flying within safe_distance*2 of another UAV
P_OUT_OF_BOUNDS = 5.0       # attempting to leave the region

BANDWIDTH = 1.0             # A2G bandwidth (Mbps); spec B = 1.0
DEFAULT_UAV_H = 1.0         # 100 m altitude in 100 m units (-> 0.1 km for map)
DEFAULT_ARRIVE_DIST = 0.5   # arrival radius in 100 m units (50 m, single-UAV)
DEFAULT_T = 200             # episode horizon in steps


class MultiUAVInspectionEnv(object):
    def __init__(self, cfg, safe_distance=0.1, points=None, data_sizes=None,
                 uav_traverse=None, T=None, depot_spacing=0.5):
        self.cfg = cfg
        self.uav_num = int(cfg["uav_num"])
        self.user_num = int(cfg["user_num"])
        self.length = float(cfg["length"])
        self.width = float(cfg["width"])
        self.v_max = float(cfg["v_max"])
        self.delta_t = float(cfg["delta_t"])
        self.tau_max = float(cfg["tau_max"])
        self.T_out_max = float(cfg["T_out_max"])
        self.lambda_E = float(cfg["lambda_E"])
        self.lambda_out = float(cfg["lambda_out"])
        self.lambda_tau = float(cfg["lambda_tau"])

        self.safe_distance = float(safe_distance)
        self.depot_spacing = float(depot_spacing)
        self.B = BANDWIDTH
        self.uav_h = DEFAULT_UAV_H
        self.arrive_dist = DEFAULT_ARRIVE_DIST
        self.T = int(T) if T is not None else int(cfg.get("T", DEFAULT_T))

        # Observation / action space sizes.
        self.local_obs_dim = 6 + 2 * (self.uav_num - 1)
        self.action_dim = 2

        # Inspection points (K, 2) and their per-point data sizes (K,) in Mbit.
        self.points = self._resolve_points(points)
        self.data_sizes = self._resolve_data_sizes(data_sizes)
        if self.data_sizes.shape[0] != self.points.shape[0]:
            raise ValueError("data_sizes length must match the number of points")
        n_pts = self.points.shape[0]
        self.data_ref = float(np.max(self.data_sizes)) if n_pts > 0 else 1.0
        if self.data_ref <= 0.0:
            self.data_ref = 1.0

        # Per-UAV target sequence (uav_id -> list of point indices).
        if uav_traverse is not None:
            self.uav_traverse = {int(k): [int(x) for x in v]
                                 for k, v in uav_traverse.items()}
        else:
            self.uav_traverse = self._default_traverse()

        # Reproducibility: the radio-map Monte-Carlo fast-fading estimation
        # draws from the global numpy RNG, so seed it once per environment.
        np.random.seed(int(cfg["seed"]))

        # Runtime state (populated by reset()).
        self.UAVs = []
        self.reset()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _resolve_points(self, points):
        if points is not None:
            pts = np.asarray(points, dtype=float)
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            if pts.shape[1] >= 3:
                pts = pts[:, :2]  # drop any height column
            return pts
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "datas", "Users_%d.txt" % self.user_num)
        return np.loadtxt(path)[:, :2]

    def _resolve_data_sizes(self, data_sizes):
        if data_sizes is not None:
            return np.asarray(data_sizes, dtype=float).reshape(-1)
        return np.asarray(
            get_data_sizes(self.user_num, seed=int(self.cfg["seed"])),
            dtype=float).reshape(-1)

    def _default_traverse(self):
        # Round-robin distribution of points across UAVs when no explicit
        # sequence has been injected.
        n = self.points.shape[0]
        return {i: list(range(i, n, self.uav_num)) for i in range(self.uav_num)}

    def set_uav_traverse(self, traverse):
        """Inject the per-UAV target sequence: ``{uav_id: [point indices]}``."""
        self.uav_traverse = {int(k): [int(x) for x in v]
                             for k, v in traverse.items()}

    # ------------------------------------------------------------------
    # Reset / observation
    # ------------------------------------------------------------------
    def reset(self):
        self.t = 0
        ini = np.asarray(self.cfg["ini_loc"], dtype=float).reshape(-1)
        # Spread initial positions along +x by a fixed offset so UAVs do not
        # overlap at the depot (avoid a spurious collision penalty at start).
        # UAV i is placed at ini_loc + [i * depot_spacing, 0.0].
        self.UAVs = []
        for i in range(self.uav_num):
            x = float(ini[0] + i * self.depot_spacing)
            y = float(ini[1])
            self.UAVs.append(UAV(x, y, self.uav_h))

        self._traverse = [list(self.uav_traverse.get(i, []))
                          for i in range(self.uav_num)]
        self._current_target = [0 for _ in range(self.uav_num)]
        self._done = [len(self._traverse[i]) == 0 for i in range(self.uav_num)]

        self._data_buffer = [0.0 for _ in range(self.uav_num)]
        self._offload_elapsed = [0.0 for _ in range(self.uav_num)]
        self._buffer_done = [False for _ in range(self.uav_num)]
        self._tau_done = [0.0 for _ in range(self.uav_num)]
        self._energy = [0.0 for _ in range(self.uav_num)]
        self._tau_history = [[] for _ in range(self.uav_num)]
        self._times = [None for _ in range(self.uav_num)]
        self._trajectories = [[(self.UAVs[i].x, self.UAVs[i].y)]
                              for i in range(self.uav_num)]

        # Pre-load the first point's data so it is offloaded en route to it.
        for i in range(self.uav_num):
            if not self._done[i]:
                self._data_buffer[i] = self._data_for(i, 0)

        return [self.get_obs(i) for i in range(self.uav_num)]

    def _data_for(self, i, k):
        """Data size (Mbit) of the k-th point in UAV i's sequence."""
        return float(self.data_sizes[self._traverse[i][k]])

    def get_obs(self, i):
        uav = self.UAVs[i]
        x_norm = uav.x / self.length
        y_norm = uav.y / self.width

        k = self._current_target[i]
        if self._done[i] or k >= len(self._traverse[i]):
            tx, ty = uav.x, uav.y
        else:
            t = self.points[self._traverse[i][k]]
            tx, ty = float(t[0]), float(t[1])
        dx = (tx - uav.x) / self.length
        dy = (ty - uav.y) / self.width

        data_norm = self._data_buffer[i] / self.data_ref
        remaining_time = (self.T - self.t) / float(self.T)

        obs = [x_norm, y_norm, dx, dy, data_norm, remaining_time]
        for j in range(self.uav_num):
            if j == i:
                continue
            other = self.UAVs[j]
            obs.append((other.x - uav.x) / self.length)
            obs.append((other.y - uav.y) / self.width)
        return np.asarray(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Radio-map queries
    # ------------------------------------------------------------------
    def _loc_km(self, i):
        uav = self.UAVs[i]
        return np.array([[uav.x / 10.0, uav.y / 10.0, uav.h / 10.0]])

    def _a2g_rate(self, i):
        """A2G uplink rate (Mbps) at UAV i's current position."""
        sinr = float(radio_map_A2G.getPointDateRate(self._loc_km(i)))
        if not np.isfinite(sinr):
            return 0.0
        return self.B * math.log2(1.0 + 10.0 ** (sinr / 10.0))

    def _get_g2a_outage(self, i):
        """G2A downlink outage probability at UAV i's current position."""
        out = radio_map_G2A.getPointMiniOutage(self._loc_km(i))
        return float(out[0][0])

    # ------------------------------------------------------------------
    # Core dynamics helpers
    # ------------------------------------------------------------------
    def _offload(self, i, dt):
        """Offload one step's worth of data from UAV i's buffer (Mbit)."""
        rate = self._a2g_rate(i)
        self._data_buffer[i] = max(0.0, self._data_buffer[i] - rate * dt)
        return rate

    def _collision_penalty(self, i):
        """Return a negative reward term if UAV i is too close to another
        active UAV, otherwise 0.0."""
        uav = self.UAVs[i]
        for j in range(self.uav_num):
            if j == i or self._done[j]:
                continue
            other = self.UAVs[j]
            d = math.hypot(uav.x - other.x, uav.y - other.y)
            if d < self.safe_distance * 2.0:
                return -P_COLLISION
        return 0.0

    def _trans_penalty(self, i):
        """Return ``P_trans`` if UAV i's remaining transmission time exceeds
        ``tau_max``, else 0.0.

        The remaining transmission time is the *instantaneous* quantity
        ``data_buffer / rate``, evaluated at the UAV's current position via the
        Shannon rate ``B*log2(1 + 10**(SINR/10))`` with a ``1e-6`` floor to
        avoid division by zero (and so a tiny-but-nonzero buffer at a near-zero
        rate is correctly flagged).
        """
        if self._data_buffer[i] <= 0.0:
            return 0.0
        rate = max(self._a2g_rate(i), 1e-6)
        if self._data_buffer[i] / rate > self.tau_max:
            return P_TRANS
        return 0.0

    def _move(self, i, v, theta):
        """Apply the (v, theta) action; cancel + penalise out-of-bounds moves."""
        uav = self.UAVs[i]
        prev_x, prev_y = uav.x, uav.y
        new_x = prev_x + v * math.cos(theta) * self.delta_t
        new_y = prev_y + v * math.sin(theta) * self.delta_t
        if (new_x < 0.0 or new_x > self.length
                or new_y < 0.0 or new_y > self.width):
            uav.x, uav.y = prev_x, prev_y
            return -P_OUT_OF_BOUNDS
        uav.x, uav.y = new_x, new_y
        return 0.0

    def _check_arrival(self, i):
        """Assess arrival at the current target; advance sequence on success."""
        uav = self.UAVs[i]
        k = self._current_target[i]
        if k >= len(self._traverse[i]):
            return 0.0
        target = self.points[self._traverse[i][k]]
        d = math.hypot(uav.x - target[0], uav.y - target[1])
        if d > self.arrive_dist:
            return 0.0

        reward = R_ARRIVE
        # Delay-QoS: penalize if the previous point's data is still not fully
        # offloaded AND its instantaneous remaining transmission time
        # (data_buffer / rate at the current position) exceeds tau_max.
        reward -= self._trans_penalty(i)
        tau_k = (self._tau_done[i] if self._buffer_done[i]
                 else self._offload_elapsed[i])
        self._tau_history[i].append(float(tau_k))

        # Advance to the next target, pre-loading its data.
        self._current_target[i] += 1
        if self._current_target[i] < len(self._traverse[i]):
            self._data_buffer[i] = self._data_for(i, self._current_target[i])
            self._offload_elapsed[i] = 0.0
            self._buffer_done[i] = False
            self._tau_done[i] = 0.0
        else:
            self._done[i] = True
            self._data_buffer[i] = 0.0
            self._times[i] = self.t
        return reward

    # ------------------------------------------------------------------
    # Gym-like step
    # ------------------------------------------------------------------
    def step(self, actions):
        self.t += 1
        actions = np.asarray(actions, dtype=float).reshape(self.uav_num, 2)

        rewards = [0.0 for _ in range(self.uav_num)]
        for i in range(self.uav_num):
            if self._done[i]:
                continue

            v = float(np.clip(actions[i, 0], 0.0, self.v_max))
            theta = float(actions[i, 1])

            # Energy (reward penalty) + accumulate for metrics.
            energy = rotary_power(v, ENERGY_PARAMS) * self.delta_t
            self._energy[i] += energy
            reward = -self.lambda_E * energy

            # Movement (out-of-bounds -> cancel + penalty).
            reward += self._move(i, v, theta)

            # Data offloading at the post-move position.
            self._offload(i, self.delta_t)
            self._offload_elapsed[i] += self.delta_t
            if not self._buffer_done[i] and self._data_buffer[i] <= 0.0:
                self._buffer_done[i] = True
                self._tau_done[i] = self._offload_elapsed[i]

            # G2A outage penalty (enabled).
            outage = self._get_g2a_outage(i)
            reward -= self.lambda_out * outage

            # Collision penalty (negative term).
            reward += self._collision_penalty(i)

            # Arrival check.
            reward += self._check_arrival(i)

            self._trajectories[i].append((self.UAVs[i].x, self.UAVs[i].y))
            rewards[i] = reward

        terminal = bool(np.all(self._done))
        truncated = (not terminal) and (self.t >= self.T)
        if terminal:
            for i in range(self.uav_num):
                rewards[i] += R_COMPLETE
        episode_done = terminal or truncated

        obs_list = [self.get_obs(i) for i in range(self.uav_num)]
        dones = [bool(self._done[i]) for i in range(self.uav_num)]
        info = {
            "terminal": bool(episode_done),
            "truncated": bool(truncated),
            "done": list(self._done),
            "t": self.t,
            "energy": list(self._energy),
            "tau_history": [list(t) for t in self._tau_history],
            "times": list(self._times),
        }
        return obs_list, rewards, dones, info
