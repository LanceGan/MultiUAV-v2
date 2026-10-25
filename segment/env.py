"""Single-segment communication-aware trajectory environment (new design).

One UAV flies from a *start* waypoint to a *goal* waypoint while uploading a
payload (the *source* point's data, per the new design's "upload the previous
point's data before arriving at the next" constraint).  The state is
goal-conditioned (relative goal + data buffer + remaining time); the radio-map /
communication information enters ONLY through the reward (upload rate + G2A
outage), matching the new design and the single-UAV project's convention.

Units follow the existing project convention (100 m units, Mbit, Mbps, seconds)
for compatibility with ``radio_map`` / ``common.config``.
"""

import math

import numpy as np

from radio_map import radio_map_A2G, radio_map_G2A

BANDWIDTH = 1.0   # A2G uplink bandwidth (MHz) -> rate in Mbps
UAV_H = 1.0       # 100 m altitude in 100 m units (-> 0.1 km for the map)
DATA_REF = 30.0   # max per-point data size (Mbit) for normalising the absolute-data state feature

# Reward weights (single-segment; communication enters via rate / outage).
PROGRESS = 60.0            # symmetric forward-progress reward (approach +, retreat -)
POTENTIAL = 35.0           # potential-based shaping toward the goal (symmetric)
UPLOAD_WEIGHT = 1.0        # reward per Mbit uploaded this step (communication-aware)
TIME_PENALTY = 1.0         # per-step time cost (minimise segment flight time)
OUTAGE_WEIGHT = 1.0        # G2A downlink outage penalty (reliability)
ARRIVE_REWARD = 400.0      # reaching the goal with the payload fully uploaded
INFEASIBLE_PENALTY = 1000.0  # reaching the goal / timing out with data still left


class CommField(object):
    """Cached statistical-average A2G rate / G2A outage field.

    Precomputes the new design's planning map (§6.3: fixed-seed statistical
    averages) on a grid ONCE, then serves bilinear-interpolated values, so the
    env avoids the per-step Monte-Carlo fast-fading cost (~3 ms/query).
    """

    def __init__(self, cfg, spacing=0.25):
        from scipy.interpolate import RegularGridInterpolator
        self.length = float(cfg["length"])
        self.width = float(cfg["width"])
        self.xs = np.arange(0.0, self.length + 1e-9, spacing)
        self.ys = np.arange(0.0, self.width + 1e-9, spacing)

        gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        h_km = UAV_H / 10.0

        rate = np.zeros(len(pts))
        outage = np.zeros(len(pts))
        for k in range(len(pts)):
            loc = np.array([[pts[k, 0] / 10.0, pts[k, 1] / 10.0, h_km]])
            sinr = float(radio_map_A2G.getPointDateRate(loc))
            rate[k] = BANDWIDTH * math.log2(1.0 + 10.0 ** (sinr / 10.0))
            outage[k] = float(radio_map_G2A.getPointMiniOutage(loc)[0][0])

        n = (len(self.xs), len(self.ys))
        self._rate_fn = RegularGridInterpolator(
            (self.xs, self.ys), rate.reshape(n), bounds_error=False, fill_value=0.0)
        self._outage_fn = RegularGridInterpolator(
            (self.xs, self.ys), outage.reshape(n), bounds_error=False, fill_value=1.0)

    def rate(self, pos):
        return float(np.clip(self._rate_fn(np.asarray(pos, float).reshape(1, -1)),
                             0.0, None)[0])

    def outage(self, pos):
        return float(np.clip(self._outage_fn(np.asarray(pos, float).reshape(1, -1)),
                             0.0, 1.0)[0])


class SingleSegmentEnv(object):
    """A single start -> goal segment with an in-flight upload requirement."""

    def __init__(self, cfg, start, goal, data_size, arrive_radius=0.15,
                 max_steps=None, field=None):
        self.length = float(cfg["length"])
        self.width = float(cfg["width"])
        self.v_max = float(cfg["v_max"])
        self.delta_t = float(cfg["delta_t"])
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.data_size = float(data_size)     # payload to upload (Mbit)
        self.arrive_radius = float(arrive_radius)
        self.field = field                     # optional cached CommField

        if max_steps is None:
            dist = float(np.linalg.norm(self.goal - self.start))
            # Enough steps to fly the segment a few times over (upload headroom).
            self.max_steps = int(max(
                60.0, dist / (self.v_max * self.delta_t) * 5.0))
        else:
            self.max_steps = int(max_steps)

        self.pos = self.start.copy()
        self.data_buffer = 0.0
        self.t = 0
        self.feasible = False

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def state_dim(self):
        return 7  # [x, y, dx_goal, dy_goal, data_frac, data_abs, time_frac]

    def _state(self):
        dx = self.goal[0] - self.pos[0]
        dy = self.goal[1] - self.pos[1]
        return np.array([
            self.pos[0] / self.length,
            self.pos[1] / self.width,
            dx / self.length,
            dy / self.width,
            self.data_buffer / max(self.data_size, 1e-6),  # fraction of THIS segment
            self.data_buffer / DATA_REF,                    # absolute data (Mbit / 30)
            self.t / max(self.max_steps, 1),
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Radio map
    # ------------------------------------------------------------------
    def _loc_km(self):
        return np.array([[self.pos[0] / 10.0, self.pos[1] / 10.0,
                          UAV_H / 10.0]])

    def _rate(self):
        if self.field is not None:
            return self.field.rate(self.pos)
        sinr = float(radio_map_A2G.getPointDateRate(self._loc_km()))
        return BANDWIDTH * math.log2(1.0 + 10.0 ** (sinr / 10.0))

    def _outage(self):
        if self.field is not None:
            return self.field.outage(self.pos)
        return float(radio_map_G2A.getPointMiniOutage(self._loc_km())[0][0])

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self):
        self.pos = self.start.copy()
        self.data_buffer = self.data_size
        self.t = 0
        self.feasible = False
        return self._state()

    def step(self, action):
        v = float(np.clip(action[0], 0.0, self.v_max))
        theta = float(action[1])

        dist_pre = float(np.linalg.norm(self.pos - self.goal))

        # Move (clamp to the region; out-of-bounds slides along the wall).
        self.pos = np.array([
            min(max(self.pos[0] + v * math.cos(theta) * self.delta_t, 0.0),
                self.length),
            min(max(self.pos[1] + v * math.sin(theta) * self.delta_t, 0.0),
                self.width),
        ])

        # Upload (automatic; the rate depends on the A2G map at the new position).
        rate = self._rate()
        data_before = self.data_buffer
        self.data_buffer = max(0.0, data_before - rate * self.delta_t)
        uploaded = data_before - self.data_buffer
        outage = self._outage()

        # Reward: symmetric forward progress + potential shaping (both arbitrage
        # free) + upload reward (communication enters here, bounded by the buffer)
        # - time - outage.
        dist_cur = float(np.linalg.norm(self.pos - self.goal))
        raw_progress = dist_pre - dist_cur  # >0 approaching, <0 retreating
        reward = (PROGRESS * raw_progress
                  + POTENTIAL * (1.0 / (dist_cur + 1.0) - 1.0 / (dist_pre + 1.0))
                  + UPLOAD_WEIGHT * uploaded
                  - TIME_PENALTY
                  - OUTAGE_WEIGHT * outage)

        done = False
        if dist_cur <= self.arrive_radius:
            done = True
            if self.data_buffer <= 0.0:
                reward += ARRIVE_REWARD
                self.feasible = True
            else:
                reward -= INFEASIBLE_PENALTY
                self.feasible = False

        self.t += 1
        if not done and self.t >= self.max_steps:
            done = True
            reward -= INFEASIBLE_PENALTY
            self.feasible = False

        info = {
            "feasible": self.feasible,
            "t": self.t,
            "rate": rate,
            "outage": outage,
            "data_buffer": self.data_buffer,
        }
        return self._state(), reward, done, info
