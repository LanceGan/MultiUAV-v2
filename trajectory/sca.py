"""Layer-3 baseline: communication-aware trajectory via SCA (SQP).

Given the layer-2 waypoint sequence, each consecutive waypoint pair is refined
from a straight line into a polyline that minimises the *same* comm-aware cost
used by the sequence layer (``edge_cost``), now evaluated along the actual
(possibly deviating) path:

    cost = flight_time + lambda_E*E + lambda_out*outage_time
           + lambda_tau*max(0, tau - tau_max)

where ``tau = data*flight_time/C`` (C = distance-weighted offload capacity along
the path), matching the sequence layer's delay definition.

The raw radio-map field is non-smooth (LoS/NLoS jumps), so we build a smoothed
surrogate: sample rate/outage on a grid and interpolate linearly.  SCA is then
successive quadratic programming (scipy SLSQP) on that surrogate, with multiple
restarts and a *keep-better* fallback to the straight line so the result never
degrades a segment.  This is the "traditional optimisation" baseline: it uses
the full radio map (same physics as the DRL) and is an upper bound on what
communication-aware deviation can achieve.
"""

import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

from common.config import ENERGY_PARAMS
from common.metrics import edge_energy
from radio_map import radio_map_A2G, radio_map_G2A
from sequence.radio_path import BANDWIDTH, C_FLOOR, DEFAULT_UAV_H


class ChannelField(object):
    """Smoothed rate (Mbps) / outage (prob) field over the inspection region.

    Samples the deterministic radio maps on a regular grid and exposes linear
    interpolants, so the SCA objective is continuous (gradient-friendly).
    """

    def __init__(self, cfg, spacing=0.5):
        self.length = float(cfg["length"])
        self.width = float(cfg["width"])
        self.h_km = float(cfg.get("uav_h", DEFAULT_UAV_H)) / 10.0
        self.xs = np.arange(0.0, self.length + 1e-9, spacing)
        self.ys = np.arange(0.0, self.width + 1e-9, spacing)

        gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        rate_g, outage_g = self._sample(pts)

        self._rate_fn = RegularGridInterpolator(
            (self.xs, self.ys), rate_g, bounds_error=False, fill_value=0.0)
        self._outage_fn = RegularGridInterpolator(
            (self.xs, self.ys), outage_g, bounds_error=False, fill_value=1.0)

    def _sample(self, pts_100m):
        """Query A2G rate and G2A outage for ``(K, 2)`` points (100m units)."""
        locs = np.hstack([pts_100m / 10.0,
                          np.full((len(pts_100m), 1), self.h_km)])
        rates = np.zeros(len(pts_100m))
        outages = np.zeros(len(pts_100m))
        for k in range(len(pts_100m)):
            sinr = float(radio_map_A2G.getPointDateRate(locs[k:k + 1]))
            rates[k] = BANDWIDTH * np.log2(1.0 + 10.0 ** (sinr / 10.0))
            outages[k] = float(radio_map_G2A.getPointMiniOutage(locs[k:k + 1])[0][0])
        n = (len(self.xs), len(self.ys))
        return rates.reshape(n), outages.reshape(n)

    def rate(self, pts_100m):
        return np.clip(self._rate_fn(pts_100m), 0.0, None)

    def outage(self, pts_100m):
        return np.clip(self._outage_fn(pts_100m), 0.0, 1.0)


def trajectory_cost(field, poly_100m, data_size, cfg, use_a2g=True):
    """Comm-aware cost of a polyline ``(M+1, 2)`` in 100m units.

    ``use_a2g=False`` drops the A2G delay term (``lam_tau=0``), so the SCA
    baseline optimises only the G2A outage + flight time (G2A-only tier).
    """
    v_max = float(cfg["v_max"])
    lam_E = float(cfg["lambda_E"])
    lam_out = float(cfg["lambda_out"])
    lam_tau = float(cfg["lambda_tau"]) if use_a2g else 0.0
    tau_max = float(cfg["tau_max"])

    mid = (poly_100m[:-1] + poly_100m[1:]) / 2.0
    d = np.linalg.norm(poly_100m[1:] - poly_100m[:-1], axis=1)
    rate = field.rate(mid)
    outage = field.outage(mid)

    total_d = float(d.sum())
    flight_time = total_d / v_max
    capacity = max(float((rate * d).sum() / v_max), C_FLOOR)
    tau = data_size * flight_time / capacity
    outage_time = float((outage * d).sum() / v_max)
    energy = edge_energy(total_d * 100.0, v_max * 100.0, ENERGY_PARAMS)

    return (flight_time + lam_E * energy + lam_out * outage_time
            + lam_tau * max(0.0, tau - tau_max))


def optimize_segment(field, a, b, data_size, cfg, m=5, n_restarts=3, seed=0,
                     use_a2g=True):
    """Refine the straight segment ``a -> b`` (SCA); never worse than straight.

    Returns ``(polyline, cost)`` where ``polyline`` has ``m + 2`` vertices.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    straight = np.vstack([a, b])
    best_cost = trajectory_cost(field, straight, data_size, cfg, use_a2g=use_a2g)
    best_poly = straight

    rng = np.random.RandomState(seed)
    bounds = [(0.0, field.length)] * (2 * m)
    for r in range(n_restarts):
        init = np.linspace(0.0, 1.0, m + 2)[:, None] * (b - a) + a
        if r > 0:
            init[1:-1] += rng.normal(0.0, 0.5, size=(m, 2))
        x0 = np.clip(init[1:-1].ravel(), 0.0, field.length)

        def obj(x):
            q = np.vstack([a, x.reshape(m, 2), b])
            return trajectory_cost(field, q, data_size, cfg, use_a2g=use_a2g)

        res = minimize(obj, x0, method="SLSQP", bounds=bounds,
                       options={"maxiter": 150, "ftol": 1e-5})
        if res.fun < best_cost:
            best_cost = res.fun
            best_poly = np.vstack([a, res.x.reshape(m, 2), b])
    return best_poly, best_cost


def plan_route(field, waypoints, data_per_segment, cfg, m=5, n_restarts=3,
               seed=0, use_a2g=True):
    """Plan a comm-optimised route through ``waypoints`` (list of (x, y)).

    ``waypoints`` is the visit sequence INCLUDING the start anchor but EXCLUDING
    the end (the env stops after the last inspection point).  ``data_per_segment``
    has length ``len(waypoints) - 1``: entry ``k`` is the data offloaded on
    segment ``waypoints[k] -> waypoints[k+1]`` (the *destination* point's data,
    matching the env's "pre-load en route to the point" semantics).  Returns a
    concatenated polyline.
    """
    full = [np.asarray(waypoints[0], dtype=float)]
    for k in range(len(waypoints) - 1):
        a = np.asarray(waypoints[k], dtype=float)
        b = np.asarray(waypoints[k + 1], dtype=float)
        data = float(data_per_segment[k])
        if np.linalg.norm(b - a) < 1e-9:
            continue
        poly, _ = optimize_segment(field, a, b, data, cfg,
                                   m=m, n_restarts=n_restarts, seed=seed + k,
                                   use_a2g=use_a2g)
        full.append(poly[1:])  # drop the duplicated start vertex
    return np.vstack(full)


class SCAPolicy(object):
    """Deterministic policy that follows a pre-planned polyline at ``v_max``.

    Used to replay an SCA-optimised route through the env (same evaluation
    pipeline as the greedy baseline and the DRL).  Each UAV must use its own
    instance (one per UAV path).
    """

    def __init__(self, path, length, width, v_max):
        self.path = np.asarray(path, dtype=float)
        self.length = float(length)
        self.width = float(width)
        self.v_max = float(v_max)
        self._idx = 0

    def select_action(self, obs):
        x = float(obs[0]) * self.length
        y = float(obs[1]) * self.width
        # Advance past vertices already reached (within a small radius).
        while self._idx < len(self.path) - 1:
            if math.hypot(self.path[self._idx][0] - x,
                          self.path[self._idx][1] - y) < 0.3:
                self._idx += 1
            else:
                break
        tx, ty = self.path[self._idx]
        theta = math.atan2(ty - y, tx - x)
        if theta < 0.0:
            theta += 2.0 * math.pi
        return np.array([self.v_max, theta], dtype=np.float32)
