"""Legal single-segment baselines for the fair comparison (P3-2).

Each baseline takes a single directed edge (``start -> goal`` with payload
``data``) and returns ``(flight_time, upload_delay, makespan, feasible)``.  They
share the same radio map (``field``) and the same hard constraint (upload must
finish before the UAV reaches the goal's 15 m arrive radius) so the comparison
against the TD3 policy is fair.

Terminology follows the review:
- ``Straight-MaxSpeed`` is a *relaxation lower bound* (it ignores the upload),
  NOT a legal feasible solution.
- ``Straight-Hold`` is the legal naive baseline (hold *before* the 15 m radius).
- ``Comm-Aware Waypoint`` is a discrete relay-point search (strong, map-aware).
"""

import numpy as np

# arrive radius in 100 m units (== 15 m), matching segment/env.py.
ARRIVE_RADIUS = 0.15


def _rate_along(field, a, b, n=20):
    """Average upload rate (Mbps) along the straight line a -> b."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    pts = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
    return float(np.mean([field.rate(p) for p in pts]))


def straight_maxspeed(cfg, field, a, b, data):
    """Relaxation lower bound: fly straight at v_max, ignore the upload.

    Not a feasible solution under the upload hard constraint — only a lower
    bound on the makespan (no feasible policy can be faster than this).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    dist = float(np.linalg.norm(b - a))
    # flight to the 15 m arrive radius (same distance convention as straight_hold)
    flight = max(0.0, dist - ARRIVE_RADIUS) / float(cfg["v_max"])
    rate = max(_rate_along(field, a, b), 1e-6)
    return flight, data / rate, flight, False


def straight_hold(cfg, field, a, b, data):
    """Fly straight; if the upload is unfinished at the 15 m radius, hold there
    until it completes, then enter (legal — always feasible)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    dist = float(np.linalg.norm(b - a))
    flight = max(0.0, dist - ARRIVE_RADIUS) / float(cfg["v_max"])
    rate = max(_rate_along(field, a, b), 1e-6)
    upload = data / rate
    makespan = max(flight, upload)   # hold before the 15 m radius if upload > flight
    return flight, upload, makespan, True


def comm_aware_waypoint(cfg, field, a, b, data, grid=20):
    """Discrete relay-point search: pick the waypoint that minimises
    flight + upload (exploits the A2G map, but only through straight legs)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    xs = np.linspace(0.0, float(cfg["length"]), grid)
    ys = np.linspace(0.0, float(cfg["width"]), grid)
    best = None
    for x in xs:
        for y in ys:
            rp = np.array([x, y])
            d1 = float(np.linalg.norm(rp - a))
            d2 = float(np.linalg.norm(b - rp))
            flight = (d1 + d2) / float(cfg["v_max"])
            r1 = max(_rate_along(field, a, rp, n=8), 1e-6)
            r2 = max(_rate_along(field, rp, b, n=8), 1e-6)
            # distance-weighted (== time-weighted at v_max) average rate
            rate = (r1 * d1 + r2 * d2) / max(d1 + d2, 1e-6)
            upload = data / max(rate, 1e-6)
            cost = flight + upload
            if best is None or cost < best[0]:
                best = (cost, flight, upload)
    _, flight, upload = best
    return flight, upload, max(flight, upload), True
