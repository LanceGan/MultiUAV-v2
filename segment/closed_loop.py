"""Closed-loop multi-UAV evaluator (new design §7 upgrade).

Replaces the v1 *static* congestion approximation (``scheduler.py``) with a
per-slot simulation: each UAV flies its single-segment TD3 trajectory (the
trajectory is fixed — the deterministic policy does not change with contention),
and at every time slot each UAV associates to its best-SINR BS sector and shares
that sector's bandwidth equally with the other UAVs currently in it.  The
contention-reduced upload rate then determines the *true* per-edge upload delay
and route feasibility.

This is the "逐时隙扇区关联 + 共享带宽" core that the outer ALNS will later
optimise against (iteration), so the route search becomes explicitly
resource-conflict-aware instead of post-hoc.
"""
import warnings
warnings.warn("segment.closed_loop is a legacy prototype; the formal pipeline lives in segment.stc (STC-ALNS).", DeprecationWarning, stacklevel=2)



import math

import numpy as np

from radio_map import radio_map_A2G
from segment.env import SingleSegmentEnv, UAV_H

SIR_THRESHOLD = 3.0  # must match radio_map_A2G.SIR_THRESHOLD


def per_sector_sinr(pos):
    """Per-BS per-sector average SINR (dB), shape (numBS, Sec_num)."""
    loc = np.array([pos[0] / 10.0, pos[1] / 10.0, UAV_H / 10.0])
    _, sinr = radio_map_A2G.getPointOutageMatrix(loc, SIR_THRESHOLD)
    return np.asarray(sinr, dtype=float)


def best_sector(pos):
    """Flat sector id (0 .. numBS*Sec_num-1) with the best SINR at ``pos``."""
    return int(np.argmax(per_sector_sinr(pos)))


def rollout_traj(cfg, route, points, data_sizes, model, field):
    """Roll out one UAV's full route -> per-step records.

    Returns
    -------
    (positions, edge_idx, data, full_rate) : each array length = total steps.
    ``edge_idx[t]`` is the segment index of step t; ``data[t]`` is the payload
    uploaded during step t (the *source* point's data); ``full_rate[t]`` is the
    single-UAV upload rate (Mbps, full bandwidth B) at step t.
    """
    ini = np.asarray(cfg["ini_loc"], float)
    end = np.asarray(cfg["end_loc"], float)
    route = [int(j) for j in route]
    wps = [ini] + [np.asarray(points[j], float) for j in route] + [end]
    datas = [0.0] + [float(data_sizes[j]) for j in route]

    mi = np.zeros(2, np.float32)
    ma = np.array([float(cfg["v_max"]), 2.0 * math.pi], np.float32)

    pos, edge, data, rate = [], [], [], []
    for r in range(len(wps) - 1):
        e = SingleSegmentEnv(cfg, wps[r], wps[r + 1], datas[r], field=field)
        e.reset()
        while True:
            a = np.clip(model.select_action(e._state()), mi, ma)
            _, _, done, info = e.step(a)
            pos.append(e.pos.copy())
            edge.append(r)
            data.append(datas[r])
            rate.append(float(info["rate"]))
            if done:
                break
    return (np.asarray(pos), np.asarray(edge), np.asarray(data), np.asarray(rate))


def _co_sector_factors(trajs, sectors):
    """Per-UAV per-step contention factor (>=1) = #UAVs sharing that sector.

    Steps are aligned by *step index* (simultaneous take-off); a UAV that has
    already finished its (shorter) trajectory is dropped from later slots.
    Only UAVs that actually have payload to upload at a step (``data > 0``)
    count as bandwidth competitors — a UAV on a depot->point leg uploads
    nothing, so it must not consume shared bandwidth.
    """
    n_uav = len(trajs)
    factors = []
    for u in range(n_uav):
        f = np.ones(len(sectors[u]), dtype=float)
        for t in range(len(sectors[u])):
            cnt = 1
            for v in range(n_uav):
                if v == u or t >= len(sectors[v]):
                    continue
                if trajs[v][2][t] <= 0.0:   # v has nothing to upload at this step
                    continue
                if sectors[v][t] == sectors[u][t]:
                    cnt += 1
            f[t] = cnt
        factors.append(f)
    return factors


def closed_loop_evaluate(cfg, routes, points, data_sizes, model, field):
    """Per-slot contention simulation -> (makespan, delay, feasible).

    All UAVs take off simultaneously.  Trajectories are fixed (single-segment
    TD3); only the upload rate is contention-reduced (divided by the number of
    co-sector UAVs).  Consistent with the single-segment environment's hard
    constraint (``Q_u[n_{u,r+1}]=0``: upload must finish *before* the UAV reaches
    the goal), an edge whose payload is not fully uploaded by the segment's last
    step is marked *infeasible* — there is no post-arrival loitering here.
    """
    n_uav = len(routes)
    dt = float(cfg["delta_t"])
    trajs = [rollout_traj(cfg, r, points, data_sizes, model, field) for r in routes]
    sectors = [np.array([best_sector(p) for p in t[0]]) for t in trajs]
    factors = _co_sector_factors(trajs, sectors)

    total_delay = 0.0
    makespan = 0.0
    feasible = True
    for u in range(n_uav):
        _, edge, data, rate = trajs[u]
        fac = factors[u]
        uav_delay = 0.0
        uav_time = 0.0
        for r in np.unique(edge):
            idx = np.where(edge == r)[0]
            d_r = float(data[idx[0]])
            if d_r <= 0.0:                      # depot->point segment: nothing to upload
                uav_time += len(idx) * dt
                continue
            # upload at contention-reduced rate along the segment's steps
            uploaded = 0.0
            t_upload = 0.0
            for t in idx:
                r_eff = rate[t] / fac[t]        # Mbps, shared bandwidth
                uploaded += r_eff * dt
                t_upload += dt
                if uploaded >= d_r:
                    break
            # makespan: the trajectory is fixed (the policy does not change under
            # contention), so the flight time is simply the segment's duration.
            uav_time += len(idx) * dt
            if uploaded < d_r:
                # hard constraint violated: payload not uploaded before reaching
                # the goal (no post-arrival loitering in the single-segment env).
                feasible = False
            uav_delay += t_upload
        total_delay += uav_delay
        makespan = max(makespan, uav_time)

    return makespan, total_delay, feasible


def congestion_factors(cfg, routes, points, data_sizes, model, field):
    """Convenience: per-UAV average co-sector factor (for edge-weight scaling)."""
    trajs = [rollout_traj(cfg, r, points, data_sizes, model, field) for r in routes]
    sectors = [np.array([best_sector(p) for p in t[0]]) for t in trajs]
    factors = _co_sector_factors(trajs, sectors)
    return [float(f.mean()) for f in factors]


def iterative_alns(cfg, points, data_sizes, flight, delay, outage, feasible,
                   K, n_uav, T_ref, D_ref, model, field, rounds=3,
                   iters=2000, seed=42, wT=0.5, wD=0.5):
    """Resource-aware iterative ALNS (new design §7 closed loop).

    Round 1 runs the plain ALNS on single-UAV edge weights.  Each subsequent
    round evaluates the current routes with the per-slot contention simulator
    (``congestion_factors``), scales each UAV's delay by its average co-sector
    factor, and re-runs the ALNS from the previous solution — so the route
    search explicitly avoids co-sector resource conflicts instead of treating
    contention as a post-hoc multiplier.
    """
    from segment.alns import ALNS

    ini = np.asarray(cfg["ini_loc"], float)
    alns = ALNS(flight, delay, outage, feasible, K, n_uav, T_ref, D_ref,
                wT=wT, wD=wD)
    routes, obj = alns.run(points, ini, iters=iters, seed=seed)
    for _ in range(rounds - 1):
        facs = congestion_factors(cfg, routes, points, data_sizes, model, field)
        alns.delay_factors = facs
        routes, obj = alns.run(points, ini, iters=iters, seed=seed,
                               init_routes=routes)
    return routes, obj
