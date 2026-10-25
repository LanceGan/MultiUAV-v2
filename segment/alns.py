"""Multi-neighborhood search (with simulated annealing) for the outer
combinatorial layer (assignment + visit sequence).

NOTE: this is NOT a strict *adaptive* LNS — the operators are selected uniformly
at random, with no operator scoring / adaptive weights / segment update periods.
It optimises which UAV visits which inspection points, and in what order, using
the pre-computed single-segment edge weights (flight time, upload delay, outage,
feasibility) from ``segment/precompute.py``.  The objective is the new design's
normalised ``min wT*T/T_ref + wD*D/D_ref``, normalised by a greedy straight-line
(no comm-awareness) baseline.  The route-level outage QoS constraint
(``O_u/T_u <= eps_out``, §9) is enforced in ``route_cost``.

Takeoff-time staggering (§11) is left as a future extension.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.run_traj import fixed_traverse
from radio_map import radio_map_A2G
from segment.env import BANDWIDTH, UAV_H

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "results", "models", "segment")


# ---------------------------------------------------------------------------
# Edge weights (pre-computed TD3 trajectories)
# ---------------------------------------------------------------------------
def load_weights(path):
    d = np.load(path)
    return d["flight"], d["delay"], d["outage"], d["feasible"]


# ---------------------------------------------------------------------------
# Straight-line (no comm-awareness) baseline
# ---------------------------------------------------------------------------
def _straight_rate(cfg, a, b, n=10):
    """Average upload rate (Mbps) sampled at ``n`` points along the straight
    segment (more accurate than a single midpoint sample)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    pts = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
    rates = []
    for p in pts:
        loc = np.array([[p[0] / 10.0, p[1] / 10.0, UAV_H / 10.0]])
        sinr = float(radio_map_A2G.getPointDateRate(loc))
        rates.append(BANDWIDTH * math.log2(1.0 + 10.0 ** (sinr / 10.0)))
    return float(np.mean(rates))


def straight_edge(cfg, a, b, data):
    dist = float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)))
    flight = dist / float(cfg["v_max"])
    rate = max(_straight_rate(cfg, a, b), 1e-6)
    return flight, data / rate


def greedy_nearest_neighbor(points, start):
    """Return a greedy nearest-neighbor order of point indices (single route)."""
    remaining = list(range(len(points)))
    order = []
    cur = start
    while remaining:
        nxt = min(remaining, key=lambda j: np.linalg.norm(points[j] - cur))
        order.append(nxt)
        cur = points[nxt]
        remaining.remove(nxt)
    return order


def straight_line_baseline(cfg, points, data_sizes, n_uav):
    """Greedy NN single route split into ``n_uav`` contiguous groups -> (T_ref, D_ref)."""
    ini = np.asarray(cfg["ini_loc"], dtype=float)
    end = np.asarray(cfg["end_loc"], dtype=float)
    order = greedy_nearest_neighbor(points, ini)
    groups = np.array_split(order, n_uav)

    max_t = 0.0
    total_d = 0.0
    for g in groups:
        g = list(g)
        if not g:
            continue
        ft, dl = straight_edge(cfg, ini, points[g[0]], 0.0)
        total_d += dl
        for r in range(len(g) - 1):
            f, d = straight_edge(cfg, points[g[r]], points[g[r + 1]],
                                 data_sizes[g[r]])
            ft += f
            total_d += d
        f, d = straight_edge(cfg, points[g[-1]], end, data_sizes[g[-1]])
        ft += f
        total_d += d
        max_t = max(max_t, ft)
    return max_t, total_d


# ---------------------------------------------------------------------------
# ALNS
# ---------------------------------------------------------------------------
class ALNS(object):
    def __init__(self, flight, delay, outage, feasible, K, n_uav, T_ref, D_ref,
                 wT=0.5, wD=0.5, eps_out=0.35):
        self.flight = flight    # (K+1, K+1): row = src+1 (depot -1 -> 0), col = dst (end K -> K)
        self.delay = delay
        self.outage = outage
        self.feasible = feasible
        self.K = K
        self.n_uav = n_uav
        self.T_ref = T_ref
        self.D_ref = D_ref
        self.wT = wT
        self.wD = wD
        self.eps_out = eps_out
        self.delay_factors = [1.0] * n_uav  # per-UAV contention scaling (iterative ALNS)

    # -- cost --------------------------------------------------------------
    def route_cost(self, route):
        """(flight_time, delay, feasible) for one route [p1, ..., pk].

        ``feasible`` = every edge feasible AND the route-level outage QoS
        constraint holds (``O_u / T_u <= eps_out``, new design §9).
        """
        if not route:
            return 0.0, 0.0, True
        ft = self.flight[0, route[0]]
        dl = self.delay[0, route[0]]
        ot = self.outage[0, route[0]]
        feas = bool(self.feasible[0, route[0]])
        for r in range(len(route) - 1):
            i, j = route[r], route[r + 1]
            ft += self.flight[i + 1, j]
            dl += self.delay[i + 1, j]
            ot += self.outage[i + 1, j]
            feas = feas and bool(self.feasible[i + 1, j])
        ft += self.flight[route[-1] + 1, self.K]
        dl += self.delay[route[-1] + 1, self.K]
        ot += self.outage[route[-1] + 1, self.K]
        feas = feas and bool(self.feasible[route[-1] + 1, self.K])
        feas = feas and (ot / max(ft, 1e-6) <= self.eps_out)
        return ft, dl, feas

    def solution_cost(self, routes):
        max_t = 0.0
        total_d = 0.0
        feas = True
        for u, r in enumerate(routes):
            ft, dl, f = self.route_cost(r)
            max_t = max(max_t, ft)
            total_d += dl * self.delay_factors[u]  # per-UAV contention scaling
            feas = feas and f
        return max_t, total_d, feas

    def objective(self, routes):
        T, D, feas = self.solution_cost(routes)
        if not feas:
            return float("inf")
        return self.wT * T / self.T_ref + self.wD * D / self.D_ref

    # -- initial solution --------------------------------------------------
    def initial_solution(self, points, ini, rng):
        # Cheapest-insertion construction (better than greedy NN): seed each
        # route with a feasible point, then repeatedly insert the point into the
        # position that minimises the objective (feasibility-first).
        remaining = list(range(self.K))
        routes = [[] for _ in range(self.n_uav)]
        for u in range(self.n_uav):
            seeds = [j for j in remaining
                     if bool(self.feasible[0, j]) and bool(self.feasible[j + 1, self.K])]
            if not seeds:
                seeds = [j for j in remaining if bool(self.feasible[0, j])] or remaining
            nxt = min(seeds, key=lambda j: np.linalg.norm(points[j] - ini))
            routes[u].append(nxt)
            remaining.remove(nxt)
        while remaining:
            best = None
            for j in remaining:
                for u in range(self.n_uav):
                    for pos in range(len(routes[u]) + 1):
                        cand = [list(r) for r in routes]
                        cand[u].insert(pos, j)
                        obj = self.objective(cand)
                        if best is None or obj < best[0]:
                            best = (obj, u, pos, j)
            _, u, pos, j = best
            routes[u].insert(pos, j)
            remaining.remove(j)
        return routes

    # -- operators ---------------------------------------------------------
    def destroy(self, routes, q, rng):
        flat = [(u, pos, routes[u][pos])
                for u in range(self.n_uav) for pos in range(len(routes[u]))]
        removed = rng.choice(len(flat), size=min(q, len(flat)), replace=False)
        removed = [flat[i] for i in sorted(removed, reverse=True)]
        removed_pts = []
        for u, pos, pt in removed:
            del routes[u][pos]
            removed_pts.append(pt)
        return removed_pts

    def two_opt(self, routes, rng):
        """Reverse a random subsequence of a random route (2-opt move)."""
        u = rng.randint(self.n_uav)
        route = routes[u]
        if len(route) < 3:
            return
        i = rng.randint(0, len(route) - 2)
        j = rng.randint(i + 2, len(route) + 1)
        route[i:j] = reversed(route[i:j])

    def swap(self, routes, rng):
        """Swap two random points (within or across routes)."""
        positions = [(u, i) for u in range(self.n_uav)
                     for i in range(len(routes[u]))]
        if len(positions) < 2:
            return
        idx = rng.choice(len(positions), size=2, replace=False)
        (u1, i1), (u2, i2) = positions[idx[0]], positions[idx[1]]
        routes[u1][i1], routes[u2][i2] = routes[u2][i2], routes[u1][i1]

    def repair_best(self, routes, removed_pts, rng):
        for pt in removed_pts:
            best = None
            for u in range(self.n_uav):
                for pos in range(len(routes[u]) + 1):
                    cand = [list(r) for r in routes]
                    cand[u].insert(pos, pt)
                    obj = self.objective(cand)
                    if best is None or obj < best[0]:
                        best = (obj, u, pos)
            _, u, pos = best
            routes[u].insert(pos, pt)

    # -- main loop ---------------------------------------------------------
    def run(self, points, ini, iters=2000, seed=42, T0=0.1, c=0.995,
            init_routes=None):
        rng = np.random.RandomState(seed)
        if init_routes is not None:
            routes = [list(r) for r in init_routes]
        else:
            routes = self.initial_solution(points, ini, rng)
        cur_obj = self.objective(routes)
        best_routes = [list(r) for r in routes]
        best_obj = cur_obj
        T = T0
        for it in range(iters):
            cand = [list(r) for r in routes]
            op = rng.choice(["destroy_repair", "two_opt", "swap"])
            if op == "destroy_repair":
                q = rng.randint(1, min(5, self.K) + 1)
                removed = self.destroy(cand, q, rng)
                self.repair_best(cand, removed, rng)
            elif op == "two_opt":
                self.two_opt(cand, rng)
            else:
                self.swap(cand, rng)
            new_obj = self.objective(cand)
            if new_obj < cur_obj or rng.rand() < math.exp(-(new_obj - cur_obj) / max(T, 1e-9)):
                routes = cand
                cur_obj = new_obj
                if new_obj < best_obj:
                    best_routes = [list(r) for r in routes]
                    best_obj = new_obj
            T *= c
        return best_routes, best_obj


# ---------------------------------------------------------------------------
def smooth_trajectory(px, py, s=0.5):
    """Fit a B-spline through a trajectory to remove TD3 waviness."""
    from scipy.interpolate import splprep, splev
    px = np.asarray(px, float)
    py = np.asarray(py, float)
    keep = np.ones(len(px), dtype=bool)
    for i in range(1, len(px)):
        if np.hypot(px[i] - px[i - 1], py[i] - py[i - 1]) < 1e-6:
            keep[i] = False
    px, py = px[keep], py[keep]
    if len(px) < 4:
        return px, py
    k = min(3, len(px) - 1)
    tck, u = splprep([px, py], s=s * len(px), k=k)
    u_new = np.linspace(0.0, 1.0, max(len(px) * 2, 20))
    sx, sy = splev(u_new, tck)
    return sx, sy


def plot_routes(routes, cfg, points, data_sizes, model_path, out_path):
    """Plot the ALNS routes (single-segment TD3 trajectories) over the G2A
    outage + A2G SINR maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from radio_map import radio_map_A2G, radio_map_G2A
    from trajectory.env import DEFAULT_UAV_H
    from segment.env import CommField, SingleSegmentEnv
    from segment.td3 import TD3

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    UNIT_TO_KM = 0.1
    UAV_H_KM = DEFAULT_UAV_H * UNIT_TO_KM
    GRID_RES = 60

    extent_km = float(cfg["length"]) * UNIT_TO_KM
    xs = np.linspace(0.0, extent_km, GRID_RES)
    ys = np.linspace(0.0, extent_km, GRID_RES)
    xx, yy = np.meshgrid(xs, ys)
    locs = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, UAV_H_KM)], axis=1)
    _, g2a_vals = radio_map_G2A.getPointMiniOutage(locs)
    _, a2g_vals = radio_map_A2G.getPointMiniOutage(locs)
    g2a_grid = np.asarray(g2a_vals, dtype=float).reshape(GRID_RES, GRID_RES)
    a2g_grid = np.asarray(a2g_vals, dtype=float).reshape(GRID_RES, GRID_RES)

    field = CommField(cfg, spacing=0.25)
    ma = np.array([float(cfg["v_max"]), 2.0 * math.pi], dtype=np.float32)
    mi = np.zeros(2, dtype=np.float32)
    model = TD3(7, 2, ma, mi)
    model.load(model_path)

    ini = np.asarray(cfg["ini_loc"], dtype=float)
    end = np.asarray(cfg["end_loc"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    ax = axes[0]
    cf = ax.contourf(xs, ys, g2a_grid, levels=np.linspace(0.0, 1.0, 11), cmap="Reds")
    fig.colorbar(cf, ax=ax, label="G2A 中断概率")
    ax = axes[1]
    cf = ax.contourf(xs, ys, a2g_grid, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="A2G SINR (dB)")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(routes), 1)))
    for u, route in enumerate(routes):
        route = [int(j) for j in route]
        wps = [ini] + [np.asarray(points[j], dtype=float) for j in route] + [end]
        datas = [0.0] + [float(data_sizes[j]) for j in route]
        for r in range(len(wps) - 1):
            e = SingleSegmentEnv(cfg, wps[r], wps[r + 1], datas[r], field=field)
            e.reset()
            px, py = [e.pos[0]], [e.pos[1]]
            while True:
                a = np.clip(model.select_action(e._state()), mi, ma)
                s_, rr, done, info = e.step(a)
                px.append(e.pos[0])
                py.append(e.pos[1])
                if done:
                    break
            for ax in axes:
                # white halo under the coloured line so it stands out on any map
                ax.plot(np.asarray(px) * UNIT_TO_KM, np.asarray(py) * UNIT_TO_KM,
                        color="white", lw=3.2, zorder=3)
                ax.plot(np.asarray(px) * UNIT_TO_KM, np.asarray(py) * UNIT_TO_KM,
                        color=colors[u], lw=1.8, zorder=4,
                        label=("UAV %d" % u) if r == 0 else None)

    for ax in axes:
        for j in range(len(points)):
            ax.plot(points[j][0] * UNIT_TO_KM, points[j][1] * UNIT_TO_KM,
                    "ko", ms=3, zorder=5)
        ax.plot(ini[0] * UNIT_TO_KM, ini[1] * UNIT_TO_KM, "r^", ms=10, label="depot")
        ax.plot(end[0] * UNIT_TO_KM, end[1] * UNIT_TO_KM, "rs", ms=10, label="end")
        ax.set_xlim(0, extent_km)
        ax.set_ylim(0, extent_km)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")

    axes[0].set_title("G2A 中断图 + ALNS 路线")
    axes[1].set_title("A2G SINR 图 + ALNS 路线")
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav_num", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--wT", type=float, default=0.5)
    parser.add_argument("--wD", type=float, default=0.5)
    parser.add_argument("--iterative", action="store_true",
                        help="use the closed-loop iterative ALNS (resource-aware)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="iterative ALNS rounds")
    args = parser.parse_args(argv)

    cfg, points, data_sizes, _ = fixed_traverse(args.uav_num, args.seed)
    K = len(points)

    flight, delay, outage, feas = load_weights(
        os.path.join(MODEL_DIR, "edge_weights_uav%d.npz" % args.uav_num))
    T_ref, D_ref = straight_line_baseline(cfg, points, data_sizes, args.uav_num)

    alns = ALNS(flight, delay, outage, feas, K, args.uav_num, T_ref, D_ref,
                wT=args.wT, wD=args.wD, eps_out=0.35)

    ini = np.asarray(cfg["ini_loc"], float)
    # Baseline: greedy nearest-neighbor route (the ALNS's own start point),
    # evaluated with the TD3 edge weights.
    base_routes = alns.initial_solution(points, ini, np.random.RandomState(args.seed))
    base_T, base_D, base_feas = alns.solution_cost(base_routes)
    base_obj = alns.objective(base_routes)
    print("baseline (greedy NN, TD3 weights): T=%.1f s  D=%.1f s  obj=%.4f  feasible=%s"
          % (base_T, base_D, base_obj, base_feas))

    best_routes, best_obj = alns.run(points, ini, iters=args.iters, seed=args.seed)

    if args.iterative:
        # Closed-loop resource-aware refinement: scale per-UAV delay by the
        # co-sector contention factor and re-run the ALNS (see closed_loop.py).
        from segment.closed_loop import iterative_alns, congestion_factors
        from segment.env import CommField
        from segment.td3 import TD3
        field = CommField(cfg, spacing=0.25)
        ma = np.array([float(cfg["v_max"]), 2.0 * math.pi], np.float32)
        mi = np.zeros(2, np.float32)
        model = TD3(7, 2, ma, mi)
        model.load(os.path.join(MODEL_DIR, "uav%d_allpairs_best.pth" % args.uav_num))
        facs_before = congestion_factors(cfg, best_routes, points, data_sizes, model, field)
        best_routes, best_obj = iterative_alns(
            cfg, points, data_sizes, flight, delay, outage, feas, K, args.uav_num,
            T_ref, D_ref, model, field, rounds=args.rounds, iters=args.iters,
            seed=args.seed, wT=args.wT, wD=args.wD)
        facs_after = congestion_factors(cfg, best_routes, points, data_sizes, model, field)
        print("iterative: avg co-sector factor %.3f -> %.3f"
              % (float(np.mean(facs_before)), float(np.mean(facs_after))))

    T, D, feas = alns.solution_cost(best_routes)
    print("baseline T_ref=%.1f s  D_ref=%.1f s" % (T_ref, D_ref))
    print("best objective = %.4f" % best_obj)
    print("best T (makespan) = %.1f s   D (delay) = %.1f s   feasible=%s" % (T, D, feas))
    print("baseline -> ALNS: obj %.4f -> %.4f   (T %.1f->%.1f s   D %.1f->%.1f s)"
          % (base_obj, best_obj, base_T, T, base_D, D))
    for u, r in enumerate(best_routes):
        print("  UAV %d: %s" % (u, [int(j) for j in r]))

    # Save routes + plot the trajectories over the radio maps.
    route_path = os.path.join(MODEL_DIR, "alns_routes_uav%d.json" % args.uav_num)
    with open(route_path, "w") as f:
        json.dump([[int(j) for j in r] for r in best_routes], f)
    fig_path = os.path.join(ROOT, "results", "figs", "alns_routes_uav%d.png" % args.uav_num)
    plot_routes(best_routes, cfg, points, data_sizes,
                os.path.join(MODEL_DIR, "uav%d_allpairs_best.pth" % args.uav_num), fig_path)
    print("saved routes -> %s" % route_path)
    print("saved figure -> %s" % fig_path)
    return 0


if __name__ == "__main__":
    main()
