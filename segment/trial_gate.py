"""P3-3 trial gate: does the single-segment TD3 beat the legal baselines?

Compares, on the *same* N=2 all-pairs edge set and the *same* radio map, the
single-segment TD3 against:
  - Straight-MaxSpeed  (relaxation lower bound)
  - Straight-Hold      (legal naive baseline, hold before 15 m)
  - Comm-Aware Waypoint (discrete relay-point search)

Metrics: average makespan, average upload delay, and the objective
``wT*T + wD*D``.  This is the small single-seed gate that decides whether the
TD3/ALNS method is worth pursuing before any large-scale multi-seed run.
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.run_traj import fixed_traverse
from segment.env import CommField, SingleSegmentEnv
from segment.td3 import TD3
from segment.train import all_pairs_segments
from segment.baselines import straight_maxspeed, straight_hold, comm_aware_waypoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "results", "models", "segment")


def td3_edge(env, model, mi, ma):
    """Roll out one edge with the TD3 policy -> (flight, delay, feasible)."""
    env.reset()
    steps = 0
    buf_empty = 0 if env.data_size <= 0.0 else None
    while True:
        a = np.clip(model.select_action(env._state()), mi, ma)
        _, _, done, info = env.step(a)
        steps += 1
        if buf_empty is None and info["data_buffer"] <= 0.0:
            buf_empty = steps
        if done:
            break
    flight = steps * env.delta_t
    delay = (buf_empty if buf_empty is not None else steps) * env.delta_t
    return flight, delay, bool(info["feasible"])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav_num", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wT", type=float, default=0.5)
    parser.add_argument("--wD", type=float, default=0.5)
    args = parser.parse_args(argv)

    cfg, points, data_sizes, _ = fixed_traverse(args.uav_num, args.seed)
    segs = all_pairs_segments(cfg, points, data_sizes)
    field = CommField(cfg, spacing=0.25)
    ma = np.array([float(cfg["v_max"]), 2.0 * math.pi], np.float32)
    mi = np.zeros(2, np.float32)
    model = TD3(7, 2, ma, mi)
    model.load(os.path.join(MODEL_DIR, "uav%d_TD3_best.pth" % args.uav_num))

    # collect per-edge (makespan, delay, feasible) for each method
    methods = ["TD3", "Straight-MaxSpeed", "Straight-Hold", "Comm-Waypoint"]
    mk = {m: [] for m in methods}
    dl = {m: [] for m in methods}
    fe = {m: [] for m in methods}
    for (a, b, d) in segs:
        # TD3 -> (flight, delay, feasible)
        e = SingleSegmentEnv(cfg, a, b, d, field=field)
        f_td, d_td, feas_td = td3_edge(e, model, mi, ma)
        mk["TD3"].append(f_td); dl["TD3"].append(d_td); fe["TD3"].append(feas_td)
        # baselines -> (flight, upload, makespan, feasible)
        fl, up, m, f = straight_maxspeed(cfg, field, a, b, d)
        mk["Straight-MaxSpeed"].append(m); dl["Straight-MaxSpeed"].append(up); fe["Straight-MaxSpeed"].append(f)
        fl, up, m, f = straight_hold(cfg, field, a, b, d)
        mk["Straight-Hold"].append(m); dl["Straight-Hold"].append(up); fe["Straight-Hold"].append(f)
        fl, up, m, f = comm_aware_waypoint(cfg, field, a, b, d)
        mk["Comm-Waypoint"].append(m); dl["Comm-Waypoint"].append(up); fe["Comm-Waypoint"].append(f)

    print("=== 单段门控（N=%d, %d 边, wT=%.1f wD=%.1f）===" % (args.uav_num, len(segs), args.wT, args.wD))
    print("%-16s %-14s %-14s %-10s %-12s" % ("method", "avg_makespan", "avg_delay", "feasible%", "objective"))
    objs = {}
    for m in methods:
        mkm = np.mean(mk[m]); mdl = np.mean(dl[m]); mfe = 100 * np.mean(fe[m])
        # objective = wT*T + wD*D (unnormalised, relative magnitudes only)
        obj = args.wT * mkm + args.wD * mdl
        objs[m] = obj
        print("%-16s %-14.1f %-14.1f %-10.1f %-12.2f" % (m, mkm, mdl, mfe, obj))
    td3_obj = objs["TD3"]
    print("\n（objective = wT*makespan + wD*delay，越低越好）")
    for m in ["Straight-Hold", "Comm-Waypoint"]:
        print("TD3 vs %s: objective %.2f vs %.2f (TD3 相对 %.1f%%)"
              % (m, td3_obj, objs[m], 100 * (td3_obj / objs[m] - 1)))
    return 0


if __name__ == "__main__":
    main()
