"""Multi-seed paired evaluation: base ALNS vs closed-loop iterative ALNS.

For each seed, runs BOTH the plain (resource-unaware) ALNS and the iterative
closed-loop ALNS, then evaluates each with the per-slot contention simulator.
Reports mean ± std and the paired improvement, which is the confidence-interval
comparison the GPT review flagged as missing.
"""
import warnings
warnings.warn("segment.run_multiseed is a legacy prototype; the formal pipeline lives in segment.stc (STC-ALNS).", DeprecationWarning, stacklevel=2)



import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.run_traj import fixed_traverse
from segment.env import CommField
from segment.td3 import TD3
from segment.alns import ALNS, load_weights, straight_line_baseline
from segment.closed_loop import iterative_alns, congestion_factors, closed_loop_evaluate

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "models", "segment")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav_num", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=2000)
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]

    cfg, points, data_sizes, _ = fixed_traverse(args.uav_num, 42)
    K = len(points)
    n_uav = args.uav_num
    flight, delay, outage, feas_arr = load_weights(
        os.path.join(MODEL_DIR, "edge_weights_uav%d.npz" % n_uav))
    T_ref, D_ref = straight_line_baseline(cfg, points, data_sizes, n_uav)

    field = CommField(cfg, spacing=0.25)
    ma = np.array([float(cfg["v_max"]), 2 * np.pi], np.float32)
    mi = np.zeros(2, np.float32)
    model = TD3(7, 2, ma, mi)
    model.load(os.path.join(MODEL_DIR, "uav%d_allpairs_best.pth" % n_uav))
    ini = np.asarray(cfg["ini_loc"], float)

    print("多 seed 配对评估（N=%d, %d seeds）: base ALNS vs 闭环迭代 ALNS" % (n_uav, len(seeds)))
    base = {"mk": [], "dl": [], "fac": []}
    ite = {"mk": [], "dl": [], "fac": []}
    for s in seeds:
        alns = ALNS(flight, delay, outage, feas_arr, K, n_uav, T_ref, D_ref)
        b_routes, _ = alns.run(points, ini, iters=args.iters, seed=s)
        bT, bD, _ = closed_loop_evaluate(cfg, b_routes, points, data_sizes, model, field)
        bfac = np.mean(congestion_factors(cfg, b_routes, points, data_sizes, model, field))

        routes, _ = iterative_alns(cfg, points, data_sizes, flight, delay, outage,
                                   feas_arr, K, n_uav, T_ref, D_ref, model, field,
                                   rounds=args.rounds, iters=args.iters, seed=s)
        T, D, _ = closed_loop_evaluate(cfg, routes, points, data_sizes, model, field)
        fac = np.mean(congestion_factors(cfg, routes, points, data_sizes, model, field))

        base["mk"].append(bT); base["dl"].append(bD); base["fac"].append(bfac)
        ite["mk"].append(T); ite["dl"].append(D); ite["fac"].append(fac)
        print("  seed %d: base(mk %.0f dl %.0f f %.2f) -> iter(mk %.0f dl %.0f f %.2f)"
              % (s, bT, bD, bfac, T, D, fac))

    def stat(x):
        return float(np.mean(x)), float(np.std(x))

    bm, bms = stat(base["mk"]); bd, bds = stat(base["dl"]); bf, bfs = stat(base["fac"])
    im, ims = stat(ite["mk"]); idl, idls = stat(ite["dl"]); iff, ifs = stat(ite["fac"])

    print("\n=== base vs 迭代（mean ± std，%d seeds）===" % len(seeds))
    print("makespan:  base %.1f ± %.1f s  ->  迭代 %.1f ± %.1f s   (%.1f%%)"
          % (bm, bms, im, ims, 100 * (im / bm - 1)))
    print("delay:     base %.1f ± %.1f s  ->  迭代 %.1f ± %.1f s   (%.1f%%)"
          % (bd, bds, idl, idls, 100 * (idl / bd - 1)))
    print("争用因子:  base %.3f ± %.3f  ->  迭代 %.3f ± %.3f   (%.1f%%)"
          % (bf, bfs, iff, ifs, 100 * (iff / bf - 1)))
    return 0


if __name__ == "__main__":
    main()
