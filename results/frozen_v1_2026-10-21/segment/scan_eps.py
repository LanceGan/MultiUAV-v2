"""Scan eps_out: feasible-rate vs outage-ratio tradeoff for the TD3 policy.

Rolls out all 420 directed edges once, then applies different route-level
outage thresholds post-hoc and reports, for each eps_out, the feasible rate
(arrive + upload + avg outage ratio <= eps_out) and the average outage ratio of
the feasible edges.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "results", "models", "segment")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav_num", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cfg, points, data_sizes, _ = fixed_traverse(args.uav_num, args.seed)
    segs = all_pairs_segments(cfg, points, data_sizes)
    field = CommField(cfg, spacing=0.25)
    ma = np.array([float(cfg["v_max"]), 2.0 * math.pi], dtype=np.float32)
    mi = np.zeros(2, dtype=np.float32)
    model = TD3(7, 2, ma, mi)
    model.load(os.path.join(MODEL_DIR, "uav2_TD3_best.pth"))

    metrics = []  # (arrive_feasible, outage_ratio)
    for (a, b, d) in segs:
        e = SingleSegmentEnv(cfg, a, b, d, field=field)
        e.reset()
        steps = 0
        outage_time = 0.0
        while True:
            ac = np.clip(model.select_action(e._state()), mi, ma)
            s_, r, done, info = e.step(ac)
            steps += 1
            outage_time += info["outage"] * e.delta_t
            if done:
                break
        ft = steps * e.delta_t
        outage_ratio = outage_time / max(ft, 1e-6)
        metrics.append((bool(info["feasible"]), outage_ratio))

    print("%-8s %-14s %-14s" % ("eps_out", "feasible", "avg_outage_ratio"))
    for eps in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 1.0]:
        feas = [m for m in metrics if m[0] and m[1] <= eps]
        n = len(feas)
        avg_ratio = float(np.mean([m[1] for m in feas])) if feas else 0.0
        print("%-8.2f %3d/420 = %5.1f%%  %.4f"
              % (eps, n, 100.0 * n / len(metrics), avg_ratio))
    return 0


if __name__ == "__main__":
    main()
