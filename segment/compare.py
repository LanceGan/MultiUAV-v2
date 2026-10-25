"""Deterministic rollout comparison of TD3 / DDPG / SAC / greedy-straight-line.

Rolls out every one of the 420 directed edges with each trained policy (no
exploration noise), and reports the feasible rate (arrive + upload + route-level
outage <= eps_out) plus the average segment flight time and upload delay.
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
from segment.ddpg import DDPG
from segment.sac import SAC
from segment.alns import straight_edge
from segment.train import all_pairs_segments

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "results", "models", "segment")
EPS_OUT = 0.35


def rollout_edge_metrics(env, model, mi, ma):
    env.reset()
    steps = 0
    outage_time = 0.0
    buf_empty = 0 if env.data_size <= 0.0 else None
    while True:
        a = np.clip(model.select_action(env._state()), mi, ma)
        s_, r, done, info = env.step(a)
        steps += 1
        outage_time += info["outage"] * env.delta_t
        if buf_empty is None and info["data_buffer"] <= 0.0:
            buf_empty = steps
        if done:
            break
    ft = steps * env.delta_t
    dl = (buf_empty if buf_empty is not None else steps) * env.delta_t
    feas = bool(info["feasible"]) and (outage_time / max(ft, 1e-6) <= EPS_OUT)
    return ft, dl, feas


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

    print("%-8s %-14s %-12s %-12s" % ("algo", "feasible", "avg_flight(s)", "avg_delay(s)"))
    for algo, cls in [("TD3", TD3), ("DDPG", DDPG), ("SAC", SAC)]:
        model = cls(7, 2, ma, mi)
        model.load(os.path.join(MODEL_DIR, "uav%d_%s_best.pth" % (args.uav_num, algo)))
        feas = 0
        flights = []
        delays = []
        for (a, b, d) in segs:
            e = SingleSegmentEnv(cfg, a, b, d, field=field)
            ft, dl, ok = rollout_edge_metrics(e, model, mi, ma)
            if ok:
                feas += 1
                flights.append(ft)
                delays.append(dl)
        print("%-8s %3d/%-4d=%5.1f%%  %-12.1f %-12.1f"
              % (algo, feas, len(segs), 100.0 * feas / len(segs),
                 float(np.mean(flights)) if flights else 0.0,
                 float(np.mean(delays)) if delays else 0.0))

    # greedy straight-line (non-DRL baseline): no feasibility, just flight/delay.
    gf = []
    gd = []
    for (a, b, d) in segs:
        ft, dl = straight_edge(cfg, a, b, d)
        gf.append(ft)
        gd.append(dl)
    print("%-8s %-14s %-12.1f %-12.1f"
          % ("greedy", "-", float(np.mean(gf)), float(np.mean(gd))))
    return 0


if __name__ == "__main__":
    main()
