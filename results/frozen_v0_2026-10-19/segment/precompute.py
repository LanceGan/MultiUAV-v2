"""Pre-compute the directed edge weights for the ALNS (new design).

Uses the trained goal/resource-conditioned TD3 policy to roll out every ordered
pair (depot->point, point->point, point->end) and record the segment metrics:
flight time, upload delay (tau), cumulative outage time, and feasibility.  This
is the "预计算边权" step: the inner DRL supplies the edge costs that the outer
ALNS combinatorial search optimises over.

Key convention for ``weights``: ``-1`` = depot (source only), ``0..K-1`` = point
indices, ``K`` = end (destination only).  ``weights[(i, j)]`` is the metrics of
the directed edge ``i -> j``.
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.run_traj import fixed_traverse
from segment.env import SingleSegmentEnv, CommField
from segment.td3 import TD3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "results", "models", "segment")


def rollout_edge(env, model, min_action, max_action):
    """Deterministic rollout of one segment -> metrics dict."""
    env.reset()
    steps = 0
    buf_empty_step = 0 if env.data_size <= 0.0 else None
    outage_time = 0.0
    while True:
        a = np.clip(model.select_action(env._state()), min_action, max_action)
        s_, r, done, info = env.step(a)
        steps += 1
        outage_time += info["outage"] * env.delta_t
        if buf_empty_step is None and info["data_buffer"] <= 0.0:
            buf_empty_step = steps
        if done:
            break
    dt = env.delta_t
    return {
        "flight_time": steps * dt,
        "delay": (buf_empty_step if buf_empty_step is not None else steps) * dt,
        "outage_time": outage_time,
        "feasible": bool(info["feasible"]),
        "steps": steps,
    }


def precompute(cfg, points, data_sizes, model_path, spacing=0.25):
    """Return ``weights[(src, dst)]`` for every directed edge."""
    field = CommField(cfg, spacing=spacing)
    ini = np.asarray(cfg["ini_loc"], dtype=float)
    end = np.asarray(cfg["end_loc"], dtype=float)
    pts = [np.asarray(p, dtype=float) for p in points]
    K = len(pts)

    ma = np.array([float(cfg["v_max"]), 2.0 * math.pi], dtype=np.float32)
    mi = np.zeros(2, dtype=np.float32)
    model = TD3(6, 2, ma, mi)
    model.load(model_path)

    weights = {}
    for j in range(K):                       # depot -> point_j (payload 0)
        e = SingleSegmentEnv(cfg, ini, pts[j], 0.0, field=field)
        weights[(-1, j)] = rollout_edge(e, model, mi, ma)
    for i in range(K):                       # point_i -> point_j (source data)
        for j in range(K):
            if i == j:
                continue
            e = SingleSegmentEnv(cfg, pts[i], pts[j], float(data_sizes[i]),
                                 field=field)
            weights[(i, j)] = rollout_edge(e, model, mi, ma)
    for i in range(K):                       # point_i -> end (source data)
        e = SingleSegmentEnv(cfg, pts[i], end, float(data_sizes[i]), field=field)
        weights[(i, K)] = rollout_edge(e, model, mi, ma)
    return weights


def save_weights(weights, K, out_path):
    """Save the weight table as dense .npz arrays for the ALNS."""
    n_src = K + 1          # depot (-1 -> row 0) + K points (rows 1..K)
    n_dst = K + 1          # K points (cols 0..K-1) + end (K -> col K)
    flight = np.full((n_src, n_dst), np.nan)
    delay = np.full((n_src, n_dst), np.nan)
    outage = np.full((n_src, n_dst), np.nan)
    feas = np.zeros((n_src, n_dst), dtype=bool)
    for (i, j), m in weights.items():
        ri = i + 1         # depot -1 -> 0, point i -> i+1
        cj = j             # point j -> j, end K -> K
        flight[ri, cj] = m["flight_time"]
        delay[ri, cj] = m["delay"]
        outage[ri, cj] = m["outage_time"]
        feas[ri, cj] = m["feasible"]
    np.savez(out_path, flight=flight, delay=delay, outage=outage, feasible=feas)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav_num", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="uav2_allpairs.pth")
    parser.add_argument("--spacing", type=float, default=0.25)
    args = parser.parse_args(argv)

    cfg, points, data_sizes, _ = fixed_traverse(args.uav_num, args.seed)
    model_path = os.path.join(MODEL_DIR, args.model)
    weights = precompute(cfg, points, data_sizes, model_path, args.spacing)
    K = len(points)

    out = os.path.join(MODEL_DIR, "edge_weights_uav%d.npz" % args.uav_num)
    save_weights(weights, K, out)

    feas = [m["feasible"] for m in weights.values()]
    ft = [m["flight_time"] for m in weights.values()]
    dl = [m["delay"] for m in weights.values()]
    print("saved -> %s" % out)
    print("edges: %d  feasible: %d/%d = %.1f%%"
          % (len(weights), sum(feas), len(feas), 100.0 * sum(feas) / len(feas)))
    print("flight_time: min %.1f  median %.1f  max %.1f (s)"
          % (min(ft), float(np.median(ft)), max(ft)))
    print("delay:       min %.1f  median %.1f  max %.1f (s)"
          % (min(dl), float(np.median(dl)), max(dl)))
    return 0


if __name__ == "__main__":
    main()
