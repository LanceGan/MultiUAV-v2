"""Plot the single-segment metric comparison (TD3/DDPG/SAC/greedy).

Three bar subplots: feasible rate, average makespan, average delay.  Uses the
final settings (7-dim state, 15m arrival, eps_out=0.35).
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
FIG_DIR = os.path.join(ROOT, "results", "figs")
EPS_OUT = 0.35


def rollout_metrics(env, model, mi, ma):
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

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    cfg, points, data_sizes, _ = fixed_traverse(args.uav_num, args.seed)
    segs = all_pairs_segments(cfg, points, data_sizes)
    field = CommField(cfg, spacing=0.25)
    ma = np.array([float(cfg["v_max"]), 2.0 * math.pi], dtype=np.float32)
    mi = np.zeros(2, dtype=np.float32)

    names = ["TD3", "DDPG", "SAC"]
    classes = [TD3, DDPG, SAC]
    feas_rate = []
    makespan = []
    delay = []
    for algo, cls in zip(names, classes):
        model = cls(7, 2, ma, mi)
        model.load(os.path.join(MODEL_DIR, "uav%d_%s_best.pth" % (args.uav_num, algo)))
        feas = 0
        fl = []
        dl = []
        for (a, b, d) in segs:
            e = SingleSegmentEnv(cfg, a, b, d, field=field)
            ft, d_, ok = rollout_metrics(e, model, mi, ma)
            if ok:
                feas += 1
                fl.append(ft)
                dl.append(d_)
        feas_rate.append(100.0 * feas / len(segs))
        makespan.append(float(np.mean(fl)))
        delay.append(float(np.mean(dl)))

    # greedy straight-line (no feasibility; flight/delay only)
    gf, gd = [], []
    for (a, b, d) in segs:
        ft, dl = straight_edge(cfg, a, b, d)
        gf.append(ft)
        gd.append(dl)
    g_makespan = float(np.mean(gf))
    g_delay = float(np.mean(gd))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(3)

    axes[0].bar(x, feas_rate, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[0].set_xticks(x); axes[0].set_xticklabels(names)
    axes[0].set_ylabel("可行率 (%)")
    axes[0].set_title("可行率 (eps_out=0.35)")

    axes[1].bar(x, makespan, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1].axhline(g_makespan, ls="--", c="gray", label="greedy %.0fs" % g_makespan)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names)
    axes[1].set_ylabel("平均 makespan (s)")
    axes[1].set_title("平均航段飞行时间")
    axes[1].legend()

    axes[2].bar(x, delay, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[2].axhline(g_delay, ls="--", c="gray", label="greedy %.0fs" % g_delay)
    axes[2].set_xticks(x); axes[2].set_xticklabels(names)
    axes[2].set_ylabel("平均时延 (s)")
    axes[2].set_title("平均上传时延")
    axes[2].legend()

    fig.suptitle("单段轨迹算法对比（TD3 / DDPG / SAC vs 贪心直线）")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "segment_compare_metrics_uav%d.png" % args.uav_num)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved -> %s" % out)
    print("feasible %%: %s" % [round(f, 1) for f in feas_rate])
    print("makespan:   %s" % [round(m, 1) for m in makespan])
    print("delay:      %s" % [round(d, 1) for d in delay])
    print("greedy: makespan %.1f  delay %.1f" % (g_makespan, g_delay))


if __name__ == "__main__":
    main()
