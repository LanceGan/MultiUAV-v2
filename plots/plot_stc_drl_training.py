"""从 results/models/stc_drl/*_rewards.npz 生成 DRL 单段重训曲线图。

纯读取训练落盘的 npz（ep_rewards / ep_feasible / eval_heldout_feasible），
不含任何硬编码指标值。输出两张图：
  1. stc_drl_reward_curves.png      —— 三算法 episode 奖励（原始淡色 + 滑动平均）
  2. stc_drl_feasibility_curves.png —— 训练滑窗可行率 + held-out 评估可行率
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALGOS = ("sac", "td3", "ddpg")
COLORS = {"sac": "tab:blue", "td3": "tab:red", "ddpg": "tab:green"}


def _moving_avg(x, w=100):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def load_curves(models_dir, run_id):
    out = {}
    for algo in ALGOS:
        p = Path(models_dir) / ("%s_u2_%s_rewards.npz" % (algo, run_id))
        if not p.is_file():
            continue
        d = np.load(p)
        out[algo] = {
            "ep_rewards": np.asarray(d["ep_rewards"], dtype=float),
            "ep_feasible": np.asarray(d["ep_feasible"], dtype=float),
            "heldout": np.asarray(d["eval_heldout_feasible"], dtype=float),
        }
    return out


def render(curves, out_dir, window=100):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for algo, c in curves.items():
        r = c["ep_rewards"]
        ax.plot(np.arange(len(r)), r, color=COLORS[algo], alpha=0.15, lw=0.6)
        ma = _moving_avg(r, window)
        ax.plot(np.arange(window - 1, window - 1 + len(ma)), ma,
                color=COLORS[algo], lw=1.8, label=algo.upper())
    ax.set_xlabel("episode")
    ax.set_ylabel("episode reward")
    ax.set_title("SI single-segment DRL training reward (raw + %d-ep moving avg)"
                 % window)
    ax.legend()
    p = out_dir / "stc_drl_reward_curves.png"
    fig.savefig(p, bbox_inches="tight", dpi=130)
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for algo, c in curves.items():
        f = c["ep_feasible"]
        ma = _moving_avg(f, window)
        ax.plot(np.arange(window - 1, window - 1 + len(ma)), 100 * ma,
                color=COLORS[algo], lw=1.6, label="%s train (%d-ep)"
                % (algo.upper(), window))
        h = c["heldout"]
        if h.ndim == 2 and len(h):
            ax.plot(h[:, 0], 100 * h[:, 1], color=COLORS[algo], lw=1.2,
                    ls="--", marker="o", ms=2.5,
                    label="%s held-out" % algo.upper())
    ax.set_xlabel("episode")
    ax.set_ylabel("feasible rate (%)")
    ax.set_ylim(0, 102)
    ax.set_title("Training-window vs held-out feasibility")
    ax.legend(fontsize=8)
    p = out_dir / "stc_drl_feasibility_curves.png"
    fig.savefig(p, bbox_inches="tight", dpi=130)
    plt.close(fig)
    written.append(p)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", default="results/models/stc_drl")
    ap.add_argument("--run-id", default="si-v1")
    ap.add_argument("--out-dir", default="results/figs")
    ap.add_argument("--window", type=int, default=100)
    a = ap.parse_args(argv)
    curves = load_curves(a.models_dir, a.run_id)
    if not curves:
        ap.error("no *_rewards.npz found under %s for run id %r"
                 % (a.models_dir, a.run_id))
    written = render(curves, a.out_dir, a.window)
    for p in written:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
