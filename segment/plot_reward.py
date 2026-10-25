"""Plot the single-segment training reward + feasibility curves (TD3/DDPG/SAC).

Reads ``results/logs/UAV_2_{algo}_rewards.npz`` (ep_rewards, ep_feasible) and
produces two smoothed subplots: per-episode reward (rolling mean) and cumulative
feasibility rate.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "results", "logs")
FIG_DIR = os.path.join(ROOT, "results", "figs")


def rolling_mean(x, w):
    if len(x) < w:
        return x
    c = np.convolve(x, np.ones(w) / w, mode="valid")
    return c


def main():
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    algos = ["TD3", "DDPG", "SAC"]
    colors = {"TD3": "#1f77b4", "DDPG": "#ff7f0e", "SAC": "#2ca02c"}
    W = 100  # rolling window

    data = {}
    for algo in algos:
        d = np.load(os.path.join(LOG_DIR, "UAV_2_%s_rewards.npz" % algo))
        data[algo] = (d["ep_rewards"], d["ep_feasible"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: smoothed reward.
    ax = axes[0]
    for algo in algos:
        rew, _ = data[algo]
        s = rolling_mean(rew, W)
        ax.plot(np.arange(W, W + len(s)), s, color=colors[algo], lw=1.6,
                label="%s (末100均值 %.0f)" % (algo, rew[-100:].mean()))
    ax.set_xlabel("episode")
    ax.set_ylabel("平均奖励（窗口 %d）" % W)
    ax.set_title("单段训练奖励曲线")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: cumulative feasibility rate.
    ax = axes[1]
    for algo in algos:
        _, feas = data[algo]
        cum = np.cumsum(feas) / np.arange(1, len(feas) + 1)
        ax.plot(np.arange(1, len(cum) + 1), cum * 100.0, color=colors[algo],
                lw=1.6, label="%s (最终 %.1f%%)" % (algo, 100.0 * feas.mean()))
    ax.set_xlabel("episode")
    ax.set_ylabel("累计可行率 (%)")
    ax.set_title("单段训练累计可行率（arrive+upload）")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("单段轨迹训练曲线（TD3 / DDPG / SAC，3000 集，新乡村地图）")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "segment_reward_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved -> %s" % out)
    return 0


if __name__ == "__main__":
    main()
