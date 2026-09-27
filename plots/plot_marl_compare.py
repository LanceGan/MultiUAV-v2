"""Plot the four MARL algorithms' training reward curves (N=2, one clean figure).

Reads ``results/logs/UAV_2_<Algo>_rewards.npz`` (``ep_rewards``) and draws the raw
curve plus a smoothed (moving-average) version for each of ILD3 / MADDPG /
MATD3 / MAPPO.  Kept separate from ``plot_results.py`` so the comparison figure
is clean (not mixed with the older Ours/Proposed/AllProposed histories).

CLI::

    F:/anaconda3/envs/pytorch/python.exe plots/plot_marl_compare.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ALGOS = ("ILD3", "MADDPG", "MATD3", "MAPPO")
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "logs")
FIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "figs")
WINDOW = 50  # moving-average window


def _moving_average(x, w):
    if len(x) < w:
        return x
    cum = np.concatenate([[0.0], np.cumsum(x)])
    return (cum[w:] - cum[:-w]) / w


def plot():
    os.makedirs(FIGS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    for algo in ALGOS:
        p = os.path.join(LOGS_DIR, "UAV_2_%s_rewards.npz" % algo)
        if not os.path.exists(p):
            continue
        r = np.asarray(np.load(p)["ep_rewards"], dtype=float)
        ep = np.arange(1, r.size + 1)
        axes[0].plot(ep, r, lw=0.8, alpha=0.5, label=algo)
        ma = _moving_average(r, WINDOW)
        axes[1].plot(ep[WINDOW - 1:], ma, lw=1.8, label=algo)

    axes[0].set_xlabel("回合 (episode)")
    axes[0].set_ylabel("回合奖励 (reward)")
    axes[0].set_title("原始奖励曲线")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("回合 (episode)")
    axes[1].set_ylabel("回合奖励 (reward)")
    axes[1].set_title("平滑奖励曲线（移动平均 %d）" % WINDOW)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("四种 MARL 算法训练奖励对比（N=2, 2000 集，通信感知观测）")
    fig.tight_layout()
    out = os.path.join(FIGS_DIR, "marl_reward_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved -> %s" % out)
    return out


if __name__ == "__main__":
    plot()
