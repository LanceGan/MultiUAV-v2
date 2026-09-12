"""Task 9: auto-generate all paper figures purely from the logs.

Reads every ``*.metrics.json`` / rollout ``*.npz`` / training-reward history in
``logs_dir`` and writes the paper figures to ``figs_dir``. No experimental
number is hardcoded anywhere: every plotted value is read from the log files.
The only literals in this module are axis labels, colors, font sizes, figure
layout, and unit-conversion / grid-resolution constants (all marked below).

Figures produced:

1. Trajectory overlays (``trajectory_<run_id>.png``): each UAV's x/y path drawn
   on top of the G2A outage map and the A2G SINR map. The radio maps are queried
   over a grid (``radio_map_G2A`` / ``radio_map_A2G``) to build the contour
   background.
2. Metric comparison (``metrics_<metric>.png``): one grouped bar chart per
   metric (makespan / total_delay / avg_uplink_rate / outage_duration /
   total_energy), comparing methods, grouped by ``uav_num``.
3. Layer-1 clustering (``assign_clusters.png``): inspection points colored by
   cluster label, one panel per clustering method x ``uav_num``, read from
   ``assign_<method>_uav<N>.json``.
4. Layer-2 sequence comparison (``sequence_comparison.png``): grouped bar chart
   of the 5 sequence methods on ``comm_cost`` / ``euclidean_length``, read from
   ``seq_<method>_uav<N>.json``.
5. Training reward curves (``training_rewards.png``): reward-vs-episode per
   method / ``uav_num``, read from ``*_rewards.npz`` (written by
   ``trajectory/train.py``) with a fallback to parsing the training ``*.log``.

CLI::

    F:/anaconda3/envs/pytorch/python.exe plots/plot_results.py
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make the project root importable when run as ``python plots/plot_results.py``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radio_map import radio_map_A2G, radio_map_G2A  # noqa: E402
from trajectory.env import DEFAULT_UAV_H  # noqa: E402

# ---------------------------------------------------------------------------
# Style / layout constants. These are the ONLY literals in this module: axis
# labels, colors, font sizes, figure geometry, and unit-conversion constants.
# They are NOT experimental results (which all come from the log files).
# ---------------------------------------------------------------------------

# CJK-capable font (same pattern as the old MultiUAV ``plot_test_statistics``).
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# The five real end-to-end metrics written by ``experiment/evaluate.py``.
METRIC_ORDER = ("makespan", "total_delay", "avg_uplink_rate",
                "outage_duration", "total_energy")
METRIC_LABELS = {
    "makespan": "任务完成时间 (s)",
    "total_delay": "总时延 (s)",
    "avg_uplink_rate": "平均上行速率 (Mbps)",
    "outage_duration": "中断时长 (s)",
    "total_energy": "总能耗 (J)",
}

# Layer-1 clustering methods (rows) and layer-2 sequence methods (bars), in
# paper order — used to lay out the assign/seq figures consistently.
ASSIGN_METHOD_ORDER = ("naive", "balanced", "jia", "4d")
SEQ_METHOD_ORDER = ("ga_euclidean", "pso", "aco", "nn_2opt", "ga_eqtsp")

UNIT_TO_KM = 0.1        # env units are 100 m; 1 unit -> 0.1 km (env._loc_km)
UAV_H_KM = DEFAULT_UAV_H * UNIT_TO_KM   # hover altitude in km (env default)
GRID_RES = 60           # radio-map contour resolution (grid points per axis)
DPI = 150

_TRAILING_TOKENS = {"best", "final", "last", "stable", "rewards", "metrics"}
_REWARD_RE = re.compile(r"ep\s+(\d+)\s*\|\s*reward\s+(-?[\d.eE+-]+)")


# ---------------------------------------------------------------------------
# Filename parsing (method + uav_num from ``UAV_<num>_<method>[_<episode>]``).
# ---------------------------------------------------------------------------
def _parse_method(stem):
    """Return the method/experiment name encoded in a log filename stem."""
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "UAV" and parts[1].isdigit():
        rest = [p for p in parts[2:] if p not in _TRAILING_TOKENS]
        if rest:
            return "_".join(rest)
        return "_".join(parts[2:])
    return stem


def _parse_uav_num(stem):
    """Return the UAV count encoded in a ``UAV_<num>_...`` filename, else None."""
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "UAV" and parts[1].isdigit():
        return int(parts[1])
    return None


# ---------------------------------------------------------------------------
# Radio-map grid queries (contour background).
# ---------------------------------------------------------------------------
def _query_radio_grid(module, field):
    """Query ``module`` over a grid; return ``(xs_km, ys_km, grid)``.

    ``field`` is ``"outage"`` (probability in [0, 1]) or ``"sinr"`` (dB).
    """
    extent_km = float(module.D)
    xs = np.linspace(0.0, extent_km, GRID_RES)
    ys = np.linspace(0.0, extent_km, GRID_RES)
    xx, yy = np.meshgrid(xs, ys)
    locs = np.stack([xx.ravel(), yy.ravel(),
                     np.full(xx.size, UAV_H_KM)], axis=1)
    outages, sinrs = module.getPointMiniOutage(locs)
    values = np.asarray(sinrs if field == "sinr" else outages, dtype=float)
    return xs, ys, values.reshape(GRID_RES, GRID_RES)


# ---------------------------------------------------------------------------
# 1. Trajectory overlays.
# ---------------------------------------------------------------------------
def _plot_trajectories(run_id, npz_path, figs_dir):
    log = np.load(npz_path)
    uav_num = int(np.asarray(log["uav_num"]).item())
    trajs = [np.asarray(log["traj_%d" % i], dtype=float) for i in range(uav_num)]

    g2a_xs, g2a_ys, g2a_grid = _query_radio_grid(radio_map_G2A, "outage")
    a2g_xs, a2g_ys, a2g_grid = _query_radio_grid(radio_map_A2G, "sinr")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: G2A downlink outage map + trajectories.
    ax = axes[0]
    cf = ax.contourf(g2a_xs, g2a_ys, g2a_grid, levels=np.linspace(0.0, 1.0, 11),
                     cmap="Reds")
    fig.colorbar(cf, ax=ax, label="G2A 中断概率")
    for i, tr in enumerate(trajs):
        x = tr[:, 0] * UNIT_TO_KM
        y = tr[:, 1] * UNIT_TO_KM
        ax.plot(x, y, lw=1.5, label="UAV %d" % i)
        ax.scatter([x[0]], [y[0]], marker="o", s=30, zorder=5)
    ax.set_title("G2A 中断图 + 轨迹 (%s)" % run_id)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.legend()

    # Right: A2G uplink SINR map + trajectories.
    ax = axes[1]
    cf = ax.contourf(a2g_xs, a2g_ys, a2g_grid, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="A2G SINR (dB)")
    for i, tr in enumerate(trajs):
        x = tr[:, 0] * UNIT_TO_KM
        y = tr[:, 1] * UNIT_TO_KM
        ax.plot(x, y, lw=1.5, label="UAV %d" % i)
        ax.scatter([x[0]], [y[0]], marker="o", s=30, zorder=5)
    ax.set_title("A2G SINR 图 + 轨迹 (%s)" % run_id)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.legend()

    fig.tight_layout()
    out = os.path.join(figs_dir, "trajectory_%s.png" % run_id)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2. Metric comparison bar charts.
# ---------------------------------------------------------------------------
def _load_metrics(logs_dir):
    """Load every ``*.metrics.json`` into a list of per-run dicts."""
    runs = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "*.metrics.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        stem = os.path.splitext(os.path.basename(path))[0]
        run_id = data.get("run_id", stem.removesuffix(".metrics"))
        runs.append({
            "run_id": run_id,
            "uav_num": int(data.get("uav_num", _parse_uav_num(stem) or 0)),
            "method": _parse_method(run_id) or _parse_method(stem),
            "metrics": {k: float(data[k]) for k in METRIC_ORDER if k in data},
        })
    return runs


def _plot_metrics(runs, figs_dir):
    if not runs:
        return []
    methods = sorted({r["method"] for r in runs})
    uav_nums = sorted({r["uav_num"] for r in runs})
    n_groups = len(methods)
    n_bars = len(uav_nums)
    width = 0.8 / max(n_bars, 1)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    outputs = []
    for metric in METRIC_ORDER:
        fig, ax = plt.subplots(figsize=(max(n_groups, 1) * 2.2 + 4, 6))
        for bi, un in enumerate(uav_nums):
            values = []
            for method in methods:
                vals = [r["metrics"][metric] for r in runs
                        if r["method"] == method and r["uav_num"] == un
                        and metric in r["metrics"]]
                values.append(float(np.mean(vals)) if vals else float("nan"))
            x = np.arange(n_groups) + (bi - (n_bars - 1) / 2.0) * width
            bars = ax.bar(x, values, width,
                          label="UAV %d" % un,
                          color=color_cycle[bi % len(color_cycle)])
            for bar, v in zip(bars, values):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2.0, v,
                            "%.3g" % v, ha="center", va="bottom", fontsize=9)

        ax.set_xticks(np.arange(n_groups))
        ax.set_xticklabels(methods)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_LABELS[metric])
        if n_bars > 1:
            ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        out = os.path.join(figs_dir, "metrics_%s.png" % metric)
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs


# ---------------------------------------------------------------------------
# 3. Layer-1 clustering figure (assign_*.json).
# ---------------------------------------------------------------------------
def _load_assign(logs_dir):
    """Load ``assign_<method>_uav<N>.json`` runs that carry per-point labels."""
    runs = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "assign_*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "points" not in data or "labels" not in data:
            continue
        runs.append(data)
    return runs


def _plot_clusters(runs, figs_dir):
    """Scatter each inspection point colored by cluster label (methods x uav_num)."""
    if not runs:
        return None
    methods = [m for m in ASSIGN_METHOD_ORDER
               if any(r.get("method") == m for r in runs)]
    uav_nums = sorted({int(r["uav_num"]) for r in runs})
    fig, axes = plt.subplots(len(methods), len(uav_nums),
                             figsize=(3.2 * len(uav_nums), 3.0 * len(methods)),
                             squeeze=False)
    for mi, method in enumerate(methods):
        for ci, un in enumerate(uav_nums):
            ax = axes[mi][ci]
            run = next((r for r in runs
                        if r.get("method") == method and int(r["uav_num"]) == un),
                       None)
            if run is None:
                ax.axis("off")
                continue
            pts = np.asarray(run["points"], dtype=float) * UNIT_TO_KM
            labels = np.asarray(run["labels"], dtype=int)
            for c in np.unique(labels):
                ax.scatter(pts[labels == c, 0], pts[labels == c, 1],
                           s=12, label="cluster %d" % c)
            ax.set_title("%s (UAV %d)" % (method, un), fontsize=10)
            ax.set_xlabel("x (km)")
            ax.set_ylabel("y (km)")
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(figs_dir, "assign_clusters.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4. Layer-2 sequence figure (seq_*.json).
# ---------------------------------------------------------------------------
def _load_seq(logs_dir):
    """Load ``seq_<method>_uav<N>.json`` runs."""
    runs = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "seq_*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        runs.append(data)
    return runs


def _plot_sequence(runs, figs_dir):
    """Grouped bar chart of the 5 sequence methods on comm_cost / euclidean_length.

    Each panel plots the per-``uav_num`` totals (``total_comm_cost`` /
    ``total_euclidean_length``) for the five methods side by side.
    """
    if not runs:
        return None
    methods = [m for m in SEQ_METHOD_ORDER
               if any(r.get("method") == m for r in runs)]
    uav_nums = sorted({int(r["uav_num"]) for r in runs})
    n_bars = len(uav_nums)
    width = 0.8 / max(n_bars, 1)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    panels = (("comm_cost", "total_comm_cost", "通信代价"),
              ("euclidean_length", "total_euclidean_length", "欧氏路径长度 (100m)"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (_key, field, label) in zip(axes, panels):
        x = np.arange(len(methods))
        for bi, un in enumerate(uav_nums):
            values = []
            for m in methods:
                vals = [float(r[field]) for r in runs
                        if r.get("method") == m and int(r["uav_num"]) == un
                        and field in r]
                values.append(float(np.mean(vals)) if vals else float("nan"))
            bars = ax.bar(x + (bi - (n_bars - 1) / 2.0) * width, values, width,
                          label="UAV %d" % un,
                          color=color_cycle[bi % len(color_cycle)])
            for bar, v in zip(bars, values):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2.0, v,
                            "%.3g" % v, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3, axis="y")
        if n_bars > 1:
            ax.legend()
    fig.tight_layout()
    out = os.path.join(figs_dir, "sequence_comparison.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4b. Layer-1 clustering within-cluster variance (the 4D objective).
# ---------------------------------------------------------------------------
def _plot_cluster_metrics(runs, figs_dir):
    """Bar chart of the 4 clustering methods on within-cluster variance — the
    objective the 4D method actually minimizes (lower = more homogeneous)."""
    if not runs:
        return None
    methods = [m for m in ASSIGN_METHOD_ORDER
               if any(r.get("method") == m for r in runs)]
    uav_nums = sorted({int(r["uav_num"]) for r in runs})
    panels = (("g2a_outage_within_var", "簇内 G2A 中断方差"),
              ("a2g_sinr_within_var", "簇内 A2G SINR 方差"),
              ("load_within_var", "簇内数据量方差"))
    n_bars = len(uav_nums)
    width = 0.8 / max(n_bars, 1)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5))
    for ax, (key, label) in zip(axes, panels):
        x = np.arange(len(methods))
        for bi, un in enumerate(uav_nums):
            values = []
            for m in methods:
                vals = [float(r[key]) for r in runs
                        if r.get("method") == m and int(r["uav_num"]) == un
                        and key in r]
                values.append(float(np.mean(vals)) if vals else float("nan"))
            bars = ax.bar(x + (bi - (n_bars - 1) / 2.0) * width, values, width,
                          label="UAV %d" % un,
                          color=color_cycle[bi % len(color_cycle)])
            for bar, v in zip(bars, values):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2.0, v,
                            "%.3g" % v, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        if n_bars > 1:
            ax.legend(fontsize=7)
    fig.suptitle("聚类簇内方差（越小越同质）")
    fig.tight_layout()
    out = os.path.join(figs_dir, "cluster_metrics.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4c. Layer-2 sequence trajectory figure (straight-line executor + G2A map).
# ---------------------------------------------------------------------------
def _plot_sequence_trajectories(seq_runs, figs_dir):
    """Trajectory figure: a straight-line executor follows each sequence's visit
    order, overlaid on the G2A outage map.  Contrasts the Euclidean sequence with
    the comm-aware GA_EQTSP sequence — the latter avoids high-outage (red) regions.
    """
    if not seq_runs:
        return None
    uav_nums = sorted({int(r["uav_num"]) for r in seq_runs})
    g2a_xs, g2a_ys, g2a_grid = _query_radio_grid(radio_map_G2A, "outage")

    outputs = []
    for un in uav_nums:
        eu = next((r for r in seq_runs
                   if r.get("method") == "ga_euclidean" and int(r["uav_num"]) == un),
                  None)
        ga = next((r for r in seq_runs
                   if r.get("method") == "ga_eqtsp" and int(r["uav_num"]) == un),
                  None)
        if eu is None or ga is None:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (run, title) in zip(axes,
                                    ((eu, "纯欧氏序列"), (ga, "GA_EQTSP 通信感知序列"))):
            cf = ax.contourf(g2a_xs, g2a_ys, g2a_grid,
                             levels=np.linspace(0.0, 1.0, 11), cmap="Reds")
            for u in run["per_uav"]:
                pts = np.asarray(u["ordered_points"], dtype=float) * UNIT_TO_KM
                ax.plot(pts[:, 0], pts[:, 1], lw=1.5, label="UAV %d" % int(u["uav"]))
            ax.set_title(title)
            ax.set_xlabel("x (km)")
            ax.set_ylabel("y (km)")
            ax.legend(fontsize=7)
        fig.colorbar(cf, ax=axes, label="G2A 中断概率")
        fig.suptitle("轨迹对比 (UAV %d)：直线执行器 + 序列" % un)
        fig.tight_layout()
        out = os.path.join(figs_dir, "trajectory_sequence_uav%d.png" % un)
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs


# ---------------------------------------------------------------------------
# 5. Training reward curves.
# ---------------------------------------------------------------------------
def _load_rewards(logs_dir):
    """Collect reward histories keyed by ``(uav_num, method)``.

    Primary source is ``*_rewards.npz`` (``ep_rewards``) written by train.py;
    fall back to parsing the training ``*.log`` text lines for runs that do not
    yet have a structured history.
    """
    histories = {}

    for path in sorted(glob.glob(os.path.join(logs_dir, "*_rewards.npz"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        data = np.load(path)
        if "ep_rewards" not in data:
            continue
        key = (_parse_uav_num(stem), _parse_method(stem))
        arr = np.asarray(data["ep_rewards"], dtype=float)
        histories[key] = (np.arange(1, arr.size + 1, dtype=float), arr)

    for path in sorted(glob.glob(os.path.join(logs_dir, "*.log"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        key = (_parse_uav_num(stem), _parse_method(stem))
        if key in histories:
            continue  # structured history already present
        episodes, rewards = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = _REWARD_RE.search(line)
                if m:
                    episodes.append(int(m.group(1)))
                    rewards.append(float(m.group(2)))
        if rewards:
            histories[key] = (np.asarray(episodes, dtype=float),
                              np.asarray(rewards, dtype=float))
    return histories


def _plot_rewards(histories, figs_dir):
    if not histories:
        return None
    keys = sorted(histories.keys(), key=lambda k: (k[0] or 0, k[1]))
    fig, ax = plt.subplots(figsize=(10, 6))
    for (un, method) in keys:
        episodes, arr = histories[(un, method)]
        label = "%s" % method if un is None else "UAV %d %s" % (un, method)
        ax.plot(episodes, arr, lw=1.5, label=label)
    ax.set_xlabel("回合 (episode)")
    ax.set_ylabel("回合奖励 (reward)")
    ax.set_title("训练奖励曲线")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(figs_dir, "training_rewards.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def plot_all(logs_dir="results/logs", figs_dir="results/figs"):
    """Read all logs in ``logs_dir`` and write the paper figures to ``figs_dir``.

    Returns the list of generated figure paths.
    """
    logs_dir = os.path.abspath(logs_dir)
    figs_dir = os.path.abspath(figs_dir)
    os.makedirs(figs_dir, exist_ok=True)

    generated = []

    # 1. Trajectory overlays (rollout ``*.npz`` with per-UAV trajectories).
    for path in sorted(glob.glob(os.path.join(logs_dir, "*.npz"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.endswith("_rewards"):
            continue
        data = np.load(path)
        if "traj_0" not in data:
            continue
        run_id = stem
        generated.append(_plot_trajectories(run_id, path, figs_dir))

    # 2. Metric comparison bar charts.
    generated.extend(_plot_metrics(_load_metrics(logs_dir), figs_dir))

    # 3. Layer-1 clustering figures: points colored by cluster label + the
    #    within-cluster variance metrics (the 4D objective).
    assign_runs = _load_assign(logs_dir)
    cluster_out = _plot_clusters(assign_runs, figs_dir)
    if cluster_out:
        generated.append(cluster_out)
    cluster_metrics_out = _plot_cluster_metrics(assign_runs, figs_dir)
    if cluster_metrics_out:
        generated.append(cluster_metrics_out)

    # 4. Layer-2 sequence figures: comm_cost / euclidean_length bars + the
    #    trajectory figure (straight-line executor + G2A outage map).
    seq_runs = _load_seq(logs_dir)
    seq_out = _plot_sequence(seq_runs, figs_dir)
    if seq_out:
        generated.append(seq_out)
    seq_traj_out = _plot_sequence_trajectories(seq_runs, figs_dir)
    if seq_traj_out:
        generated.extend(seq_traj_out)

    # 5. Training reward curves.
    reward_out = _plot_rewards(_load_rewards(logs_dir), figs_dir)
    if reward_out:
        generated.append(reward_out)

    print("generated %d figure(s) in %s" % (len(generated), figs_dir))
    for p in generated:
        print("  - %s" % p)
    return generated


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Auto-generate all paper figures from the logs (Task 9).")
    parser.add_argument("--logs_dir", type=str, default="results/logs")
    parser.add_argument("--figs_dir", type=str, default="results/figs")
    args = parser.parse_args(argv)
    plot_all(args.logs_dir, args.figs_dir)


if __name__ == "__main__":
    main()
