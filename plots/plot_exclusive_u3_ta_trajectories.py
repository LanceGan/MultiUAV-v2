"""Plot sealed U=3 TA-SharedTD3-BC and TA-MAPPO trajectories.

Both scale probes are rejected: all 30 tasks are inspected and uploaded, but
the deterministic best checkpoints contain five collision slots and per-UAV
contiguous-outage excursions. The figure therefore presents the final
trajectories together with the measured failure boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = Path(
    "/Users/ganruijie.1/.codex/skills/nature-figure/scripts"
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SKILL_SCRIPTS))

from audit_panel_alignment import require_matplotlib_panel_alignment
from segment.stc.scenario import load_stc_scenario


RUNS = {
    "sharedtd3": {
        "label": "TA-SharedTD3-BC",
        "run_id": "win-formal-u3-s42-sharedtd3bc-ta",
    },
    "mappo": {
        "label": "TA-MAPPO",
        "run_id": "win-formal-u3-s42-mappo-ta",
    },
}

COLORS = {
    "uav0": "#0F4D92",
    "uav1": "#42949E",
    "uav2": "#D08B22",
    "task": "#767676",
    "anchor": "#272727",
    "collision": "#B64342",
    "grid": "#D8D8D8",
    "terminal": "#FFD166",
}


def _apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "axes.titlesize": 6.8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
    })


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _panel_label(ax, label: str) -> None:
    offset = ScaledTranslation(-5 / 72, 4 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0.0, 1.0, label, transform=ax.transAxes + offset,
        fontsize=8, fontweight="bold", ha="left", va="bottom",
    )


def _load_run(analysis_root: Path, formal_root: Path, key: str):
    meta = RUNS[key]
    analysis = analysis_root / meta["run_id"] / "trajectory_best_constraint.npz"
    summary_path = formal_root / meta["run_id"] / "summary.json"
    if not analysis.is_file() or not summary_path.is_file():
        raise FileNotFoundError("missing U=3 trajectory or summary for %s" % key)
    arrays = np.load(analysis)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return arrays, summary, analysis, summary_path


def _collision_points(positions_m: np.ndarray, safe_distance_m=30.0):
    positions = positions_m / 1000.0
    points = []
    for t in range(positions.shape[0]):
        for i in range(positions.shape[1]):
            for j in range(i + 1, positions.shape[1]):
                if np.linalg.norm(positions[t, i] - positions[t, j]) < safe_distance_m / 1000.0:
                    points.append((positions[t, i] + positions[t, j]) / 2.0)
    if not points:
        return np.empty((0, 2))
    raw = np.asarray(points)
    kept = []
    for point in raw:
        if not kept or min(np.linalg.norm(point - previous) for previous in kept) > 0.16:
            kept.append(point)
    return np.asarray(kept)


def _write_source_data(out_dir: Path, loaded) -> None:
    with (out_dir / "u3_ta_trajectories.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "step", "uav_id", "x_km", "y_km"))
        for key, payload in loaded.items():
            positions = payload["arrays"]["positions_m"] / 1000.0
            for step in range(positions.shape[0]):
                for uav_id in range(positions.shape[1]):
                    writer.writerow((
                        RUNS[key]["label"], step, uav_id,
                        positions[step, uav_id, 0], positions[step, uav_id, 1],
                    ))


def _plot(loaded, scenario, out_dir: Path) -> Path:
    _apply_style()
    fig = plt.figure(figsize=(7.09, 3.72), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, left=0.055, right=0.985, bottom=0.19, top=0.80, wspace=0.12,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]

    task = np.asarray(scenario.task_positions_m) / 1000.0
    depot = np.asarray(scenario.depot_m) / 1000.0
    terminal = np.asarray(scenario.terminal_m) / 1000.0

    for ax, key in zip(axes, RUNS):
        arrays = loaded[key]["arrays"]
        summary = loaded[key]["summary"]
        metrics = summary["best_metrics"]
        positions = arrays["positions_m"] / 1000.0
        for uav_id in range(positions.shape[1]):
            ax.plot(
                positions[:, uav_id, 0], positions[:, uav_id, 1],
                color=COLORS[f"uav{uav_id}"], linewidth=1.15, alpha=0.94,
            )
            ax.scatter(
                positions[0, uav_id, 0], positions[0, uav_id, 1],
                s=15, color=COLORS[f"uav{uav_id}"],
                edgecolor="white", linewidth=0.35, zorder=5,
            )
        ax.scatter(
            task[:, 0], task[:, 1], s=18,
            facecolor="#F2F2F2", edgecolor=COLORS["task"],
            linewidth=0.6, zorder=3,
        )
        ax.scatter(
            [depot[0]], [depot[1]], marker="s", s=24,
            facecolor="white", edgecolor=COLORS["anchor"],
            linewidth=0.8, zorder=4,
        )
        ax.scatter(
            [terminal[0]], [terminal[1]], marker="*", s=52,
            facecolor=COLORS["terminal"], edgecolor=COLORS["anchor"],
            linewidth=0.5, zorder=4,
        )
        collisions = _collision_points(arrays["positions_m"])
        if len(collisions):
            ax.scatter(
                collisions[:, 0], collisions[:, 1], marker="x", s=27,
                color=COLORS["collision"], linewidth=1.0, zorder=6,
            )

        outage = metrics["max_contiguous_outage_s"]
        ax.set_title(
            f"{RUNS[key]['label']}  |  REJECTED\n"
            f"30/30 tasks · 30/30 uploads · 5 collision slots\n"
            f"max outage = {outage[0]:.1f}/{outage[1]:.1f}/{outage[2]:.1f} s",
            loc="right", pad=5,
        )
        ax.set_xlim(0, scenario.length_m / 1000.0)
        ax.set_ylim(0, scenario.width_m / 1000.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (km)")
        ax.grid(color=COLORS["grid"], linewidth=0.45, alpha=0.62)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("y (km)")
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")

    handles = [
        plt.Line2D([], [], color=COLORS["uav0"], lw=1.4, label="UAV 1"),
        plt.Line2D([], [], color=COLORS["uav1"], lw=1.4, label="UAV 2"),
        plt.Line2D([], [], color=COLORS["uav2"], lw=1.4, label="UAV 3"),
        plt.Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="#F2F2F2",
            markeredgecolor=COLORS["task"], markersize=4.0, label="Task",
        ),
        plt.Line2D(
            [], [], marker="x", linestyle="", color=COLORS["collision"],
            markersize=4.6, markeredgewidth=1.0, label="Collision",
        ),
        plt.Line2D(
            [], [], marker="*", linestyle="", markerfacecolor=COLORS["terminal"],
            markeredgecolor=COLORS["anchor"], markersize=6, label="Terminal",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.045),
        ncol=6, fontsize=6.1, handlelength=1.7, columnspacing=1.15,
    )
    fig.text(
        0.5, 0.012,
        "Hard constraints failed: collision and per-UAV outage > 20 s.",
        ha="center", va="bottom", fontsize=6.1, color=COLORS["collision"],
    )

    fig.canvas.draw()
    base = out_dir / "exclusive_u3_ta_best_trajectories"
    require_matplotlib_panel_alignment(
        fig,
        axes=axes,
        panel_ids=["a", "b"],
        row_groups=[["a", "b"]],
        column_groups=[],
        json_out=str(base) + ".alignment-layout.json",
        overlay_svg=str(base) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(base.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(base.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    return base.with_suffix(".png")


def run(args):
    scenario = load_stc_scenario(3, task_seed=42, map_seed=42)
    loaded = {}
    for key in RUNS:
        arrays, summary, analysis_path, summary_path = _load_run(
            Path(args.analysis_root).resolve(),
            Path(args.formal_root).resolve(),
            key,
        )
        loaded[key] = {
            "arrays": arrays,
            "summary": summary,
            "analysis_path": analysis_path,
            "summary_path": summary_path,
        }

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_source_data(out_dir, loaded)
    png = _plot(loaded, scenario, out_dir)

    provenance = {
        "figure_claim": (
            "The deterministic U=3 best checkpoints complete all tasks and "
            "uploads but remain infeasible due to collisions and outage."
        ),
        "runs": {
            key: {
                "run_id": RUNS[key]["run_id"],
                "analysis_sha256": _sha256(value["analysis_path"]),
                "summary_sha256": _sha256(value["summary_path"]),
                "best_metrics": value["summary"]["best_metrics"],
            }
            for key, value in loaded.items()
        },
        "source_data": ["u3_ta_trajectories.csv"],
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(png)
    return png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root", default=str(ROOT / "results/formal_win_analysis")
    )
    parser.add_argument(
        "--formal-root", default=str(ROOT / "results/formal_win")
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results/figs_u3_ta_20261113"),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    main()
