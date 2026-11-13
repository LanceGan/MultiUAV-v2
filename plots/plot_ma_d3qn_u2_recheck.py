"""Render the U=2 MA-D3QN best-checkpoint trajectory and stability check.

The left panel replays the saved ``best_constraint.pth`` deterministically and
draws the resulting full mission trajectories. The right panel uses the 100-set
diagnostic evaluation history to show why a feasible checkpoint alone does not
establish stable training feasibility.
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
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = Path(
    "/Users/ganruijie.1/.codex/skills/nature-figure/scripts"
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SKILL_SCRIPTS))

from audit_panel_alignment import require_matplotlib_panel_alignment
from experiment.run_exclusive_marl_pilot import (
    _deterministic_actions,
    _device,
    _load_routes,
    _make_model,
    _set_device,
)
from segment.stc.e2e_exclusive import ExclusiveSectorFixedRouteEnv
from segment.stc.radio import PerSectorRadioField
from segment.stc.scenario import load_stc_scenario


DEFAULT_RUN = (
    ROOT
    / "results/runs_ma_d3qn_recheck_v1"
    / "ma-d3qn-recheck-v1-u2-s42-e100-s2000"
)

COLORS = {
    "uav0": "#0F4D92",
    "uav1": "#42949E",
    "task": "#767676",
    "anchor": "#272727",
    "fail": "#B64342",
    "feasible": "#2E7D32",
    "threshold": "#B64342",
    "grid": "#D8D8D8",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
    })


def _panel_label(ax, label: str) -> None:
    offset = ScaledTranslation(-5 / 72, 4 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0.0, 1.0, label, transform=ax.transAxes + offset,
        fontsize=8, fontweight="bold", ha="left", va="bottom",
    )


def _replay(checkpoint: Path, routes_json: Path, summary: dict):
    device = _device()
    _set_device(device)
    scenario = load_stc_scenario(2, task_seed=42, map_seed=42)
    field = PerSectorRadioField.from_scenario(scenario, spacing_m=200.0)
    routes = _load_routes(routes_json, 2)
    env = ExclusiveSectorFixedRouteEnv(
        scenario,
        field,
        routes,
        horizon_steps=2000,
        arrival_radius_m=10.0,
        safe_distance_m=30.0,
        outage_probability_threshold=0.5,
        outage_max_s=20.0,
    )
    model = _make_model(
        "MA-D3QN", env, actor_lr=1e-4, critic_lr=1e-3,
        teacher_anchor_weight=0.0,
    )
    model.load(str(checkpoint))
    obs = env.reset()
    step_rows = []
    for uav_id in range(env.n_uav):
        step_rows.append({
            "uav_id": uav_id,
            "step": 0,
            "time_s": 0.0,
            "x_m": float(env.positions[uav_id, 0]),
            "y_m": float(env.positions[uav_id, 1]),
        })
    while True:
        actions = _deterministic_actions("MA-D3QN", model, obs)
        actions = env.project_safe_actions(actions, outage_threshold=0.2)
        obs, _rewards, terminal, info = env.step(actions)
        for uav_id in range(env.n_uav):
            step_rows.append({
                "uav_id": uav_id,
                "step": int(env.t),
                "time_s": float(env.t * env.dt),
                "x_m": float(env.positions[uav_id, 0]),
                "y_m": float(env.positions[uav_id, 1]),
            })
        if terminal or info["truncated"]:
            break
    metrics = env.metrics()
    expected = summary["best_metrics"]
    if not metrics["feasible"]:
        raise RuntimeError(
            "best checkpoint replay is not feasible: %r" % metrics["failure_reasons"]
        )
    if abs(float(metrics["makespan_s"]) - float(expected["makespan_s"])) > 1e-6:
        raise RuntimeError(
            "replay makespan changed: %.6f != %.6f"
            % (metrics["makespan_s"], expected["makespan_s"])
        )
    return scenario, routes, env, metrics, step_rows


def _write_source_data(out_dir: Path, trajectory_rows, evaluations) -> None:
    with (out_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=("uav_id", "step", "time_s", "x_m", "y_m")
        )
        writer.writeheader()
        writer.writerows(trajectory_rows)

    with (out_dir / "evaluations.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = (
            "episode", "feasible", "makespan_s", "total_delay_s",
            "outage_uav0_s", "outage_uav1_s", "failure_reasons",
        )
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for item in evaluations:
            metrics = item["metrics"]
            outage = metrics["max_contiguous_outage_s"]
            writer.writerow({
                "episode": item["episode"],
                "feasible": bool(metrics["feasible"]),
                "makespan_s": metrics["makespan_s"],
                "total_delay_s": metrics["total_delay_s"],
                "outage_uav0_s": outage[0],
                "outage_uav1_s": outage[1],
                "failure_reasons": ";".join(metrics["failure_reasons"]),
            })


def _plot(out_dir: Path, scenario, routes, env, metrics, evaluations) -> Path:
    _apply_style()
    fig = plt.figure(figsize=(7.09, 3.45), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, width_ratios=(1.72, 1.0),
        left=0.065, right=0.985, bottom=0.25, top=0.90, wspace=0.28,
    )
    ax_map = fig.add_subplot(grid[0, 0])
    ax_eval = fig.add_subplot(grid[0, 1])

    task_pos = np.asarray(scenario.task_positions_m)
    ax_map.scatter(
        task_pos[:, 0] / 1000.0, task_pos[:, 1] / 1000.0,
        s=26, facecolor="#F2F2F2", edgecolor=COLORS["task"],
        linewidth=0.7, zorder=2, label="Inspection task",
    )
    depot = np.asarray(scenario.depot_m) / 1000.0
    terminal = np.asarray(scenario.terminal_m) / 1000.0
    ax_map.scatter(
        [depot[0]], [depot[1]], marker="s", s=31,
        facecolor="white", edgecolor=COLORS["anchor"], linewidth=0.9, zorder=4,
    )
    ax_map.scatter(
        [terminal[0]], [terminal[1]], marker="*", s=62,
        facecolor="#FFD166", edgecolor=COLORS["anchor"], linewidth=0.6, zorder=4,
    )
    for uav_id, route in routes.items():
        traj = np.asarray(env.trajectories[uav_id], dtype=float) / 1000.0
        ax_map.plot(
            traj[:, 0], traj[:, 1], color=COLORS[f"uav{uav_id}"],
            linewidth=1.45 if uav_id == 0 else 1.25,
            alpha=0.95, zorder=3,
            label=f"UAV {uav_id + 1}",
        )
        ax_map.scatter(
            traj[0, 0], traj[0, 1], s=24,
            color=COLORS[f"uav{uav_id}"], edgecolor="white", linewidth=0.5,
            zorder=6,
        )
    ax_map.set_xlim(0, scenario.length_m / 1000.0)
    ax_map.set_ylim(0, scenario.width_m / 1000.0)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("x (km)")
    ax_map.set_ylabel("y (km)")
    ax_map.set_title("Best feasible checkpoint", loc="right", pad=4, fontsize=7.5)
    ax_map.grid(color=COLORS["grid"], linewidth=0.45, alpha=0.6)
    ax_map.set_axisbelow(True)
    outage = metrics["max_contiguous_outage_s"]
    fig.text(
        0.065, 0.115,
        "20/20 tasks and uploads · zero collisions\n"
        f"T = {metrics['makespan_s']:.1f} s · D = {metrics['total_delay_s']:.1f} s\n"
        f"max outage = {outage[0]:.1f}/{outage[1]:.1f} s",
        ha="left", va="bottom", fontsize=6.0,
        linespacing=1.25,
    )
    map_handles = [
        plt.Line2D([], [], color=COLORS["uav0"], lw=1.6, label="UAV 1"),
        plt.Line2D([], [], color=COLORS["uav1"], lw=1.6, label="UAV 2"),
        plt.Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="#F2F2F2",
            markeredgecolor=COLORS["task"], markersize=4.2, label="Task",
        ),
        plt.Line2D(
            [], [], marker="*", linestyle="", markerfacecolor="#FFD166",
            markeredgecolor=COLORS["anchor"], markersize=6, label="Terminal",
        ),
    ]
    fig.legend(
        handles=map_handles, loc="lower left", bbox_to_anchor=(0.065, 0.012),
        fontsize=5.8, ncol=4, handlelength=1.7, columnspacing=1.0,
    )
    _panel_label(ax_map, "a")

    episodes = np.asarray([row["episode"] for row in evaluations], dtype=float)
    feasible = np.asarray(
        [bool(row["metrics"]["feasible"]) for row in evaluations], dtype=bool
    )
    outage_eval = np.asarray(
        [row["metrics"]["max_contiguous_outage_s"] for row in evaluations],
        dtype=float,
    )
    for uav_id in range(2):
        ax_eval.plot(
            episodes, outage_eval[:, uav_id], color=COLORS[f"uav{uav_id}"],
            linewidth=1.0, alpha=0.65, zorder=2,
        )
        for is_ok, marker, face in (
                (True, "o", COLORS[f"uav{uav_id}"]),
                (False, "o", "white")):
            mask = feasible == is_ok
            ax_eval.scatter(
                episodes[mask], outage_eval[mask, uav_id],
                s=22, marker=marker, facecolor=face,
                edgecolor=COLORS[f"uav{uav_id}"], linewidth=0.85, zorder=4,
            )
    ax_eval.axhline(
        20.0, color=COLORS["threshold"], linestyle="--", linewidth=0.9,
        label="Outage limit (20 s)", zorder=1,
    )
    ax_eval.axhspan(20.0, max(140.0, float(outage_eval.max()) + 5.0),
                    color="#F6CFCB", alpha=0.38, zorder=0)
    early = next(
        (row for row in evaluations
         if "early_service_arrival" in row["metrics"]["failure_reasons"]),
        None,
    )
    if early is not None:
        ax_eval.annotate(
            "early-service\nfailure", xy=(early["episode"], 20.0),
            xytext=(-8, 28), textcoords="offset points",
            fontsize=5.8, color=COLORS["fail"], ha="right",
            arrowprops={"arrowstyle": "-", "color": COLORS["fail"], "lw": 0.6},
        )
    ax_eval.set_xlim(0, 105)
    ax_eval.set_ylim(0, max(145.0, float(outage_eval.max()) + 8.0))
    ax_eval.set_xlabel("Training episode")
    ax_eval.set_ylabel("Maximum contiguous outage (s)")
    ax_eval.set_title(
        f"Checkpoint stability: {int(feasible.sum())}/{len(feasible)} feasible",
        loc="right", pad=4, fontsize=7.5,
    )
    ax_eval.grid(axis="y", color=COLORS["grid"], linewidth=0.45, alpha=0.6)
    ax_eval.set_axisbelow(True)
    eval_handles = [
        plt.Line2D([], [], color=COLORS["uav0"], marker="o", lw=1.0, label="UAV 1"),
        plt.Line2D([], [], color=COLORS["uav1"], marker="o", lw=1.0, label="UAV 2"),
        plt.Line2D(
            [], [], color=COLORS["threshold"], linestyle="--", lw=0.9,
            label="Outage limit (20 s)",
        ),
    ]
    fig.legend(
        handles=eval_handles, loc="lower right", bbox_to_anchor=(0.985, 0.012),
        fontsize=5.8, ncol=3, handlelength=1.7, columnspacing=1.0,
    )
    _panel_label(ax_eval, "b")

    fig.canvas.draw()
    base = out_dir / "ma_d3qn_u2_best_trajectory_and_stability"
    require_matplotlib_panel_alignment(
        fig,
        axes=[ax_map, ax_eval],
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


def run(args) -> Path:
    run_dir = Path(args.run_dir).resolve()
    checkpoint = run_dir / "best_constraint.pth"
    summary_path = run_dir / "summary.json"
    if not checkpoint.is_file() or not summary_path.is_file():
        raise FileNotFoundError("missing checkpoint or summary in %s" % run_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["algorithm"] != "MA-D3QN":
        raise ValueError("expected MA-D3QN summary")

    scenario, routes, env, metrics, trajectory_rows = _replay(
        checkpoint, Path(args.routes_json).resolve(), summary
    )
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_source_data(out_dir, trajectory_rows, summary["evaluations"])
    png = _plot(
        out_dir, scenario, routes, env, metrics, summary["evaluations"]
    )

    provenance = {
        "source_run": str(run_dir),
        "checkpoint_sha256": _sha256(checkpoint),
        "summary_sha256": _sha256(summary_path),
        "routes_json": str(Path(args.routes_json).resolve()),
        "replay_device": str(_device()),
        "metrics": metrics,
        "evaluation_definition": (
            "deterministic policy, MapGuide disabled, safety and upload guards enabled"
        ),
        "source_data": ["trajectory.csv", "evaluations.csv"],
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(png)
    return png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument(
        "--routes-json",
        default=str(ROOT / "configs/exclusive/u2_s42_4d_ga_routes.json"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results/figs_ma_d3qn_u2_recheck_20261111"),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    main()
