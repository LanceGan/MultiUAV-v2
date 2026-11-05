"""Plot sealed STC final routes on a common scenario.

The figure uses only persisted ``solutions.json``/``timelines.json`` records.
For U=2 s42, GATSAC is read from its sealed supplement run while the other five
systems come from the sealed main run.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import get_scenario
from segment.stc.scenario import load_stc_scenario

qa_dir = os.environ.get("NATURE_FIGURE_SCRIPTS")
if qa_dir:
    sys.path.insert(0, qa_dir)
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None


METHODS = (
    ("region_distance_straight", "Region-Distance-Straight", "main"),
    ("shortest_balanced_graph", "Shortest-Balanced-Graph", "main"),
    ("jia_ebtas_style", "Jia-EBTAS-style", "main"),
    ("gatsac_style", "GATSAC-style", "gatsac"),
    ("library_open_loop", "Library-Open-Loop", "main"),
    ("proposed_stc_alns", "Proposed STC-ALNS", "main"),
)

UAV_COLORS = ("#1f77b4", "#e69f00")


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _record_map(records, method_seed):
    return {
        rec["method"]: rec
        for rec in records
        if rec.get("method_seed") == method_seed
    }


def _artifact_map(items):
    return {item["record_id"]: item for item in items}


def _violation_text(record):
    if record["feasible"]:
        return "Feasible  |  J = %.4f" % record["objective"]
    violations = record.get("violations") or {}
    parts = []
    for key, label in (
        ("unserved_tasks", "unserved"),
        ("upload_failures", "upload fail"),
        ("deadline_exceeded", "deadline"),
        ("wait_exceeded", "wait slots"),
    ):
        value = violations.get(key, 0)
        if value:
            parts.append("%s %s" % (label, value))
    excess = float(violations.get("outage_excess", 0.0) or 0.0)
    if excess > 0:
        parts.append("outage +%.3f" % excess)
    return "Infeasible  |  " + (", ".join(parts) or "hard constraint")


def _source_paths(main_run, gatsac_run):
    return {
        "main": {
            "raw": main_run / "raw_records.json",
            "solutions": main_run / "solutions.json",
            "timelines": main_run / "timelines.json",
        },
        "gatsac": {
            "raw": gatsac_run / "raw_records.json",
            "solutions": gatsac_run / "solutions.json",
            "timelines": gatsac_run / "timelines.json",
        },
    }


def _route_paths(timeline_record):
    if timeline_record is None:
        return []
    paths = []
    for timeline in timeline_record.get("timelines") or ():
        positions = [slot["position_m"] for slot in timeline.get("slots") or ()]
        if positions:
            paths.append(positions)
    return paths


def _wait_violation_points(timeline_record, max_wait_s):
    points = []
    if timeline_record is None:
        return points
    dt = float(timeline_record.get("delta_t_s", 0.5))
    for timeline in timeline_record.get("timelines") or ():
        held_by_edge = {}
        for slot in timeline.get("slots") or ():
            if slot.get("holding"):
                edge_id = int(slot["edge_id"])
                held_by_edge[edge_id] = held_by_edge.get(edge_id, 0.0) + dt
                if held_by_edge[edge_id] > max_wait_s + 1e-9:
                    points.append(tuple(slot["position_m"]))
            else:
                held_by_edge.clear()
    return points


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_figure(main_run, gatsac_run, out_dir, method_seed=0):
    main_run = Path(main_run).resolve()
    gatsac_run = Path(gatsac_run).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = _source_paths(main_run, gatsac_run)
    raw = {k: _load_json(v["raw"]) for k, v in sources.items()}
    solutions = {k: _artifact_map(_load_json(v["solutions"]))
                 for k, v in sources.items()}
    timelines = {k: _artifact_map(_load_json(v["timelines"]))
                 for k, v in sources.items()}
    record_by_source = {
        key: _record_map(items, method_seed) for key, items in raw.items()
    }

    scenario = load_stc_scenario(2, task_seed=42, map_seed=42)
    legacy = get_scenario(2)
    tasks = scenario.task_positions_m
    base_positions = tuple(
        (float(x) * 100.0, float(y) * 100.0) for x, y in legacy["bs_loc"])

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.88, bottom=0.17,
                        wspace=0.055, hspace=0.22)
    axes = axes.ravel()

    for panel_idx, (method, label, source_key) in enumerate(METHODS):
        ax = axes[panel_idx]
        index = chr(ord("a") + panel_idx)
        record = record_by_source[source_key].get(method)
        if record is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "%s unavailable" % label,
                    ha="center", va="center", transform=ax.transAxes)
            continue
        rec_id = record["record_id"]
        timeline_record = timelines[source_key].get(rec_id)
        paths = _route_paths(timeline_record)

        ax.scatter(
            [p[0] for p in tasks], [p[1] for p in tasks],
            s=12, c="#c9cdd2", edgecolors="#5a6169", linewidths=0.35,
            zorder=2)
        if base_positions:
            ax.scatter(
                [p[0] for p in base_positions], [p[1] for p in base_positions],
                marker="^", s=28, c="#7b6ba8", edgecolors="white",
                linewidths=0.35, zorder=4)
        ax.scatter(*scenario.depot_m, marker="s", s=28, c="#222222",
                   edgecolors="white", linewidths=0.35, zorder=5)
        ax.scatter(*scenario.terminal_m, marker="*", s=42, c="#222222",
                   edgecolors="white", linewidths=0.35, zorder=5)

        for uav, path in enumerate(paths):
            xs, ys = zip(*path)
            ax.plot(xs, ys, color=UAV_COLORS[uav % len(UAV_COLORS)],
                    linewidth=1.2, alpha=0.95, zorder=3)
        hold_points = _wait_violation_points(
            timeline_record, scenario.limits.max_wait_s)
        if hold_points:
            ax.scatter(
                [p[0] for p in hold_points], [p[1] for p in hold_points],
                marker="X", s=22, c="#d62728", edgecolors="white",
                linewidths=0.3, zorder=5)

        status_text = _violation_text(record)
        title_color = "#1b6b3a" if record["feasible"] else "#b22222"
        ax.set_title("%s  %s\n%s" % (index, label, status_text),
                     loc="left", fontsize=7.1, color=title_color,
                     fontweight="bold", pad=3.5)
        ax.set_xlim(-100, max(scenario.length_m, 1) + 100)
        ax.set_ylim(-100, max(scenario.width_m, 1) + 100)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    legend_handles = [
        Line2D([0], [0], color=UAV_COLORS[0], lw=1.4, label="UAV 1 route"),
        Line2D([0], [0], color=UAV_COLORS[1], lw=1.4, label="UAV 2 route"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#c9cdd2",
               markeredgecolor="#5a6169", markersize=4.5, label="Inspection point"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#7b6ba8",
               markeredgecolor="white", markersize=5, label="Base station"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#222222",
               markeredgecolor="white", markersize=4.5, label="Depot"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#222222",
               markeredgecolor="white", markersize=6, label="Terminal"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#d62728",
               markeredgecolor="white", markersize=6, label="Wait violation"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.018), fontsize=6.4)
    fig.suptitle(
        "U=2 inspection plan on s42 (method seed 0): final routes and hard feasibility",
        fontsize=8.5, fontweight="bold")

    stem = out_dir / "stc_u2_s42_seed0_routes"
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            fig, json_out=str(stem) + ".alignment-layout.json",
            overlay_svg=str(stem) + ".alignment.svg",
            tolerance_pt=1.5, gutter_tolerance_pt=1.5,
            require_panel_labels=False, strict=True)
    fig.savefig(str(stem) + ".png", dpi=600)
    fig.savefig(str(stem) + ".pdf")
    fig.savefig(str(stem) + ".svg")
    fig.savefig(str(stem) + ".tiff", dpi=600)

    provenance = {
        "figure": stem.name + ".png",
        "scenario": {"uav_num": 2, "map_seed": 42, "task_seed": 42,
                     "method_seed": method_seed},
        "sources": {
            key: {name: {"path": str(path), "sha256": _sha256(path)}
                  for name, path in paths.items()}
            for key, paths in sources.items()
        },
        "methods": [
            {"method": method, "label": label, "source": source}
            for method, label, source in METHODS
        ],
    }
    (out_dir / "stc_u2_s42_seed0_routes.provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    plt.close(fig)
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-run", required=True)
    parser.add_argument("--gatsac-run", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method-seed", type=int, default=0)
    args = parser.parse_args()
    stem = make_figure(args.main_run, args.gatsac_run, args.out_dir,
                       method_seed=args.method_seed)
    print("wrote %s.{png,pdf,svg}" % stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
