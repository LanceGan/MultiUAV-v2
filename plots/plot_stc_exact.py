"""Summarise sealed exact-benchmark runs without overstating optimality.

The figure separates certification/feasibility from wall-clock scaling. A
complete discrete search that finds no feasible plan is distinct from a
timeout and from a feasible but unproven incumbent.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

qa_dir = os.environ.get("NATURE_FIGURE_SCRIPTS")
if qa_dir:
    sys.path.insert(0, qa_dir)
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None


METHOD_LABELS = {
    "exact_discrete_enumeration": "Exact discrete",
    "proposed_stc_alns": "Proposed STC-ALNS",
}
METHOD_COLORS = {
    "exact_discrete_enumeration": "#4c72b0",
    "proposed_stc_alns": "#dd8452",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_runs(runs_dir, pattern):
    runs = []
    for run_dir in sorted(Path(runs_dir).glob(pattern)):
        manifest_path = run_dir / "manifest.json"
        records_path = run_dir / "records.json"
        if not manifest_path.is_file() or not records_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "success":
            continue
        records = json.loads(records_path.read_text(encoding="utf-8"))
        by_task = {}
        for record in records:
            if record.get("status") != "evaluated":
                continue
            method = record.get("method")
            if method not in METHOD_LABELS:
                continue
            task_count = int(record["n_tasks"])
            by_task.setdefault(task_count, {})[method] = record
        for task_count, methods in by_task.items():
            missing = sorted(set(METHOD_LABELS) - set(methods))
            if missing:
                raise ValueError(
                    "%s K=%d missing methods: %s"
                    % (run_dir, task_count, ", ".join(missing)))
        runs.append({
            "run_dir": run_dir,
            "run_id": manifest.get("run_id", run_dir.name),
            "manifest": manifest_path,
            "records_path": records_path,
            "by_task": by_task,
        })
    if not runs:
        raise ValueError("no sealed stc_exact runs found")
    return runs


def _exact_state(record):
    if record.get("timed_out"):
        return "timeout"
    if record.get("proven_optimal"):
        return "optimal"
    if record.get("discrete_search_complete") and not record.get("feasible"):
        return "no feasible"
    if record.get("feasible"):
        return "feasible incumbent"
    return "not certified"


def summarise(runs):
    rows = []
    for run in runs:
        for task_count, methods in sorted(run["by_task"].items()):
            exact = methods["exact_discrete_enumeration"]
            proposed = methods["proposed_stc_alns"]
            rows.append({
                "run_id": run["run_id"],
                "n_tasks": task_count,
                "exact_feasible": bool(exact.get("feasible")),
                "exact_objective": exact.get("objective"),
                "exact_runtime_s": float(exact["runtime_s"]),
                "exact_proven_optimal": bool(exact.get("proven_optimal")),
                "exact_discrete_search_complete": bool(
                    exact.get("discrete_search_complete")),
                "exact_timed_out": bool(exact.get("timed_out")),
                "exact_state": _exact_state(exact),
                "exact_explored": int(exact.get("explored", 0)),
                "exact_gap": exact.get("gap"),
                "proposed_feasible": bool(proposed.get("feasible")),
                "proposed_objective": proposed.get("objective"),
                "proposed_makespan_s": proposed.get("makespan_s"),
                "proposed_total_delay_s": proposed.get("total_delay_s"),
                "proposed_runtime_s": float(proposed["runtime_s"]),
                "proposed_iterations": int(proposed.get("iterations", 0)),
                "proposed_iterations_executed": int(
                    proposed.get("iterations_executed", 0)),
            })
    rows.sort(key=lambda row: row["n_tasks"])
    task_counts = sorted({row["n_tasks"] for row in rows})
    if len(task_counts) != len(rows):
        raise ValueError("multiple exact runs found for the same task count")
    return {
        "n_task_counts": len(task_counts),
        "task_counts": task_counts,
        "rows": rows,
        "optimality_scope": (
            "optimal only within the declared discrete candidate space"),
    }


def _style():
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


def make_figure(runs, out_dir):
    stats = summarise(runs)
    rows = stats["rows"]
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _style()
    fig, (ax_state, ax_time) = plt.subplots(
        1, 2, figsize=(7.2, 3.35), gridspec_kw={"width_ratios": [1.0, 1.08]})
    fig.subplots_adjust(left=0.15, right=0.985, top=0.84, bottom=0.21,
                        wspace=0.28)

    y = np.arange(len(rows))
    for idx, row in enumerate(rows):
        if row["exact_timed_out"]:
            ax_state.scatter(
                0, idx, marker="X", s=46, c="#b22222",
                edgecolors="white", linewidths=0.5, zorder=4)
        elif row["exact_proven_optimal"]:
            ax_state.scatter(
                0, idx, marker="o", s=44,
                c=METHOD_COLORS["exact_discrete_enumeration"],
                edgecolors="white", linewidths=0.5, zorder=4)
        elif row["exact_discrete_search_complete"]:
            ax_state.scatter(
                0, idx, marker="s", s=42, c="#8a8f98",
                edgecolors="white", linewidths=0.5, zorder=4)
        else:
            ax_state.scatter(
                0, idx, marker="D", s=40, c="#7f7f7f",
                edgecolors="white", linewidths=0.5, zorder=4)
        ax_state.text(
            0.08, idx, row["exact_state"], ha="left", va="center",
            fontsize=6.3, color="#333333")

        proposed_color = (METHOD_COLORS["proposed_stc_alns"]
                          if row["proposed_feasible"] else "#b22222")
        ax_state.scatter(
            1, idx,
            marker="o" if row["proposed_feasible"] else "X",
            s=44, c=proposed_color, edgecolors="white", linewidths=0.5,
            zorder=4)
        proposed_text = (
            "feasible, J=%.3f" % row["proposed_objective"]
            if row["proposed_feasible"] else "infeasible")
        ax_state.text(
            1.08, idx, proposed_text, ha="left", va="center",
            fontsize=6.3, color="#333333")

    ax_state.set_xlim(-0.45, 1.70)
    ax_state.set_xticks([0, 1])
    ax_state.set_xticklabels(["Exact discrete", "Proposed STC-ALNS"])
    ax_state.set_yticks(y)
    ax_state.set_yticklabels(["K=%d" % task for task in stats["task_counts"]])
    ax_state.invert_yaxis()
    ax_state.set_title(
        "a  Certification and feasibility\n(single deterministic prefix per K)",
        loc="left", fontsize=7.2, fontweight="bold", pad=4)
    ax_state.grid(axis="x", color="#eceef1", linewidth=0.5, zorder=0)

    x = np.arange(len(rows))
    exact_runtime = np.asarray(
        [row["exact_runtime_s"] for row in rows], dtype=float)
    proposed_runtime = np.asarray(
        [row["proposed_runtime_s"] for row in rows], dtype=float)
    if np.any(exact_runtime <= 0.0) or np.any(proposed_runtime <= 0.0):
        raise ValueError("runtime must be positive for log10 display")
    ax_time.plot(
        x, np.log10(exact_runtime), color=METHOD_COLORS["exact_discrete_enumeration"],
        marker="o", lw=1.2, ms=4.5, label="Exact discrete")
    ax_time.plot(
        x, np.log10(proposed_runtime),
        color=METHOD_COLORS["proposed_stc_alns"], marker="s", lw=1.2,
        ms=4.2, label="Proposed STC-ALNS")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(["K=%d" % task for task in stats["task_counts"]])
    ax_time.set_ylabel("Runtime (s, log scale)")
    ax_time.set_title(
        "b  Wall-clock scaling\n(same hardware/session; labels are measured)",
        loc="left", fontsize=7.2, fontweight="bold", pad=4)
    ax_time.grid(axis="y", color="#e6e8eb", linewidth=0.5, zorder=0)
    ax_time.legend(loc="upper left", fontsize=6.2)

    fig.suptitle(
        "Exact discrete benchmark: certification is separate from feasibility",
        fontsize=8.3, fontweight="bold", y=0.985)

    stem = out_dir / "stc_exact_k4_k8"
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            fig,
            json_out=str(stem) + ".alignment-layout.json",
            overlay_svg=str(stem) + ".alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True)
    fig.savefig(str(stem) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight")
    fig.savefig(str(stem) + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)

    stats_path = out_dir / "stc_exact_k4_k8.statistics.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    stats_csv_path = out_dir / "stc_exact_k4_k8.statistics.csv"
    with stats_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "figure": stem.name + ".png",
        "runs": [
            {
                "run_id": run["run_id"],
                "manifest": str(run["manifest"]),
                "manifest_sha256": _sha256(run["manifest"]),
                "records": str(run["records_path"]),
                "records_sha256": _sha256(run["records_path"]),
            }
            for run in runs
        ],
        "statistics_file": stats_path.name,
        "statistics_sha256": _sha256(stats_path),
        "statistics_csv_file": stats_csv_path.name,
        "statistics_csv_sha256": _sha256(stats_csv_path),
        "optimality_scope": stats["optimality_scope"],
    }
    provenance_path = out_dir / (stem.name + ".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8")
    png_path = Path(str(stem) + ".png")
    bundle = {
        "figures": {
            png_path.name: {
                "bytes": png_path.stat().st_size,
                "sha256": _sha256(png_path),
            }
        },
        "n_figures": 1,
        "provenance_sha256": _sha256(provenance_path),
    }
    (out_dir / "reporting_bundle.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    return stem, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--pattern", default="formal-exact-k*-s42")
    parser.add_argument("--out-dir", default="results/figs_exact_20261107")
    args = parser.parse_args(argv)
    runs = _load_runs(args.runs_dir, args.pattern)
    stem, stats = make_figure(runs, args.out_dir)
    print("wrote %s.{png,pdf,svg,tiff}" % stem)
    print("Exact task counts=%s" % stats["task_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
