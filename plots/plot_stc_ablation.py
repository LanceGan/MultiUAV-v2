"""Plot the sealed RQ5 mechanism ablation from records.json only.

Core conclusion: disabling a genuinely contributing mechanism should increase
the paired objective relative to the same-seed full solver.
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

qa_dir = os.environ.get("NATURE_FIGURE_SCRIPTS")
if qa_dir:
    sys.path.insert(0, qa_dir)
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None


ARM_LABELS = {
    "full": "Full STC-ALNS",
    "ebtas_init": "w/o EBTAS initialisation",
    "multi_candidate": "w/o multi-candidate",
    "hotspot_destroy": "w/o hotspot destroy",
    "conflict_repair": "w/o conflict-aware repair",
    "path_switch": "w/o path switch",
    "adaptive_weights": "w/o adaptive weights",
    "incremental_eval": "Full replay (efficiency)",
}
ARM_ORDER = tuple(ARM_LABELS)
COLORS = {
    "full": "#333333",
    "worse": "#c44e52",
    "better": "#4c72b0",
    "neutral": "#8a8f98",
}
SEED_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_records(runs_dir, pattern):
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
        by_seed = {}
        for record in records:
            if record.get("status") != "evaluated":
                continue
            arm = record.get("ablation")
            seed = record.get("method_seed")
            objective = record.get("objective")
            if arm not in ARM_LABELS or seed is None or objective is None:
                continue
            by_seed.setdefault(int(seed), {})[arm] = float(objective)
        for seed, by_arm in by_seed.items():
            missing = [arm for arm in ARM_ORDER if arm not in by_arm]
            if missing:
                raise ValueError(
                    "%s seed %d missing arms: %s"
                    % (run_dir, seed, ", ".join(missing)))
        runs.append({
            "run_dir": run_dir,
            "manifest": manifest_path,
            "records_path": records_path,
            "records": records,
            "run_id": manifest.get("run_id", run_dir.name),
            "by_seed": by_seed,
        })
    if not runs:
        raise ValueError("no sealed stc_ablation runs found")
    return runs


def _exact_sign_flip_p(deltas):
    """One-sided paired sign-flip test for H1: mean(delta) > 0."""
    values = np.asarray(deltas, dtype=float)
    observed = float(values.mean())
    if values.size == 0 or observed <= 0.0:
        return 1.0
    if values.size > 15:
        raise ValueError("exact sign-flip test supports at most 15 replicates")
    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=values.size)))
    permuted = np.mean(signs * values, axis=1)
    return float(np.mean(permuted >= observed - 1e-12))


def _holm_adjust(pvalues):
    items = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    count = len(items)
    for rank, (idx, pvalue) in enumerate(items):
        running = max(running, min(1.0, (count - rank) * pvalue))
        adjusted[idx] = running
    return adjusted


def summarize(runs):
    rows = []
    all_seeds = sorted({seed for run in runs for seed in run["by_seed"]})
    for arm in ARM_ORDER:
        objectives = []
        deltas = []
        seeds = []
        feasible = []
        for run in runs:
            for seed, by_arm in run["by_seed"].items():
                record = next(
                    item for item in run["records"]
                    if item.get("method_seed") == seed
                    and item.get("ablation") == arm)
                objectives.append(by_arm[arm])
                deltas.append(by_arm[arm] - by_arm["full"])
                seeds.append(seed)
                feasible.append(bool(record.get("feasible")))
        values = np.asarray(objectives, dtype=float)
        diffs = np.asarray(deltas, dtype=float)
        sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
        delta_sd = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
        delta_se = delta_sd / np.sqrt(diffs.size) if diffs.size else 0.0
        critical = (float(student_t.ppf(0.975, diffs.size - 1))
                    if diffs.size > 1 else 0.0)
        mean_delta = float(diffs.mean())
        rows.append({
            "ablation": arm,
            "label": ARM_LABELS[arm],
            "n": int(values.size),
            "seeds": seeds,
            "objective_mean": float(values.mean()),
            "objective_sd": sd,
            "paired_delta_mean": mean_delta,
            "paired_delta_sd": delta_sd,
            "paired_delta_ci95": [
                mean_delta - critical * delta_se,
                mean_delta + critical * delta_se,
            ],
            "paired_deltas": [float(value) for value in diffs],
            "worse_seeds": int(np.sum(diffs > 1e-12)),
            "better_seeds": int(np.sum(diffs < -1e-12)),
            "tied_seeds": int(np.sum(np.abs(diffs) <= 1e-12)),
            "all_feasible": bool(all(feasible)),
            "raw_p_one_sided": _exact_sign_flip_p(diffs)
            if arm != "full" else None,
        })

    non_full = [row for row in rows if row["ablation"] != "full"]
    adjusted = _holm_adjust(
        [row["raw_p_one_sided"] for row in non_full])
    for row, pvalue in zip(non_full, adjusted):
        row["holm_p_one_sided"] = float(pvalue)
    rows[0]["holm_p_one_sided"] = None
    return {
        "n_method_seeds": len(all_seeds),
        "method_seeds": all_seeds,
        "primary_test": (
            "one-sided exact paired sign-flip test; H1 mean(removed-full)>0"),
        "multiplicity": "Holm correction across seven non-full arms",
        "alpha": 0.05,
        "rows": rows,
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
    stats = summarize(runs)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_arm = {row["ablation"]: row for row in stats["rows"]}
    ordered_arms = ["full"] + sorted(
        (arm for arm in ARM_ORDER if arm != "full"),
        key=lambda arm: rows_by_arm[arm]["paired_delta_mean"],
        reverse=True)
    display_rows = [rows_by_arm[arm] for arm in ordered_arms]

    _style()
    fig, (ax_obj, ax_delta) = plt.subplots(
        1, 2, figsize=(7.2, 3.55), gridspec_kw={"width_ratios": [1.0, 1.12]})
    fig.subplots_adjust(left=0.23, right=0.985, top=0.84, bottom=0.18,
                        wspace=0.16)
    y = np.arange(len(display_rows))
    full_mean = rows_by_arm["full"]["objective_mean"]

    obj_means = np.asarray(
        [row["objective_mean"] for row in display_rows], dtype=float)
    obj_sd = np.asarray(
        [row["objective_sd"] for row in display_rows], dtype=float)
    obj_colors = [
        COLORS["full"] if row["ablation"] == "full" else "#8d99ae"
        for row in display_rows]
    ax_obj.barh(
        y, obj_means, height=0.62, color=obj_colors, alpha=0.90,
        edgecolor="white", linewidth=0.4)
    ax_obj.errorbar(
        obj_means, y, xerr=obj_sd, fmt="none", ecolor="#343a40",
        elinewidth=0.65, capsize=1.8, capthick=0.65, zorder=4)
    for idx, row in enumerate(display_rows):
        values = np.asarray(
            [run["by_seed"][seed][row["ablation"]] for run in runs
             for seed in sorted(run["by_seed"])], dtype=float)
        offsets = np.linspace(-0.16, 0.16, values.size)
        ax_obj.scatter(
            values, idx + offsets, s=9, c="#20242a", alpha=0.72,
            linewidths=0, zorder=5)
    ax_obj.axvline(full_mean, color=COLORS["full"], lw=0.8,
                   ls=(0, (2, 2)), zorder=1)
    ax_obj.set_yticks(y)
    ax_obj.set_yticklabels([row["label"] for row in display_rows])
    ax_obj.invert_yaxis()
    ax_obj.set_xlabel("Mean objective J (lower is better)")
    ax_obj.set_title(
        "a  Absolute objective\n(n=%d method seeds; whiskers: 1 SD)"
        % stats["n_method_seeds"],
        loc="left", fontsize=7.2, fontweight="bold", pad=4)
    ax_obj.grid(axis="x", color="#e6e8eb", linewidth=0.5, zorder=0)

    seed_order = sorted({seed for run in runs for seed in run["by_seed"]})
    for idx, row in enumerate(display_rows):
        for seed_idx, (run, seed) in enumerate(
                [(run, seed) for run in runs for seed in sorted(run["by_seed"])]):
            deltas = np.asarray(row["paired_deltas"], dtype=float)
            jitter = (seed_idx - (len(deltas) - 1) / 2.0) * 0.025
            marker = SEED_MARKERS[seed_order.index(seed) % len(SEED_MARKERS)]
            color = (COLORS["worse"] if deltas[seed_idx] > 1e-12
                     else COLORS["better"] if deltas[seed_idx] < -1e-12
                     else COLORS["neutral"])
            ax_delta.scatter(
                deltas[seed_idx], idx + jitter, marker=marker, s=18,
                c=color, edgecolors="white", linewidths=0.3, zorder=4)
        mean = row["paired_delta_mean"]
        lo, hi = row["paired_delta_ci95"]
        color = (COLORS["worse"] if mean > 1e-12
                 else COLORS["better"] if mean < -1e-12
                 else COLORS["neutral"])
        ax_delta.errorbar(
            mean, idx, xerr=[[mean - lo], [hi - mean]], fmt="D",
            ms=4.2, color=color, ecolor=color, elinewidth=0.8,
            capsize=2.0, capthick=0.7, zorder=6)
    ax_delta.axvline(0.0, color="#333333", lw=0.8, zorder=1)
    ax_delta.set_yticks(y)
    ax_delta.set_yticklabels([])
    ax_delta.invert_yaxis()
    ax_delta.set_xlabel(r"Paired change vs full, $\Delta J$  (>0 worse)")
    ax_delta.set_title(
        "b  Same-seed paired effect\n(points: seeds; diamond/bar: mean and 95% t CI)",
        loc="left", fontsize=7.2, fontweight="bold", pad=4)
    ax_delta.grid(axis="x", color="#e6e8eb", linewidth=0.5, zorder=0)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=COLORS["worse"], markeredgecolor="white",
               markersize=4.5, label="Removal worsens"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=COLORS["better"], markeredgecolor="white",
               markersize=4.5, label="Removal improves"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=COLORS["neutral"], markeredgecolor="white",
               markersize=4.5, label="No change"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3,
        bbox_to_anchor=(0.60, 0.015), fontsize=6.2)
    fig.suptitle(
        "RQ5 mechanism ablation on s42: paired evidence across %d method seeds"
        % stats["n_method_seeds"],
        fontsize=8.4, fontweight="bold", y=0.985)

    stem = out_dir / "stc_rq5_ablation_s42"
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

    stats_payload = {
        **stats,
        "runs": [
            {"run_id": run["run_id"], "run_dir": str(run["run_dir"])}
            for run in runs
        ],
    }
    stats_path = out_dir / "stc_rq5_ablation_s42.statistics.json"
    stats_path.write_text(
        json.dumps(stats_payload, indent=2, ensure_ascii=False),
        encoding="utf-8")
    stats_csv_path = out_dir / "stc_rq5_ablation_s42.statistics.csv"
    with stats_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "ablation", "label", "n", "objective_mean", "objective_sd",
            "paired_delta_mean", "paired_delta_sd", "delta_ci95_low",
            "delta_ci95_high", "worse_seeds", "better_seeds", "tied_seeds",
            "raw_p_one_sided", "holm_p_one_sided", "all_feasible"])
        for row in stats["rows"]:
            writer.writerow([
                row["ablation"], row["label"], row["n"],
                row["objective_mean"], row["objective_sd"],
                row["paired_delta_mean"], row["paired_delta_sd"],
                row["paired_delta_ci95"][0], row["paired_delta_ci95"][1],
                row["worse_seeds"], row["better_seeds"], row["tied_seeds"],
                row["raw_p_one_sided"], row["holm_p_one_sided"],
                row["all_feasible"],
            ])
    provenance = {
        "figure": stem.name + ".png",
        "figure_files": [
            stem.name + suffix for suffix in
            (".png", ".pdf", ".svg", ".tiff")],
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
        "methods": [
            {"ablation": arm, "label": ARM_LABELS[arm]}
            for arm in ARM_ORDER
        ],
        "statistics_file": stats_path.name,
        "statistics_sha256": _sha256(stats_path),
        "statistics_csv_file": stats_csv_path.name,
        "statistics_csv_sha256": _sha256(stats_csv_path),
        "core_conclusion": (
            "The paired objective shows which ablation mechanisms contribute; "
            "non-positive deltas are reported without relabelling them as gains."),
    }
    provenance_path = out_dir / (
        stem.name + ".provenance.json")
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
        "statistics_sha256": _sha256(stats_path),
        "statistics_csv_sha256": _sha256(stats_csv_path),
    }
    (out_dir / "reporting_bundle.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return stem, stats_payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument(
        "--pattern", default="formal-ablation-s42-m*-r3")
    parser.add_argument(
        "--out-dir", default="results/figs_ablation_s42_20261107")
    args = parser.parse_args(argv)
    runs = _load_records(args.runs_dir, args.pattern)
    stem, stats = make_figure(runs, args.out_dir)
    print("wrote %s.{png,pdf,svg,tiff}" % stem)
    print("RQ5 rows=%d seeds=%d" % (
        len(stats["rows"]), stats["n_method_seeds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
