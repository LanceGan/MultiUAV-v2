"""Regenerate STC formal figures from finalized run directories (Task 10).

Every plotted value is derived from the finalized (sealed, status ``success``)
run artifacts; there are no hard-coded result values.  Runs that are not
sealed/consistent are skipped, and a figure whose input is empty is not written
(it never fabricates an empty placeholder).
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from experiment.run_stc_gates import load_run  # noqa: F401  (re-exported)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Pure extraction (no matplotlib) — every value from the records
# --------------------------------------------------------------------------- #

def extract_td_points(records):
    """Normalized (T/T_ref, D/D_ref) points for evaluated feasible records."""
    out = []
    for r in records:
        if not isinstance(r, dict) or r.get("status") != "evaluated":
            continue
        if not r.get("feasible"):
            continue
        ref = (r.get("reference") or {}).get("reference") or {}
        t_ref, d_ref = ref.get("t_ref_s"), ref.get("d_ref_s")
        ms, td = r.get("makespan_s"), r.get("total_delay_s")
        if None in (t_ref, d_ref, ms, td):
            continue
        if not all(math.isfinite(float(x)) for x in (t_ref, d_ref, ms, td)):
            continue
        if float(t_ref) <= 0 or float(d_ref) <= 0:
            continue
        out.append({"method": r.get("method"), "label": r.get("label"),
                    "proposed": bool(r.get("proposed")),
                    "t_norm": float(ms) / float(t_ref),
                    "d_norm": float(td) / float(d_ref)})
    return out


def extract_feasibility(records):
    """Per-method success rate over evaluated (non-unavailable) records."""
    per = {}
    for r in records:
        if not isinstance(r, dict) or r.get("status") != "evaluated":
            continue
        m = r.get("method")
        d = per.setdefault(m, {"evaluated": 0, "feasible": 0})
        d["evaluated"] += 1
        if r.get("feasible"):
            d["feasible"] += 1
    return {m: {"evaluated": d["evaluated"], "feasible": d["feasible"],
                "success_rate": d["feasible"] / d["evaluated"]}
            for m, d in per.items() if d["evaluated"]}


def _method_mean(records, key):
    per = {}
    for r in records:
        if not isinstance(r, dict) or r.get("status") != "evaluated":
            continue
        v = r.get(key)
        if v is None or not math.isfinite(float(v)):
            continue
        per.setdefault(r.get("method"), []).append(float(v))
    return {m: sum(vs) / len(vs) for m, vs in per.items()}


def extract_outage(records):
    return _method_mean(records, "outage_ratio")


def extract_conflict(records):
    return _method_mean(records, "conflict_lost_bits")


def extract_waiting(records):
    return _method_mean(records, "waiting_s")


def extract_balance(records):
    """Per-method per-UAV completion times (flattened)."""
    per = {}
    for r in records:
        if not isinstance(r, dict) or r.get("status") != "evaluated":
            continue
        times = ((r.get("balance") or {}).get("completion_times_s") or [])
        vals = [float(t) for t in times if t is not None]
        if vals:
            per.setdefault(r.get("method"), []).extend(vals)
    return per


def extract_ablation(records):
    out = []
    for r in records:
        if not isinstance(r, dict) or r.get("status") != "evaluated":
            continue
        if r.get("objective") is None:
            continue
        out.append({"ablation": r.get("ablation"), "objective": float(r["objective"])})
    return out


def extract_scale(records):
    out = []
    for r in records:
        if not isinstance(r, dict) or r.get("status") != "evaluated":
            continue
        if r.get("objective") is None:
            continue
        out.append({"variant": r.get("variant"), "objective": float(r["objective"])})
    return out


def extract_segment(records):
    """Generator -> feasible fraction (segment-layer evidence)."""
    out = {}
    for r in records:
        if not isinstance(r, dict) or r.get("status") != "evaluated":
            continue
        if r.get("feasible_frac") is None:
            continue
        out[r.get("generator")] = float(r["feasible_frac"])
    return out


# --------------------------------------------------------------------------- #
# Rendering (headless)
# --------------------------------------------------------------------------- #

def _concat(runs, name):
    out = []
    for run in runs:
        if not run.valid:
            continue
        value = run.artifacts.get(name)
        if isinstance(value, list):
            out.extend(value)
    return out


def _save(fig, out_dir, name):
    path = Path(out_dir) / name
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return path


def _bar(out_dir, name, data, ylabel, title):
    if not data:
        return []
    labels = list(data)
    values = [data[l] for l in labels]
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return [_save(fig, out_dir, name)]


def _render_main(records, out_dir):
    written = []

    td = extract_td_points(records)
    if td:
        fig, ax = plt.subplots()
        for p in td:
            ax.scatter(p["t_norm"], p["d_norm"],
                       marker="*" if p["proposed"] else "o",
                       label=p["label"] or p["method"])
        ax.set_xlabel("T / T_ref")
        ax.set_ylabel("D / D_ref")
        ax.set_title("T-D normalized")
        ax.legend()
        written.append(_save(fig, out_dir, "stc_td_pareto.png"))

    feas = extract_feasibility(records)
    written += _bar(out_dir, "stc_success_rate.png",
                    {m: v["success_rate"] for m, v in feas.items()},
                    "success rate", "System success rate")

    written += _bar(out_dir, "stc_outage.png", extract_outage(records),
                    "mean outage ratio", "G2A outage")
    written += _bar(out_dir, "stc_conflict_lost_bits.png",
                    extract_conflict(records), "lost bits", "Conflict lost bits")
    written += _bar(out_dir, "stc_waiting.png", extract_waiting(records),
                    "waiting (s)", "Waiting time")

    balance = extract_balance(records)
    if balance:
        labels = list(balance)
        data = [balance[l] for l in labels]
        fig, ax = plt.subplots()
        ax.boxplot(data, labels=labels)
        ax.set_ylabel("completion time (s)")
        ax.set_title("Per-UAV completion time balance")
        written.append(_save(fig, out_dir, "stc_balance.png"))

    return written


def _render_ablation(records, out_dir):
    data = {a["ablation"]: a["objective"] for a in extract_ablation(records)}
    return _bar(out_dir, "stc_ablation.png", data, "objective", "Ablation")


def _render_scale(records, out_dir):
    data = {s["variant"]: s["objective"] for s in extract_scale(records)}
    return _bar(out_dir, "stc_scale.png", data, "objective", "Scale")


def _render_segment(records, out_dir):
    data = extract_segment(records)
    return _bar(out_dir, "stc_segment_feasibility.png", data,
                "feasible fraction", "Segment generator feasibility")


def render_figures(runs_by_experiment, out_dir):
    """Render every available figure from finalized runs.

    ``runs_by_experiment`` maps an experiment name to a list of
    :class:`~experiment.run_stc_gates.LoadedRun`.  Returns the list of written
    figure paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for experiment, runs in runs_by_experiment.items():
        if experiment == "stc_main":
            written += _render_main(_concat(runs, "raw_records.json"), out_dir)
        elif experiment == "stc_ablation":
            written += _render_ablation(_concat(runs, "records.json"), out_dir)
        elif experiment == "stc_scale":
            written += _render_scale(_concat(runs, "records.json"), out_dir)
        elif experiment == "stc_segment_gate":
            written += _render_segment(_concat(runs, "records.json"), out_dir)
    return written


def load_valid_runs(run_paths):
    """Load finalized (sealed, success) runs, skipping invalid/non-run paths."""
    out = []
    for path in run_paths:
        run = load_run(path)
        if run is not None and run.valid:
            out.append(run)
    return out


def _group(runs):
    grouped = {}
    for run in runs:
        grouped.setdefault(run.experiment, []).append(run)
    return grouped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=None,
                        help="finalized run directories")
    parser.add_argument("--runs-dir", default=None,
                        help="directory of finalized runs (results/runs)")
    parser.add_argument("--out-dir", default="results/figs")
    args = parser.parse_args(argv)
    if not args.runs and not args.runs_dir:
        parser.error("one of --runs or --runs-dir is required")

    if args.runs_dir:
        from experiment.run_stc_gates import scan_runs
        runs = scan_runs(args.runs_dir)
    else:
        runs = load_valid_runs(args.runs)

    grouped = _group(runs)
    written = render_figures(grouped, args.out_dir)
    print("wrote %d figure(s) to %s" % (len(written), args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
