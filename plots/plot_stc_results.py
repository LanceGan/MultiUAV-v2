"""Regenerate STC formal figures from finalized run directories (Task 10).

Every plotted value is derived from the finalized (sealed, status ``success``)
run artifacts; there are no hard-coded result values.  Runs that are not
sealed/consistent are skipped.  Before mixing records, the plotter verifies they
are compatible (same reference / library / world hash); multi-seed records are
aggregated hierarchically (mean across method seeds within a scenario, then
mean across scenarios).  Each figure records its provenance (consumed run ids
and input hashes) in ``figures_provenance.json``.
"""

import argparse
import json
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
# Pure helpers / extraction (no matplotlib) — every value from the records
# --------------------------------------------------------------------------- #

def _finite(value):
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _scalar(value):
    """Coerce a metric value (scalar or per-UAV list) to a finite float."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        vals = [float(x) for x in value if x is not None]
        if not vals:
            return None
        value = sum(vals) / len(vals)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _compat_key(rec):
    return (rec.get("reference_hash"), rec.get("library_hash"),
            rec.get("world_hash"))


def check_compatible(records):
    """Raise if records from the SAME scenario mix incompatible runs.

    Records across different scenarios legitimately differ (different map/task
    seeds); only records within one scenario must share reference/library/world
    hashes.  Mixing scenarios is expected and handled by hierarchical
    aggregation, not rejected.
    """
    by_scen = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        scen = (r.get("uav_num"), r.get("map_seed"), r.get("task_seed"))
        by_scen.setdefault(scen, set()).add(_compat_key(r))
    problems = []
    for scen, keys in by_scen.items():
        keys.discard((None, None, None))
        if len(keys) > 1:
            problems.append("scenario %r mixes incompatible runs: %r"
                            % (scen, sorted(keys)))
    if problems:
        raise ValueError("; ".join(problems))
    return True


def check_same_source(runs):
    """Raise if valid runs come from different source/evaluator versions.

    The git head is the source (and therefore evaluator) version: figures must
    never mix records computed by different code versions.
    """
    heads = {r.manifest.get("git", {}).get("head") for r in runs if r.valid}
    heads.discard(None)
    heads.discard("unknown")
    if len(heads) > 1:
        raise ValueError("runs mix incompatible source versions: %r"
                         % sorted(heads))
    return True


def _hierarchical_mean(records, value_fn):
    """Mean across method seeds within a scenario, then mean across the COMMON
    scenarios (only scenarios where every method has a value), so methods are
    compared on identical scenario sets — never flat-averaged over different
    scenario sets."""
    by_scen = {}
    methods = set()
    for r in records:
        if not isinstance(r, dict) or r.get("status") != "evaluated":
            continue
        scen = (r.get("uav_num"), r.get("map_seed"), r.get("task_seed"))
        v = value_fn(r)
        if v is None:
            continue
        m = r.get("method")
        by_scen.setdefault((scen, m), []).append(v)
        methods.add(m)
    scen_means = {}
    for (scen, m), vals in by_scen.items():
        scen_means.setdefault(scen, {})[m] = sum(vals) / len(vals)
    common = [s for s, mm in scen_means.items() if set(mm) >= methods]
    out = {}
    for m in methods:
        vals = [scen_means[s][m] for s in common if m in scen_means[s]]
        if vals:
            out[m] = sum(vals) / len(vals)
    return out


def extract_td_points(records):
    """Per-method normalized (T/T_ref, D/D_ref), aggregated over the
    common-feasible scenario set (only scenarios feasible for EVERY method), so
    methods are never compared on scenarios where some are infeasible."""
    by_scen = {}
    methods = set()
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
        if not all(_finite(x) for x in (t_ref, d_ref, ms, td)):
            continue
        if float(t_ref) <= 0 or float(d_ref) <= 0:
            continue
        scen = (r.get("uav_num"), r.get("map_seed"), r.get("task_seed"))
        m = r.get("method")
        by_scen.setdefault((scen, m), []).append(
            (float(ms) / float(t_ref), float(td) / float(d_ref)))
        methods.add(m)

    scen_means = {}
    for (scen, m), pts in by_scen.items():
        scen_means.setdefault(scen, {})[m] = (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts))
    common = [s for s, mm in scen_means.items() if set(mm) >= methods]

    labels, proposed = {}, {}
    for r in records:
        if isinstance(r, dict) and r.get("method") is not None \
                and r["method"] not in labels:
            labels[r["method"]] = r.get("label")
            proposed[r["method"]] = bool(r.get("proposed"))

    out = []
    for m in methods:
        pts = [scen_means[s][m] for s in common if m in scen_means[s]]
        if not pts:
            continue
        out.append({"method": m, "label": labels.get(m),
                    "proposed": proposed.get(m, False),
                    "t_norm": sum(p[0] for p in pts) / len(pts),
                    "d_norm": sum(p[1] for p in pts) / len(pts)})
    return out


def extract_feasibility(records):
    """Per-method success rate over evaluated (non-unavailable) records (all
    paired instances)."""
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


def extract_outage(records):
    return _hierarchical_mean(records, lambda r: _scalar(r.get("outage_ratio")))


def extract_conflict(records):
    return _hierarchical_mean(records, lambda r: _scalar(r.get("conflict_lost_bits")))


def extract_waiting(records):
    return _hierarchical_mean(records, lambda r: _scalar(r.get("waiting_s")))


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


def _mean(values):
    return sum(values) / len(values)


def extract_ablation(records):
    """Mean objective per ablation name (replicates are averaged, never
    silently overwritten by the last one)."""
    acc = {}
    for r in records:
        if not isinstance(r, dict) or r.get("status") != "evaluated":
            continue
        if not _finite(r.get("objective")):
            continue
        acc.setdefault(r.get("ablation"), []).append(float(r["objective"]))
    return [{"ablation": k, "objective": _mean(v)}
            for k, v in acc.items() if k is not None]


def extract_scale(records):
    """Mean objective per (method, family, variant) — method/family identity is
    retained so a scale figure never conflates the open-loop baseline with the
    Proposed solver or one family with another."""
    acc = {}
    for r in records:
        if not isinstance(r, dict) or r.get("status") != "evaluated":
            continue
        if not _finite(r.get("objective")):
            continue
        key = (r.get("method"), r.get("family"), r.get("variant"))
        acc.setdefault(key, []).append(float(r["objective"]))
    return [{"method": k[0], "family": k[1], "variant": k[2],
             "objective": _mean(v)} for k, v in acc.items()]


def extract_segment(records):
    """Generator -> mean feasible fraction (replicates averaged)."""
    acc = {}
    for r in records:
        if not isinstance(r, dict) or r.get("status") != "evaluated":
            continue
        if not _finite(r.get("feasible_frac")):
            continue
        acc.setdefault(r.get("generator"), []).append(float(r["feasible_frac"]))
    return {k: _mean(v) for k, v in acc.items() if k is not None}


def extract_rq2(runs):
    """RQ2 assignment / sequence mean objective per method, across runs."""
    assign, seq = {}, {}
    for run in runs:
        if not run.valid:
            continue
        rq2 = run.artifacts.get("rq2.json")
        if isinstance(rq2, dict):
            rq2 = [rq2]
        for entry in (rq2 or []):
            if not isinstance(entry, dict):
                continue
            for layer, target in (("assignment", assign), ("sequence", seq)):
                for rec in (entry.get(layer) or []):
                    if not isinstance(rec, dict) or rec.get("status") != "evaluated":
                        continue
                    m, o = rec.get("method"), rec.get("objective")
                    if m is None or not _finite(o):
                        continue
                    target.setdefault(m, []).append(float(o))
    return {"assignment": {m: _mean(v) for m, v in assign.items()},
            "sequence": {m: _mean(v) for m, v in seq.items()}}


def extract_rq3(runs):
    """RQ3 mean metric error per approximation mode.metric, across runs."""
    acc = {}
    for run in runs:
        if not run.valid:
            continue
        for rec in (run.artifacts.get("records.json") or []):
            if not isinstance(rec, dict):
                continue
            me = rec.get("metric_error")
            if not isinstance(me, dict):
                continue
            for mode, metrics in me.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, value in metrics.items():
                    if _finite(value):
                        acc.setdefault("%s.%s" % (mode, metric), []).append(float(value))
    return {k: _mean(v) for k, v in acc.items()}


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
        ax.boxplot(data, tick_labels=labels)
        ax.set_ylabel("completion time (s)")
        ax.set_title("Per-UAV completion time balance")
        written.append(_save(fig, out_dir, "stc_balance.png"))

    return written


def _render_ablation(records, out_dir):
    data = {a["ablation"]: a["objective"] for a in extract_ablation(records)}
    return _bar(out_dir, "stc_ablation.png", data, "objective", "Ablation")


def _render_scale(records, out_dir):
    data = {"%s | %s | %s" % (s["method"] or "?", s["family"] or "?",
                              s["variant"] or "?"): s["objective"]
            for s in extract_scale(records)}
    return _bar(out_dir, "stc_scale.png", data, "objective",
                "Scale (method | family | variant)")


def _render_segment(records, out_dir):
    data = extract_segment(records)
    return _bar(out_dir, "stc_segment_feasibility.png", data,
                "feasible fraction", "Segment generator feasibility")


def _render_rq2(runs, out_dir):
    written = []
    rq2 = extract_rq2(runs)
    for layer, name in (("assignment", "stc_rq2_assignment.png"),
                        ("sequence", "stc_rq2_sequence.png")):
        written += _bar(out_dir, name, rq2[layer], "objective", "RQ2 " + layer)
    return written


def _render_rq3(runs, out_dir):
    data = extract_rq3(runs)
    return _bar(out_dir, "stc_rq3_metric_error.png", data,
                "mean absolute error", "RQ3 evaluator approximation error")


def render_figures(runs_by_experiment, out_dir):
    """Render every available figure from finalized, compatible runs.

    ``runs_by_experiment`` maps an experiment name to a list of
    :class:`~experiment.run_stc_gates.LoadedRun`.  Runs within one experiment
    must share the same source/evaluator version; records within one scenario
    must share hashes.  Returns the list of written figure paths, writes
    ``figures_provenance.json``, and writes an immutable content-addressed
    ``reporting_bundle.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    provenance = {}
    for experiment, runs in runs_by_experiment.items():
        valid = [r for r in runs if r.valid]
        if not valid:
            continue
        check_same_source(valid)
        if experiment == "stc_main":
            records = _concat(runs, "raw_records.json")
            check_compatible(records)
            figs = _render_main(records, out_dir)
            figs += _render_rq2(valid, out_dir)
        elif experiment == "stc_ablation":
            records = _concat(runs, "records.json")
            figs = _render_ablation(records, out_dir)
        elif experiment == "stc_scale":
            records = _concat(runs, "records.json")
            figs = _render_scale(records, out_dir)
        elif experiment == "stc_segment_gate":
            records = _concat(runs, "records.json")
            figs = _render_segment(records, out_dir)
        elif experiment == "stc_evaluator_audit":
            figs = _render_rq3(valid, out_dir)
        else:
            continue
        written += figs
        for fig in figs:
            provenance[fig.name] = {
                "experiment": experiment,
                "runs": [r.run_id for r in valid],
                "inputs": {r.run_id: r.manifest.get("inputs", {}) for r in valid},
            }
    _write_reporting_bundle(out_dir, written, provenance)
    return written


def _write_reporting_bundle(out_dir, written, provenance):
    """Write an immutable content-addressed reporting bundle.

    Records every figure's SHA-256 and byte length plus the provenance JSON's
    hash, so any tampering with the figure bundle is detectable.  The bundle is
    write-once (an existing bundle is never silently overwritten) and an empty
    bundle (no figures rendered) is refused.
    """
    import hashlib
    if not written:
        raise ValueError("refusing to write an empty reporting bundle "
                         "(no figures were rendered)")
    prov_path = out_dir / "figures_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True),
                         encoding="utf-8")
    bundle = {
        "figures": {},
        "provenance_sha256": hashlib.sha256(prov_path.read_bytes()).hexdigest(),
        "n_figures": len(written),
    }
    for fig in written:
        data = fig.read_bytes()
        bundle["figures"][fig.name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    bundle_path = out_dir / "reporting_bundle.json"
    if bundle_path.exists():
        raise FileExistsError(
            "reporting bundle already exists and is immutable "
            "(delete it to regenerate): %s" % bundle_path)
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True),
                           encoding="utf-8")
    return bundle_path


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
