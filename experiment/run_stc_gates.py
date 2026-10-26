"""Gate 0-4 orchestration: a freeze-time evidence check over finalized runs.

Each gate inspects the finalized (sealed, ``status == "success"``) run
directories under ``results/runs/`` and reports PASS / FAIL / NOT_RUN:

- Gate 0  evaluator / full test suite
- Gate 1  segment-layer evidence (RQ1)
- Gate 2  assignment/sequence (RQ2) + synchronous shared-bandwidth (RQ3) evidence
- Gate 3  proposed-method contrast + at least two contributing ablations (RQ5)
- Gate 4  provenance / reproduction (sealed manifests, hash-consistent)

A gate whose required evidence is *entirely absent* is ``NOT_RUN``; a gate whose
evidence is *present but invalid/incomplete* is ``FAIL``; only complete valid
evidence is ``PASS``.  Evidence is validated, not merely counted: feasibility
must be finite, both sides of an RQ2 comparison must contain evaluated records,
ablations must be compared against a ``full`` baseline and genuinely degrade,
and provenance must record a real git head and input hashes.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.run_stc_segment_gate import DETERMINISTIC_GENERATORS
from segment.stc.system_baselines import SYSTEM_BASELINE_KEYS

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"

GATE_ORDER = ("gate_0_evaluator", "gate_1_segment",
              "gate_2_assignment_sequence", "gate_3_proposed_ablation",
              "gate_4_provenance")

PROPOSED_METHODS = ("proposed_stc_alns",)
BASELINE_METHODS = tuple(m for m in SYSTEM_BASELINE_KEYS
                         if m not in PROPOSED_METHODS)

# Registered RQ2 assignment / sequence method names (must match run_stc_main.run_rq2).
ASSIGNMENT_METHODS = frozenset({
    "region_partition", "spatial_kmeans", "shortest_balanced",
    "jia_ebtas", "library_aware_ebtas",
})
SEQUENCE_METHODS = frozenset({
    "distance_nn_2opt", "jia_time_energy_tsp", "ga_eqtsp_sequence",
    "library_aware_sequence",
})
# RQ3 approximation modes reported by the evaluator audit.
AUDIT_MODES = frozenset({"independent", "static_factor"})


@dataclass
class LoadedRun:
    run_id: str
    experiment: str
    status: str
    valid: bool
    manifest: dict
    artifacts: dict
    problems: list = field(default_factory=list)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_run(run_dir):
    """Load a single run directory into a :class:`LoadedRun`.

    Returns ``None`` when the directory is not a run at all (no
    ``manifest.json``); such directories are ignored by :func:`scan_runs`.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return LoadedRun(run_id=run_dir.name, experiment=None, status=None,
                         valid=False, manifest={}, artifacts={},
                         problems=["manifest.json unreadable"])
    if not isinstance(manifest, dict):
        return LoadedRun(run_id=run_dir.name, experiment=None, status=None,
                         valid=False, manifest={}, artifacts={},
                         problems=["manifest.json is not a JSON object"])

    status = manifest.get("status")
    expected = manifest.get("expected", [])
    hashes = manifest.get("hashes", {})
    problems = []
    if status != "success":
        problems.append("status=%r (not success)" % (status,))

    artifacts = {}
    for name in expected:
        path = run_dir / name
        if not path.is_file():
            problems.append("missing expected artifact %s" % name)
            continue
        data = path.read_bytes()
        if name in hashes and _sha256(data) != hashes[name]:
            problems.append("hash mismatch for %s" % name)
        try:
            artifacts[name] = json.loads(data.decode("utf-8"))
        except (OSError, ValueError):
            artifacts[name] = None
            problems.append("artifact %s unparseable" % name)

    valid = status == "success" and not problems
    return LoadedRun(run_id=run_dir.name, experiment=manifest.get("experiment"),
                     status=status, valid=valid, manifest=manifest,
                     artifacts=artifacts, problems=problems)


def scan_runs(runs_dir):
    """Return every run directory under ``runs_dir`` (direct children)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    out = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        run = load_run(child)
        if run is not None:
            out.append(run)
    return out


def classify(runs):
    """Group runs by their manifest ``experiment`` key."""
    grouped = {}
    for run in runs:
        grouped.setdefault(run.experiment, []).append(run)
    return grouped


def _ok(run):
    return run.valid


def _finite(value):
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _valid_sha256(value):
    """A registered input/artifact hash must be a 64-char SHA-256 hex string."""
    s = str(value)
    return (len(s) == 64
            and all(c in "0123456789abcdefABCDEF" for c in s))


def _valid_git_head(value):
    """A real git head is a 40-char SHA-1 hex string (not an arbitrary label)."""
    s = str(value)
    return (len(s) == 40
            and all(c in "0123456789abcdefABCDEF" for c in s))


def _legal_edge_set(n_tasks):
    """The exact ``K(K+1)`` legal directed edges of a K-task scenario."""
    edges = {(-1, i) for i in range(n_tasks)}
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i != j:
                edges.add((i, j))
    for i in range(n_tasks):
        edges.add((i, n_tasks))
    return edges


def _infer_n_tasks(n_edges):
    """Infer K from ``n_edges == K(K+1)``; ``None`` when it is not of that form."""
    disc = 1 + 4 * int(n_edges)
    r = int(math.isqrt(disc))
    if r * r != disc:
        return None
    k = (r - 1) // 2
    return k if k * (k + 1) == int(n_edges) else None


def _valid_rq2_record(r, layer):
    """An RQ2 record is valid evidence only with a REGISTERED method name for its
    layer and a finite objective/makespan (an evaluated-but-forged record is not
    evidence)."""
    known = ASSIGNMENT_METHODS if layer == "assignment" else SEQUENCE_METHODS
    return (isinstance(r, dict)
            and r.get("status") == "evaluated"
            and isinstance(r.get("method"), str) and r.get("method") in known
            and (_finite(r.get("objective")) or _finite(r.get("makespan_s"))))


def _valid_metric_error(rec):
    """An audit record is valid only if ``metric_error`` reports a known
    approximation mode with finite metric deviations (NaN / forged modes are not
    evidence)."""
    me = rec.get("metric_error")
    if not isinstance(me, dict) or not me:
        return False
    for mode, metrics in me.items():
        if mode not in AUDIT_MODES or not isinstance(metrics, dict):
            continue
        if any(_finite(v) for v in metrics.values()):
            return True
    return False


def _gate(status, detail, evidence=()):
    return {"status": status, "detail": detail, "evidence": list(evidence)}


def _requirement(present, valid, pass_detail, fail_detail, missing_detail):
    """One evidence requirement -> ``(status, detail)``."""
    if not present:
        return (NOT_RUN, missing_detail)
    if valid:
        return (PASS, pass_detail)
    return (FAIL, fail_detail)


def _combine(components):
    """Combine requirement statuses: FAIL wins, then NOT_RUN, else PASS."""
    statuses = [c[0] for c in components]
    if FAIL in statuses:
        status = FAIL
    elif NOT_RUN in statuses:
        status = NOT_RUN
    else:
        status = PASS
    if status == PASS:
        detail = "; ".join(c[1] for c in components)
    else:
        detail = "; ".join(c[1] for c in components if c[0] != PASS)
    return _gate(status, detail)


# --------------------------------------------------------------------------- #
# Gate evaluations
# --------------------------------------------------------------------------- #

def evaluate_gate_0(tests_exit_code):
    """Gate 0: full test suite.  ``tests_exit_code`` is the pytest exit code
    (``None`` = not run)."""
    if tests_exit_code is None:
        return _gate(NOT_RUN, "full test suite not run")
    if tests_exit_code == 0:
        return _gate(PASS, "full test suite passed")
    return _gate(FAIL, "full test suite failed (pytest exit %d)" % tests_exit_code)


def evaluate_gate_1(segment_runs):
    """Gate 1: segment-layer evidence.

    Requires BOTH a straight-hold baseline and at least one communication-aware
    generator (finite feasibility), plus a real common-feasible edge
    intersection: the reported common-feasible IDs must be present in every
    evaluated generator's per-edge records with ``feasible`` true, and each
    generator must cover the full edge set (``n_edges`` == ``len(per_edge)``).
    A lone communication-aware record (no straight-hold baseline) is NOT
    evidence.
    """
    if not segment_runs:
        return _gate(NOT_RUN, "no stc_segment_gate run found")
    valid = [r for r in segment_runs if _ok(r)]
    if not valid:
        return _gate(FAIL, "stc_segment_gate run present but not finalized/valid",
                     ["%s: %s" % (r.run_id, "; ".join(r.problems))
                      for r in segment_runs if r.problems])

    evaluated = {}
    for run in valid:
        for rec in (run.artifacts.get("records.json") or []):
            if not isinstance(rec, dict) or rec.get("status") != "evaluated":
                continue
            gen = rec.get("generator")
            if gen is None or not _finite(rec.get("feasible_frac")):
                continue  # NaN / missing feasibility is not evidence
            evaluated[gen] = rec

    problems = []
    # generator names must be registered (deterministic families or a drl: label)
    for gen in evaluated:
        if gen not in DETERMINISTIC_GENERATORS and not gen.startswith("drl:"):
            problems.append("unknown generator %s" % gen)
    if "straight_hold" not in evaluated:
        problems.append("no straight_hold baseline evaluated")
    comm_aware = [g for g in evaluated if g != "straight_hold"]
    if not comm_aware:
        problems.append("no communication-aware generator evaluated")

    # the common-feasible edge IDs must be real (feasible in every generator's
    # per-edge records), every generator must cover the full legal edge set (no
    # self-loops / forged edges), and every generator must declare the SAME
    # intersection
    common = None
    declared_sets = []
    n_tasks = None
    for gen, rec in evaluated.items():
        per_edge = rec.get("per_edge") or []
        n_edges = rec.get("n_edges")
        if not per_edge or n_edges is None or len(per_edge) != n_edges:
            problems.append("generator %s per-edge coverage incomplete" % gen)
            continue
        k = _infer_n_tasks(n_edges)
        if k is None:
            problems.append("generator %s n_edges is not K(K+1)" % gen)
            continue
        if n_tasks is None:
            n_tasks = k
        elif n_tasks != k:
            problems.append("generators disagree on the scenario size")
        edge_ids = {tuple(pe["edge"]) for pe in per_edge
                    if isinstance(pe, dict) and "edge" in pe}
        if edge_ids != _legal_edge_set(k):
            problems.append("generator %s per-edge set is not the legal edge set "
                            "(self-loops / forged edges)" % gen)
            continue
        feasible_ids = {tuple(pe["edge"]) for pe in per_edge
                        if isinstance(pe, dict) and pe.get("feasible")}
        common = feasible_ids if common is None else (common & feasible_ids)
        declared_sets.append({tuple(e) for e in (rec.get("common_feasible_edges") or [])})
    common = common or set()
    if evaluated:
        if not common:
            problems.append("no common-feasible edge intersection (T/D/O not comparable)")
        elif not declared_sets or any(d != common for d in declared_sets):
            problems.append("common-feasible edge IDs inconsistent with per-edge records")

    if problems:
        return _gate(FAIL, "; ".join(problems), sorted(evaluated))
    return _gate(PASS,
                 "%d generator(s) with a real common-feasible edge intersection"
                 % len(evaluated), sorted(evaluated))


def _rq2_valid(run):
    """Valid RQ2 evidence: BOTH an assignment and a sequence comparison with at
    least one evaluated, metric-carrying record each (a list or single dict is
    accepted).  An evaluated record without a method or finite metric is not
    evidence."""
    rq2 = run.artifacts.get("rq2.json")
    if isinstance(rq2, dict):
        rq2 = [rq2]
    if not isinstance(rq2, list) or not rq2:
        return False
    for entry in rq2:
        if not isinstance(entry, dict):
            continue
        assign = entry.get("assignment") or []
        seq = entry.get("sequence") or []
        if (any(_valid_rq2_record(r, "assignment") for r in assign)
                and any(_valid_rq2_record(r, "sequence") for r in seq)):
            return True
    return False


def evaluate_gate_2(audit_runs, main_runs):
    """Gate 2: synchronous shared-bandwidth audit (RQ3) + assignment/sequence
    evidence (RQ2)."""
    sync_present = bool(audit_runs)
    sync_valid = False
    for run in audit_runs:
        if not _ok(run):
            continue
        for rec in (run.artifacts.get("records.json") or []):
            if isinstance(rec, dict) and _valid_metric_error(rec):
                sync_valid = True
                break
        if sync_valid:
            break

    rq2_present = bool(main_runs)
    rq2_valid = any(_ok(r) and _rq2_valid(r) for r in main_runs)

    return _combine([
        _requirement(sync_present, sync_valid,
                     "synchronous evaluator audited",
                     "synchronous evaluator audit invalid (no metric_error records)",
                     "no stc_evaluator_audit run"),
        _requirement(rq2_present, rq2_valid,
                     "assignment and sequence comparisons present",
                     "assignment/sequence evidence invalid (need evaluated records "
                     "in both assignment and sequence)",
                     "no stc_main run with rq2 evidence"),
    ])


def _is_proposed(rec):
    # only the registered proposed method key counts; a ``proposed=True`` flag on
    # an arbitrary method is NOT evidence of the proposed method
    return rec.get("method") in PROPOSED_METHODS


def _scenario_key(rec):
    return (rec.get("uav_num"), rec.get("map_seed"), rec.get("task_seed"))


def _method_key(rec):
    """A (scenario, method_seed) pairing key: a contrast must share the seed."""
    return (rec.get("uav_num"), rec.get("map_seed"), rec.get("task_seed"),
            rec.get("method_seed"))


def evaluate_gate_3(main_runs, ablation_runs):
    """Gate 3: proposed-method contrast (vs >=2 real baselines with finite
    objective) + at least two mechanisms that genuinely degrade when removed.

    Both checks are evaluated WITHIN a single scenario: evidence is never
    spliced across scenarios (proposed from one scenario + baselines from
    another is not a contrast).  Only the declared ablation mechanism names
    count as contributions; an arbitrary name never does.
    """
    from experiment.run_stc_ablation import ABLATIONS as KNOWN_ABLATIONS

    contrast_present = bool(main_runs)
    contrast_valid = False
    n_baselines = 0
    for run in main_runs:
        if not _ok(run):
            continue
        by_key = {}
        for rec in (run.artifacts.get("raw_records.json") or []):
            if not isinstance(rec, dict) or rec.get("status") != "evaluated":
                continue
            if not _finite(rec.get("objective")):
                continue
            d = by_key.setdefault(_method_key(rec),
                                  {"proposed": False, "baselines": set()})
            if _is_proposed(rec):
                d["proposed"] = True
            elif rec.get("method") in BASELINE_METHODS:
                d["baselines"].add(rec.get("method"))
        for d in by_key.values():
            if d["proposed"] and len(d["baselines"]) >= 2:
                contrast_valid = True
                n_baselines = max(n_baselines, len(d["baselines"]))

    ablation_present = bool(ablation_runs)
    ablation_valid = False
    n_contributing = 0
    for run in ablation_runs:
        if not _ok(run):
            continue
        by_scen = {}
        for rec in (run.artifacts.get("records.json") or []):
            if not isinstance(rec, dict) or rec.get("status") != "evaluated":
                continue
            if not _finite(rec.get("objective")):
                continue  # NaN ablation is not evidence
            name = rec.get("ablation")
            if name != "full" and name not in KNOWN_ABLATIONS:
                continue  # arbitrary names never count as contributions
            d = by_scen.setdefault(_scenario_key(rec), {"full": None, "mech": {}})
            if name == "full":
                d["full"] = rec["objective"]
            else:
                d["mech"][name] = rec["objective"]
        for d in by_scen.values():
            if d["full"] is None:
                continue
            contrib = [n for n, o in d["mech"].items() if o > d["full"]]
            if len(contrib) >= 2:
                ablation_valid = True
                n_contributing = max(n_contributing, len(contrib))

    return _combine([
        _requirement(contrast_present, contrast_valid,
                     "proposed contrast vs %d baselines" % n_baselines,
                     "proposed contrast invalid (need proposed + >=2 real baselines "
                     "with finite objective within one scenario)",
                     "no stc_main run"),
        _requirement(ablation_present, ablation_valid,
                     "%d contributing mechanisms" % n_contributing,
                     "ablation invalid (need a 'full' comparator + >=2 declared "
                     "mechanisms that degrade when removed, within one scenario)",
                     "no stc_ablation run"),
    ])


def _valid_reporting_bundle(path):
    """Return ``True`` for a valid non-empty reporting bundle, ``False`` for an
    invalid/empty one, ``None`` when absent."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(bundle, dict):
        return False
    figures = bundle.get("figures")
    if not isinstance(figures, dict) or not figures:
        return False
    for name, info in figures.items():
        if not isinstance(info, dict) or not _valid_sha256(info.get("sha256")) \
                or not isinstance(info.get("bytes"), int) or info["bytes"] <= 0:
            return False
    return True


def evaluate_gate_4(all_runs, reporting_bundle_path=None):
    """Gate 4: provenance/reproduction — every run is sealed, hash-consistent,
    records a real git head and registered input hashes, and (when a path is
    given) the immutable reporting bundle exists and is valid."""
    if not all_runs:
        return _gate(NOT_RUN, "no runs found")
    invalid = [r for r in all_runs if not _ok(r)]
    if invalid:
        return _gate(FAIL,
                     "non-finalized/valid run(s): %s" % ", ".join(r.run_id for r in invalid),
                     ["%s: %s" % (r.run_id, "; ".join(r.problems))
                      for r in invalid])
    problems = []
    for run in all_runs:
        head = (run.manifest.get("git") or {}).get("head", "")
        if not _valid_git_head(head):
            problems.append("%s: git head not a valid SHA-1 (got %r)"
                            % (run.run_id, head))
        inputs = run.manifest.get("inputs") or {}
        if not inputs:
            problems.append("%s: no registered input hashes" % run.run_id)
        bad_inputs = [k for k, v in inputs.items() if not _valid_sha256(v)]
        if bad_inputs:
            problems.append("%s: invalid input hash (not SHA-256 hex): %s"
                            % (run.run_id, ", ".join(bad_inputs)))
        expected = run.manifest.get("expected") or []
        if not expected:
            problems.append("%s: no expected artifacts declared" % run.run_id)
        hashes = run.manifest.get("hashes") or {}
        if not hashes:
            problems.append("%s: no artifact hashes recorded" % run.run_id)
        # expected artifacts and recorded hashes must correspond exactly, so an
        # unverifiable expected artifact (or a hash for a non-artifact) cannot
        # slip through as "sealed"
        missing = set(expected) - set(hashes)
        extra = set(hashes) - set(expected)
        if missing:
            problems.append("%s: expected artifact(s) without a recorded hash: %s"
                            % (run.run_id, ", ".join(sorted(missing))))
        if extra:
            problems.append("%s: recorded hash for a non-expected artifact: %s"
                            % (run.run_id, ", ".join(sorted(extra))))
    if problems:
        return _gate(FAIL, "; ".join(problems))
    if reporting_bundle_path is not None:
        bundle_state = _valid_reporting_bundle(reporting_bundle_path)
        if bundle_state is None:
            return _gate(FAIL, "reporting bundle missing",
                         [str(reporting_bundle_path)])
        if bundle_state is False:
            return _gate(FAIL, "reporting bundle invalid or empty",
                         [str(reporting_bundle_path)])
    return _gate(PASS, "%d finalized run(s) with sealed provenance" % len(all_runs),
                 [r.run_id for r in all_runs])


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #

def build_report(runs, tests_exit_code=None, reporting_bundle_path=None):
    """Assemble the machine-readable Gate 0-4 report from loaded runs."""
    grouped = classify(runs)
    gates = {
        "gate_0_evaluator": evaluate_gate_0(tests_exit_code),
        "gate_1_segment": evaluate_gate_1(grouped.get("stc_segment_gate", [])),
        "gate_2_assignment_sequence": evaluate_gate_2(
            grouped.get("stc_evaluator_audit", []), grouped.get("stc_main", [])),
        "gate_3_proposed_ablation": evaluate_gate_3(
            grouped.get("stc_main", []), grouped.get("stc_ablation", [])),
        "gate_4_provenance": evaluate_gate_4(runs, reporting_bundle_path),
    }
    statuses = [gates[name]["status"] for name in GATE_ORDER]
    if FAIL in statuses:
        overall = FAIL
    elif all(s == PASS for s in statuses):
        overall = PASS
    else:
        overall = NOT_RUN
    return {
        "gates": gates,
        "overall": overall,
        "runs_scanned": [r.run_id for r in runs],
        "runs_valid": [r.run_id for r in runs if r.valid],
    }


def _run_tests(pytest_args):
    if not pytest_args:
        return None
    cmd = [sys.executable, "-m", "pytest"] + list(pytest_args)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--report", default=None,
                        help="write the JSON report to this path (else stdout)")
    parser.add_argument("--tests", action="store_true",
                        help="run the full test suite for Gate 0")
    parser.add_argument("--tests-args", default=None,
                        help="extra pytest args for Gate 0 (space-separated)")
    parser.add_argument("--skip-tests", action="store_true",
                        help="do not run tests (Gate 0 = NOT_RUN)")
    parser.add_argument("--reporting-bundle", default="results/figs/reporting_bundle.json",
                        help="path to the immutable reporting bundle (Gate 4)")
    args = parser.parse_args(argv)

    tests_exit_code = None
    if args.tests and not args.skip_tests:
        pytest_args = ["tests/"]
        if args.tests_args:
            pytest_args += args.tests_args.split()
        tests_exit_code = _run_tests(pytest_args)

    runs = scan_runs(args.runs_dir)
    report = build_report(runs, tests_exit_code=tests_exit_code,
                          reporting_bundle_path=args.reporting_bundle)
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    if args.report:
        Path(args.report).write_text(payload, encoding="utf-8")
        print(payload)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
