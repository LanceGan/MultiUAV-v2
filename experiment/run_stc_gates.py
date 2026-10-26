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
    """Gate 1: segment-layer evidence (finite feasibility + both families +
    a non-empty common-feasible edge intersection)."""
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

    deterministic = [g for g in DETERMINISTIC_GENERATORS if g in evaluated]
    comm_aware = [g for g in evaluated if g != "straight_hold"]
    common_ok = any(isinstance(r.get("common_feasible_n"), (int, float))
                    and r["common_feasible_n"] > 0 for r in evaluated.values())

    if deterministic and comm_aware and common_ok:
        return _gate(PASS,
                     "%d deterministic and %d communication-aware generators, "
                     "common-feasible edges present"
                     % (len(deterministic), len(comm_aware)),
                     sorted(evaluated))
    problems = []
    if not deterministic:
        problems.append("no deterministic generator evaluated with finite feasibility")
    if not comm_aware:
        problems.append("no communication-aware generator evaluated")
    if not common_ok:
        problems.append("no common-feasible edge intersection (T/D/O not comparable)")
    return _gate(FAIL, "; ".join(problems), sorted(evaluated))


def _rq2_valid(run):
    """Valid RQ2 evidence: BOTH an assignment and a sequence comparison with at
    least one evaluated record each (a list or single dict is accepted)."""
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
        assign_ok = any(isinstance(r, dict) and r.get("status") == "evaluated"
                        for r in assign)
        seq_ok = any(isinstance(r, dict) and r.get("status") == "evaluated"
                     for r in seq)
        if assign_ok and seq_ok:
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
            if isinstance(rec, dict) and rec.get("metric_error"):
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
    return bool(rec.get("proposed")) or rec.get("method") in PROPOSED_METHODS


def evaluate_gate_3(main_runs, ablation_runs):
    """Gate 3: proposed-method contrast (vs >=2 real baselines with finite
    objective) + at least two mechanisms that genuinely degrade when removed."""
    contrast_present = bool(main_runs)
    proposed_ok = False
    baselines = set()
    for run in main_runs:
        if not _ok(run):
            continue
        for rec in (run.artifacts.get("raw_records.json") or []):
            if not isinstance(rec, dict) or rec.get("status") != "evaluated":
                continue
            if not _finite(rec.get("objective")):
                continue
            if _is_proposed(rec):
                proposed_ok = True
            elif rec.get("method") in BASELINE_METHODS:
                baselines.add(rec.get("method"))
    contrast_valid = proposed_ok and len(baselines) >= 2

    ablation_present = bool(ablation_runs)
    full_obj = None
    mechanisms = {}
    for run in ablation_runs:
        if not _ok(run):
            continue
        for rec in (run.artifacts.get("records.json") or []):
            if not isinstance(rec, dict) or rec.get("status") != "evaluated":
                continue
            if not _finite(rec.get("objective")):
                continue  # NaN ablation is not evidence
            name = rec.get("ablation")
            if name == "full":
                if full_obj is None:
                    full_obj = rec["objective"]
            elif name:
                mechanisms[name] = rec["objective"]
    contributing = [n for n, o in mechanisms.items()
                    if full_obj is not None and o > full_obj]
    ablation_valid = full_obj is not None and len(contributing) >= 2

    return _combine([
        _requirement(contrast_present, contrast_valid,
                     "proposed contrast vs %d baselines" % len(baselines),
                     "proposed contrast invalid (need proposed + >=2 real baselines "
                     "with finite objective)",
                     "no stc_main run"),
        _requirement(ablation_present, ablation_valid,
                     "%d contributing mechanisms" % len(contributing),
                     "ablation invalid (need a 'full' comparator + >=2 mechanisms "
                     "that degrade when removed)",
                     "no stc_ablation run"),
    ])


def evaluate_gate_4(all_runs):
    """Gate 4: provenance/reproduction — every run is sealed, hash-consistent,
    and records a real git head and registered input hashes."""
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
        if not head or head == "unknown":
            problems.append("%s: git head missing/unknown" % run.run_id)
        inputs = run.manifest.get("inputs") or {}
        if not inputs or not any(str(v) for v in inputs.values()):
            problems.append("%s: no registered input hashes" % run.run_id)
    if problems:
        return _gate(FAIL, "; ".join(problems))
    return _gate(PASS, "%d finalized run(s) with sealed provenance" % len(all_runs),
                 [r.run_id for r in all_runs])


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #

def build_report(runs, tests_exit_code=None):
    """Assemble the machine-readable Gate 0-4 report from loaded runs."""
    grouped = classify(runs)
    gates = {
        "gate_0_evaluator": evaluate_gate_0(tests_exit_code),
        "gate_1_segment": evaluate_gate_1(grouped.get("stc_segment_gate", [])),
        "gate_2_assignment_sequence": evaluate_gate_2(
            grouped.get("stc_evaluator_audit", []), grouped.get("stc_main", [])),
        "gate_3_proposed_ablation": evaluate_gate_3(
            grouped.get("stc_main", []), grouped.get("stc_ablation", [])),
        "gate_4_provenance": evaluate_gate_4(runs),
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
    args = parser.parse_args(argv)

    tests_exit_code = None
    if args.tests and not args.skip_tests:
        pytest_args = ["tests/"]
        if args.tests_args:
            pytest_args += args.tests_args.split()
        tests_exit_code = _run_tests(pytest_args)

    runs = scan_runs(args.runs_dir)
    report = build_report(runs, tests_exit_code=tests_exit_code)
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    if args.report:
        Path(args.report).write_text(payload, encoding="utf-8")
        print(payload)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
