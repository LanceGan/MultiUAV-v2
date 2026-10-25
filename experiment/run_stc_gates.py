"""Gate 0-4 orchestration: a freeze-time evidence check over finalized runs.

Each gate inspects the finalized (sealed, ``status == "success"``) run
directories under ``results/runs/`` and reports PASS / FAIL / NOT_RUN:

- Gate 0  evaluator / full test suite
- Gate 1  segment-layer evidence (RQ1)
- Gate 2  assignment/sequence (RQ2) + synchronous shared-bandwidth (RQ3) evidence
- Gate 3  proposed-method contrast + at least two contributing ablations (RQ5)
- Gate 4  provenance / reproduction (sealed manifests, hash-consistent)

The report is machine-readable JSON.  A gate whose evidence is entirely absent
is ``NOT_RUN``; a gate whose evidence is present but invalid or incomplete is
``FAIL``; only complete valid evidence is ``PASS``.  Nothing here fabricates a
result: it only verifies what finalized runs already contain.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.run_stc_segment_gate import DETERMINISTIC_GENERATORS

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"

GATE_ORDER = ("gate_0_evaluator", "gate_1_segment",
              "gate_2_assignment_sequence", "gate_3_proposed_ablation",
              "gate_4_provenance")

PROPOSED_METHODS = ("proposed_stc_alns",)


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
    A run whose manifest exists but is not sealed/consistent is returned with
    ``valid=False`` and its problems recorded.
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


def _nonempty_list(run, name):
    value = run.artifacts.get(name)
    return isinstance(value, list) and bool(value)


def _gate(status, detail, evidence=()):
    return {"status": status, "detail": detail, "evidence": list(evidence)}


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
    """Gate 1: segment-layer evidence from finalized ``stc_segment_gate`` runs."""
    if not segment_runs:
        return _gate(NOT_RUN, "no stc_segment_gate run found")
    valid = [r for r in segment_runs if _ok(r)]
    if not valid:
        return _gate(FAIL, "stc_segment_gate run present but not finalized/valid",
                     ["%s: %s" % (r.run_id, "; ".join(r.problems))
                      for r in segment_runs if r.problems])

    evaluated = set()
    for run in valid:
        for rec in (run.artifacts.get("records.json") or []):
            if not isinstance(rec, dict):
                continue
            if rec.get("status") == "evaluated" and rec.get("feasible_frac") is not None:
                evaluated.add(rec.get("generator"))

    deterministic = evaluated & set(DETERMINISTIC_GENERATORS)
    comm_aware = evaluated - {"straight_hold"}
    if deterministic and comm_aware:
        return _gate(PASS,
                     "%d deterministic and %d communication-aware segment "
                     "generators evaluated" % (len(deterministic), len(comm_aware)),
                     sorted(evaluated))
    if not deterministic:
        return _gate(FAIL, "no deterministic segment generator evaluated with "
                           "feasibility", sorted(evaluated))
    return _gate(FAIL, "no communication-aware segment generator evaluated "
                       "(only straight_hold)", sorted(evaluated))


def _has_rq2_evidence(run):
    rq2 = run.artifacts.get("rq2.json")
    if not isinstance(rq2, dict):
        return False
    return bool(rq2.get("assignment")) or bool(rq2.get("sequence"))


def evaluate_gate_2(audit_runs, main_runs):
    """Gate 2: synchronous shared-bandwidth audit (RQ3) + assignment/sequence
    evidence (RQ2)."""
    if not (audit_runs or main_runs):
        return _gate(NOT_RUN, "no stc_evaluator_audit or stc_main run found")
    sync_ok = any(_ok(r) and _nonempty_list(r, "records.json")
                  for r in audit_runs)
    rq2_ok = any(_ok(r) and _has_rq2_evidence(r) for r in main_runs)
    if sync_ok and rq2_ok:
        return _gate(PASS, "synchronous evaluator audited and assignment/sequence "
                           "evidence present")
    missing = []
    if not sync_ok:
        missing.append("synchronous shared-bandwidth audit (stc_evaluator_audit)")
    if not rq2_ok:
        missing.append("assignment/sequence evidence (rq2.json in stc_main)")
    return _gate(FAIL, "missing evidence: " + ", ".join(missing))


def _is_proposed(rec):
    return bool(rec.get("proposed")) or rec.get("method") in PROPOSED_METHODS


def evaluate_gate_3(main_runs, ablation_runs):
    """Gate 3: proposed-method contrast + at least two contributing ablations."""
    if not (main_runs or ablation_runs):
        return _gate(NOT_RUN, "no stc_main or stc_ablation run found")

    proposed_seen = False
    baseline_methods = set()
    for run in main_runs:
        if not _ok(run):
            continue
        for rec in (run.artifacts.get("raw_records.json") or []):
            if not isinstance(rec, dict):
                continue
            if rec.get("status") != "evaluated":
                continue
            if _is_proposed(rec):
                proposed_seen = True
            else:
                baseline_methods.add(rec.get("method"))

    ablation_contributions = set()
    for run in ablation_runs:
        if not _ok(run):
            continue
        for rec in (run.artifacts.get("records.json") or []):
            if not isinstance(rec, dict):
                continue
            if rec.get("ablation") == "full":
                continue
            if rec.get("status") == "evaluated" and rec.get("objective") is not None:
                ablation_contributions.add(rec.get("ablation"))

    contrast_ok = proposed_seen and len(baseline_methods) >= 2
    ablation_ok = len(ablation_contributions) >= 2
    if contrast_ok and ablation_ok:
        return _gate(PASS,
                     "proposed contrast against %d baselines and %d contributing "
                     "ablations" % (len(baseline_methods),
                                    len(ablation_contributions)),
                     sorted(ablation_contributions))
    missing = []
    if not contrast_ok:
        missing.append("proposed-method contrast (proposed + >=2 evaluated baselines)")
    if not ablation_ok:
        missing.append(">=2 contributing ablations")
    return _gate(FAIL, "missing evidence: " + ", ".join(missing))


def evaluate_gate_4(all_runs):
    """Gate 4: provenance/reproduction — every finalized run is sealed with
    hash-consistent artifacts, a recorded git head, and registered inputs."""
    if not all_runs:
        return _gate(NOT_RUN, "no runs found")
    valid = [r for r in all_runs if _ok(r)]
    invalid = [r for r in all_runs if not _ok(r)]
    if not valid:
        return _gate(FAIL, "no finalized/sealed run among %d run(s)" % len(all_runs),
                     ["%s: %s" % (r.run_id, "; ".join(r.problems))
                      for r in invalid])
    problems = []
    for run in valid:
        git = run.manifest.get("git") or {}
        if not git.get("head"):
            problems.append("%s: no git head recorded" % run.run_id)
        if not run.manifest.get("inputs"):
            problems.append("%s: no registered input hashes" % run.run_id)
    if problems:
        return _gate(FAIL, "; ".join(problems))
    return _gate(PASS, "%d finalized run(s) with sealed provenance" % len(valid),
                 [r.run_id for r in valid])


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
