"""Task 10: Gate 0-4 orchestration (experiment/run_stc_gates.py).

The gates are a freeze-time evidence checker over finalized (sealed, status
``success``) run directories.  They never fabricate evidence: a gate with no
evidence is ``NOT_RUN``, a gate whose required evidence is present but invalid
or incomplete is ``FAIL``, and only a gate with complete valid evidence is
``PASS``.
"""

import hashlib
import json

import pytest


def _sha256(data):
    return hashlib.sha256(data if isinstance(data, bytes)
                          else data.encode("utf-8")).hexdigest()


def _write_run(root, run_id, experiment, artifacts, status="success",
               git_head="abc123", inputs=None, hashes=None, expected=None,
               extra_manifest=None):
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, value in artifacts.items():
        data = json.dumps(value, sort_keys=True).encode("utf-8")
        (run_dir / name).write_bytes(data)
        written[name] = _sha256(data)
    manifest = {
        "run_id": run_id,
        "experiment": experiment,
        "status": status,
        "git": {"head": git_head},
        "inputs": inputs or {},
        "hashes": hashes if hashes is not None else written,
        "expected": expected if expected is not None else sorted(artifacts),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (run_dir / "manifest.json").write_bytes(
        json.dumps(manifest, sort_keys=True).encode("utf-8"))
    return run_dir


_EDGES = [[-1, 0], [0, 1], [1, 2], [2, 3]]


def _segment_record(generator, status="evaluated", feasible_frac=0.9,
                    per_edge=None, common_feasible_edges=None):
    edges = per_edge if per_edge is not None else _EDGES
    common = (common_feasible_edges if common_feasible_edges is not None
              else edges)
    return {"generator": generator, "status": status,
            "feasible_frac": feasible_frac, "n_edges": len(edges),
            "common_feasible_n": len(common), "t_mean_s": 1.0,
            "d_mean_s": 1.0, "o_mean": 0.0,
            "per_edge": [{"edge": e, "feasible": True} for e in edges],
            "common_feasible_edges": common}


def _main_record(method, proposed=False, status="evaluated", feasible=True):
    rec = {"method": method, "proposed": proposed, "status": status,
           "feasible": feasible, "objective": 0.9 if feasible else None,
           "makespan_s": 7.0, "total_delay_s": 2.0}
    if status == "unavailable":
        rec.update({"feasible": None, "objective": None,
                    "makespan_s": None, "total_delay_s": None,
                    "failure_reason": "unavailable"})
    return rec


def _audit_record(i=0):
    return {"uav_num": 2, "map_seed": 42, "task_seed": 42,
            "n_plans": 2, "metric_error": {"independent": {"makespan_s_abs": 0.0}}}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def test_load_run_returns_none_without_manifest(tmp_path):
    from experiment import run_stc_gates as gates
    d = tmp_path / "not-a-run"
    d.mkdir()
    assert gates.load_run(d) is None


def test_load_run_valid(tmp_path):
    from experiment import run_stc_gates as gates
    _write_run(tmp_path, "r1", "stc_main",
               {"raw_records.json": [], "summary.json": {}})
    run = gates.load_run(tmp_path / "r1")
    assert run is not None
    assert run.valid is True
    assert run.experiment == "stc_main"
    assert run.status == "success"
    assert run.problems == []


def test_load_run_hash_mismatch(tmp_path):
    from experiment import run_stc_gates as gates
    _write_run(tmp_path, "r1", "stc_main",
               {"raw_records.json": [{"a": 1}], "summary.json": {}})
    (tmp_path / "r1" / "raw_records.json").write_text("[{\"a\": 2}]")
    run = gates.load_run(tmp_path / "r1")
    assert run.valid is False
    assert any("hash" in p for p in run.problems)


def test_load_run_status_failed(tmp_path):
    from experiment import run_stc_gates as gates
    _write_run(tmp_path, "r1", "stc_main",
               {"raw_records.json": []}, status="failed")
    run = gates.load_run(tmp_path / "r1")
    assert run.valid is False
    assert any("status" in p for p in run.problems)


def test_scan_runs_skips_non_runs(tmp_path):
    from experiment import run_stc_gates as gates
    _write_run(tmp_path, "r1", "stc_main", {"summary.json": {}})
    (tmp_path / "empty").mkdir()
    runs = gates.scan_runs(tmp_path)
    assert {r.run_id for r in runs} == {"r1"}


# --------------------------------------------------------------------------- #
# Gate 0
# --------------------------------------------------------------------------- #

def test_gate_0_statuses():
    from experiment import run_stc_gates as gates
    assert gates.evaluate_gate_0(None)["status"] == "NOT_RUN"
    assert gates.evaluate_gate_0(0)["status"] == "PASS"
    assert gates.evaluate_gate_0(1)["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# Gate 1: segment evidence
# --------------------------------------------------------------------------- #

def test_gate_1_not_run_without_segment_run():
    from experiment import run_stc_gates as gates
    assert gates.evaluate_gate_1([])["status"] == "NOT_RUN"


def test_gate_1_fail_without_comm_aware():
    from experiment import run_stc_gates as gates
    import tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp())
    # only straight_hold evaluated -> no communication-aware generator
    artifact = {"records.json": [_segment_record("straight_hold")]}
    run = gates.load_run(_write_run(root, "seg", "stc_segment_gate", artifact))
    assert gates.evaluate_gate_1([run])["status"] == "FAIL"


def test_gate_1_pass_with_comm_aware():
    from experiment import run_stc_gates as gates
    import tempfile
    root = tempfile.mkdtemp()
    from pathlib import Path
    root = Path(root)
    artifact = {"records.json": [_segment_record("straight_hold"),
                                 _segment_record("comm_waypoint")]}
    run = gates.load_run(_write_run(root, "seg", "stc_segment_gate", artifact))
    assert gates.evaluate_gate_1([run])["status"] == "PASS"


# --------------------------------------------------------------------------- #
# Gate 2: assignment/sequence + synchronous evaluator
# --------------------------------------------------------------------------- #

def test_gate_2_not_run_without_both_families():
    from experiment import run_stc_gates as gates
    assert gates.evaluate_gate_2([], [])["status"] == "NOT_RUN"


def test_gate_2_fail_missing_rq2():
    from experiment import run_stc_gates as gates
    import tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp())
    audit = gates.load_run(_write_run(root, "audit", "stc_evaluator_audit",
                                      {"records.json": [_audit_record()]}))
    main = gates.load_run(_write_run(root, "main", "stc_main",
                                     {"raw_records.json": [_main_record("proposed_stc_alns", True)]}))
    result = gates.evaluate_gate_2([audit], [main])
    assert result["status"] == "FAIL"
    assert "assignment" in result["detail"].lower() or "rq2" in result["detail"].lower()


def test_gate_2_pass_with_both():
    from experiment import run_stc_gates as gates
    import tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp())
    audit = gates.load_run(_write_run(root, "audit", "stc_evaluator_audit",
                                      {"records.json": [_audit_record()]}))
    main = gates.load_run(_write_run(
        root, "main", "stc_main",
        {"raw_records.json": [_main_record("proposed_stc_alns", True)],
         "rq2.json": {"assignment": [{"method": "jia_ebtas", "status": "evaluated",
                                      "objective": 0.9}],
                      "sequence": [{"method": "distance_nn_2opt", "status": "evaluated",
                                    "objective": 0.8}]}}))
    assert gates.evaluate_gate_2([audit], [main])["status"] == "PASS"


def test_gate_2_fail_on_invalid_audit_and_rq2():
    from experiment import run_stc_gates as gates
    import tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp())
    # audit with only an empty record (no metric_error) is invalid evidence
    audit = gates.load_run(_write_run(root, "audit", "stc_evaluator_audit",
                                      {"records.json": [{}]}))
    # rq2 whose assignment is only "unavailable" and sequence is empty
    main = gates.load_run(_write_run(
        root, "main", "stc_main",
        {"raw_records.json": [_main_record("proposed_stc_alns", True)],
         "rq2.json": {"assignment": [{"method": "jia_ebtas", "status": "unavailable"}],
                      "sequence": []}}))
    assert gates.evaluate_gate_2([audit], [main])["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# Gate 3: proposed contrast + >=2 ablations
# --------------------------------------------------------------------------- #

def test_gate_3_not_run():
    from experiment import run_stc_gates as gates
    assert gates.evaluate_gate_3([], [])["status"] == "NOT_RUN"


def test_gate_3_not_run_without_ablations():
    from experiment import run_stc_gates as gates
    import tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp())
    main = gates.load_run(_write_run(
        root, "main", "stc_main",
        {"raw_records.json": [_main_record("proposed_stc_alns", True),
                              _main_record("jia_ebtas_style"),
                              _main_record("gatsac_style")]}))
    # ablation evidence is entirely absent -> NOT_RUN, not FAIL
    assert gates.evaluate_gate_3([main], [])["status"] == "NOT_RUN"


def test_gate_3_fail_on_fake_baselines():
    from experiment import run_stc_gates as gates
    import tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp())
    # fake baseline names (not in the registered set) are not real contrast
    main = gates.load_run(_write_run(
        root, "main", "stc_main",
        {"raw_records.json": [_main_record("proposed_stc_alns", True),
                              _main_record("totally_fake_baseline"),
                              _main_record("another_fake")]}))
    ablation = gates.load_run(_write_run(
        root, "ablation", "stc_ablation",
        {"records.json": [{"ablation": "full", "status": "evaluated", "objective": 0.8},
                          {"ablation": "a", "status": "evaluated", "objective": 0.9},
                          {"ablation": "b", "status": "evaluated", "objective": 0.85}]}))
    assert gates.evaluate_gate_3([main], [ablation])["status"] == "FAIL"


def test_gate_3_fail_on_no_full_comparator_or_nan():
    from experiment import run_stc_gates as gates
    import tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp())
    main = gates.load_run(_write_run(
        root, "main", "stc_main",
        {"raw_records.json": [_main_record("proposed_stc_alns", True),
                              _main_record("jia_ebtas_style"),
                              _main_record("library_open_loop")]}))
    # no "full" comparator; one ablation objective is None (NaN)
    ablation = gates.load_run(_write_run(
        root, "ablation", "stc_ablation",
        {"records.json": [{"ablation": "a", "status": "evaluated", "objective": 0.9},
                          {"ablation": "b", "status": "evaluated", "objective": None}]}))
    assert gates.evaluate_gate_3([main], [ablation])["status"] == "FAIL"


def test_gate_1_fail_on_nan_feasibility():
    from experiment import run_stc_gates as gates
    import tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp())
    # single comm-waypoint with NaN feasibility is not valid evidence
    run = gates.load_run(_write_run(
        root, "seg", "stc_segment_gate",
        {"records.json": [{"generator": "comm_waypoint", "status": "evaluated",
                           "feasible_frac": None, "common_feasible_n": 0}]}))
    assert gates.evaluate_gate_1([run])["status"] == "FAIL"


def test_gate_3_pass():
    from experiment import run_stc_gates as gates
    import tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp())
    main = gates.load_run(_write_run(
        root, "main", "stc_main",
        {"raw_records.json": [_main_record("proposed_stc_alns", True),
                              _main_record("jia_ebtas_style"),
                              _main_record("gatsac_style")]}))
    ablation = gates.load_run(_write_run(
        root, "ablation", "stc_ablation",
        {"records.json": [{"ablation": "full", "status": "evaluated", "objective": 0.8},
                          {"ablation": "ebtas_init", "status": "evaluated", "objective": 0.9},
                          {"ablation": "path_switch", "status": "evaluated", "objective": 0.85}]}))
    assert gates.evaluate_gate_3([main], [ablation])["status"] == "PASS"


# --------------------------------------------------------------------------- #
# Gate 4: provenance / reproduction
# --------------------------------------------------------------------------- #

def test_gate_4_not_run():
    from experiment import run_stc_gates as gates
    assert gates.evaluate_gate_4([])["status"] == "NOT_RUN"


def test_gate_4_fail_on_invalid_run():
    from experiment import run_stc_gates as gates
    import tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp())
    good = gates.load_run(_write_run(root, "good", "stc_main",
                                     {"summary.json": {}}))
    bad = gates.load_run(_write_run(root, "bad", "stc_main",
                                    {"summary.json": {}}, status="failed"))
    result = gates.evaluate_gate_4([good, bad])
    assert result["status"] == "FAIL"


def test_gate_4_pass():
    from experiment import run_stc_gates as gates
    import tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp())
    good = gates.load_run(_write_run(root, "good", "stc_main",
                                     {"summary.json": {}},
                                     inputs={"scenario": "a" * 64}))
    assert gates.evaluate_gate_4([good])["status"] == "PASS"


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #

def test_build_report_overall_not_run(tmp_path):
    from experiment import run_stc_gates as gates
    report = gates.build_report([], tests_exit_code=None)
    assert report["overall"] == "NOT_RUN"
    assert set(report["gates"]) == set(gates.GATE_ORDER)


def test_build_report_overall_fail_when_any_fail(tmp_path):
    from experiment import run_stc_gates as gates
    import tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp())
    bad = gates.load_run(_write_run(root, "bad", "stc_main",
                                    {"summary.json": {}}, status="failed"))
    report = gates.build_report([bad], tests_exit_code=0)
    # gate 0 passes, gate 4 fails -> overall FAIL
    assert report["gates"]["gate_0_evaluator"]["status"] == "PASS"
    assert report["gates"]["gate_4_provenance"]["status"] == "FAIL"
    assert report["overall"] == "FAIL"
