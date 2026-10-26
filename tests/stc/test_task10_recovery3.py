"""Task 10 recovery round 3: authenticity hardening after the third review.

The reviewer constructed an obviously-invalid evidence set (self-loop edge
inventory, forged generators / RQ2 methods, NaN audit, a ``proposed=True`` fake
Proposed, disjoint method seeds, fake git identity, empty input hash, artifact
hash unrelated to ``expected``) that still passed Gates 0-4.  These tests pin the
authenticity checks, plus paired plot aggregation and the write-once reporting
bundle.
"""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest


def _sha256(data):
    return hashlib.sha256(data if isinstance(data, bytes)
                          else data.encode("utf-8")).hexdigest()


def _write_run(root, run_id, experiment, artifacts, status="success",
               git_head="a" * 40, inputs=None, hashes=None, expected=None):
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, value in artifacts.items():
        data = json.dumps(value, sort_keys=True).encode("utf-8")
        (run_dir / name).write_bytes(data)
        written[name] = _sha256(data)
    manifest = {
        "run_id": run_id, "experiment": experiment, "status": status,
        "git": {"head": git_head},
        "inputs": inputs if inputs is not None else {},
        "hashes": hashes if hashes is not None else written,
        "expected": expected if expected is not None else sorted(artifacts),
    }
    (run_dir / "manifest.json").write_bytes(
        json.dumps(manifest, sort_keys=True).encode("utf-8"))
    return run_dir


_EDGES = [[-1, 0], [-1, 1], [0, 1], [1, 0], [0, 2], [1, 2]]  # K=2 legal set


def _segment_record(generator, status="evaluated", feasible_frac=0.9,
                    per_edge=None, common_feasible_edges=None):
    edges = per_edge if per_edge is not None else _EDGES
    common = (common_feasible_edges if common_feasible_edges is not None
              else edges)
    return {"generator": generator, "status": status,
            "feasible_frac": feasible_frac, "n_edges": len(edges),
            "common_feasible_n": len(common), "t_mean_s": 1.0,
            "d_mean_s": 1.0, "o_mean": 0.0,
            "per_edge": [{"edge": e, "feasible": True,
                          "candidate_id": "c", "trajectory_hash": "h" * 64}
                         for e in edges],
            "common_feasible_edges": common}


def _main_record(method, proposed=False, status="evaluated", feasible=True,
                 scen=(None, None, None), method_seed=None):
    rec = {"method": method, "proposed": proposed, "status": status,
           "feasible": feasible, "objective": 0.9 if feasible else None,
           "makespan_s": 7.0, "total_delay_s": 2.0,
           "uav_num": scen[0], "map_seed": scen[1], "task_seed": scen[2],
           "method_seed": method_seed}
    if status == "unavailable":
        rec.update({"feasible": None, "objective": None,
                    "makespan_s": None, "total_delay_s": None})
    return rec


# --------------------------------------------------------------------------- #
# Gate 1: self-loops / forged generators are rejected
# --------------------------------------------------------------------------- #

def test_gate_1_fail_self_loop_edges():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    self_loop = [{"edge": [-1, 0], "feasible": True},
                 {"edge": [0, 0], "feasible": True},  # self-loop
                 {"edge": [0, 1], "feasible": True},
                 {"edge": [1, 0], "feasible": True},
                 {"edge": [0, 2], "feasible": True},
                 {"edge": [1, 2], "feasible": True}]
    rec = _segment_record("straight_hold", per_edge=self_loop,
                          common_feasible_edges=[[-1, 0]])
    rec2 = _segment_record("comm_waypoint")
    run = gates.load_run(_write_run(root, "seg", "stc_segment_gate",
                                    {"records.json": [rec, rec2]}))
    assert gates.evaluate_gate_1([run])["status"] == "FAIL"


def test_gate_1_fail_unknown_generator():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    fake = _segment_record("totally_fake_generator")
    run = gates.load_run(_write_run(root, "seg", "stc_segment_gate",
                                    {"records.json": [fake]}))
    assert gates.evaluate_gate_1([run])["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# Gate 2: forged RQ2 method names are rejected
# --------------------------------------------------------------------------- #

def test_gate_2_fail_forged_rq2_method():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    audit = gates.load_run(_write_run(
        root, "audit", "stc_evaluator_audit",
        {"records.json": [{"metric_error": {"independent": {"makespan_s_abs": 0.0}}}]}))
    main = gates.load_run(_write_run(
        root, "main", "stc_main",
        {"raw_records.json": [_main_record("proposed_stc_alns", True)],
         "rq2.json": {"assignment": [{"method": "forged_assign", "status": "evaluated",
                                      "objective": 0.9}],
                      "sequence": [{"method": "distance_nn_2opt", "status": "evaluated",
                                    "objective": 0.8}]}}))
    assert gates.evaluate_gate_2([audit], [main])["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# Gate 3: proposed=True on a fake method / disjoint method seeds are rejected
# --------------------------------------------------------------------------- #

def test_gate_3_fail_fake_proposed_flag():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    # a fake method claiming proposed=True is not the Proposed method
    main = gates.load_run(_write_run(
        root, "main", "stc_main",
        {"raw_records.json": [
            _main_record("some_fake", proposed=True, scen=(2, 42, 42)),
            _main_record("jia_ebtas_style", scen=(2, 42, 42)),
            _main_record("library_open_loop", scen=(2, 42, 42)),
        ]}))
    ablation = gates.load_run(_write_run(
        root, "ablation", "stc_ablation",
        {"records.json": [
            {"ablation": "full", "status": "evaluated", "objective": 0.8,
             "uav_num": 2, "map_seed": 42, "task_seed": 42},
            {"ablation": "ebtas_init", "status": "evaluated", "objective": 0.9,
             "uav_num": 2, "map_seed": 42, "task_seed": 42},
            {"ablation": "path_switch", "status": "evaluated", "objective": 0.85,
             "uav_num": 2, "map_seed": 42, "task_seed": 42},
        ]}))
    assert gates.evaluate_gate_3([main], [ablation])["status"] == "FAIL"


def test_gate_3_fail_disjoint_method_seeds():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    # proposed uses method_seed 0, baselines use method_seed 1 -> not a paired contrast
    main = gates.load_run(_write_run(
        root, "main", "stc_main",
        {"raw_records.json": [
            _main_record("proposed_stc_alns", True, scen=(2, 42, 42), method_seed=0),
            _main_record("jia_ebtas_style", scen=(2, 42, 42), method_seed=1),
            _main_record("library_open_loop", scen=(2, 42, 42), method_seed=1),
        ]}))
    ablation = gates.load_run(_write_run(
        root, "ablation", "stc_ablation",
        {"records.json": [
            {"ablation": "full", "status": "evaluated", "objective": 0.8,
             "uav_num": 2, "map_seed": 42, "task_seed": 42},
            {"ablation": "ebtas_init", "status": "evaluated", "objective": 0.9,
             "uav_num": 2, "map_seed": 42, "task_seed": 42},
            {"ablation": "path_switch", "status": "evaluated", "objective": 0.85,
             "uav_num": 2, "map_seed": 42, "task_seed": 42},
        ]}))
    assert gates.evaluate_gate_3([main], [ablation])["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# Gate 4: fake git identity / empty input hash / unrelated artifact hash /
#         reporting bundle
# --------------------------------------------------------------------------- #

def test_gate_4_fail_fake_git_identity():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    run = gates.load_run(_write_run(root, "r", "stc_main", {"summary.json": {}},
                                    git_head="abc123",
                                    inputs={"scenario": "a" * 64}))
    assert gates.evaluate_gate_4([run])["status"] == "FAIL"


def test_gate_4_fail_empty_input_hash():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    run = gates.load_run(_write_run(root, "r", "stc_main", {"summary.json": {}},
                                    inputs={"scenario": ""}))
    assert gates.evaluate_gate_4([run])["status"] == "FAIL"


def test_gate_4_fail_unrelated_artifact_hash():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    run = gates.load_run(_write_run(root, "r", "stc_main", {"summary.json": {}},
                                    inputs={"scenario": "a" * 64},
                                    hashes={"unrelated": "a" * 64}))
    assert gates.evaluate_gate_4([run])["status"] == "FAIL"


def test_gate_4_fail_missing_reporting_bundle():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    run = gates.load_run(_write_run(root, "r", "stc_main", {"summary.json": {}},
                                    inputs={"scenario": "a" * 64}))
    result = gates.evaluate_gate_4([run], reporting_bundle_path="missing.json")
    assert result["status"] == "FAIL"


def test_gate_4_fail_empty_reporting_bundle(tmp_path):
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    run = gates.load_run(_write_run(root, "r", "stc_main", {"summary.json": {}},
                                    inputs={"scenario": "a" * 64}))
    bundle = tmp_path / "reporting_bundle.json"
    bundle.write_text(json.dumps({"figures": {}}))
    result = gates.evaluate_gate_4([run], reporting_bundle_path=str(bundle))
    assert result["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# Plot: common-feasible pairing
# --------------------------------------------------------------------------- #

def _td_record(method, scen, feasible=True):
    return {"method": method, "status": "evaluated", "feasible": feasible,
            "makespan_s": 7.0, "total_delay_s": 2.0,
            "reference": {"reference": {"t_ref_s": 8.0, "d_ref_s": 2.0}},
            "uav_num": 2, "map_seed": scen, "task_seed": 1}


def test_extract_td_points_requires_common_feasible_scenario():
    from plots import plot_stc_results as plots
    records = [
        _td_record("a", scen=1, feasible=True),
        _td_record("b", scen=1, feasible=False),
        _td_record("a", scen=2, feasible=False),
        _td_record("b", scen=2, feasible=True),
    ]
    # no scenario is feasible for both methods -> no comparison
    assert plots.extract_td_points(records) == []


# --------------------------------------------------------------------------- #
# Reporting bundle: write-once + refuse empty
# --------------------------------------------------------------------------- #

def test_reporting_bundle_refuses_empty(tmp_path):
    from plots import plot_stc_results as plots
    with pytest.raises(ValueError):
        plots._write_reporting_bundle(tmp_path, [], {})


def test_reporting_bundle_is_write_once(tmp_path):
    from plots import plot_stc_results as plots
    fig = tmp_path / "f.png"
    fig.write_bytes(b"fake")
    plots._write_reporting_bundle(tmp_path, [fig], {"f.png": {}})
    with pytest.raises(FileExistsError):
        plots._write_reporting_bundle(tmp_path, [fig], {"f.png": {}})
