"""Task 10 recovery round 2: fixes for the second independent review.

Covers: Gate 1 requiring a straight-hold baseline + real common-feasible edge
IDs, Gate 2 rejecting metric-less evaluated records / NaN audit, Gate 3 not
splicing evidence across scenarios nor counting arbitrary ablation names,
Gate 4 rejecting empty expected/hashes and invalid input hashes, plot
compatibility accepting valid multi-scenario but rejecting mixed source
versions, ablation/scale/segment aggregation without overwriting replicates
(scale retains method/family identity), and an immutable content-addressed
reporting bundle.
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
        "run_id": run_id,
        "experiment": experiment,
        "status": status,
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
            "per_edge": [{"edge": e, "feasible": True} for e in edges],
            "common_feasible_edges": common}


def _main_record(method, proposed=False, status="evaluated", feasible=True,
                 scen=(None, None, None)):
    rec = {"method": method, "proposed": proposed, "status": status,
           "feasible": feasible, "objective": 0.9 if feasible else None,
           "makespan_s": 7.0, "total_delay_s": 2.0,
           "uav_num": scen[0], "map_seed": scen[1], "task_seed": scen[2]}
    if status == "unavailable":
        rec.update({"feasible": None, "objective": None,
                    "makespan_s": None, "total_delay_s": None})
    return rec


def _audit_record(metric_error=None):
    return {"uav_num": 2, "map_seed": 42, "task_seed": 42, "n_plans": 2,
            "metric_error": metric_error}


# --------------------------------------------------------------------------- #
# Gate 1: straight-hold baseline + real common-feasible IDs
# --------------------------------------------------------------------------- #

def test_gate_1_fail_lone_comm_waypoint_no_baseline():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    run = gates.load_run(_write_run(root, "seg", "stc_segment_gate",
                                    {"records.json": [_segment_record("comm_waypoint")]}))
    assert gates.evaluate_gate_1([run])["status"] == "FAIL"


def test_gate_1_fail_inconsistent_common_ids():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    artifact = {"records.json": [
        _segment_record("straight_hold", common_feasible_edges=_EDGES),
        _segment_record("comm_waypoint", common_feasible_edges=_EDGES[:2]),
    ]}
    run = gates.load_run(_write_run(root, "seg", "stc_segment_gate", artifact))
    assert gates.evaluate_gate_1([run])["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# Gate 2: metric-less evaluated records / NaN audit are not evidence
# --------------------------------------------------------------------------- #

def test_gate_2_fail_evaluated_rq2_without_metric():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    audit = gates.load_run(_write_run(
        root, "audit", "stc_evaluator_audit",
        {"records.json": [_audit_record({"independent": {"makespan_s_abs": 0.0}})]}))
    main = gates.load_run(_write_run(
        root, "main", "stc_main",
        {"raw_records.json": [_main_record("proposed_stc_alns", True)],
         "rq2.json": {"assignment": [{"method": "jia_ebtas", "status": "evaluated"}],
                      "sequence": [{"method": "distance_nn_2opt", "status": "evaluated"}]}}))
    assert gates.evaluate_gate_2([audit], [main])["status"] == "FAIL"


def test_gate_2_fail_nan_audit():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    audit = gates.load_run(_write_run(
        root, "audit", "stc_evaluator_audit",
        {"records.json": [_audit_record({"independent": {"makespan_s_abs": None}})]}))
    main = gates.load_run(_write_run(
        root, "main", "stc_main",
        {"raw_records.json": [_main_record("proposed_stc_alns", True)],
         "rq2.json": {"assignment": [{"method": "jia_ebtas", "status": "evaluated",
                                      "objective": 0.9}],
                      "sequence": [{"method": "distance_nn_2opt", "status": "evaluated",
                                    "objective": 0.8}]}}))
    assert gates.evaluate_gate_2([audit], [main])["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# Gate 3: no cross-scenario splicing, no arbitrary ablation names
# --------------------------------------------------------------------------- #

def test_gate_3_fail_cross_scenario_splice():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    main = gates.load_run(_write_run(
        root, "main", "stc_main",
        {"raw_records.json": [
            _main_record("proposed_stc_alns", True, scen=(2, 42, 42)),
            _main_record("jia_ebtas_style", scen=(2, 42, 42)),
            _main_record("library_open_loop", scen=(2, 42, 42)),
        ]}))
    # "full" in scenario A, mechanisms in scenario B -> no valid ablation
    ablation = gates.load_run(_write_run(
        root, "ablation", "stc_ablation",
        {"records.json": [
            {"ablation": "full", "status": "evaluated", "objective": 0.8,
             "uav_num": 2, "map_seed": 42, "task_seed": 42},
            {"ablation": "ebtas_init", "status": "evaluated", "objective": 0.9,
             "uav_num": 2, "map_seed": 43, "task_seed": 42},
            {"ablation": "path_switch", "status": "evaluated", "objective": 0.85,
             "uav_num": 2, "map_seed": 43, "task_seed": 42},
        ]}))
    assert gates.evaluate_gate_3([main], [ablation])["status"] == "FAIL"


def test_gate_3_fail_arbitrary_ablation_name():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    main = gates.load_run(_write_run(
        root, "main", "stc_main",
        {"raw_records.json": [
            _main_record("proposed_stc_alns", True, scen=(2, 42, 42)),
            _main_record("jia_ebtas_style", scen=(2, 42, 42)),
            _main_record("library_open_loop", scen=(2, 42, 42)),
        ]}))
    ablation = gates.load_run(_write_run(
        root, "ablation", "stc_ablation",
        {"records.json": [
            {"ablation": "full", "status": "evaluated", "objective": 0.8,
             "uav_num": 2, "map_seed": 42, "task_seed": 42},
            {"ablation": "totally_fake", "status": "evaluated", "objective": 0.9,
             "uav_num": 2, "map_seed": 42, "task_seed": 42},
            {"ablation": "also_fake", "status": "evaluated", "objective": 0.85,
             "uav_num": 2, "map_seed": 42, "task_seed": 42},
        ]}))
    assert gates.evaluate_gate_3([main], [ablation])["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# Gate 4: empty expected/hashes, invalid input hash
# --------------------------------------------------------------------------- #

def test_gate_4_fail_empty_expected_and_hashes():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    run = gates.load_run(_write_run(root, "r", "stc_main", {"summary.json": {}},
                                    inputs={"scenario": "a" * 64},
                                    expected=[], hashes={}))
    assert gates.evaluate_gate_4([run])["status"] == "FAIL"


def test_gate_4_fail_invalid_input_hash():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    run = gates.load_run(_write_run(root, "r", "stc_main", {"summary.json": {}},
                                    inputs={"scenario": "not-a-hash"}))
    assert gates.evaluate_gate_4([run])["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# Plot compatibility: multi-scenario OK, mixed source versions rejected
# --------------------------------------------------------------------------- #

def test_check_compatible_accepts_multi_scenario():
    from plots import plot_stc_results as plots
    records = [
        {"method": "a", "status": "evaluated", "reference_hash": "h1",
         "library_hash": "l1", "world_hash": "w1",
         "uav_num": 2, "map_seed": 1, "task_seed": 1},
        {"method": "b", "status": "evaluated", "reference_hash": "h2",
         "library_hash": "l2", "world_hash": "w2",
         "uav_num": 2, "map_seed": 2, "task_seed": 1},
    ]
    assert plots.check_compatible(records) is True


def test_check_same_source_raises_mixed_heads():
    from plots import plot_stc_results as plots
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    r1 = gates.load_run(_write_run(root, "r1", "stc_main", {"summary.json": {}},
                                   git_head="a" * 40))
    r2 = gates.load_run(_write_run(root, "r2", "stc_main", {"summary.json": {}},
                                   git_head="b" * 40))
    with pytest.raises(ValueError):
        plots.check_same_source([r1, r2])


# --------------------------------------------------------------------------- #
# Aggregation without overwriting replicates; scale retains identity
# --------------------------------------------------------------------------- #

def test_extract_ablation_averages_replicates():
    from plots import plot_stc_results as plots
    records = [
        {"ablation": "full", "status": "evaluated", "objective": 0.8},
        {"ablation": "full", "status": "evaluated", "objective": 0.6},
    ]
    out = plots.extract_ablation(records)
    assert out == [{"ablation": "full", "objective": 0.7}]


def test_extract_scale_keeps_method_family_and_averages():
    from plots import plot_stc_results as plots
    records = [
        {"method": "proposed_stc_alns", "family": "uav", "variant": "U=2",
         "status": "evaluated", "objective": 0.5},
        {"method": "proposed_stc_alns", "family": "uav", "variant": "U=2",
         "status": "evaluated", "objective": 0.7},
        {"method": "library_open_loop", "family": "uav", "variant": "U=2",
         "status": "evaluated", "objective": 0.9},
    ]
    out = plots.extract_scale(records)
    by_key = {(r["method"], r["family"], r["variant"]): r["objective"]
              for r in out}
    assert by_key[("proposed_stc_alns", "uav", "U=2")] == 0.6
    assert by_key[("library_open_loop", "uav", "U=2")] == 0.9


# --------------------------------------------------------------------------- #
# Reporting bundle: immutable content-addressed
# --------------------------------------------------------------------------- #

def _plot_record(method, proposed=False, feasible=True, makespan=7.0, delay=2.0,
                 t_ref=8.0, d_ref=2.0):
    rec = {"method": method, "label": method, "proposed": proposed,
           "status": "evaluated", "feasible": feasible, "makespan_s": makespan,
           "total_delay_s": delay,
           "reference": {"reference": {"t_ref_s": t_ref, "d_ref_s": d_ref},
                         "w_t": 0.5, "w_d": 0.5}}
    return rec


def test_render_writes_reporting_bundle(tmp_path):
    from plots import plot_stc_results as plots
    from experiment import run_stc_gates as gates

    run = tmp_path / "rq4"
    run.mkdir()
    data = json.dumps([_plot_record("proposed_stc_alns", proposed=True),
                       _plot_record("jia_ebtas_style")]).encode()
    (run / "raw_records.json").write_bytes(data)
    (run / "manifest.json").write_text(json.dumps({
        "run_id": "rq4", "experiment": "stc_main", "status": "success",
        "git": {"head": "abc"}, "inputs": {"x": "a" * 64},
        "expected": ["raw_records.json"],
        "hashes": {"raw_records.json": _sha256(data)}}))
    loaded = gates.load_run(run)
    out_dir = tmp_path / "figs"
    written = plots.render_figures({"stc_main": [loaded]}, out_dir=out_dir)
    assert written

    bundle = json.loads((out_dir / "reporting_bundle.json").read_text())
    assert bundle["n_figures"] == len(written)
    assert bundle["provenance_sha256"]
    for name, info in bundle["figures"].items():
        assert len(info["sha256"]) == 64
        assert info["bytes"] > 0
