"""Task 10 recovery round 4: independent-review Important findings.

The fourth review attacked the gates with synthetic evidence and obtained
three false PASSes plus one unchecked plot path:

1. Gate 1 merged generator records from DIFFERENT sealed runs sharing seed keys
   (straight-hold from one world + comm-aware from another) into one "complete"
   scenario.
2. Gate 3's ablation check ignored ``method_seed`` and depended on record
   order: a mechanism objective could be compared against another seed's
   'full' comparator, so a dataset where NO seed had two contributing
   mechanisms still passed.
3. The reporting bundle accepted a byte-valid figure with no
   ``figures_provenance.json`` entry (figure <-> provenance correspondence was
   never enforced).
4. ``check_compatible`` ran only for ``stc_main``; ablation/scale/segment
   figures concatenated records across runs without any record-level
   reference/library/world hash check.

These tests pin the fixes.
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
               git_head="a" * 40):
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, value in artifacts.items():
        data = json.dumps(value, sort_keys=True).encode("utf-8")
        (run_dir / name).write_bytes(data)
        written[name] = _sha256(data)
    manifest = {
        "run_id": run_id, "experiment": experiment, "status": status,
        "git": {"head": git_head}, "inputs": {},
        "hashes": written, "expected": sorted(artifacts),
    }
    (run_dir / "manifest.json").write_bytes(
        json.dumps(manifest, sort_keys=True).encode("utf-8"))
    return run_dir


_EDGES = [[-1, 0], [-1, 1], [0, 1], [1, 0], [0, 2], [1, 2]]  # K=2 legal set


def _segment_record(generator, world_hash="w" * 64, scenario_hash="s" * 64,
                    scen=(2, 42, 42)):
    rec = {"generator": generator, "status": "evaluated", "feasible_frac": 1.0,
           "n_edges": len(_EDGES), "common_feasible_n": len(_EDGES),
           "per_edge": [{"edge": e, "feasible": True} for e in _EDGES],
           "common_feasible_edges": [list(e) for e in _EDGES],
           "uav_num": scen[0], "map_seed": scen[1], "task_seed": scen[2]}
    if world_hash is not None:
        rec["world_hash"] = world_hash
    if scenario_hash is not None:
        rec["scenario_hash"] = scenario_hash
    return rec


# --------------------------------------------------------------------------- #
# Finding 1: Gate 1 must not splice generators across runs
# --------------------------------------------------------------------------- #

def test_gate_1_fail_cross_run_splice():
    # run A holds only the straight_hold baseline (world A), run B holds only a
    # comm-aware generator (world B); identical seed keys.  Neither run alone is
    # complete and the pair must never be merged into one "complete" scenario.
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    run_a = gates.load_run(_write_run(
        root, "runA", "stc_segment_gate",
        {"records.json": [_segment_record("straight_hold",
                                          world_hash="a" * 64,
                                          scenario_hash="c" * 64)]}))
    run_b = gates.load_run(_write_run(
        root, "runB", "stc_segment_gate",
        {"records.json": [_segment_record("astar",
                                          world_hash="b" * 64,
                                          scenario_hash="d" * 64)]}))
    result = gates.evaluate_gate_1([run_a, run_b])
    assert result["status"] == "FAIL"


def test_gate_1_fail_mixed_world_identity_in_one_run():
    # one run whose generators report different world hashes for the same
    # scenario is spliced evidence -> authenticity failure
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    run = gates.load_run(_write_run(
        root, "seg", "stc_segment_gate",
        {"records.json": [_segment_record("straight_hold", world_hash="a" * 64),
                          _segment_record("astar", world_hash="b" * 64)]}))
    result = gates.evaluate_gate_1([run])
    assert result["status"] == "FAIL"
    assert "identities" in result["detail"]


def test_gate_1_fail_missing_world_identity():
    # a complete generator pair without world/scenario identity is not
    # PASS-carrying evidence (it cannot be shown unspliced)
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    run = gates.load_run(_write_run(
        root, "seg", "stc_segment_gate",
        {"records.json": [
            _segment_record("straight_hold", world_hash=None, scenario_hash=None),
            _segment_record("astar", world_hash=None, scenario_hash=None)]}))
    assert gates.evaluate_gate_1([run])["status"] == "FAIL"


def test_gate_1_pass_single_run_with_identity():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    run = gates.load_run(_write_run(
        root, "seg", "stc_segment_gate",
        {"records.json": [_segment_record("straight_hold"),
                          _segment_record("comm_waypoint")]}))
    assert gates.evaluate_gate_1([run])["status"] == "PASS"


# --------------------------------------------------------------------------- #
# Finding 2: Gate 3 ablation evidence must pair by (scenario, method_seed)
# --------------------------------------------------------------------------- #

def _main_record(method, objective=0.9):
    return {"method": method, "status": "evaluated", "feasible": True,
            "objective": objective, "makespan_s": 7.0, "total_delay_s": 2.0,
            "uav_num": 2, "map_seed": 42, "task_seed": 42, "method_seed": 0}


def _valid_main_run(root):
    from experiment import run_stc_gates as gates
    return gates.load_run(_write_run(
        root, "main", "stc_main",
        {"raw_records.json": [_main_record("proposed_stc_alns", 0.9),
                              _main_record("jia_ebtas_style", 1.0),
                              _main_record("library_open_loop", 1.1)]}))


def _abl_record(method_seed, name, objective):
    return {"status": "evaluated", "ablation": name, "objective": objective,
            "uav_num": 2, "map_seed": 42, "task_seed": 42,
            "method_seed": method_seed}


def test_gate_3_fail_ablation_cross_seed_pairing():
    # seed 0: both mechanisms IMPROVE when removed (no contribution);
    # seed 1: only a 'full' record.  No seed shows two contributing mechanisms,
    # so comparing seed-0 mechanisms against seed-1's comparator must FAIL.
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    main = _valid_main_run(root)
    ablation = gates.load_run(_write_run(
        root, "abl", "stc_ablation",
        {"records.json": [_abl_record(0, "full", 1.0),
                          _abl_record(0, "hotspot_destroy", 0.9),
                          _abl_record(0, "conflict_repair", 0.95),
                          _abl_record(1, "full", 0.5)]}))
    result = gates.evaluate_gate_3([main], [ablation])
    assert result["status"] == "FAIL"
    assert "ablation invalid" in result["detail"]


def test_gate_3_pass_ablation_within_seed():
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    main = _valid_main_run(root)
    ablation = gates.load_run(_write_run(
        root, "abl", "stc_ablation",
        {"records.json": [_abl_record(0, "full", 1.0),
                          _abl_record(0, "hotspot_destroy", 1.1),
                          _abl_record(0, "conflict_repair", 1.05)]}))
    assert gates.evaluate_gate_3([main], [ablation])["status"] == "PASS"


def test_gate_3_fail_ablation_duplicated_comparator():
    # two 'full' records for the same (scenario, method_seed) pair make the
    # comparison order-dependent -> not paired evidence
    from experiment import run_stc_gates as gates
    root = Path(tempfile.mkdtemp())
    main = _valid_main_run(root)
    ablation = gates.load_run(_write_run(
        root, "abl", "stc_ablation",
        {"records.json": [_abl_record(0, "full", 2.0),
                          _abl_record(0, "full", 1.0),
                          _abl_record(0, "hotspot_destroy", 1.1),
                          _abl_record(0, "conflict_repair", 1.05)]}))
    assert gates.evaluate_gate_3([main], [ablation])["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# Finding 3: every bundle figure needs its provenance entry
# --------------------------------------------------------------------------- #

def _loaded_run(run_id, inputs):
    from experiment.run_stc_gates import LoadedRun
    return LoadedRun(run_id=run_id, experiment="stc_main", status="success",
                     valid=True, manifest={"inputs": inputs}, artifacts={})


def _write_bundle(td, figures, provenance):
    prov_path = td / "figures_provenance.json"
    prov_path.write_text(json.dumps(provenance), encoding="utf-8")
    bundle = {"figures": figures,
              "provenance_sha256": _sha256(prov_path.read_bytes()),
              "n_figures": len(figures)}
    bundle_path = td / "reporting_bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    return bundle_path


def test_bundle_fail_unprovenanced_figure(tmp_path):
    from experiment.run_stc_gates import _valid_reporting_bundle
    (tmp_path / "x.png").write_bytes(b"PNG-X")
    inputs = {"resolved_config": "b" * 64}
    run = _loaded_run("r1", inputs)
    # provenance describes y.png only; the bundle ships x.png
    provenance = {"y.png": {"experiment": "stc_main", "runs": ["r1"],
                            "inputs": {"r1": inputs}}}
    figures = {"x.png": {"sha256": _sha256(b"PNG-X"), "bytes": 5}}
    bundle_path = _write_bundle(tmp_path, figures, provenance)
    assert _valid_reporting_bundle(bundle_path, [run]) is False


def test_bundle_pass_exact_figure_provenance_match(tmp_path):
    from experiment.run_stc_gates import _valid_reporting_bundle
    (tmp_path / "x.png").write_bytes(b"PNG-X")
    inputs = {"resolved_config": "b" * 64}
    run = _loaded_run("r1", inputs)
    provenance = {"x.png": {"experiment": "stc_main", "runs": ["r1"],
                            "inputs": {"r1": inputs}}}
    figures = {"x.png": {"sha256": _sha256(b"PNG-X"), "bytes": 5}}
    bundle_path = _write_bundle(tmp_path, figures, provenance)
    assert _valid_reporting_bundle(bundle_path, [run]) is True


# --------------------------------------------------------------------------- #
# Finding 4: plot compatibility check covers every record family
# --------------------------------------------------------------------------- #

def _ablation_run(run_id, library_hash):
    from experiment.run_stc_gates import LoadedRun
    rec = {"status": "evaluated", "ablation": "full", "objective": 0.8,
           "uav_num": 2, "map_seed": 42, "task_seed": 42, "method_seed": 0,
           "reference_hash": "r" * 64, "library_hash": library_hash,
           "world_hash": "w" * 64}
    return LoadedRun(run_id=run_id, experiment="stc_ablation",
                     status="success", valid=True,
                     manifest={"git": {"head": "a" * 40}, "inputs": {}},
                     artifacts={"records.json": [rec]})


def test_render_rejects_incompatible_ablation_records(tmp_path):
    from plots.plot_stc_results import render_figures
    runs = {"stc_ablation": [_ablation_run("a1", "1" * 64),
                             _ablation_run("a2", "2" * 64)]}
    with pytest.raises(ValueError):
        render_figures(runs, tmp_path)


def test_check_compatible_accepts_multicandidate_ablation_arm():
    # the multi_candidate arm's REDUCED library is the treatment, not the
    # environment: arms sharing reference/reference-library/world hashes are
    # compatible even though their own library_hash differs
    from plots.plot_stc_results import check_compatible
    base = {"status": "evaluated", "uav_num": 2, "map_seed": 42, "task_seed": 42,
            "reference_hash": "r" * 64, "reference_library_hash": "L" * 64,
            "world_hash": "w" * 64}
    records = [dict(base, ablation="full", library_hash="L" * 64),
               dict(base, ablation="multi_candidate", library_hash="m" * 64)]
    assert check_compatible(records) is True


def test_check_compatible_accepts_scale_variants_with_distinct_libraries():
    # scale families rebuild the library (and reference) per variant: variants
    # are distinct conditions, not incompatible runs of one condition
    from plots.plot_stc_results import check_compatible
    def rec(variant, h):
        return {"status": "evaluated", "uav_num": 2, "map_seed": 42,
                "task_seed": 42, "family": "payload", "variant": variant,
                "reference_hash": h * 64, "library_hash": h * 64,
                "world_hash": "w" * 64}
    assert check_compatible([rec("x0.5", "1"), rec("x1.0", "2")]) is True


def test_check_compatible_rejects_mixed_hashes_within_one_variant():
    from plots.plot_stc_results import check_compatible
    def rec(h):
        return {"status": "evaluated", "uav_num": 2, "map_seed": 42,
                "task_seed": 42, "family": "payload", "variant": "x1.0",
                "reference_hash": h * 64, "library_hash": h * 64,
                "world_hash": "w" * 64}
    with pytest.raises(ValueError):
        check_compatible([rec("1"), rec("2")])


# --------------------------------------------------------------------------- #
# check_same_source：默认严格；显式声明的已证等价 head 组可通过
# --------------------------------------------------------------------------- #

def _run_with_head(run_id, head):
    from experiment.run_stc_gates import LoadedRun
    return LoadedRun(run_id=run_id, experiment="stc_main", status="success",
                     valid=True, manifest={"git": {"head": head}, "inputs": {}},
                     artifacts={})


def test_same_source_rejects_mixed_heads_by_default():
    from plots.plot_stc_results import check_same_source
    runs = [_run_with_head("a", "1" * 40), _run_with_head("b", "2" * 40)]
    with pytest.raises(ValueError):
        check_same_source(runs)


def test_same_source_accepts_declared_equivalent_heads():
    from plots.plot_stc_results import check_same_source
    runs = [_run_with_head("a", "1" * 40), _run_with_head("b", "2" * 40)]
    assert check_same_source(runs, equivalent_heads=("1" * 40, "2" * 40)) is True


def test_same_source_rejects_head_outside_declared_set():
    from plots.plot_stc_results import check_same_source
    runs = [_run_with_head("a", "1" * 40), _run_with_head("b", "2" * 40),
            _run_with_head("c", "3" * 40)]
    with pytest.raises(ValueError):
        check_same_source(runs, equivalent_heads=("1" * 40, "2" * 40))
