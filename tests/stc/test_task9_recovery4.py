"""Round 4 regressions: direct replay and durable, content-bound evidence."""

import json
from dataclasses import replace

import numpy as np
import pytest

from test_task9_recovery3 import _scenario, _limits, _field_for, _dummy_spec
from segment.stc.types import SegmentCandidate, RadioSample


def _boundary_candidate():
    return SegmentCandidate(
        candidate_id="same-id", source=0, target=1, generator="fixture",
        positions_m=np.array([[0., 0.], [1., 0.], [2., 0.]]),
        full_rate_bps=np.full((3, 3), 10.), sector_ids=np.zeros(3, dtype=int),
        g2a_outage=np.array([[0., 0., 0.], [1., 1., 1.], [0., 0., 0.]]),
        flight_steps=2, hold_position_m=(1., 0.), source_data_bits=1.)


class BoundaryField:
    world_hash = "boundary-world"
    n_sectors = 3

    def sample(self, p):
        return RadioSample(np.ones(3), np.full(3, float(p[0] == 1.)))


def test_direct_segment_has_no_depot_slot_and_matches_authoritative_replay():
    from experiment.stc_common import evaluate_segment
    from segment.stc import evaluator
    sc = _scenario(1, 2, limits=_limits(delta_t_s=1., eps_out=.4))
    cand = _boundary_candidate()
    metrics = evaluate_segment(sc, cand, BoundaryField(), _dummy_spec())
    # Two actual slots: outage 1 then 0. A third depot slot gives 1/3 < .4.
    assert metrics["feasible"] is False
    assert metrics["flight_time_s"] == 2.
    assert metrics["upload_delay_s"] == 1.
    assert metrics["outage"] == .5
    direct = evaluator.evaluate_direct_segment(sc, cand, BoundaryField(), _dummy_spec())
    assert len(direct.timelines[0].slots) == 2
    assert direct.timelines[0].slots[0].pre_buffer_bits == 1.
    assert metrics["violations"] == direct.violations
    assert metrics["failure_reason"] == "outage_excess"


@pytest.mark.parametrize("direct", [True, False])
def test_authoritative_replay_rejects_completion_in_slot_beyond_deadline(direct):
    from segment.stc.evaluator import evaluate_direct_segment, evaluate_solution
    from segment.stc.types import PlanSolution
    sc = _scenario(1, 1, task_data_bits=(1.,),
                   limits=_limits(delta_t_s=1., mission_deadline_s=1. if direct else 2., eps_out=.9))
    cand = _boundary_candidate()
    if direct:
        result = evaluate_direct_segment(sc, cand, BoundaryField(), _dummy_spec())
    else:
        depot = replace(cand, source=-1, target=0, candidate_id="depot",
                        positions_m=cand.positions_m[:2], full_rate_bps=cand.full_rate_bps[:2],
                        sector_ids=cand.sector_ids[:2], g2a_outage=cand.g2a_outage[:2],
                        source_data_bits=0.)
        result = evaluate_solution(sc, {(-1, 0): (depot,), (0, 1): (cand,)},
                                   PlanSolution(((0,),), (("depot", cand.candidate_id),)),
                                   BoundaryField(), _dummy_spec())
    assert result.feasible is False
    assert result.violations.deadline_exceeded == 1


def test_segment_gate_failure_survives_json_with_typed_violations_and_identity(monkeypatch):
    from experiment import run_stc_segment_gate as gate
    from experiment.stc_common import evaluate_segment, json_bytes
    sc = _scenario(1, 2, limits=_limits(delta_t_s=1., eps_out=.4))
    metrics = evaluate_segment(sc, _boundary_candidate(), BoundaryField(), _dummy_spec())
    rec = gate._aggregate("fixture", {(0, 1): metrics}, 1, .1, set(),
                          BoundaryField.world_hash, checkpoint_hash="checkpoint")
    saved = json.loads(json_bytes(rec))
    pe = saved["per_edge"][0]
    assert pe["failure_reason"] == "outage_excess"
    assert pe["violations"]["outage_excess"] == pytest.approx(.1)
    assert pe["scenario_hash"] == saved["scenario_hash"]
    assert pe["scenario_hash"]
    assert pe["world_hash"] == "boundary-world"
    assert pe["checkpoint_hash"] == "checkpoint"
    assert pe["generator"] == "fixture"
    assert saved["n_edges"] == 1 and saved["common_feasible_n"] == 0


def test_reference_binds_candidate_content_and_all_scenario_limits():
    from segment.stc.library import build_all_pairs_library
    from segment.stc.evaluator import compute_reference
    sc = _scenario(1, 2)
    lib = build_all_pairs_library(sc, _field_for(sc),
                                  generator_config={"straight_hold": True}).candidates
    original = compute_reference(sc, lib)
    changed = {e: tuple(replace(c, full_rate_bps=c.full_rate_bps * 2)
                        for c in cs) for e, cs in lib.items()}
    faster = compute_reference(sc, changed)
    assert faster.d_ref_s == pytest.approx(original.d_ref_s / 2)
    assert faster.source_hash != original.source_hash
    # Trace content must bind even if proxy T_ref/D_ref and IDs are unchanged.
    trace = {e: tuple(replace(c, positions_m=c.positions_m + .01)
                      for c in cs) for e, cs in lib.items()}
    assert compute_reference(sc, trace).source_hash != original.source_hash
    altered_limit = replace(sc, limits=replace(sc.limits, eps_out=.3))
    assert compute_reference(altered_limit, lib).source_hash != original.source_hash


def _assert_provenance(rec):
    for key in ("reference_hash", "scenario_hash", "library_hash", "world_hash"):
        assert rec[key]
    for key in ("t_ref_s", "d_ref_s", "runtime_s"):
        assert np.isfinite(rec[key])
    assert rec["t_ref_s"] > 0 and rec["d_ref_s"] > 0
    assert rec["iterations"] == 0
    assert rec["iterations_executed"] >= 0
    assert isinstance(rec["evaluator_calls"], dict)


def test_scale_all_families_and_density_persist_references_and_content_hashes(monkeypatch):
    from experiment import run_stc_scale as scale
    from segment.stc.evaluator import compute_reference
    sc = _scenario(1, 2)
    monkeypatch.setattr(scale, "DENSITY_FAMILIES", ((1, 2, 400., 1),))
    records = scale.run_scale(sc, _field_for, seed=11, iterations=0)["records"]
    records += scale.run_density_scale(sc, _field_for, seed=11, iterations=0)["records"]
    assert {r["family"] for r in records} == set(scale.SCALE_FAMILIES) | {"density"}
    for rec in records:
        _assert_provenance(rec)
        assert rec["method_seed"] == 11 and rec["task_seed"] == sc.task_seed
    real_build = scale.build_all_pairs_library
    lib = real_build(sc, _field_for(sc), generator_config={"straight_hold": True})
    monkeypatch.setattr(scale, "build_all_pairs_library", lambda *a, **k: lib)
    before = scale._eval_methods(sc, _field_for(sc), {}, 11, 0)[0]
    ref = compute_reference(sc, lib.candidates)
    assert before["t_ref_s"] == ref.t_ref_s and before["d_ref_s"] == ref.d_ref_s
    altered = replace(lib, candidates={e: tuple(replace(c, full_rate_bps=c.full_rate_bps * 2)
                                              for c in cs) for e, cs in lib.candidates.items()})
    monkeypatch.setattr(scale, "build_all_pairs_library", lambda *a, **k: altered)
    after = scale._eval_methods(sc, _field_for(sc), {}, 11, 0)[0]
    assert before["library_hash"] != after["library_hash"]
    assert before["reference_hash"] != after["reference_hash"]


def test_ablation_cli_executes_each_configured_method_seed_with_provenance(monkeypatch, tmp_path):
    from experiment import run_stc_ablation as ablation
    sc = _scenario(1, 2)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ablation, "load_stc_scenario", lambda *a, **kw: sc)
    monkeypatch.setattr(ablation.PerSectorRadioField, "from_scenario",
                        lambda *a, **kw: _field_for(sc))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"uav_nums": [1], "map_seeds": [0],
        "task_seeds": [42], "method_seeds": [11, 29], "iterations": 0,
        "generator_config": {"straight_hold": True}}))
    assert ablation.main(["--config", str(config), "--run-id", "ablation"]) == 0
    run = tmp_path / "results/runs/ablation"
    records = json.loads((run / "records.json").read_text())
    from collections import Counter
    assert Counter((r["ablation"], r["method_seed"]) for r in records) == {
        (name, seed): 1 for name in ["full"] + list(ablation.ABLATIONS)
        for seed in [11, 29]}
    assert json.loads((run / "resolved_config.json").read_text())["method_seeds"] == [11, 29]
    for rec in records:
        _assert_provenance(rec)
        assert rec["map_seed"] == 0 and rec["task_seed"] == 42
    assert len({r["reference_hash"] for r in records}) == 1


@pytest.mark.parametrize("failure", [RuntimeError("solver boom"), KeyboardInterrupt()])
def test_main_checkpoints_each_method_and_fails_honestly(monkeypatch, tmp_path, failure):
    from types import SimpleNamespace
    from experiment import run_stc_main as main
    from segment.stc.library import build_all_pairs_library
    from segment.stc.evaluator import default_objective_spec, evaluate_solution
    from segment.stc.operators import build_initial_solution
    sc = _scenario(1, 2)
    field = _field_for(sc)
    library = build_all_pairs_library(sc, field, generator_config={"straight_hold": True})
    reference = default_objective_spec(sc, library.candidates)
    sol = build_initial_solution(sc, library.candidates, seed=0)
    result = evaluate_solution(sc, library.candidates, sol, field, reference)
    sac = SimpleNamespace(status="unavailable", reason="missing", checkpoint_path="")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_build_context", lambda *a: (sc, field, library, reference, sac))
    monkeypatch.setattr(main, "validate_baseline_availability", lambda *a, **kw: None)

    def execute(method, *a, **kw):
        if method == "proposed_stc_alns":
            # Prior method artifacts must already exist BEFORE the failure.
            saved = json.loads((tmp_path / "results/runs/failure/raw_records.json").read_text())
            assert saved[0]["method"] == "library_open_loop"
            raise failure
        return SimpleNamespace(method=method, result=result, solution=sol,
                               evaluator_calls={"full": 1, "incremental": 0}, provenance={})

    monkeypatch.setattr(main, "run_system_baseline", execute)
    config = tmp_path / "config.json"
    cfg = {"uav_nums": [1], "map_seeds": [0], "task_seeds": [42],
           "method_seeds": [11], "methods": ["library_open_loop", "proposed_stc_alns"],
           "iterations": 0, "generator_config": {"straight_hold": True}}
    config.write_text(json.dumps(cfg))
    with pytest.raises(type(failure)):
        main.main(["--config", str(config), "--run-id", "failure"])
    run = tmp_path / "results/runs/failure"
    assert json.loads((run / "manifest.json").read_text())["status"] == "failed"
    records = json.loads((run / "raw_records.json").read_text())
    assert [r["status"] for r in records] == ["evaluated", "failed"]
    bad = records[1]
    assert bad["method"] == "proposed_stc_alns" and bad["proposed"] is True
    assert bad["availability_declared"] is False
    assert bad["evaluator_calls"] is None
    assert bad["failure_type"] == type(failure).__name__
    assert len(json.loads((run / "solutions.json").read_text())) == 2
    assert len(json.loads((run / "timelines.json").read_text())) == 2
    with pytest.raises(RuntimeError, match="coverage"):
        main._assert_coverage(records, cfg)


def test_declared_unavailability_preserves_method_identity():
    from experiment import run_stc_main as main
    rec = main._unavailable_payload("proposed_stc_alns", "declared missing", {"sac_checkpoint": None})
    assert rec["status"] == "unavailable" and rec["availability_declared"] is True
    assert rec["proposed"] is True


def test_ablation_search_uses_full_library_frozen_spec_for_every_variant(monkeypatch):
    from experiment import run_stc_ablation as ablation
    from segment.stc.library import build_all_pairs_library
    from segment.stc.evaluator import default_objective_spec
    sc = _scenario(1, 2)
    field = _field_for(sc)
    library = build_all_pairs_library(sc, field, generator_config={"straight_hold": True})
    # Deliberately different references expose any internal recomputation.
    frozen = _dummy_spec()
    monkeypatch.setattr(ablation, "default_objective_spec", lambda *a: frozen)
    real_solve = ablation.STCALNS.solve
    seen = []

    def solve(self, *a, **kw):
        result = real_solve(self, *a, **kw)
        seen.append(self.checkpoint.objective_spec)
        return result

    monkeypatch.setattr(ablation.STCALNS, "solve", solve)
    out = ablation.run_ablation(sc, field, library, iterations=0)
    assert seen == [frozen] * (1 + len(ablation.ABLATIONS))
    for rec in out["records"]:
        assert rec["objective"] == frozen.score(rec["makespan_s"], rec["total_delay_s"])
    # Unchanged callers still derive their own library reference.
    from segment.stc.operators import build_initial_solution
    solver = ablation.STCALNS(sc, library.candidates, field)
    solver.solve(build_initial_solution(sc, library.candidates), iterations=0)
    assert solver.checkpoint.objective_spec == default_objective_spec(sc, library.candidates)


def test_reference_hash_binds_exact_reference_values():
    from segment.stc.evaluator import _reference_hash
    sc = _scenario(1, 2)
    trace = [(0, 1, "candidate", "content")]
    assert _reference_hash(sc, trace, 1., 2.) != _reference_hash(sc, trace, 1., 3.)
    assert _reference_hash(sc, trace, 1., 2.) != _reference_hash(sc, trace, 2., 2.)


def test_ablation_runtime_failure_is_persisted_and_rethrown(monkeypatch, tmp_path):
    from experiment import run_stc_ablation as ablation
    sc = _scenario(1, 2)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ablation, "load_stc_scenario", lambda *a, **kw: sc)
    monkeypatch.setattr(ablation.PerSectorRadioField, "from_scenario", lambda *a, **kw: _field_for(sc))
    real_solve = ablation.STCALNS.solve
    executed = []

    def solve(self, *a, **kw):
        executed.append(kw["method_seed"])
        if len(executed) == 2:
            raise RuntimeError("ablation execution failed")
        return real_solve(self, *a, **kw)

    monkeypatch.setattr(ablation.STCALNS, "solve", solve)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"uav_nums": [1], "map_seeds": [0], "task_seeds": [42],
                                  "method_seeds": [11, 29], "iterations": 0,
                                  "generator_config": {"straight_hold": True}}))
    with pytest.raises(RuntimeError, match="ablation execution failed"):
        ablation.main(["--config", str(config), "--run-id", "ablation-failed"])
    run = tmp_path / "results/runs/ablation-failed"
    assert json.loads((run / "manifest.json").read_text())["status"] == "failed"
    records = json.loads((run / "records.json").read_text())
    assert [r["status"] for r in records] == ["evaluated", "failed"]
    assert records[1]["ablation"] == "ebtas_init"
    assert records[1]["availability_declared"] is False
    assert records[1]["evaluator_calls"] is None
    assert records[1]["method_seed"] == 11 and records[1]["scenario_hash"]
