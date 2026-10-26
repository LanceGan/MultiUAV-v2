"""Task 9 recovery round 3: fixes for the third independent review.

Covers: RQ1 segment feasibility must come from the authoritative slot machine
(source upload / endpoint drain / hold / max-wait / deadline), RQ5 must record
full method identity (method seed, runtime, iterations, evaluator calls,
reference), per-edge evidence must carry candidate/trajectory/world identity,
main-runner per-method failure isolation, and timeout enforced after summary_fn.
"""

import json
import time as _time

import numpy as np
import pytest

from segment.stc.types import RadioSample, ScenarioLimits, STCScenario


def _limits(**kw):
    base = dict(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                arrive_radius_m=15.0, max_wait_s=60.0, mission_deadline_s=7200.0,
                bandwidth_hz=1e6, uplink_power_w=0.2, eps_out=0.35,
                w_t=0.5, w_d=0.5)
    base.update(kw)
    return ScenarioLimits(**base)


def _scenario(n_uav=2, n_tasks=3, **kw):
    positions = tuple((50.0 + (i % 2) * 60.0, 50.0 + (i // 2) * 60.0)
                      for i in range(n_tasks))
    base = dict(n_uav=n_uav, n_tasks=n_tasks, task_positions_m=positions,
                task_data_bits=tuple([1e6] * n_tasks),
                depot_m=(10.0, 10.0), terminal_m=(390.0, 390.0),
                limits=_limits(), length_m=400.0, width_m=400.0, n_bs=1,
                task_seed=42, map_seed=0)
    base.update(kw)
    return STCScenario(**base)


class _ConstField:
    def __init__(self, n_sectors=3, a2g=1.0, g2a=0.0):
        self.n_sectors = n_sectors
        self.world_hash = "const-field"
        self._a2g = a2g
        self._g2a = g2a

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.full(self.n_sectors, self._a2g),
                           g2a_outage=np.full(self.n_sectors, self._g2a))


def _field_for(sc):
    return _ConstField(n_sectors=sc.n_sectors)


def _dummy_spec():
    from segment.stc.types import ObjectiveSpec, ReferenceValues
    return ObjectiveSpec(reference=ReferenceValues(t_ref_s=1.0, d_ref_s=1.0,
                                                  source_hash="test"),
                         w_t=0.5, w_d=0.5)


# --------------------------------------------------------------------------- #
# 1. Segment feasibility from the authoritative slot machine
# --------------------------------------------------------------------------- #

def test_segment_deadline_exceeded_is_infeasible():
    from experiment.stc_common import evaluate_segment
    from segment.stc.generators import generate_straight_hold

    sc = _scenario(1, 3, limits=_limits(mission_deadline_s=1.0))
    field = _field_for(sc)
    cand = generate_straight_hold(sc, field, 0, 1)
    result = evaluate_segment(sc, cand, field, _dummy_spec())
    assert result["feasible"] is False


def test_segment_upload_incomplete_is_infeasible():
    from experiment.stc_common import evaluate_segment
    from segment.stc.generators import generate_straight_hold

    # huge payload + tiny max_wait -> cannot drain -> upload/wait failure
    sc = _scenario(1, 3, task_data_bits=tuple([5e7] * 3),
                   limits=_limits(max_wait_s=1.0))
    field = _field_for(sc)
    cand = generate_straight_hold(sc, field, 0, 1)
    result = evaluate_segment(sc, cand, field, _dummy_spec())
    assert result["feasible"] is False


# --------------------------------------------------------------------------- #
# 2. RQ5 records full method identity
# --------------------------------------------------------------------------- #

def test_scale_records_full_method_identity():
    from experiment import run_stc_scale as scale_mod

    base = _scenario(2, 3)
    out = scale_mod.run_scale(base, _field_for, seed=0, iterations=3)
    assert out["records"]
    for rec in out["records"]:
        assert "method_seed" in rec
        assert "iterations" in rec
        assert "iterations_executed" in rec
        assert rec["reference_hash"]
        assert np.isfinite(rec["runtime_s"])
        assert isinstance(rec["evaluator_calls"], dict)


def test_scale_default_config_has_method_seeds():
    from experiment import run_stc_scale as scale_mod
    assert scale_mod.DEFAULT_CONFIG["method_seeds"] == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# 3. per-edge evidence carries candidate / trajectory / world identity
# --------------------------------------------------------------------------- #

def test_segment_gate_per_edge_has_identity():
    from experiment import run_stc_segment_gate as gate

    sc = _scenario(1, 3)
    field = _field_for(sc)
    out = gate.run_segment_gate(sc, field)
    for rec in out["records"]:
        if rec["status"] != "evaluated":
            continue
        assert "world_hash" in rec
        assert "checkpoint_hash" in rec
        for pe in rec["per_edge"]:
            assert "candidate_id" in pe
            assert pe["trajectory_hash"]


# --------------------------------------------------------------------------- #
# 4. main runner isolates a per-method failure
# --------------------------------------------------------------------------- #

def test_main_isolates_method_failure(monkeypatch):
    from experiment import run_stc_main as main
    from segment.stc.library import build_all_pairs_library
    from segment.stc.evaluator import default_objective_spec
    from segment.stc.run_context import RunContext

    sc = _scenario(2, 3)
    field = _field_for(sc)
    library = build_all_pairs_library(sc, field,
                                      generator_config={"straight_hold": True})
    reference = default_objective_spec(sc, library.candidates)

    class SacRes:
        status = "unavailable"
        candidates = {}
        reason = "none"
        checkpoint_hash = ""
        checkpoint_path = ""

    class FakeResult:
        timelines = ()

    class FakeR:
        def __init__(self, method):
            self.method = method
            self.result = FakeResult()
            self.solution = None

    def fake_run(method, *args, **kwargs):
        if method == "jia_ebtas_style":
            raise RuntimeError("boom")
        return FakeR(method)

    monkeypatch.setattr(main, "validate_baseline_availability",
                        lambda *a, **k: (None, None))
    monkeypatch.setattr(main, "run_system_baseline", fake_run)
    monkeypatch.setattr(main, "_result_payload", lambda r, sc: {
        "method": r.method, "status": "evaluated", "feasible": True,
        "objective": 0.5, "makespan_s": 1.0, "total_delay_s": 0.5,
        "outage_ratio": (0.0,), "violations": None, "balance": None,
        "conflict_lost_bits": 0.0, "conflict_duration_s": 0.0,
        "waiting_s": 0.0, "evaluator_calls": {"full": 1, "incremental": 0},
        "provenance": {}, "failure_reason": None})

    cfg = {"sac_checkpoint": None, "graph_generators": ["astar"],
           "methods": ["region_distance_straight", "jia_ebtas_style"],
           "method_seeds": [0], "iterations": 2,
           "solver": {"rho": 0.1, "T0": 0.1, "cooling": 0.995, "segment": 100,
                      "audit_interval": 50},
           "score": {"new_best": 3., "improve": 1., "accept": .5, "reject": 0.}}

    import tempfile
    rc = RunContext.create(tempfile.mkdtemp(), "stc_main", cfg, run_id="t")

    records, _sol, _tl = main._run_scenario(
        sc, field, library, reference, SacRes(), 2, 42, 42, cfg, rc)
    by_method = {r["method"]: r for r in records}
    assert by_method["region_distance_straight"]["status"] == "evaluated"
    assert by_method["jia_ebtas_style"]["status"] == "unavailable"
    assert "runtime failure" in by_method["jia_ebtas_style"]["failure_reason"]


# --------------------------------------------------------------------------- #
# 5. timeout enforced after summary_fn
# --------------------------------------------------------------------------- #

def test_lifecycle_timeout_after_summary_fn_seals_failed(tmp_path, monkeypatch):
    from experiment.stc_common import run_lifecycle

    monkeypatch.chdir(tmp_path)
    cfg = {"uav_nums": [2], "timeout_s": 0.05}

    def per_scenario(u, m, t):
        return [{"u": u, "m": m, "t": t}]

    def summary_fn(records):
        _time.sleep(0.3)  # overruns the timeout during summary
        return {"record_count": len(records)}

    with pytest.raises(TimeoutError):
        run_lifecycle("stc_test", cfg, "lifecycle_timeout_summary", [(2, 7, 1)],
                      per_scenario, summary_fn)

    manifest = json.loads((tmp_path / "results/runs/lifecycle_timeout_summary/"
                           "manifest.json").read_text())
    assert manifest["status"] == "failed"
