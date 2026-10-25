"""Task 7: normalized objective, reference, and feasibility rank."""

import pytest

from segment.stc.evaluator import (
    compute_reference,
    evaluate_solution,
    feasibility_rank,
    objective,
)
from segment.stc.generators import generate_straight_hold
from segment.stc.types import (
    EvaluationResult,
    ObjectiveSpec,
    ReferenceValues,
    STCScenario,
    ScenarioLimits,
    ViolationVector,
)


def _spec(t_ref=200.0, d_ref=100.0):
    return ObjectiveSpec(
        reference=ReferenceValues(t_ref_s=t_ref, d_ref_s=d_ref, source_hash="test"),
        w_t=0.5, w_d=0.5)


def _result(feasible, unserved=0, makespan=100.0, delay=50.0):
    return EvaluationResult(
        feasible=feasible, makespan_s=makespan, total_delay_s=delay,
        outage_ratio=(0.1,), violations=ViolationVector(unserved, 0, 0.0, 0, 0),
        timelines=(), objective=0.0,
    )


def test_objective_matches_hand_calculation():
    r = _result(True, makespan=100.0, delay=50.0)
    spec = _spec(200.0, 100.0)
    assert objective(r, spec) == pytest.approx(0.5 * 100 / 200 + 0.5 * 50 / 100)


def test_infeasible_does_not_outrank_feasible():
    feas = _result(True)
    infeas = _result(False, unserved=1)
    assert feasibility_rank(feas) < feasibility_rank(infeas)


def _scenario():
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    return STCScenario(n_uav=2, n_tasks=4,
                       task_positions_m=tuple((1000.0 + i * 1000.0, 2000.0) for i in range(4)),
                       task_data_bits=tuple([1e6] * 4), depot_m=(0.0, 2000.0),
                       terminal_m=(4000.0, 2000.0), limits=limits, length_m=4000.0,
                       width_m=4000.0, n_bs=4, task_seed=42, map_seed=0)


class _MockField:
    def __init__(self, n_sectors):
        self.n_sectors = n_sectors

    def sample(self, pos):
        from segment.stc.types import RadioSample
        import numpy as np
        return RadioSample(a2g_efficiency=np.full(self.n_sectors, 1.0),
                           g2a_outage=np.full(self.n_sectors, 0.1))


def _library(sc):
    field = _MockField(sc.n_sectors)
    lib = {}
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src != tgt:
                lib[(src, tgt)] = [generate_straight_hold(sc, field, src, tgt)]
    return lib


def test_reference_is_positive_and_deterministic():
    sc = _scenario()
    lib = _library(sc)
    r1 = compute_reference(sc, lib)
    r2 = compute_reference(sc, lib)
    assert r1 == r2
    assert r1.t_ref_s > 0.0 and r1.d_ref_s > 0.0
    assert isinstance(r1.source_hash, str) and r1.source_hash


def test_evaluator_uses_normalized_objective():
    sc = _scenario()
    lib = _library(sc)
    from segment.stc.operators import build_initial_solution
    sol = build_initial_solution(sc, lib)
    spec = _spec(200.0, 100.0)
    res = evaluate_solution(sc, lib, sol, _MockField(sc.n_sectors), spec)
    expected = spec.score(res.makespan_s, res.total_delay_s)
    assert res.objective == pytest.approx(expected)


# ---- Step 1: failing normalized-objective integration tests ----

def test_evaluation_result_uses_frozen_normalized_objective():
    """The objective on a feasible result equals the ObjectiveSpec score."""
    sc = _scenario()
    lib = _library(sc)
    field = _MockField(sc.n_sectors)
    from segment.stc.operators import build_initial_solution
    sol = build_initial_solution(sc, lib)
    ref = ReferenceValues(t_ref_s=300.0, d_ref_s=150.0, source_hash="frozen")
    spec = ObjectiveSpec(reference=ref, w_t=0.5, w_d=0.5)
    res = evaluate_solution(sc, lib, sol, field, spec)
    assert res.objective == pytest.approx(
        0.5 * res.makespan_s / 300.0 + 0.5 * res.total_delay_s / 150.0)


def test_solver_inputs_cannot_omit_reference_values():
    """Missing or nonpositive references must raise before any search."""
    with pytest.raises(ValueError):
        ReferenceValues(t_ref_s=0.0, d_ref_s=100.0, source_hash="x")
    with pytest.raises(ValueError):
        ReferenceValues(t_ref_s=100.0, d_ref_s=-1.0, source_hash="x")
    with pytest.raises(ValueError):
        ObjectiveSpec(reference=ReferenceValues(100.0, 100.0, "x"), w_t=0.0, w_d=0.5)
    sc = _scenario()
    lib = _library(sc)
    field = _MockField(sc.n_sectors)
    from segment.stc.operators import build_initial_solution
    sol = build_initial_solution(sc, lib)
    with pytest.raises(ValueError):
        evaluate_solution(sc, lib, sol, field, None)


def test_changing_reference_changes_score_not_raw_metrics():
    sc = _scenario()
    lib = _library(sc)
    field = _MockField(sc.n_sectors)
    from segment.stc.operators import build_initial_solution
    sol = build_initial_solution(sc, lib)
    spec_a = _spec(200.0, 100.0)
    spec_b = _spec(400.0, 200.0)
    res_a = evaluate_solution(sc, lib, sol, field, spec_a)
    res_b = evaluate_solution(sc, lib, sol, field, spec_b)
    assert res_a.makespan_s == res_b.makespan_s
    assert res_a.total_delay_s == res_b.total_delay_s
    assert res_a.objective == pytest.approx(2.0 * res_b.objective)
