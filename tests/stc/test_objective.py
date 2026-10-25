"""Task 7: normalized objective, reference, and feasibility rank."""

import pytest

from segment.stc.evaluator import compute_reference, feasibility_rank, objective
from segment.stc.generators import generate_straight_hold
from segment.stc.types import (
    EvaluationResult,
    STCScenario,
    ScenarioLimits,
    ViolationVector,
)


def _result(feasible, unserved=0, makespan=100.0, delay=50.0):
    return EvaluationResult(
        feasible=feasible, makespan_s=makespan, total_delay_s=delay,
        outage_ratio=(0.1,), violations=ViolationVector(unserved, 0, 0.0, 0, 0),
        timelines=(), objective=0.0,
    )


def test_objective_matches_hand_calculation():
    r = _result(True, makespan=100.0, delay=50.0)
    assert objective(r, 200.0, 100.0, 0.5, 0.5) == pytest.approx(
        0.5 * 100 / 200 + 0.5 * 50 / 100)


def test_infeasible_does_not_outrank_feasible():
    feas = _result(True)
    infeas = _result(False, unserved=1)
    assert feasibility_rank(feas) < feasibility_rank(infeas)


def _scenario():
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, departure_step_s=1.0,
                            departure_max_s=60.0, bandwidth_hz=1e6,
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


def test_reference_is_positive_and_deterministic():
    sc = _scenario()
    field = _MockField(sc.n_sectors)
    lib = {}
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src != tgt:
                lib[(src, tgt)] = [generate_straight_hold(sc, field, src, tgt)]
    r1 = compute_reference(sc, lib)
    r2 = compute_reference(sc, lib)
    assert r1 == r2
    assert r1[0] > 0.0 and r1[1] > 0.0


def test_evaluator_uses_normalized_objective():
    from segment.stc.evaluator import evaluate_solution
    sc = _scenario()
    field = _MockField(sc.n_sectors)
    lib = {}
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src != tgt:
                lib[(src, tgt)] = [generate_straight_hold(sc, field, src, tgt)]
    from segment.stc.operators import build_initial_solution
    sol = build_initial_solution(sc, lib)
    res = evaluate_solution(sc, lib, sol, field, t_ref=200.0, d_ref=100.0)
    expected = sc.limits.w_t * res.makespan_s / 200.0 + sc.limits.w_d * res.total_delay_s / 100.0
    import pytest as _pytest
    assert res.objective == _pytest.approx(expected)
