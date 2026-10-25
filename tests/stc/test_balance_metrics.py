"""Task 5: task-balance reporting (spec §1.5).

The evaluator reports per-UAV completion times ``T_u``, their average, standard
deviation, and range as secondary metrics that are NOT folded into the main
objective (which remains ``w_t*T/T_ref + w_d*D/D_ref``).
"""

import math

import numpy as np
import pytest

from segment.stc.evaluator import evaluate_solution
from segment.stc.types import (
    ObjectiveSpec,
    PlanSolution,
    RadioSample,
    ReferenceValues,
    STCScenario,
    ScenarioLimits,
    SegmentCandidate,
)


def _spec(t_ref=1.0, d_ref=1.0):
    return ObjectiveSpec(
        reference=ReferenceValues(t_ref_s=t_ref, d_ref_s=d_ref, source_hash="hand"),
        w_t=0.5, w_d=0.5)


class _Field:
    def __init__(self, n_sectors):
        self.n_sectors = n_sectors

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.full(self.n_sectors, 1.0),
                           g2a_outage=np.zeros(self.n_sectors))


def _cand(cid, src, tgt, pts, data=0.0, n_sectors=3, rate=1e6):
    pts = np.asarray(pts, dtype=float)
    T = len(pts)
    return SegmentCandidate(
        candidate_id=cid, source=src, target=tgt, generator="hand",
        positions_m=pts, full_rate_bps=np.full((T, n_sectors), rate),
        sector_ids=np.zeros(T, dtype=int),
        g2a_outage=np.zeros((T, n_sectors)),
        flight_steps=T, hold_position_m=tuple(pts[-2] if T > 1 else pts[0]),
        source_data_bits=data)


def _scenario(n_uav=2, n_tasks=2, data_bits=(0.0, 0.0), max_wait=10.0,
              deadline=1000.0):
    limits = ScenarioLimits(delta_t_s=1.0, v_max_mps=10.0, uav_height_m=100.0,
                            arrive_radius_m=5.0, max_wait_s=max_wait,
                            mission_deadline_s=deadline, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    return STCScenario(n_uav=n_uav, n_tasks=n_tasks,
                       task_positions_m=((30.0, 0.0), (30.0, 0.0)),
                       task_data_bits=data_bits, depot_m=(0.0, 0.0),
                       terminal_m=(60.0, 0.0), limits=limits, length_m=100.0,
                       width_m=100.0, n_bs=1, task_seed=42, map_seed=0)


def test_balance_metrics_two_uavs_different_completion():
    sc = _scenario()
    field = _Field(sc.n_sectors)
    # UAV 0 uses 5-point traces -> 8 slots; UAV 1 uses 3-point traces -> 4 slots.
    lib = {
        (-1, 0): [_cand("d0", -1, 0, [(0, 0), (10, 0), (20, 0), (25, 0), (30, 0)])],
        (0, 2): [_cand("t02", 0, 2, [(30, 0), (40, 0), (50, 0), (55, 0), (60, 0)])],
        (-1, 1): [_cand("d1", -1, 1, [(0, 0), (15, 0), (30, 0)])],
        (1, 2): [_cand("t12", 1, 2, [(30, 0), (45, 0), (60, 0)])],
    }
    sol = PlanSolution(routes=((0,), (1,)),
                       candidate_ids=(("d0", "t02"), ("d1", "t12")))
    res = evaluate_solution(sc, lib, sol, field, _spec())
    assert res.feasible is True
    bal = res.balance
    assert bal.completion_times_s == pytest.approx((8.0, 4.0))
    assert bal.t_avg_s == pytest.approx(6.0)
    assert bal.sigma_t_s == pytest.approx(2.0)   # population std of {8, 4}
    assert bal.t_range_s == pytest.approx(4.0)


def test_balance_is_reported_not_folded_into_objective():
    sc = _scenario()
    field = _Field(sc.n_sectors)
    lib = {
        (-1, 0): [_cand("d0", -1, 0, [(0, 0), (10, 0), (20, 0), (25, 0), (30, 0)])],
        (0, 2): [_cand("t02", 0, 2, [(30, 0), (40, 0), (50, 0), (55, 0), (60, 0)])],
        (-1, 1): [_cand("d1", -1, 1, [(0, 0), (15, 0), (30, 0)])],
        (1, 2): [_cand("t12", 1, 2, [(30, 0), (45, 0), (60, 0)])],
    }
    sol = PlanSolution(routes=((0,), (1,)),
                       candidate_ids=(("d0", "t02"), ("d1", "t12")))
    res = evaluate_solution(sc, lib, sol, field, _spec())
    # objective = 0.5*T/1.0 + 0.5*D/1.0 with no payload D=0 and T=8 -> 4.0.
    # T_avg=6.0 and sigma=2.0 play no part in the objective.
    assert res.objective == pytest.approx(0.5 * 8.0 + 0.5 * 0.0)
    assert res.balance.t_avg_s == pytest.approx(6.0)


def test_balance_reports_nan_for_unfinished_uav():
    sc = _scenario(n_uav=1, n_tasks=1, data_bits=(1e6,), max_wait=10.0)
    field = _Field(sc.n_sectors)
    # zero-rate upload: the UAV holds forever and never completes.
    lib = {
        (-1, 0): [_cand("d0", -1, 0, [(0, 0), (10, 0), (20, 0), (25, 0), (30, 0)])],
        (0, 1): [_cand("t01", 0, 1, [(30, 0), (40, 0), (50, 0), (55, 0), (60, 0)],
                       data=1e6, rate=0.0)],
    }
    sol = PlanSolution(routes=((0,),), candidate_ids=(("d0", "t01"),))
    res = evaluate_solution(sc, lib, sol, field, _spec())
    assert res.feasible is False
    (t_u,) = res.balance.completion_times_s
    assert math.isnan(t_u)
    assert res.balance.t_avg_s == 0.0
