"""Task 6: synchronous shared-bandwidth evaluator + slot-semantics regressions."""

import numpy as np
import pytest

from segment.stc.evaluator import evaluate_solution, select_serving_sector
from segment.stc.generators import generate_straight_hold
from segment.stc.types import (
    EvaluationResult,
    ObjectiveSpec,
    PlanSolution,
    RadioSample,
    ReferenceValues,
    STCScenario,
    ScenarioLimits,
    ViolationVector,
)


def _sample(eff, outage):
    return RadioSample(a2g_efficiency=np.array(eff, dtype=float),
                       g2a_outage=np.array(outage, dtype=float))


def test_select_serving_sector_prefers_reliable_high_rate():
    s = _sample([1.0, 2.0, 0.5], [0.1, 0.4, 0.3])
    # sector 1 has the highest rate but outage 0.4 > 0.35 (unreliable)
    assert select_serving_sector(s, True, 0.35) == 0


def test_select_serving_sector_falls_back_to_min_outage():
    s = _sample([1.0, 2.0, 0.5], [0.5, 0.6, 0.7])
    assert select_serving_sector(s, True, 0.35) == 0


def test_select_serving_sector_no_payload_ignores_rate():
    s = _sample([0.0, 3.0, 1.0], [0.5, 0.6, 0.7])
    assert select_serving_sector(s, False, 0.35) == 0


def _spec(t_ref=1.0, d_ref=1.0):
    return ObjectiveSpec(
        reference=ReferenceValues(t_ref_s=t_ref, d_ref_s=d_ref, source_hash="hand"),
        w_t=0.5, w_d=0.5)


class _MockField:
    def __init__(self, n_sectors, eff=1.0, outage=0.1):
        self.n_sectors = n_sectors
        self._eff = eff
        self._outage = outage

    def sample(self, pos):
        return _sample([self._eff] * self.n_sectors, [self._outage] * self.n_sectors)


def _scenario(n_tasks=2):
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    positions = tuple((1000.0 + i * 1000.0, 2000.0) for i in range(n_tasks))
    return STCScenario(n_uav=2, n_tasks=n_tasks, task_positions_m=positions,
                       task_data_bits=tuple([1e6] * n_tasks), depot_m=(0.0, 2000.0),
                       terminal_m=(4000.0, 2000.0), limits=limits, length_m=4000.0,
                       width_m=4000.0, n_bs=4, task_seed=42, map_seed=0)


def test_evaluate_solution_returns_feasible_result():
    sc = _scenario(2)
    field = _MockField(sc.n_sectors, eff=1.0, outage=0.1)
    lib = {}
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src != tgt:
                lib[(src, tgt)] = [generate_straight_hold(sc, field, src, tgt)]
    sol = PlanSolution(routes=((0,), (1,)),
                       candidate_ids=(("straight_hold_-1_0", "straight_hold_0_2"),
                                      ("straight_hold_-1_1", "straight_hold_1_2")))
    res = evaluate_solution(sc, lib, sol, field, _spec())
    assert res.feasible is True
    assert res.makespan_s > 0.0
    assert res.total_delay_s > 0.0


# ---- Step 2: failing slot-semantics tests ----

def _degenerate_single_slot_scenario():
    """1 UAV, 0 tasks, depot == terminal == (0,0): completes in the first slot."""
    limits = ScenarioLimits(delta_t_s=1.0, v_max_mps=10.0, uav_height_m=100.0,
                            arrive_radius_m=5.0, max_wait_s=10.0,
                            mission_deadline_s=100.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    return STCScenario(n_uav=1, n_tasks=0, task_positions_m=(), task_data_bits=(),
                       depot_m=(0.0, 0.0), terminal_m=(0.0, 0.0), limits=limits,
                       length_m=100.0, width_m=100.0, n_bs=1, task_seed=42, map_seed=0)


def test_idle_uav_completes_at_t_zero():
    sc = _degenerate_single_slot_scenario()
    field = _MockField(sc.n_sectors, eff=1.0, outage=0.0)
    # empty route = idle UAV: no segments, no slots, complete at t=0.
    sol = PlanSolution(routes=((),), candidate_ids=((),))
    res = evaluate_solution(sc, {}, sol, field, _spec())
    assert res.feasible is True
    assert res.makespan_s == 0.0
    assert res.timelines[0].slots == ()


def test_final_upload_slot_records_prebuffer_active_and_conflicting():
    """A buffer that drains to zero mid-slot is still an active uploader."""
    from segment.stc.types import TimelineSlot, UAVTimeline
    from segment.stc.conflicts import build_conflict_report

    def slot(u, pre, post, sector, alloc):
        return TimelineSlot(
            time_start_s=0.0, time_end_s=0.5, edge_id=0, candidate_id="c",
            position_m=(0.0, 0.0), airborne=True, pre_buffer_bits=pre,
            post_buffer_bits=post, active_uploader=(pre > 0.0),
            serving_sector=sector, allocated_bandwidth_hz=alloc,
            rate_bps=1e6, uploaded_bits=pre - post, g2a_outage=0.1,
            holding=False, service_event="")

    # both UAVs have pre-buffer > 0 but drain to zero in this (final) slot
    tl0 = UAVTimeline(0, (slot(0, 1e6, 0.0, 0, 5e5),))
    tl1 = UAVTimeline(1, (slot(1, 1e6, 0.0, 0, 5e5),))
    rep = build_conflict_report(
        EvaluationResult(feasible=True, makespan_s=0.0, total_delay_s=0.0,
                         outage_ratio=(0.1,),
                         violations=ViolationVector(0, 0, 0.0, 0, 0),
                         timelines=(tl0, tl1), objective=0.0),
        1e6, 0.5)
    assert rep.total_overlap_s == pytest.approx(0.5)


def test_keep_timelines_false_omits_slots_but_preserves_metrics():
    sc = _scenario(2)
    field = _MockField(sc.n_sectors, eff=1.0, outage=0.1)
    lib = {}
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src != tgt:
                lib[(src, tgt)] = [generate_straight_hold(sc, field, src, tgt)]
    sol = PlanSolution(routes=((0,), (1,)),
                       candidate_ids=(("straight_hold_-1_0", "straight_hold_0_2"),
                                      ("straight_hold_-1_1", "straight_hold_1_2")))
    res_keep = evaluate_solution(sc, lib, sol, field, _spec(), keep_timelines=True)
    res_drop = evaluate_solution(sc, lib, sol, field, _spec(), keep_timelines=False)
    assert res_drop.timelines == ()
    assert res_drop.makespan_s == res_keep.makespan_s
    assert res_drop.total_delay_s == res_keep.total_delay_s
    assert res_drop.outage_ratio == res_keep.outage_ratio
    assert res_drop.violations == res_keep.violations
    assert res_drop.objective == res_keep.objective


def test_deadline_with_residual_buffer_records_upload_failure():
    """Deadline expiry with un-uploaded data records an upload failure."""
    limits = ScenarioLimits(delta_t_s=1.0, v_max_mps=10.0, uav_height_m=100.0,
                            arrive_radius_m=5.0, max_wait_s=10.0,
                            mission_deadline_s=5.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    sc = STCScenario(n_uav=1, n_tasks=1, task_positions_m=((30.0, 0.0),),
                     task_data_bits=(10e6,), depot_m=(0.0, 0.0),
                     terminal_m=(60.0, 0.0), limits=limits, length_m=100.0,
                     width_m=100.0, n_bs=1, task_seed=42, map_seed=0)
    field = _MockField(sc.n_sectors, eff=1.0, outage=0.0)
    from segment.stc.types import SegmentCandidate

    def hand_cand(cid, pts, data):
        pts = np.asarray(pts, dtype=float)
        T = len(pts)
        return SegmentCandidate(
            candidate_id=cid, source=-1, target=0, generator="hand",
            positions_m=pts, full_rate_bps=np.full((T, sc.n_sectors), 1e6),
            sector_ids=np.zeros(T, dtype=int),
            g2a_outage=np.zeros((T, sc.n_sectors)),
            flight_steps=T, hold_position_m=tuple(pts[-2]),
            source_data_bits=data)

    lib = {
        (-1, 0): [hand_cand("d_t", [(0, 0), (10, 0), (20, 0), (25, 0), (30, 0)], 0.0)],
        (0, 1): [hand_cand("t_t", [(30, 0), (40, 0), (50, 0), (55, 0), (60, 0)], 10e6)],
    }
    sol = PlanSolution(routes=((0,),), candidate_ids=(("d_t", "t_t"),))
    res = evaluate_solution(sc, lib, sol, field, _spec())
    assert res.feasible is False
    assert res.violations.upload_failures >= 1
    assert res.violations.deadline_exceeded >= 1
