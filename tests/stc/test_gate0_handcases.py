"""Gate 0 hand-calculated evaluator fixtures.

Each fixture uses hand-built SI candidate traces so the expected makespan,
upload delay, holding behaviour, outage ratio, and conflict attribution can be
computed by hand and asserted exactly.
"""

import numpy as np
import pytest

from segment.stc.conflicts import build_conflict_report
from segment.stc.evaluator import evaluate_solution
from segment.stc.types import (
    EdgeOccurrence,
    ObjectiveSpec,
    PairwiseConflictWindow,
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
    def __init__(self, n_sectors, eff=1.0, outage=0.0):
        self.n_sectors = n_sectors
        self.eff = eff
        self.outage = outage

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.full(self.n_sectors, self.eff),
                           g2a_outage=np.full(self.n_sectors, self.outage))


class _SplitField:
    """Sector 0 on the left half (x<50), sector 1 on the right (x>=50)."""
    n_sectors = 3

    def sample(self, pos):
        if float(np.asarray(pos)[0]) < 50.0:
            return RadioSample(a2g_efficiency=np.array([1.0, 0.0, 0.0]),
                               g2a_outage=np.array([0.0, 1.0, 1.0]))
        return RadioSample(a2g_efficiency=np.array([0.0, 1.0, 0.0]),
                           g2a_outage=np.array([1.0, 0.0, 1.0]))


def _scenario(n_uav, n_tasks, task_positions, data_bits, depot=(0.0, 0.0),
              terminal=(60.0, 0.0), dt=1.0, v_max=10.0, arrive_radius=5.0,
              max_wait=10.0, bandwidth=1e6, mission_deadline=1000.0):
    limits = ScenarioLimits(delta_t_s=dt, v_max_mps=v_max, uav_height_m=100.0,
                            arrive_radius_m=arrive_radius, max_wait_s=max_wait,
                            mission_deadline_s=mission_deadline, bandwidth_hz=bandwidth,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    return STCScenario(n_uav=n_uav, n_tasks=n_tasks, task_positions_m=task_positions,
                       task_data_bits=data_bits, depot_m=depot, terminal_m=terminal,
                       limits=limits, length_m=100.0, width_m=100.0, n_bs=1,
                       task_seed=42, map_seed=0)


def _hand_cand(cid, src, tgt, pts, data=0.0, n_sectors=3, rate=1e6):
    pts = np.asarray(pts, dtype=float)
    T = len(pts)
    return SegmentCandidate(
        candidate_id=cid, source=src, target=tgt, generator="hand",
        positions_m=pts, full_rate_bps=np.full((T, n_sectors), rate),
        sector_ids=np.zeros(T, dtype=int),
        g2a_outage=np.zeros((T, n_sectors)),
        flight_steps=T, hold_position_m=tuple(pts[-2] if T > 1 else pts[0]),
        source_data_bits=data)


def _sector_cand(cid, src, tgt, pts, data, active_sector, n_sectors=3, rate=1e6):
    pts = np.asarray(pts, dtype=float)
    T = len(pts)
    fr = np.zeros((T, n_sectors))
    fr[:, active_sector] = rate
    return SegmentCandidate(
        candidate_id=cid, source=src, target=tgt, generator="hand",
        positions_m=pts, full_rate_bps=fr,
        sector_ids=np.full(T, active_sector, dtype=int),
        g2a_outage=np.zeros((T, n_sectors)),
        flight_steps=T, hold_position_m=tuple(pts[-2] if T > 1 else pts[0]),
        source_data_bits=data)


# standard single-task traces: 3 flight steps to the hold point, 1 connector into
# the anchor (len 5).  depot(0,0) -> task(30,0) -> terminal(60,0).
_D = [(0, 0), (10, 0), (20, 0), (25, 0), (30, 0)]
_T = [(30, 0), (40, 0), (50, 0), (55, 0), (60, 0)]


def _single_task_lib(sc, data, rate=1e6):
    return {
        (-1, 0): [_hand_cand("d0", -1, 0, _D, 0.0, sc.n_sectors, rate)],
        (0, 1): [_hand_cand("t1", 0, 1, _T, data, sc.n_sectors, rate)],
    }


def test_one_uav_one_task_no_payload():
    sc = _scenario(1, 1, ((30.0, 0.0),), (0.0,))
    field = _Field(sc.n_sectors, eff=1.0, outage=0.0)
    lib = _single_task_lib(sc, 0.0)
    sol = PlanSolution(routes=((0,),), candidate_ids=(("d0", "t1"),))
    res = evaluate_solution(sc, lib, sol, field, _spec())
    assert res.feasible is True
    assert res.makespan_s == pytest.approx(8.0)   # serve at slot 3, terminal at slot 7
    assert res.total_delay_s == pytest.approx(0.0)  # no payload, no upload
    assert len(res.timelines[0].slots) == 8
    assert res.balance.completion_times_s == pytest.approx((8.0,))
    assert res.balance.t_avg_s == pytest.approx(8.0)
    assert res.balance.sigma_t_s == pytest.approx(0.0)
    assert res.balance.t_range_s == pytest.approx(0.0)


def test_one_uav_exactly_two_upload_slots():
    sc = _scenario(1, 1, ((30.0, 0.0),), (1.5e6,))
    field = _Field(sc.n_sectors, eff=1.0, outage=0.0)
    lib = _single_task_lib(sc, 1.5e6)
    sol = PlanSolution(routes=((0,),), candidate_ids=(("d0", "t1"),))
    res = evaluate_solution(sc, lib, sol, field, _spec())
    slots = res.timelines[0].slots
    active = [s for s in slots if s.active_uploader]
    assert len(active) == 2                      # 1e6 then 0.5e6 at 1e6 bps
    assert res.total_delay_s == pytest.approx(2.0)
    # the final upload slot drains the buffer to zero but is still active
    assert active[-1].pre_buffer_bits > 0.0
    assert active[-1].post_buffer_bits == pytest.approx(0.0)


def test_two_uavs_share_one_sector_one_slot():
    sc = _scenario(2, 2, ((30.0, 0.0), (30.0, 5.0)), (5e5, 5e5))
    field = _Field(sc.n_sectors, eff=1.0, outage=0.0)
    lib = {
        (-1, 0): [_hand_cand("d0", -1, 0, _D, 0.0, sc.n_sectors)],
        (-1, 1): [_hand_cand("d1", -1, 1, [(0, 0), (10, 0), (20, 0), (25, 0), (30, 5)], 0.0, sc.n_sectors)],
        (0, 2): [_hand_cand("t02", 0, 2, _T, 5e5, sc.n_sectors)],
        (1, 2): [_hand_cand("t12", 1, 2, [(30, 5), (40, 5), (50, 5), (55, 5), (60, 0)], 5e5, sc.n_sectors)],
    }
    sol = PlanSolution(routes=((0,), (1,)),
                       candidate_ids=(("d0", "t02"), ("d1", "t12")))
    res = evaluate_solution(sc, lib, sol, field, _spec())
    assert res.feasible is True
    rep = build_conflict_report(res, 1e6, 1.0)
    assert rep.total_overlap_s == pytest.approx(1.0)   # exactly one shared slot
    assert rep.lost_bits == pytest.approx(1.0e6)       # 2 x 5e5
    # the conflict is attributed to each UAV's exact edge occurrence
    assert set(rep.edge_hotspots) == {
        EdgeOccurrence(0, 1, 0, 2, "t02"),
        EdgeOccurrence(1, 1, 1, 2, "t12"),
    }
    for occ, val in rep.edge_hotspots.items():
        assert val == pytest.approx(5.0e5)
    assert rep.pairwise_windows == (
        PairwiseConflictWindow(0, 1, 0, 1, 1, 4.0, 5.0, 1.0e6),)
    # the shared slot is each UAV's final upload slot (buffer drains to zero)
    for tl in res.timelines:
        active = [s for s in tl.slots if s.active_uploader]
        assert len(active) == 1
        assert active[0].post_buffer_bits == pytest.approx(0.0)


def test_two_uavs_different_sectors_no_conflict():
    sc = _scenario(2, 2, ((30.0, 0.0), (70.0, 0.0)), (1e6, 1e6),
                   depot=(50.0, 0.0), terminal=(50.0, 0.0))
    field = _SplitField()
    lib = {
        (-1, 0): [_sector_cand("d0", -1, 0, [(50, 0), (40, 0), (35, 0), (30, 0)], 0.0, 0)],
        (0, 2): [_sector_cand("t02", 0, 2, [(30, 0), (40, 0), (45, 0), (50, 0)], 1e6, 0)],
        (-1, 1): [_sector_cand("d1", -1, 1, [(50, 0), (60, 0), (65, 0), (70, 0)], 0.0, 1)],
        (1, 2): [_sector_cand("t12", 1, 2, [(70, 0), (60, 0), (55, 0), (50, 0)], 1e6, 1)],
    }
    sol = PlanSolution(routes=((0,), (1,)),
                       candidate_ids=(("d0", "t02"), ("d1", "t12")))
    res = evaluate_solution(sc, lib, sol, field, _spec())
    assert res.feasible is True
    assert res.makespan_s == pytest.approx(6.0)
    rep = build_conflict_report(res, 1e6, 1.0)
    assert rep.total_overlap_s == 0.0
    assert rep.lost_bits == 0.0
    assert rep.edge_hotspots == {}
    assert rep.pairwise_windows == ()


def test_one_uav_waiting_outside_arrival_radius():
    sc = _scenario(1, 1, ((30.0, 0.0),), (3.5e6,))
    field = _Field(sc.n_sectors, eff=1.0, outage=0.0)
    lib = _single_task_lib(sc, 3.5e6)
    sol = PlanSolution(routes=((0,),), candidate_ids=(("d0", "t1"),))
    res = evaluate_solution(sc, lib, sol, field, _spec())
    assert res.feasible is True
    assert res.makespan_s == pytest.approx(9.0)   # one extra slot of waiting
    holding = [s for s in res.timelines[0].slots if s.holding]
    assert len(holding) == 1
    assert holding[0].pre_buffer_bits > 0.0


def test_zero_rate_max_wait_failure():
    sc = _scenario(1, 1, ((30.0, 0.0),), (1e6,), max_wait=10.0)
    field = _Field(sc.n_sectors, eff=0.0, outage=0.0)
    lib = _single_task_lib(sc, 1e6, rate=0.0)
    sol = PlanSolution(routes=((0,),), candidate_ids=(("d0", "t1"),))
    res = evaluate_solution(sc, lib, sol, field, _spec())
    assert res.feasible is False
    assert res.violations.wait_exceeded >= 1
    assert res.violations.upload_failures == 1
    assert res.violations.deadline_exceeded == 1
