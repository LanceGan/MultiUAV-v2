"""Task 5: edge hotspots and pairwise conflict windows (spec §6).

Adjacent conflict slots for the same UAV pair + sector + route-edges merge into
one window; a sector or edge change splits the window; an empty-buffer overlap
creates no window.  Lost bits are keyed to the exact edge occurrence
``(uav_id, route_edge_index, source, target, candidate_id)`` and conserved
across per-UAV and per-edge sums.
"""

import pytest

from segment.stc.conflicts import build_conflict_report
from segment.stc.types import (
    EdgeOccurrence,
    EvaluationResult,
    PairwiseConflictWindow,
    TimelineSlot,
    UAVTimeline,
    ViolationVector,
)


def _slot(t, dt, sector, pre_bits, alloc_hz, edge_id=0, source=-1, target=1,
          cid="c", full_rate=1e6):
    return TimelineSlot(
        time_start_s=t, time_end_s=t + dt, edge_id=edge_id, candidate_id=cid,
        source=source, target=target, position_m=(0.0, 0.0), airborne=True,
        pre_buffer_bits=pre_bits,
        post_buffer_bits=0.0 if pre_bits > 0.0 else 0.0,
        active_uploader=(pre_bits > 0.0), serving_sector=sector,
        allocated_bandwidth_hz=alloc_hz, rate_bps=full_rate,
        uploaded_bits=0.0, g2a_outage=0.0, holding=False, service_event="")


def _result(timelines):
    return EvaluationResult(
        feasible=True, makespan_s=0.0, total_delay_s=0.0, outage_ratio=(0.1,),
        violations=ViolationVector(0, 0, 0.0, 0, 0), timelines=timelines,
        objective=0.0,
    )


# Two UAVs both upload in sector 0 with half bandwidth each (B = 1e6, dt = 0.5).
_B = 1e6
_DT = 0.5


def test_adjacent_conflict_slots_merge_into_single_window():
    tl0 = UAVTimeline(0, (
        _slot(0.0, _DT, 0, 1e6, 5e5),
        _slot(0.5, _DT, 0, 1e6, 5e5),
    ))
    tl1 = UAVTimeline(1, (
        _slot(0.0, _DT, 0, 1e6, 5e5),
        _slot(0.5, _DT, 0, 1e6, 5e5),
    ))
    rep = build_conflict_report(_result((tl0, tl1)), _B, _DT)
    # two adjacent same-pair/sector/edge slots merge into one interval
    assert rep.pairwise_windows == (
        PairwiseConflictWindow(0, 1, 0, 0, 0, 0.0, 1.0, 1e6),)


def test_sector_change_splits_window():
    tl0 = UAVTimeline(0, (
        _slot(0.0, _DT, 0, 1e6, 5e5),
        _slot(0.5, _DT, 1, 1e6, 5e5),
    ))
    tl1 = UAVTimeline(1, (
        _slot(0.0, _DT, 0, 1e6, 5e5),
        _slot(0.5, _DT, 1, 1e6, 5e5),
    ))
    rep = build_conflict_report(_result((tl0, tl1)), _B, _DT)
    assert rep.pairwise_windows == (
        PairwiseConflictWindow(0, 1, 0, 0, 0, 0.0, 0.5, 5e5),
        PairwiseConflictWindow(0, 1, 1, 0, 0, 0.5, 1.0, 5e5),
    )


def test_edge_change_splits_window():
    tl0 = UAVTimeline(0, (
        _slot(0.0, _DT, 0, 1e6, 5e5, edge_id=0),
        _slot(0.5, _DT, 0, 1e6, 5e5, edge_id=1),
    ))
    tl1 = UAVTimeline(1, (
        _slot(0.0, _DT, 0, 1e6, 5e5, edge_id=0),
        _slot(0.5, _DT, 0, 1e6, 5e5, edge_id=1),
    ))
    rep = build_conflict_report(_result((tl0, tl1)), _B, _DT)
    assert rep.pairwise_windows == (
        PairwiseConflictWindow(0, 1, 0, 0, 0, 0.0, 0.5, 5e5),
        PairwiseConflictWindow(0, 1, 0, 1, 1, 0.5, 1.0, 5e5),
    )


def test_empty_buffer_overlap_creates_no_window():
    tl0 = UAVTimeline(0, (_slot(0.0, _DT, 0, 1e6, 5e5),))    # active
    tl1 = UAVTimeline(1, (_slot(0.0, _DT, 0, 0.0, 0.0),))    # empty buffer
    rep = build_conflict_report(_result((tl0, tl1)), _B, _DT)
    assert rep.pairwise_windows == ()
    assert rep.edge_hotspots == {}
    assert rep.lost_bits == 0.0


def test_edge_hotspots_keyed_by_occurrence_and_conserved():
    tl0 = UAVTimeline(0, (
        _slot(0.0, _DT, 0, 1e6, 5e5, edge_id=0, source=-1, target=0, cid="d0"),
        _slot(0.5, _DT, 0, 1e6, 5e5, edge_id=1, source=0, target=1, cid="t0"),
    ))
    tl1 = UAVTimeline(1, (
        _slot(0.0, _DT, 0, 1e6, 5e5, edge_id=0, source=-1, target=2, cid="d2"),
        _slot(0.5, _DT, 0, 1e6, 5e5, edge_id=1, source=2, target=1, cid="t2"),
    ))
    rep = build_conflict_report(_result((tl0, tl1)), _B, _DT)

    # each active shared slot: rate_bps * (1 - alloc/B) * dt = 1e6 * 0.5 * 0.5
    per_slot = 2.5e5
    assert rep.lost_bits == pytest.approx(4 * per_slot)
    assert list(rep.per_uav_loss) == pytest.approx([2 * per_slot, 2 * per_slot])

    assert set(rep.edge_hotspots) == {
        EdgeOccurrence(0, 0, -1, 0, "d0"),
        EdgeOccurrence(0, 1, 0, 1, "t0"),
        EdgeOccurrence(1, 0, -1, 2, "d2"),
        EdgeOccurrence(1, 1, 2, 1, "t2"),
    }
    for occ, val in rep.edge_hotspots.items():
        assert val == pytest.approx(per_slot)
    # system loss == sum over UAVs == sum over edge occurrences
    assert sum(rep.per_uav_loss) == pytest.approx(rep.lost_bits)
    assert sum(rep.edge_hotspots.values()) == pytest.approx(rep.lost_bits)
