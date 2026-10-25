"""Task 8: conflict attribution on typed timelines."""

import pytest

from segment.stc.conflicts import build_conflict_report
from segment.stc.types import (
    ConflictReport,
    EvaluationResult,
    TimelineSlot,
    UAVTimeline,
    ViolationVector,
)


def _slot(t, sector, pre_bits, alloc_hz, full_rate=1e6):
    return TimelineSlot(
        time_start_s=t, time_end_s=t + 0.5, edge_id=0, candidate_id="c",
        position_m=(0.0, 0.0), airborne=True, pre_buffer_bits=pre_bits,
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


def test_no_overlap_different_sectors():
    tl0 = UAVTimeline(0, (_slot(0.0, 0, 1e6, 1e6),))   # UAV 0 in sector 0
    tl1 = UAVTimeline(1, (_slot(0.0, 1, 1e6, 1e6),))   # UAV 1 in sector 1
    rep = build_conflict_report(_result((tl0, tl1)), 1e6, 0.5)
    assert rep.total_overlap_s == 0.0
    assert rep.lost_bits == 0.0


def test_same_sector_empty_buffer_no_conflict():
    tl0 = UAVTimeline(0, (_slot(0.0, 0, 1e6, 1e6),))   # active
    tl1 = UAVTimeline(1, (_slot(0.0, 0, 0.0, 0.0),))   # empty buffer: not an uploader
    rep = build_conflict_report(_result((tl0, tl1)), 1e6, 0.5)
    assert rep.total_overlap_s == 0.0
    assert rep.lost_bits == 0.0


def test_two_active_same_sector_conflict():
    tl0 = UAVTimeline(0, (_slot(0.0, 0, 1e6, 5e5),))   # half bandwidth
    tl1 = UAVTimeline(1, (_slot(0.0, 0, 1e6, 5e5),))   # half bandwidth
    rep = build_conflict_report(_result((tl0, tl1)), 1e6, 0.5)
    assert rep.total_overlap_s == pytest.approx(0.5)  # one slot, two uploaders
    # each loses 1e6 * (1 - 0.5) * 0.5 = 2.5e5 bits
    assert rep.lost_bits == pytest.approx(5.0e5)
