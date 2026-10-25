"""Task 8: conflict attribution on synthetic timelines."""

import pytest

from segment.stc.conflicts import build_conflict_report
from segment.stc.types import (
    ConflictReport,
    EvaluationResult,
    ViolationVector,
)


def _result(timelines):
    return EvaluationResult(
        feasible=True, makespan_s=0.0, total_delay_s=0.0, outage_ratio=(0.1,),
        violations=ViolationVector(0, 0, 0.0, 0, 0), timelines=timelines,
        objective=0.0,
    )


def test_no_overlap_different_sectors():
    tl0 = ((0.0, 0, 1e6, 1e6, 1e6),)   # UAV 0 in sector 0
    tl1 = ((0.0, 1, 1e6, 1e6, 1e6),)   # UAV 1 in sector 1
    rep = build_conflict_report(_result((tl0, tl1)), 1e6, 0.5)
    assert rep.total_overlap_s == 0.0
    assert rep.lost_bits == 0.0


def test_same_sector_empty_buffer_no_conflict():
    tl0 = ((0.0, 0, 1e6, 1e6, 1e6),)   # active
    tl1 = ((0.0, 0, 0.0, 0.0, 1e6),)   # empty buffer: not an uploader
    rep = build_conflict_report(_result((tl0, tl1)), 1e6, 0.5)
    assert rep.total_overlap_s == 0.0
    assert rep.lost_bits == 0.0


def test_two_active_same_sector_conflict():
    tl0 = ((0.0, 0, 1e6, 5e5, 1e6),)   # half bandwidth
    tl1 = ((0.0, 0, 1e6, 5e5, 1e6),)   # half bandwidth
    rep = build_conflict_report(_result((tl0, tl1)), 1e6, 0.5)
    assert rep.total_overlap_s == pytest.approx(0.5)  # one slot, two uploaders
    # each loses 1e6 * (1 - 0.5) * 0.5 = 2.5e5 bits
    assert rep.lost_bits == pytest.approx(5.0e5)
