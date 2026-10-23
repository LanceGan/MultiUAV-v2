"""Task 2: typed data contracts (immutable dataclasses + validation)."""

import numpy as np
import pytest

from segment.stc.types import (
    PlanSolution,
    ScenarioLimits,
    SegmentCandidate,
)


def test_segment_candidate_rejects_mismatched_trace_lengths():
    pos = np.zeros((10, 2))
    rate = np.zeros((9, 3))          # wrong length
    sec = np.zeros(10, dtype=int)
    out = np.zeros((10, 3))
    with pytest.raises(ValueError):
        SegmentCandidate(
            candidate_id="c0", source=0, target=1, generator="straight",
            positions_m=pos, full_rate_bps=rate, sector_ids=sec,
            g2a_outage=out, flight_steps=10, hold_position_m=(0.0, 0.0),
            source_data_bits=100.0,
        )


def test_plan_solution_requires_every_task_exactly_once():
    with pytest.raises(ValueError):
        PlanSolution(
            routes=((0, 1, 2), (0, 3)),   # task 0 appears twice, task 4 missing
            candidate_ids=(("c0", "c1", "c2"), ("c3", "c4")),
            departure_steps=(0, 0),
        )


def test_scenario_requires_positive_finite_limits():
    with pytest.raises(ValueError):
        ScenarioLimits(
            delta_t_s=0.5, v_max_mps=0.0, uav_height_m=100.0,
            arrive_radius_m=15.0, max_wait_s=60.0, mission_deadline_s=7200.0,
            departure_step_s=1.0, departure_max_s=60.0, bandwidth_hz=1e6,
            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5,
        )
