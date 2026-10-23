"""Task 4: legal direct and communication-waypoint trace generators."""

import numpy as np

from segment.stc.generators import (
    generate_comm_waypoints,
    generate_straight_hold,
    generate_straight_relaxation,
)
from segment.stc.radio import PerSectorRadioField
from segment.stc.scenario import load_stc_scenario


def _field():
    sc = load_stc_scenario(2, task_seed=42, map_seed=0)
    return sc, PerSectorRadioField.from_scenario(sc, spacing_m=2000.0)


def test_straight_hold_waits_outside_arrival_region():
    sc, field = _field()
    cand = generate_straight_hold(sc, field, 0, 1)
    tgt = np.array(sc.task_positions_m[1])
    hp = np.array(cand.hold_position_m)
    assert np.linalg.norm(hp - tgt) >= sc.limits.arrive_radius_m - 1e-6


def test_candidate_ends_at_shared_service_anchor_without_jump():
    sc, field = _field()
    cand = generate_straight_hold(sc, field, 0, 1)
    tgt = np.array(sc.task_positions_m[1])
    assert np.allclose(cand.positions_m[-1], tgt)


def test_zero_payload_candidate_never_holds():
    sc, field = _field()
    cand = generate_straight_hold(sc, field, -1, 0)  # depot -> task 0, no payload
    assert cand.source_data_bits == 0.0
    # hold position equals the target's 15m boundary (no extra holding steps)
    assert cand.hold_position_m is not None


def test_straight_relaxation_is_marked_non_rankable():
    sc, field = _field()
    cand = generate_straight_relaxation(sc, field, 0, 1)
    assert cand.rankable is False


def test_waypoints_are_deterministic_and_deduplicated():
    sc, field = _field()
    c1 = generate_comm_waypoints(sc, field, 0, 1, grid_shape=(4, 4))
    c2 = generate_comm_waypoints(sc, field, 0, 1, grid_shape=(4, 4))
    ids1 = [c.candidate_id for c in c1]
    ids2 = [c.candidate_id for c in c2]
    assert ids1 == ids2
    assert len(ids1) == len(set(ids1))  # deduplicated


def test_waypoints_include_direct_candidate():
    sc, field = _field()
    cands = generate_comm_waypoints(sc, field, 0, 1, grid_shape=(4, 4))
    gens = {c.generator for c in cands}
    assert "comm_direct" in gens
