"""Task 5: Pareto pruning and library persistence."""

import numpy as np

from segment.stc.library import pareto_prune
from segment.stc.radio import PerSectorRadioField
from segment.stc.scenario import load_stc_scenario
from segment.stc.generators import generate_comm_waypoints


def _cands():
    sc = load_stc_scenario(2, task_seed=42, map_seed=0)
    field = PerSectorRadioField.from_scenario(sc, spacing_m=2000.0)
    return sc, field, generate_comm_waypoints(sc, field, 0, 1, grid_shape=(4, 4))


def _cost(c):
    flight = c.flight_steps * 0.5
    upload = c.source_data_bits / max(np.mean(c.full_rate_bps.max(axis=1)), 1e-9)
    return flight, upload


def test_pareto_prune_removes_strictly_dominated():
    _, _, cands = _cands()
    pruned = pareto_prune(cands, max_candidates=10, min_sector_distance=0.0)
    costs = [_cost(c) for c in pruned]
    # no candidate should be strictly dominated by another in the pruned set
    for i in range(len(costs)):
        for j in range(len(costs)):
            if i != j:
                dom = (costs[j][0] <= costs[i][0] and costs[j][1] <= costs[i][1]
                       and (costs[j][0] < costs[i][0] or costs[j][1] < costs[i][1]))
                assert not dom
    assert len(pruned) <= 10


def test_library_round_trip_preserves_candidates(tmp_path):
    from segment.stc.library import load_library, save_library
    sc, _, cands = _cands()
    lib = {(0, 1): tuple(cands[:3])}
    world_hash = "test-hash"
    path = str(tmp_path / "lib.npz")
    save_library(lib, sc, world_hash, path)
    loaded = load_library(path)
    assert loaded.world_hash == world_hash
    assert len(loaded.candidates[(0, 1)]) == 3
    np.testing.assert_array_equal(loaded.candidates[(0, 1)][0].positions_m, cands[0].positions_m)
    np.testing.assert_array_equal(loaded.candidates[(0, 1)][0].full_rate_bps, cands[0].full_rate_bps)
