"""Task 5: grid A* candidate generation."""

import numpy as np

from segment.stc.generators import _node_pos
from segment.stc.graph_search import astar_candidates
from segment.stc.radio import PerSectorRadioField
from segment.stc.scenario import load_stc_scenario


def _field():
    sc = load_stc_scenario(2, task_seed=42, map_seed=0)
    return sc, PerSectorRadioField.from_scenario(sc, spacing_m=2000.0)


def test_astar_returns_path_with_correct_endpoints():
    sc, field = _field()
    cand = astar_candidates(sc, field, 0, 1, grid=8)
    src = _node_pos(sc, 0)
    tgt = _node_pos(sc, 1)
    assert np.allclose(cand.positions_m[0], src, atol=1e-6)
    assert np.allclose(cand.positions_m[-1], tgt, atol=1e-6)


def test_astar_is_deterministic():
    sc, field = _field()
    c1 = astar_candidates(sc, field, 0, 1, grid=8)
    c2 = astar_candidates(sc, field, 0, 1, grid=8)
    assert np.array_equal(c1.positions_m, c2.positions_m)
