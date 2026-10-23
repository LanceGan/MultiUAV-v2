"""Task 5: SLSQP waypoint refinement with fallback."""

from segment.stc.generators import generate_comm_waypoints
from segment.stc.radio import PerSectorRadioField
from segment.stc.scenario import load_stc_scenario
from segment.stc.slsqp_refine import refine_candidate, candidate_cost


def _field():
    sc = load_stc_scenario(2, task_seed=42, map_seed=0)
    return sc, PerSectorRadioField.from_scenario(sc, spacing_m=2000.0)


def test_refine_never_worsens_cost():
    sc, field = _field()
    cands = generate_comm_waypoints(sc, field, 0, 1, grid_shape=(4, 4))
    cand = cands[-1]  # a relay candidate
    refined = refine_candidate(cand, sc, field, max_waypoints=1)
    assert candidate_cost(refined, sc.limits.delta_t_s) <= candidate_cost(
        cand, sc.limits.delta_t_s) + 1e-6


def test_refine_returns_original_on_failure():
    sc, field = _field()
    # a direct (no waypoint) candidate cannot be refined -> returns original
    cands = generate_comm_waypoints(sc, field, 0, 1, grid_shape=(2, 2))
    direct = next(c for c in cands if c.generator == "comm_direct")
    refined = refine_candidate(direct, sc, field, max_waypoints=1)
    assert refined is direct or refined.candidate_id == direct.candidate_id
