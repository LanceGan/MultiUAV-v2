"""Task 7: tiny-instance exact benchmark (honest bound, gap, timeout)."""

import itertools

import numpy as np
import pytest

from segment.stc.evaluator import evaluate_solution
from segment.stc.exact import _no_contention_lower_bound, solve_exact_tiny
from segment.stc.generators import generate_straight_hold
from segment.stc.types import PlanSolution, RadioSample, STCScenario, ScenarioLimits


class _MockField:
    def __init__(self, n_sectors):
        self.n_sectors = n_sectors

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.full(self.n_sectors, 1.0),
                           g2a_outage=np.full(self.n_sectors, 0.1))


def _scenario(n_tasks=3):
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    return STCScenario(n_uav=1, n_tasks=n_tasks,
                       task_positions_m=tuple((1000.0 + i * 800.0, 2000.0) for i in range(n_tasks)),
                       task_data_bits=tuple([1e6] * n_tasks), depot_m=(0.0, 2000.0),
                       terminal_m=(4000.0, 2000.0), limits=limits, length_m=4000.0,
                       width_m=4000.0, n_bs=4, task_seed=42, map_seed=0)


def _library(sc):
    field = _MockField(sc.n_sectors)
    lib = {}
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src != tgt:
                lib[(src, tgt)] = [generate_straight_hold(sc, field, src, tgt)]
    return field, lib


def _edges(route, K):
    if not route:
        return []
    return ([(-1, route[0])]
            + [(route[i], route[i + 1]) for i in range(len(route) - 1)]
            + [(route[-1], K)])


def _bruteforce(sc, field, lib):
    """Independent oracle: enumerate orders, take each edge's only candidate."""
    from segment.stc.evaluator import default_objective_spec
    spec = default_objective_spec(sc, lib)
    best = None
    for order in itertools.permutations(range(sc.n_tasks)):
        route = (tuple(order),)
        cids = (tuple(lib[s, t][0].candidate_id for s, t in _edges(order, sc.n_tasks)),)
        sol = PlanSolution(routes=route, candidate_ids=cids)
        res = evaluate_solution(sc, lib, sol, field, spec, keep_timelines=False)
        if res.feasible and (best is None or res.objective < best[0]):
            best = (res.objective, sol)
    return best


def test_exact_matches_bruteforce():
    sc = _scenario(3)
    field, lib = _library(sc)
    res = solve_exact_tiny(sc, lib, field)
    assert res.proven_optimal is True
    bf = _bruteforce(sc, field, lib)
    np.testing.assert_approx_equal(res.objective, bf[0])


def test_exact_finds_feasible():
    sc = _scenario(3)
    field, lib = _library(sc)
    res = solve_exact_tiny(sc, lib, field)
    assert res.objective < float("inf")
    assert res.explored > 0


def test_proven_optimal_has_tight_bound_and_zero_gap():
    sc = _scenario(3)
    field, lib = _library(sc)
    res = solve_exact_tiny(sc, lib, field)
    assert res.proven_optimal is True
    assert res.lower_bound == pytest.approx(res.objective)
    assert res.gap == 0.0


def test_no_contention_lower_bound_is_admissible():
    sc = _scenario(3)
    field, lib = _library(sc)
    from segment.stc.evaluator import default_objective_spec
    spec = default_objective_spec(sc, lib)
    lb = _no_contention_lower_bound(sc, lib, spec)
    assert np.isfinite(lb)
    assert lb > 0.0
    res = solve_exact_tiny(sc, lib, field)
    assert lb <= res.objective + 1e-9


def test_timeout_returns_valid_bound_and_infinite_gap():
    sc = _scenario(3)
    field, lib = _library(sc)
    res = solve_exact_tiny(sc, lib, field, time_limit_s=-1.0)
    assert res.proven_optimal is False
    assert np.isfinite(res.lower_bound)
    assert res.lower_bound > 0.0
    assert res.gap == float("inf")


def test_max_tasks_guard_raises_before_enumeration():
    sc = _scenario(10)
    field, lib = _library(sc)
    with pytest.raises(ValueError):
        solve_exact_tiny(sc, lib, field, max_tasks=5)


def test_default_max_tasks_guards_large_instance():
    sc = _scenario(5)
    field, lib = _library(sc)
    with pytest.raises(ValueError):
        solve_exact_tiny(sc, lib, field)
