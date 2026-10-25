"""Task 11: conflict-aware operators preserve task coverage."""

import numpy as np

from segment.stc.generators import generate_straight_hold
from segment.stc.operators import build_initial_solution
from segment.stc.stc_operators import (
    SearchContext,
    cheapest_insertion,
    conflict_aware_insertion,
    conflict_hotspot_removal,
    destroy_operators,
    local_operators,
    path_switch,
    random_removal,
    repair_operators,
    swap,
)
from segment.stc.types import RadioSample, STCScenario, ScenarioLimits


class _MockField:
    def __init__(self, n_sectors):
        self.n_sectors = n_sectors

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.full(self.n_sectors, 1.0),
                           g2a_outage=np.full(self.n_sectors, 0.1))


def _setup(n_tasks=6):
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    sc = STCScenario(n_uav=2, n_tasks=n_tasks,
                     task_positions_m=tuple((800.0 + i * 600.0, 2000.0) for i in range(n_tasks)),
                     task_data_bits=tuple([1e6] * n_tasks), depot_m=(0.0, 2000.0),
                     terminal_m=(4000.0, 2000.0), limits=limits, length_m=4000.0,
                     width_m=4000.0, n_bs=4, task_seed=42, map_seed=0)
    field = _MockField(sc.n_sectors)
    lib = {}
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src != tgt:
                lib[(src, tgt)] = [generate_straight_hold(sc, field, src, tgt)]
    return sc, field, lib


def _coverage(sol):
    return set(t for r in sol.routes for t in r)


def test_destroy_then_repair_restores_coverage():
    sc, field, lib = _setup(6)
    ctx = SearchContext(scenario=sc, library=lib, field=field)
    sol = build_initial_solution(sc, lib)
    rng = np.random.RandomState(0)
    destroyed = random_removal(sol, ctx, rng, q=2)
    assert len(_coverage(destroyed)) == sc.n_tasks - 2
    repaired = cheapest_insertion(destroyed, ctx, rng)
    assert _coverage(repaired) == set(range(sc.n_tasks))


def test_conflict_operators_round_trip():
    sc, field, lib = _setup(6)
    ctx = SearchContext(scenario=sc, library=lib, field=field)
    sol = build_initial_solution(sc, lib)
    rng = np.random.RandomState(1)
    destroyed = conflict_hotspot_removal(sol, ctx, rng, q=2)
    repaired = conflict_aware_insertion(destroyed, ctx, rng)
    assert _coverage(repaired) == set(range(sc.n_tasks))


def test_local_moves_preserve_coverage():
    sc, field, lib = _setup(6)
    ctx = SearchContext(scenario=sc, library=lib, field=field)
    sol = build_initial_solution(sc, lib)
    rng = np.random.RandomState(2)
    for op in (swap, path_switch):
        moved = op(sol, ctx, rng)
        assert _coverage(moved) == _coverage(sol)
    assert set(local_operators) == {
        "relocate", "swap", "two_opt", "route_tail_exchange", "path_switch"}
    assert set(repair_operators) == {
        "cheapest_insertion", "regret2_insertion", "makespan_balancing_insertion",
        "conflict_aware_insertion", "sector_diversity_insertion"}
    assert set(destroy_operators) == {
        "random_removal", "worst_cost_removal", "critical_route_removal",
        "conflict_hotspot_removal", "sector_related_removal", "infeasibility_removal"}
