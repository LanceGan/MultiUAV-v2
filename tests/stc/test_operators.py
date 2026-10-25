"""Task 6: conflict-aware operators preserve coverage and candidate choices."""

import numpy as np

from segment.stc.generators import generate_straight_hold
from segment.stc.operators import build_initial_solution
from segment.stc.stc_operators import (
    SearchContext,
    cheapest_proxy_insertion,
    conflict_aware_insertion,
    conflict_hotspot_removal,
    destroy_operators,
    local_operators,
    path_switch,
    random_removal,
    repair_operators,
    swap,
)
from segment.stc.types import (
    PlanSolution,
    RadioSample,
    STCScenario,
    ScenarioLimits,
    SegmentCandidate,
)


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
    repaired = cheapest_proxy_insertion(destroyed, ctx, rng)
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
        "cheapest_proxy_insertion", "regret2_insertion",
        "makespan_balancing_insertion", "conflict_aware_insertion",
        "sector_diversity_insertion"}
    assert set(destroy_operators) == {
        "random_removal", "worst_cost_removal", "critical_route_removal",
        "conflict_hotspot_removal", "sector_related_removal", "infeasibility_removal"}


def test_cheapest_proxy_insertion_named_honestly():
    # the proxy-only insertion must not claim to be a "feasible" insertion
    assert "cheapest_proxy_insertion" in repair_operators
    assert "cheapest_insertion" not in repair_operators
    doc = repair_operators["cheapest_proxy_insertion"].__doc__ or ""
    assert "proxy" in doc and "feasible" in doc


def _cand(cid, src, tgt, steps, data, n_sectors=3):
    return SegmentCandidate(
        candidate_id=cid, source=src, target=tgt, generator="hand",
        positions_m=np.zeros((steps, 2)),
        full_rate_bps=np.full((steps, n_sectors), 1e6),
        sector_ids=np.zeros(steps, dtype=int),
        g2a_outage=np.zeros((steps, n_sectors)),
        flight_steps=steps, hold_position_m=(0.0, 0.0),
        source_data_bits=float(data), rankable=True)


def _multi_cand_fixture():
    limits = ScenarioLimits(delta_t_s=1.0, v_max_mps=10.0, uav_height_m=100.0,
                            arrive_radius_m=5.0, max_wait_s=10.0,
                            mission_deadline_s=1000.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    sc = STCScenario(n_uav=2, n_tasks=4,
                     task_positions_m=((10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (40.0, 0.0)),
                     task_data_bits=(0.0, 0.0, 0.0, 0.0), depot_m=(0.0, 0.0),
                     terminal_m=(60.0, 0.0), limits=limits, length_m=100.0,
                     width_m=100.0, n_bs=1, task_seed=42, map_seed=0)
    field = _MockField(sc.n_sectors)
    lib = {}
    for t in range(sc.n_tasks):
        lib[(-1, t)] = [_cand("d_%d" % t, -1, t, 3, 0.0)]
    for i in range(sc.n_tasks):
        for j in range(sc.n_tasks):
            if i != j:
                lib[(i, j)] = [_cand("e_%d_%d" % (i, j), i, j, 3, 0.0)]
    for i in range(sc.n_tasks):
        lib[(i, sc.n_tasks)] = [_cand("t_%d" % i, i, sc.n_tasks, 3, 0.0)]
    # edge (0,1) gets a slower (more expensive) second candidate
    lib[(0, 1)] = [lib[(0, 1)][0], _cand("exp_0_1", 0, 1, 20, 0.0)]
    return sc, field, lib


def test_repair_preserves_path_switched_candidates():
    sc, field, lib = _multi_cand_fixture()
    partial = PlanSolution(routes=((0, 1), (3,)),
                           candidate_ids=(("d_0", "exp_0_1", "t_1"), ("d_3", "t_3")))
    ctx = SearchContext(scenario=sc, library=lib, field=field)
    rng = np.random.RandomState(0)
    repaired = cheapest_proxy_insertion(partial, ctx, rng)

    def edge_map(sol):
        m = {}
        for r, cids in zip(sol.routes, sol.candidate_ids):
            edges = ([(-1, r[0])] + [(r[i], r[i + 1]) for i in range(len(r) - 1)]
                     + [(r[-1], sc.n_tasks)]) if r else []
            for (s, t), cid in zip(edges, cids):
                m[(s, t)] = cid
        return m

    pm, rm = edge_map(partial), edge_map(repaired)
    # every unchanged edge keeps its (possibly path-switched) candidate
    for (s, t), cid in pm.items():
        if (s, t) in rm:
            assert rm[(s, t)] == cid
    # the specific non-cheapest edge (0,1) survives and is preserved
    assert rm[(0, 1)] == "exp_0_1"
    # coverage restored
    assert set(t for r in repaired.routes for t in r) == set(range(sc.n_tasks))
