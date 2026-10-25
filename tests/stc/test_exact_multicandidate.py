"""Task 7: independent multi-candidate exact-benchmark oracle.

The brute-force oracle here deliberately does NOT call ``_pick_candidate`` or
any production branch helper: it re-implements assignment/order/candidate
enumeration with its own ``itertools`` products and only calls the
authoritative evaluator (``evaluate_solution``) for leaf objectives.
"""

import itertools

import numpy as np

from segment.stc.evaluator import default_objective_spec, evaluate_solution
from segment.stc.exact import solve_exact_tiny
from segment.stc.generators import (
    generate_comm_waypoints,
    generate_straight_hold,
    generate_straight_relaxation,
)
from segment.stc.types import PlanSolution, RadioSample, STCScenario, ScenarioLimits


class _MockField:
    def __init__(self, n_sectors):
        self.n_sectors = n_sectors

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.full(self.n_sectors, 1.0),
                           g2a_outage=np.full(self.n_sectors, 0.1))


def _scenario(n_tasks=2, n_uav=1):
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    return STCScenario(n_uav=n_uav, n_tasks=n_tasks,
                       task_positions_m=tuple((800.0 + i * 800.0, 2000.0) for i in range(n_tasks)),
                       task_data_bits=tuple([1e6] * n_tasks), depot_m=(0.0, 2000.0),
                       terminal_m=(4000.0, 2000.0), limits=limits, length_m=4000.0,
                       width_m=4000.0, n_bs=4, task_seed=42, map_seed=0)


def _edges(route, K):
    if not route:
        return []
    return ([(-1, route[0])]
            + [(route[i], route[i + 1]) for i in range(len(route) - 1)]
            + [(route[-1], K)])


def _rankable(library, src, tgt):
    cs = [c for c in library[(src, tgt)] if c.rankable]
    return cs if cs else list(library[(src, tgt)])


def _oracle(sc, field, lib):
    """Independent brute-force oracle (no production branch helpers)."""
    spec = default_objective_spec(sc, lib)
    K = sc.n_tasks
    U = sc.n_uav
    best = None
    leaves = 0
    for assign in itertools.product(range(U), repeat=K):
        routes = [[] for _ in range(U)]
        for t, u in enumerate(assign):
            routes[u].append(t)
        for combo in itertools.product(*[itertools.permutations(r) for r in routes]):
            ordered = [list(c) for c in combo]
            per_route = []
            for r in ordered:
                ids_per_edge = [[c.candidate_id for c in _rankable(lib, s, t)]
                                for s, t in _edges(r, K)]
                per_route.append(ids_per_edge)
            for cids in itertools.product(*[itertools.product(*l) for l in per_route]):
                sol = PlanSolution(routes=tuple(tuple(r) for r in ordered),
                                   candidate_ids=tuple(cids))
                res = evaluate_solution(sc, lib, sol, field, spec, keep_timelines=False)
                leaves += 1
                if res.feasible and (best is None or res.objective < best[0]):
                    best = (res.objective, sol)
    return best, leaves


def _library(sc, field):
    lib = {}
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src != tgt:
                lib[(src, tgt)] = generate_comm_waypoints(sc, field, src, tgt,
                                                          grid_shape=(2, 2))
    return lib


def test_exact_matches_independent_oracle():
    sc = _scenario(2, 1)
    field = _MockField(sc.n_sectors)
    lib = _library(sc, field)
    res = solve_exact_tiny(sc, lib, field)
    assert res.proven_optimal is True
    bf, leaves = _oracle(sc, field, lib)
    assert bf is not None
    np.testing.assert_approx_equal(res.objective, bf[0])
    assert res.explored == leaves


def test_exact_enumerates_every_candidate_combo():
    sc = _scenario(2, 1)
    field = _MockField(sc.n_sectors)
    lib = _library(sc, field)
    res = solve_exact_tiny(sc, lib, field)
    # 1 assignment (single UAV), 2 orders, 3 edges each with 5 candidates
    assert res.explored == 2 * 5 ** 3


def test_exact_excludes_non_rankable_candidates():
    sc = _scenario(2, 1)
    field = _MockField(sc.n_sectors)
    lib = {}
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src != tgt:
                lib[(src, tgt)] = [generate_straight_hold(sc, field, src, tgt),
                                   generate_straight_relaxation(sc, field, src, tgt)]
    res = solve_exact_tiny(sc, lib, field)
    # only the rankable straight_hold candidate is enumerated (1 per edge)
    assert res.explored == 2 * 1 ** 3
    flat = [cid for cids in res.incumbent.candidate_ids for cid in cids]
    assert all("relaxation" not in cid for cid in flat)
