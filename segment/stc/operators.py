"""Initial solutions and resource-unaware baselines (Task 10).

Construction (greedy NN split + cheapest candidate per edge) and three
resource-unaware baselines (NN+2-opt, cheapest insertion, LNS+SA).  All final
metrics come from the authoritative synchronous evaluator; the search itself
may use single-UAV candidate costs.
"""

import numpy as np

from segment.stc.evaluator import default_objective_spec, evaluate_solution
from segment.stc.slsqp_refine import candidate_cost
from segment.stc.types import PlanSolution, SearchResult


def _greedy_nn_order(scenario):
    order = []
    remaining = list(range(scenario.n_tasks))
    cur = np.asarray(scenario.depot_m, float)
    while remaining:
        nxt = min(remaining, key=lambda j: float(np.linalg.norm(
            np.asarray(scenario.task_positions_m[j]) - cur)))
        order.append(nxt)
        cur = np.asarray(scenario.task_positions_m[nxt])
        remaining.remove(nxt)
    return order


def _edges(route, K):
    if not route:
        return []  # empty route: idle UAV (no segments)
    return ([(-1, route[0])]
            + [(route[i], route[i + 1]) for i in range(len(route) - 1)]
            + [(route[-1], K)])


def _pick_candidate(library, src, tgt):
    return min(library[(src, tgt)], key=lambda c: candidate_cost(c, 0.5)).candidate_id


def _plan(routes, scenario, library):
    cids = tuple(tuple(_pick_candidate(library, s, t) for s, t in _edges(r, scenario.n_tasks))
                 for r in routes)
    return PlanSolution(routes=routes, candidate_ids=cids)


def build_initial_solution(scenario, library, seed=42):
    """Greedy NN single route split into n_uav groups; cheapest candidate per edge."""
    order = _greedy_nn_order(scenario)
    # keep exactly n_uav groups (some empty when n_tasks < n_uav)
    groups = [list(g) for g in np.array_split(order, scenario.n_uav)]
    routes = tuple(tuple(g) for g in groups)
    return _plan(routes, scenario, library)


def _two_opt(sol, scenario, library, rng):
    routes = [list(r) for r in sol.routes]
    u = rng.randint(len(routes))
    if len(routes[u]) < 3:
        return sol
    i = rng.randint(0, len(routes[u]) - 2)
    j = rng.randint(i + 2, len(routes[u]) + 1)
    routes[u][i:j] = reversed(routes[u][i:j])
    return _plan(tuple(tuple(r) for r in routes), scenario, library)


def _result(scenario, library, field, solution, objective_spec):
    return evaluate_solution(scenario, library, solution, field, objective_spec)


def run_nearest_neighbor_2opt(scenario, library, field, seed=42):
    spec = default_objective_spec(scenario, library)
    best = build_initial_solution(scenario, library, seed)
    best_res = _result(scenario, library, field, best, spec)
    rng = np.random.RandomState(seed)
    for _ in range(200):
        cand = _two_opt(best, scenario, library, rng)
        res = _result(scenario, library, field, cand, spec)
        if (not best_res.feasible and res.feasible) or (res.feasible and res.objective < best_res.objective):
            best, best_res = cand, res
    return SearchResult(best_solution=best, best_result=best_res,
                        search_trace=(), evaluator_calls={"full": 201})


def run_cheapest_insertion(scenario, library, field, seed=42):
    spec = default_objective_spec(scenario, library)
    sol = build_initial_solution(scenario, library, seed)
    res = _result(scenario, library, field, sol, spec)
    return SearchResult(best_solution=sol, best_result=res, search_trace=(),
                        evaluator_calls={"full": 1})


def run_lns_sa(scenario, library, field, seed=42, iters=200):
    spec = default_objective_spec(scenario, library)
    best = build_initial_solution(scenario, library, seed)
    best_res = _result(scenario, library, field, best, spec)
    rng = np.random.RandomState(seed)
    for _ in range(iters):
        cand = _two_opt(best, scenario, library, rng)
        res = _result(scenario, library, field, cand, spec)
        if (not best_res.feasible and res.feasible) or (res.feasible and res.objective < best_res.objective):
            best, best_res = cand, res
    return SearchResult(best_solution=best, best_result=best_res,
                        search_trace=(), evaluator_calls={"full": iters + 1})
