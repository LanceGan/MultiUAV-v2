"""Conflict-aware STC operators (Task 11).

Destroy/repair/local operators with the common signature
``operator(solution, context, rng) -> PlanSolution``, registered in named
dicts.  The conflict-aware variants use the ``ConflictReport`` from the
synchronous evaluator rather than a uniform per-UAV average factor.
"""

from dataclasses import dataclass

import numpy as np

from segment.stc.operators import _edges, _plan
from segment.stc.slsqp_refine import candidate_cost
from segment.stc.types import PlanSolution


@dataclass
class SearchContext:
    scenario: object
    library: dict
    field: object
    cache: object = None
    conflict: object = None


def _plan_routes(routes, scenario, library):
    return _plan(tuple(tuple(r) for r in routes), scenario, library)


def _missing_tasks(solution, K):
    present = {t for r in solution.routes for t in r}
    return [t for t in range(K) if t not in present]


def _routes_cost(routes, scenario, library):
    total = 0.0
    for r in routes:
        for src, tgt in _edges(tuple(r), scenario.n_tasks):
            total += candidate_cost(
                min(library[(src, tgt)], key=lambda c: candidate_cost(c, 0.5)), 0.5)
    return total


# --- destroy ---

def random_removal(solution, context, rng, q=2):
    routes = [list(r) for r in solution.routes]
    flat = [(u, i) for u in range(len(routes)) for i in range(len(routes[u]))]
    if len(flat) <= q:
        return solution
    removed = [flat[i] for i in rng.choice(len(flat), size=q, replace=False)]
    for u, i in sorted(removed, reverse=True):
        del routes[u][i]
    return _plan_routes(routes, context.scenario, context.library)


def conflict_hotspot_removal(solution, context, rng, q=2):
    routes = [list(r) for r in solution.routes]
    if context.conflict is not None and context.conflict.per_uav_loss:
        u = int(np.argmax(context.conflict.per_uav_loss))
    else:
        u = rng.randint(len(routes))
    for _ in range(min(q, len(routes[u]))):
        routes[u].pop()
    return _plan_routes(routes, context.scenario, context.library)


# --- repair ---

def cheapest_insertion(solution, context, rng):
    K = context.scenario.n_tasks
    routes = [list(r) for r in solution.routes]
    for t in _missing_tasks(solution, K):
        best = None
        for u in range(len(routes)):
            for pos in range(len(routes[u]) + 1):
                cand = [list(r) for r in routes]
                cand[u].insert(pos, t)
                c = _routes_cost(cand, context.scenario, context.library)
                if best is None or c < best[0]:
                    best = (c, u, pos)
        _, u, pos = best
        routes[u].insert(pos, t)
    return _plan_routes(routes, context.scenario, context.library)


def conflict_aware_insertion(solution, context, rng):
    K = context.scenario.n_tasks
    routes = [list(r) for r in solution.routes]
    penalty = np.zeros(len(routes))
    if context.conflict is not None and context.conflict.per_uav_loss:
        total = sum(context.conflict.per_uav_loss) + 1e-9
        penalty = np.array(context.conflict.per_uav_loss) / total
    for t in _missing_tasks(solution, K):
        best = None
        for u in range(len(routes)):
            for pos in range(len(routes[u]) + 1):
                cand = [list(r) for r in routes]
                cand[u].insert(pos, t)
                c = _routes_cost(cand, context.scenario, context.library) * (1.0 + 0.1 * penalty[u])
                if best is None or c < best[0]:
                    best = (c, u, pos)
        _, u, pos = best
        routes[u].insert(pos, t)
    return _plan_routes(routes, context.scenario, context.library)


# --- local / resource moves ---

def swap(solution, context, rng):
    routes = [list(r) for r in solution.routes]
    pos = [(u, i) for u in range(len(routes)) for i in range(len(routes[u]))]
    if len(pos) < 2:
        return solution
    idx = rng.choice(len(pos), size=2, replace=False)
    (u1, i1), (u2, i2) = pos[idx[0]], pos[idx[1]]
    routes[u1][i1], routes[u2][i2] = routes[u2][i2], routes[u1][i1]
    return _plan_routes(routes, context.scenario, context.library)


def path_switch(solution, context, rng):
    u = rng.randint(len(solution.routes))
    if not solution.routes[u]:
        return solution
    edges = _edges(solution.routes[u], context.scenario.n_tasks)
    e = rng.randint(len(edges))
    src, tgt = edges[e]
    cands = context.library[(src, tgt)]
    cids = [c.candidate_id for c in cands]
    cur = list(solution.candidate_ids[u])
    if len(cids) > 1 and cur[e] in cids:
        i = cids.index(cur[e])
        cur[e] = cids[(i + 1) % len(cids)]
    new_cids = list(solution.candidate_ids)
    new_cids[u] = tuple(cur)
    return PlanSolution(routes=solution.routes, candidate_ids=tuple(new_cids))


destroy_operators = {"random_removal": random_removal,
                     "conflict_hotspot_removal": conflict_hotspot_removal}
repair_operators = {"cheapest_insertion": cheapest_insertion,
                    "conflict_aware_insertion": conflict_aware_insertion}
local_operators = {"swap": swap, "path_switch": path_switch}
