"""Conflict-aware STC operators (Task 6 Step 2/3).

Destroy/repair/local operators with the common signature
``operator(solution, context, rng) -> PlanSolution``, registered in named dicts.
Destroy operators additionally accept ``q`` (number of tasks to remove).

The conflict-aware variants consume the real :class:`ConflictReport` /
:class:`EvaluationResult` carried on :class:`SearchContext` — edge-time-sector
hotspots, pairwise windows, per-UAV loss, and completion times — rather than a
uniform per-UAV average factor.  ``conflict_aware_insertion`` ranks candidate
insertions with the authoritative synchronous evaluator (real marginal
shared-bandwidth cost), never a static multiplier.
"""

from dataclasses import dataclass

import numpy as np

from segment.stc.evaluator import (
    default_objective_spec,
    evaluate_solution,
)
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
    result: object = None
    objective_spec: object = None


def _plan_routes(routes, scenario, library):
    return _plan(tuple(tuple(r) for r in routes), scenario, library)


def _missing_tasks(solution, K):
    present = {t for r in solution.routes for t in r}
    return [t for t in range(K) if t not in present]


def _routes_cost(routes, scenario, library):
    dt = scenario.limits.delta_t_s
    total = 0.0
    for r in routes:
        for src, tgt in _edges(tuple(r), scenario.n_tasks):
            rankable = [c for c in library[(src, tgt)] if c.rankable] or list(library[(src, tgt)])
            total += min(candidate_cost(c, dt) for c in rankable)
    return total


def _sync_rank(result):
    """Lexicographic synchronous rank: feasible-first, then raw makespan+delay."""
    if result.feasible:
        return (0, result.objective)
    v = result.violations
    return (1, v.unserved_tasks, v.upload_failures, v.outage_excess,
            v.deadline_exceeded, v.wait_exceeded,
            result.makespan_s + result.total_delay_s)


def _eval_solution(context, solution):
    """Real synchronous marginal evaluation through the authoritative evaluator."""
    spec = context.objective_spec
    if spec is None:
        spec = default_objective_spec(context.scenario, context.library)
    return evaluate_solution(context.scenario, context.library, solution,
                             context.field, spec)


# --------------------------------------------------------------------------- #
# destroy
# --------------------------------------------------------------------------- #

def random_removal(solution, context, rng, q=2):
    routes = [list(r) for r in solution.routes]
    flat = [(u, i) for u in range(len(routes)) for i in range(len(routes[u]))]
    if len(flat) <= q:
        return solution
    removed = [flat[i] for i in rng.choice(len(flat), size=q, replace=False)]
    for u, i in sorted(removed, reverse=True):
        del routes[u][i]
    return _plan_routes(routes, context.scenario, context.library)


def worst_cost_removal(solution, context, rng, q=2):
    """Remove the tasks with the largest marginal single-UAV edge cost."""
    scenario = context.scenario
    dt = scenario.limits.delta_t_s
    routes = [list(r) for r in solution.routes]
    scored = []
    for u, r in enumerate(routes):
        edges = _edges(tuple(r), scenario.n_tasks)
        for i, t in enumerate(r):
            cost = 0.0
            for e in (i, i + 1):
                src, tgt = edges[e]
                rankable = [c for c in context.library[(src, tgt)] if c.rankable] \
                    or list(context.library[(src, tgt)])
                cost += min(candidate_cost(c, dt) for c in rankable)
            scored.append((cost, u, i))
    scored.sort(key=lambda x: -x[0])
    removals = sorted(((u, i) for _, u, i in scored[:q]),
                      key=lambda x: (x[0], -x[1]))
    for u, i in removals:
        del routes[u][i]
    return _plan_routes(routes, scenario, context.library)


def critical_route_removal(solution, context, rng, q=2):
    """Remove tasks from the makespan-critical UAV (largest completion time)."""
    routes = [list(r) for r in solution.routes]
    if context.result is not None and context.result.balance is not None:
        comps = np.asarray(context.result.balance.completion_times_s, dtype=float)
        if np.all(np.isnan(comps)):
            u = int(np.argmax([len(r) for r in routes]))
        else:
            u = int(np.nanargmax(comps))
    else:
        u = int(np.argmax([len(r) for r in routes]))
    for _ in range(min(q, len(routes[u]))):
        routes[u].pop()
    return _plan_routes(routes, context.scenario, context.library)


def conflict_hotspot_removal(solution, context, rng, q=2):
    """Remove tasks adjacent to the highest lost-bit edge occurrences."""
    K = context.scenario.n_tasks
    if context.conflict is None or not context.conflict.edge_hotspots:
        return random_removal(solution, context, rng, q)
    routes = [list(r) for r in solution.routes]
    order = sorted(context.conflict.edge_hotspots.items(), key=lambda kv: -kv[1])
    targets = []
    seen = set()
    for occ, _loss in order:
        for t in (occ.source, occ.target):
            if 0 <= t < K and t not in seen:
                seen.add(t)
                targets.append(t)
            if len(targets) >= q:
                break
        if len(targets) >= q:
            break
    for t in targets:
        for u, r in enumerate(routes):
            if t in r:
                r.remove(t)
                break
    return _plan_routes(routes, context.scenario, context.library)


def sector_related_removal(solution, context, rng, q=2):
    """Remove a group of tasks congesting the same sector in overlapping time."""
    if context.conflict is None or not context.conflict.pairwise_windows:
        return random_removal(solution, context, rng, q)
    by_sector = {}
    for w in context.conflict.pairwise_windows:
        by_sector[w.sector] = by_sector.get(w.sector, 0.0) + w.lost_bits
    top = max(by_sector, key=by_sector.get)
    uavs = {w.uav_a for w in context.conflict.pairwise_windows if w.sector == top}
    uavs |= {w.uav_b for w in context.conflict.pairwise_windows if w.sector == top}
    routes = [list(r) for r in solution.routes]
    removed = 0
    for u in sorted(uavs):
        while removed < q and routes[u]:
            routes[u].pop()
            removed += 1
        if removed >= q:
            break
    return _plan_routes(routes, context.scenario, context.library)


def infeasibility_removal(solution, context, rng, q=2):
    """Remove tasks from UAVs that violate upload/outage/wait constraints."""
    scenario = context.scenario
    routes = [list(r) for r in solution.routes]
    viol = []
    if context.result is not None:
        comps = context.result.balance.completion_times_s
        for u in range(len(routes)):
            bad = (np.isnan(comps[u])
                   or context.result.outage_ratio[u] > scenario.limits.eps_out)
            if bad:
                viol.append(u)
    if not viol:
        return random_removal(solution, context, rng, q)
    removed = 0
    for u in viol:
        while removed < q and routes[u]:
            routes[u].pop()
            removed += 1
        if removed >= q:
            break
    return _plan_routes(routes, scenario, context.library)


# --------------------------------------------------------------------------- #
# repair
# --------------------------------------------------------------------------- #

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


def regret2_insertion(solution, context, rng):
    K = context.scenario.n_tasks
    routes = [list(r) for r in solution.routes]
    missing = _missing_tasks(solution, K)
    while missing:
        best_choice = None
        for t in missing:
            costs = []
            for u in range(len(routes)):
                for pos in range(len(routes[u]) + 1):
                    cand = [list(r) for r in routes]
                    cand[u].insert(pos, t)
                    c = _routes_cost(cand, context.scenario, context.library)
                    costs.append((c, u, pos))
            costs.sort(key=lambda x: x[0])
            regret = (costs[1][0] - costs[0][0]) if len(costs) >= 2 else float("inf")
            if best_choice is None or regret > best_choice[0]:
                best_choice = (regret, t, costs[0])
        _, t, (c, u, pos) = best_choice
        routes[u].insert(pos, t)
        missing.remove(t)
    return _plan_routes(routes, context.scenario, context.library)


def makespan_balancing_insertion(solution, context, rng):
    """Insert missing tasks into the least-loaded (earliest-finishing) UAV."""
    K = context.scenario.n_tasks
    routes = [list(r) for r in solution.routes]
    for t in _missing_tasks(solution, K):
        if context.result is not None and context.result.balance is not None:
            comps = list(context.result.balance.completion_times_s)
            comps = [c if not np.isnan(c) else float("inf") for c in comps]
        else:
            comps = [len(r) for r in routes]
        u = int(np.argmin(comps))
        routes[u].append(t)
    return _plan_routes(routes, context.scenario, context.library)


def conflict_aware_insertion(solution, context, rng):
    """Insert each missing task by real synchronous marginal evaluation (§7.4).

    For every ``(uav, position)`` candidate the full synchronous shared-bandwidth
    evaluation is computed through :func:`evaluate_solution`; the best candidate
    by feasibility rank then normalized objective is chosen.  This is the
    conflict-aware repair — a static ``1 + 0.1 * per_uav_loss`` multiplier must
    fail the reversal fixture (independent edge cost prefers A, shared
    evaluation prefers B, and this operator must choose B).
    """
    K = context.scenario.n_tasks
    routes = [list(r) for r in solution.routes]
    for t in _missing_tasks(solution, K):
        best = None
        for u in range(len(routes)):
            for pos in range(len(routes[u]) + 1):
                cand = [list(r) for r in routes]
                cand[u].insert(pos, t)
                sol = _plan_routes(cand, context.scenario, context.library)
                res = _eval_solution(context, sol)
                key = _sync_rank(res)
                if best is None or key < best[0]:
                    best = (key, u, pos)
        _, u, pos = best
        routes[u].insert(pos, t)
    return _plan_routes(routes, context.scenario, context.library)


def _sector_entropy(candidate):
    if len(candidate.sector_ids) == 0:
        return 0.0
    n_sectors = candidate.full_rate_bps.shape[1]
    sig = np.bincount(np.asarray(candidate.sector_ids, dtype=int),
                      minlength=n_sectors) / len(candidate.sector_ids)
    sig = sig[sig > 0]
    if n_sectors <= 1:
        return 0.0
    return float(-np.sum(sig * np.log(sig)) / np.log(n_sectors))


def sector_diversity_insertion(solution, context, rng):
    """Insert missing tasks, rewarding sector-diverse edge candidates (§7.4)."""
    scenario = context.scenario
    K = scenario.n_tasks
    dt = scenario.limits.delta_t_s
    routes = [list(r) for r in solution.routes]
    for t in _missing_tasks(solution, K):
        best = None
        for u in range(len(routes)):
            for pos in range(len(routes[u]) + 1):
                cand = [list(r) for r in routes]
                cand[u].insert(pos, t)
                c = _routes_cost(cand, scenario, context.library)
                edges = _edges(tuple(cand[u]), K)
                entropy = 0.0
                for e in (pos, pos + 1):
                    src, tgt = edges[e]
                    rankable = [cc for cc in context.library[(src, tgt)] if cc.rankable] \
                        or list(context.library[(src, tgt)])
                    cbest = min(rankable, key=lambda cc: candidate_cost(cc, dt))
                    entropy += _sector_entropy(cbest)
                score = c - 0.1 * entropy
                if best is None or score < best[0]:
                    best = (score, u, pos)
        _, u, pos = best
        routes[u].insert(pos, t)
    return _plan_routes(routes, scenario, context.library)


# --------------------------------------------------------------------------- #
# local / resource moves
# --------------------------------------------------------------------------- #

def relocate(solution, context, rng):
    routes = [list(r) for r in solution.routes]
    nonempty = [u for u in range(len(routes)) if routes[u]]
    if not nonempty:
        return solution
    src_u = int(rng.choice(nonempty))
    i = int(rng.randint(len(routes[src_u])))
    t = routes[src_u].pop(i)
    dst_u = int(rng.randint(len(routes)))
    pos = int(rng.randint(len(routes[dst_u]) + 1))
    routes[dst_u].insert(pos, t)
    return _plan_routes(routes, context.scenario, context.library)


def swap(solution, context, rng):
    routes = [list(r) for r in solution.routes]
    pos = [(u, i) for u in range(len(routes)) for i in range(len(routes[u]))]
    if len(pos) < 2:
        return solution
    idx = rng.choice(len(pos), size=2, replace=False)
    (u1, i1), (u2, i2) = pos[idx[0]], pos[idx[1]]
    routes[u1][i1], routes[u2][i2] = routes[u2][i2], routes[u1][i1]
    return _plan_routes(routes, context.scenario, context.library)


def two_opt(solution, context, rng):
    routes = [list(r) for r in solution.routes]
    u = int(rng.randint(len(routes)))
    if len(routes[u]) < 3:
        return solution
    i = int(rng.randint(0, len(routes[u]) - 2))
    j = int(rng.randint(i + 2, len(routes[u]) + 1))
    routes[u][i:j] = reversed(routes[u][i:j])
    return _plan_routes(routes, context.scenario, context.library)


def route_tail_exchange(solution, context, rng):
    routes = [list(r) for r in solution.routes]
    nonempty = [u for u in range(len(routes)) if routes[u]]
    if len(nonempty) < 2:
        return solution
    u1, u2 = rng.choice(nonempty, size=2, replace=False)
    i1 = int(rng.randint(len(routes[u1])))
    i2 = int(rng.randint(len(routes[u2])))
    tail1 = routes[u1][i1:]
    tail2 = routes[u2][i2:]
    routes[u1] = routes[u1][:i1] + tail2
    routes[u2] = routes[u2][:i2] + tail1
    return _plan_routes(routes, context.scenario, context.library)


def path_switch(solution, context, rng):
    u = int(rng.randint(len(solution.routes)))
    if not solution.routes[u]:
        return solution
    edges = _edges(solution.routes[u], context.scenario.n_tasks)
    e = int(rng.randint(len(edges)))
    src, tgt = edges[e]
    cands = [c for c in context.library[(src, tgt)] if c.rankable] \
        or list(context.library[(src, tgt)])
    cids = [c.candidate_id for c in cands]
    cur = list(solution.candidate_ids[u])
    if len(cids) > 1 and cur[e] in cids:
        i = cids.index(cur[e])
        cur[e] = cids[(i + 1) % len(cids)]
    new_cids = list(solution.candidate_ids)
    new_cids[u] = tuple(cur)
    return PlanSolution(routes=solution.routes, candidate_ids=tuple(new_cids))


destroy_operators = {
    "random_removal": random_removal,
    "worst_cost_removal": worst_cost_removal,
    "critical_route_removal": critical_route_removal,
    "conflict_hotspot_removal": conflict_hotspot_removal,
    "sector_related_removal": sector_related_removal,
    "infeasibility_removal": infeasibility_removal,
}
repair_operators = {
    "cheapest_insertion": cheapest_insertion,
    "regret2_insertion": regret2_insertion,
    "makespan_balancing_insertion": makespan_balancing_insertion,
    "conflict_aware_insertion": conflict_aware_insertion,
    "sector_diversity_insertion": sector_diversity_insertion,
}
local_operators = {
    "relocate": relocate,
    "swap": swap,
    "two_opt": two_opt,
    "route_tail_exchange": route_tail_exchange,
    "path_switch": path_switch,
}
