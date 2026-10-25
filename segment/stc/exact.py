"""Tiny-instance exact discrete benchmark (Task 7).

Exhaustively enumerates assignment, visit order, and candidate choice over the
declared discrete domain (there is no departure dimension), returning the best
feasible solution with a proven-optimal flag.  ``proven_optimal`` is ``True``
only when the *entire* declared domain has been covered; a timeout returns the
incumbent plus a valid admissible no-contention lower bound and a finite gap
when both are finite, with ``proven_optimal=False``.

A pre-allocation size guard refuses to start any search whose declared leaf
count (assignment x permutation x candidate) exceeds ``max_leaves``, so an
intractable instance fails fast instead of grinding.  The reported
``lower_bound`` is a genuine admissible bound (never over-estimates the true
optimum), not a placeholder.
"""

import itertools
import math
import time
from dataclasses import dataclass

import numpy as np

from segment.stc.evaluator import default_objective_spec, evaluate_solution
from segment.stc.types import PlanSolution


@dataclass
class ExactResult:
    incumbent: PlanSolution
    objective: float
    lower_bound: float
    proven_optimal: bool
    explored: int
    pruned: int
    gap: float
    runtime_s: float


def _edge_candidates(library, src, tgt):
    """Candidates for a directed edge (empty when the edge is not in the library)."""
    try:
        return library[(src, tgt)]
    except KeyError:
        return ()


def _rankable(library, src, tgt):
    """Strict rankable-only candidates for an edge (empty when none exist)."""
    return [c for c in _edge_candidates(library, src, tgt) if c.rankable]


def _route_edges(route, K):
    if not route:
        return []
    return ([(-1, route[0])]
            + [(route[i], route[i + 1]) for i in range(len(route) - 1)]
            + [(route[-1], K)])


def _leaf_upper_bound(scenario, library):
    """Cheap upper bound on the declared leaf count (never under-estimates).

    For each assignment the number of leaves is the product over routes of
    ``len(route)!`` (permutations) times the candidate-product over its edges.
    The candidate-product per edge is bounded above by ``c_max ** (len(route)+1)``
    where ``c_max`` is the maximum rankable-candidate count over all edges, so
    the sum over assignments is an upper bound on the true leaf count.
    """
    K = scenario.n_tasks
    U = scenario.n_uav
    c_max = 1
    for s in range(-1, K + 1):
        for t in range(K + 1):
            if s != t:
                c_max = max(c_max, len(_rankable(library, s, t)))
    total = 0
    for assign in itertools.product(range(U), repeat=K):
        lens = [0] * U
        for t, u in enumerate(assign):
            lens[u] += 1
        leaves = 1
        for L in lens:
            cand = 1 if L == 0 else c_max ** (L + 1)
            leaves *= math.factorial(L) * cand
        total += leaves
    return total


def _no_contention_lower_bound(scenario, library, spec):
    """Admissible no-contention lower bound on the objective (never over-estimates).

    The objective ``w_t*T/T_ref + w_d*D/D_ref`` is decomposed into two
    independent admissible component bounds:

    * makespan ``T >= (total flight distance lower bound) / (U * v_max)``: every
      task is reached along an incoming edge whose Euclidean length is at least
      the distance from its nearest predecessor (depot or another task), so the
      total flight distance is at least the sum of those minima; the makespan is
      at least the average UAV flight time.
    * upload delay ``D >= sum_t data_t / r_max_t``: each task's data is uploaded
      exactly once, at a rate never exceeding the best full rate available on
      any edge leaving that task, so its upload duration is at least
      ``data_t / r_max_t``.

    Both terms are relaxations (no bandwidth contention, pure Euclidean flight,
    best-possible rate), so their weighted sum is a lower bound on every
    feasible objective.  Returns ``inf`` when some task's data has no candidate
    with positive rate (i.e. the instance is provably infeasible).
    """
    K = scenario.n_tasks
    U = scenario.n_uav
    if K == 0:
        return 0.0
    depot = np.asarray(scenario.depot_m, float)
    pos = [np.asarray(scenario.task_positions_m[t], float) for t in range(K)]

    total_flight_lb = 0.0
    for t in range(K):
        best = float(np.linalg.norm(depot - pos[t]))
        for p in range(K):
            if p != t:
                best = min(best, float(np.linalg.norm(pos[p] - pos[t])))
        total_flight_lb += best
    lb_t = total_flight_lb / (U * scenario.limits.v_max_mps)

    lb_d = 0.0
    for t in range(K):
        data = float(scenario.task_data_bits[t])
        if data <= 0.0:
            continue
        r_max = 0.0
        for j in list(range(K)) + [K]:
            if j == t:
                continue
            for c in _edge_candidates(library, t, j):
                r_max = max(r_max, float(np.max(c.full_rate_bps)))
        if r_max <= 0.0:
            return float("inf")
        lb_d += data / r_max

    return (spec.w_t * (lb_t / spec.reference.t_ref_s)
            + spec.w_d * (lb_d / spec.reference.d_ref_s))


def _gap(upper, lower):
    """Relative gap ``(upper - lower) / lower``; ``inf`` unless both finite."""
    if not np.isfinite(upper) or not np.isfinite(lower) or lower <= 0.0:
        return float("inf")
    return max(0.0, (upper - lower) / lower)


def _plan_variants(routes, scenario, library):
    """Lazily yield every PlanSolution for the given routes (one per candidate combo).

    Strict rankable-only: if any traversed edge has no *rankable* candidate, the
    route is outside the declared selectable domain and yields no solution (never
    fall back to a non-rankable relaxation reference).
    """
    per_route = []
    for r in routes:
        edges = _route_edges(r, scenario.n_tasks)
        id_lists = []
        for e in edges:
            cands = _rankable(library, e[0], e[1])
            if not cands:
                return  # no selectable candidate on this edge -> no solutions
            id_lists.append([c.candidate_id for c in cands])
        per_route.append(id_lists)
    route_choices = [itertools.product(*lists) for lists in per_route]
    for combo in itertools.product(*route_choices):
        yield PlanSolution(routes=tuple(tuple(r) for r in routes),
                           candidate_ids=tuple(combo))


def solve_exact_tiny(scenario, library, field, max_tasks=4, time_limit_s=3600.0,
                     objective_spec=None, max_leaves=2_000_000):
    """Guarded exact enumeration of assignment x order x candidate.

    Raises ``ValueError`` *before* any permutation/product allocation when
    ``K > max_tasks`` or when the declared leaf-count upper bound exceeds
    ``max_leaves``.  Enumerates lazily with a deadline check at every nested
    level (assignment, permutation, candidate).
    """
    start = time.time()
    K = scenario.n_tasks
    U = scenario.n_uav
    if K > max_tasks:
        raise ValueError("K=%d exceeds max_tasks=%d; exact enumeration intractable"
                         % (K, max_tasks))
    leaves = _leaf_upper_bound(scenario, library)
    if leaves > max_leaves:
        raise ValueError(
            "declared leaf count upper bound %d exceeds max_leaves=%d; "
            "refusing to start an intractable exact search" % (leaves, max_leaves))

    spec = (objective_spec if objective_spec is not None
            else default_objective_spec(scenario, library))
    lb = _no_contention_lower_bound(scenario, library, spec)

    best_sol = None
    best_obj = float("inf")
    explored = 0
    timed_out = False

    def deadline_hit():
        return time.time() - start > time_limit_s

    for assign in itertools.product(range(U), repeat=K):
        if deadline_hit():
            timed_out = True
            break
        routes = [[] for _ in range(U)]
        for t, u in enumerate(assign):
            routes[u].append(t)
        perms = (itertools.permutations(r) for r in routes)
        for combo in itertools.product(*perms):
            ordered = [list(c) for c in combo]
            if deadline_hit():
                timed_out = True
                break
            for sol in _plan_variants(ordered, scenario, library):
                if deadline_hit():
                    timed_out = True
                    break
                res = evaluate_solution(scenario, library, sol, field, spec,
                                        keep_timelines=False)
                explored += 1
                if res.feasible and res.objective < best_obj:
                    best_obj = res.objective
                    best_sol = sol
            if timed_out:
                break
        if timed_out:
            break

    runtime = time.time() - start
    empty = PlanSolution(routes=tuple(() for _ in range(U)),
                         candidate_ids=tuple(() for _ in range(U)))

    if timed_out:
        if best_sol is None:
            return ExactResult(incumbent=empty, objective=float("inf"),
                               lower_bound=lb, proven_optimal=False,
                               explored=explored, pruned=0, gap=float("inf"),
                               runtime_s=runtime)
        return ExactResult(incumbent=best_sol, objective=best_obj,
                           lower_bound=lb, proven_optimal=False,
                           explored=explored, pruned=0, gap=_gap(best_obj, lb),
                           runtime_s=runtime)

    # full domain covered without timeout
    if best_sol is None:
        # proven: no feasible solution exists in the declared discrete domain
        return ExactResult(incumbent=empty, objective=float("inf"),
                           lower_bound=float("inf"), proven_optimal=False,
                           explored=explored, pruned=0, gap=float("inf"),
                           runtime_s=runtime)

    return ExactResult(incumbent=best_sol, objective=best_obj,
                       lower_bound=best_obj, proven_optimal=True,
                       explored=explored, pruned=0, gap=0.0,
                       runtime_s=runtime)
