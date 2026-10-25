"""Tiny-instance exact discrete benchmark (Task 13).

Exhaustively enumerates assignment, visit order, and candidate choice over the
discrete candidate space, returning the best feasible solution with a
proven-optimal flag.  Optimality is *within the discrete candidate space*
only, not the continuous space.  ``lower_bound`` is currently a placeholder
(0.0) — the admissible no-contention bound is not yet implemented, so it must
not be reported as a valid bound.
"""

import itertools
import time
from dataclasses import dataclass

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


def _plan_variants(routes, scenario, library):
    """Yield every PlanSolution for the given routes (one per candidate combo)."""
    from itertools import product
    per_route = []
    for r in routes:
        edges = ([] if not r else
                 [(-1, r[0])] + [(r[i], r[i + 1]) for i in range(len(r) - 1)]
                 + [(r[-1], scenario.n_tasks)])
        per_route.append([[c.candidate_id for c in library[e]] for e in edges])
    route_combos = [list(product(*lists)) for lists in per_route]
    for combo in product(*route_combos):
        yield PlanSolution(routes=tuple(tuple(r) for r in routes),
                           candidate_ids=tuple(combo))


def solve_exact_tiny(scenario, library, field, max_tasks=8, time_limit_s=3600.0):
    start = time.time()
    K = scenario.n_tasks
    U = scenario.n_uav
    if K > max_tasks:
        raise ValueError("K=%d exceeds max_tasks=%d; exact enumeration intractable"
                         % (K, max_tasks))

    best_sol = None
    best_res = None
    explored = 0
    timed_out = False
    spec = default_objective_spec(scenario, library)

    for assign in itertools.product(range(U), repeat=K):
        routes = [[] for _ in range(U)]
        for t, u in enumerate(assign):
            routes[u].append(t)
        perms = [list(itertools.permutations(r)) for r in routes]
        for combo in itertools.product(*perms):
            ordered = [list(c) for c in combo]
            if time.time() - start > time_limit_s:
                timed_out = True
                break
            for sol in _plan_variants(ordered, scenario, library):
                res = evaluate_solution(scenario, library, sol, field, spec)
                explored += 1
                if res.feasible and (best_res is None or res.objective < best_res.objective):
                    best_sol, best_res = sol, res
        if timed_out:
            break

    proven = not timed_out
    if best_sol is None:
        empty = PlanSolution(
            routes=tuple(() for _ in range(U)),
            candidate_ids=tuple(() for _ in range(U)))
        return ExactResult(incumbent=empty, objective=float("inf"), lower_bound=0.0,
                           proven_optimal=False, explored=explored, pruned=0,
                           gap=float("inf"), runtime_s=time.time() - start)

    return ExactResult(incumbent=best_sol, objective=best_res.objective,
                       lower_bound=0.0,  # placeholder — admissible bound not implemented
                       proven_optimal=proven, explored=explored, pruned=0,
                       gap=0.0 if proven else float("inf"),
                       runtime_s=time.time() - start)
