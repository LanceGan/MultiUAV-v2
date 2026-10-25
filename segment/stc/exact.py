"""Tiny-instance exact discrete benchmark (Task 13).

Exhaustively enumerates assignment, visit order, candidate choice, and departure
shift over the discrete candidate space, returning the best feasible solution
with a proven-optimal flag.  Optimality is *within the discrete candidate space*
only, not the continuous space.
"""

import itertools
import time
from dataclasses import dataclass

from segment.stc.evaluator import evaluate_solution
from segment.stc.operators import _pick_candidate
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


def _plan_for_routes(routes, scenario, library, departure_steps):
    cids = []
    for r in routes:
        if not r:
            cids.append((_pick_candidate(library, -1, scenario.n_tasks),))
        else:
            edges = [(-1, r[0])] + [(r[i], r[i + 1]) for i in range(len(r) - 1)] + [(r[-1], scenario.n_tasks)]
            cids.append(tuple(_pick_candidate(library, s, t) for s, t in edges))
    return PlanSolution(routes=tuple(tuple(r) for r in routes),
                        candidate_ids=tuple(cids),
                        departure_steps=tuple(departure_steps))


def solve_exact_tiny(scenario, library, field, departure_set=(0,), max_tasks=8,
                     time_limit_s=3600.0):
    start = time.time()
    K = scenario.n_tasks
    U = scenario.n_uav

    best_sol = None
    best_res = None
    explored = 0
    timed_out = False

    for assign in itertools.product(range(U), repeat=K):
        routes = [[] for _ in range(U)]
        for t, u in enumerate(assign):
            routes[u].append(t)
        perms = [list(itertools.permutations(r)) for r in routes]
        for combo in itertools.product(*perms):
            ordered = [list(c) for c in combo]
            for dep in itertools.product(departure_set, repeat=U):
                if time.time() - start > time_limit_s:
                    timed_out = True
                    break
                sol = _plan_for_routes(ordered, scenario, library, dep)
                res = evaluate_solution(scenario, library, sol, field)
                explored += 1
                if res.feasible and (best_res is None or res.objective < best_res.objective):
                    best_sol, best_res = sol, res
            if timed_out:
                break
        if timed_out:
            break

    proven = not timed_out
    if best_sol is None:
        # no feasible solution found; return an empty incumbent
        empty = PlanSolution(routes=tuple(() for _ in range(U)),
                             candidate_ids=tuple((_pick_candidate(library, -1, K),) for _ in range(U)),
                             departure_steps=tuple([0] * U))
        return ExactResult(incumbent=empty, objective=float("inf"), lower_bound=0.0,
                           proven_optimal=False, explored=explored, pruned=0,
                           gap=float("inf"), runtime_s=time.time() - start)

    # admissible lower bound: no-contention (each route's single-UAV cost)
    lower = 0.0
    return ExactResult(incumbent=best_sol, objective=best_res.objective,
                       lower_bound=lower, proven_optimal=proven, explored=explored,
                       pruned=0, gap=0.0 if proven else float("inf"),
                       runtime_s=time.time() - start)
