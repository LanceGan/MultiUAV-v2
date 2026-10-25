"""Adaptive conflict-aware STC-ALNS solver (Task 12).

Roulette operator selection with adaptive weights (spec §7.6), feasibility-first
simulated-annealing acceptance, and a separately-maintained best-feasible
solution.  Deterministic under a fixed ``method_seed``.
"""

import numpy as np

from segment.stc.conflicts import build_conflict_report
from segment.stc.evaluator import compute_reference, feasibility_rank
from segment.stc.incremental import EvaluationCache
from segment.stc.stc_operators import (
    SearchContext,
    destroy_operators,
    local_operators,
    repair_operators,
)
from segment.stc.types import SearchResult


class STCALNS:
    def __init__(self, scenario, library, field, rho=0.1, T0=0.1, cooling=0.995,
                 segment=100, score_new_best=3.0, score_improve=1.0, score_accept=0.5):
        self.scenario = scenario
        self.library = library
        self.field = field
        self.rho = rho
        self.T0 = T0
        self.cooling = cooling
        self.segment = segment
        self.score_new_best = score_new_best
        self.score_improve = score_improve
        self.score_accept = score_accept

    def _pick(self, weights, rng):
        names = list(weights)
        w = np.array([weights[n] for n in names], dtype=float)
        w = np.clip(w, 1e-9, None)
        p = w / w.sum()
        return names[int(rng.choice(len(names), p=p))]

    def solve(self, initial, iterations=1000, method_seed=42):
        rng = np.random.RandomState(method_seed)
        ctx = SearchContext(scenario=self.scenario, library=self.library, field=self.field)
        t_ref, d_ref = compute_reference(self.scenario, self.library)
        cache = EvaluationCache.from_full(self.scenario, self.library, self.field,
                                          initial, t_ref=t_ref, d_ref=d_ref)

        wd = {n: 1.0 for n in destroy_operators}
        wr = {n: 1.0 for n in repair_operators}
        wl = {n: 1.0 for n in local_operators}
        scores = {n: 0.0 for n in list(destroy_operators) + list(repair_operators) + list(local_operators)}
        uses = {n: 0 for n in scores}

        cur = initial
        cur_res = cache.base_result
        ctx.conflict = build_conflict_report(cur_res, self.scenario.limits.bandwidth_hz,
                                             self.scenario.limits.delta_t_s)
        best_feasible = initial if cur_res.feasible else None
        best_feasible_res = cur_res if cur_res.feasible else None
        T = self.T0
        trace = []

        for it in range(iterations):
            d = self._pick(wd, rng)
            r = self._pick(wr, rng)
            partial = destroy_operators[d](cur, ctx, rng)
            new = repair_operators[r](partial, ctx, rng)
            chosen = [d, r]
            if rng.rand() < 0.5:
                l = self._pick(wl, rng)
                new = local_operators[l](new, ctx, rng)
                chosen.append(l)
            new_res = cache.evaluate_after_move(new, frozenset(range(self.scenario.n_uav)))

            # feasibility-first acceptance via lexicographic rank (spec §7.7):
            # a feasibility-level improvement is always accepted; at the same
            # level (both feasible), SA on the objective.  Infeasible->infeasible
            # at the same level is rejected (no finite objective to compare).
            cur_rank = feasibility_rank(cur_res)[:5]
            new_rank = feasibility_rank(new_res)[:5]
            better = new_rank < cur_rank
            same_level = new_rank == cur_rank
            if better:
                accept = True
            elif same_level and cur_res.feasible:
                dj = new_res.objective - cur_res.objective
                better = dj < 0.0
                accept = better or rng.rand() < np.exp(-dj / max(T, 1e-9))
            else:
                accept = False
            score = 0.0
            if accept:
                is_new_best = (new_res.feasible and
                               (best_feasible_res is None or new_res.objective < best_feasible_res.objective))
                if is_new_best:
                    best_feasible, best_feasible_res = new, new_res
                    score = self.score_new_best
                elif better:
                    score = self.score_improve
                else:
                    score = self.score_accept
                cur, cur_res = new, new_res
                ctx.conflict = build_conflict_report(new_res, self.scenario.limits.bandwidth_hz,
                                                     self.scenario.limits.delta_t_s)
            for op in chosen:
                scores[op] += score
                uses[op] += 1
            T *= self.cooling

            if (it + 1) % self.segment == 0:
                for op in scores:
                    wd.get(op)  # no-op to keep names consistent
                    target = wd if op in wd else (wr if op in wr else wl)
                    target[op] = (1 - self.rho) * target[op] + self.rho * scores[op] / max(1, uses[op])
                scores = {n: 0.0 for n in scores}
                uses = {n: 0 for n in uses}
            trace.append((d, r, float(cur_res.objective)))

        best = best_feasible if best_feasible is not None else cur
        best_res = best_feasible_res if best_feasible_res is not None else cur_res
        return SearchResult(best_solution=best, best_result=best_res,
                            search_trace=tuple(trace),
                            evaluator_calls={"full": cache.full_calls,
                                             "incremental": cache.incremental_calls})
