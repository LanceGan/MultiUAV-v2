"""Adaptive conflict-aware STC-ALNS solver (Task 6 Step 5).

Roulette operator selection with adaptive weights (spec §7.6), feasibility-first
simulated-annealing acceptance (spec §7.7), a separately-maintained
best-feasible solution, real synchronous conflict feedback, honest incremental
evaluation, and a checkpoint/resume path that reproduces an uninterrupted run
bit-for-bit.  Deterministic under a fixed ``method_seed``.
"""

from dataclasses import dataclass

import numpy as np

from segment.stc.conflicts import build_conflict_report
from segment.stc.evaluator import (
    default_objective_spec,
    evaluate_solution,
    feasibility_rank,
)
from segment.stc.incremental import EvaluationCache, solution_diff
from segment.stc.stc_operators import (
    SearchContext,
    destroy_operators,
    local_operators,
    repair_operators,
)
from segment.stc.types import SearchResult


@dataclass
class SolverCheckpoint:
    """Complete in-memory solver state for exact resume (Task 6 Step 5)."""
    iterations_done: int
    rng_state: tuple
    cur: object
    cur_res: object
    best_feasible: object
    best_feasible_res: object
    wd: dict
    wr: dict
    wl: dict
    scores: dict
    uses: dict
    T: float
    trace: list
    cache: object
    objective_spec: object
    conflict: object
    marginal_full_calls: int


class STCALNS:
    def __init__(self, scenario, library, field, rho=0.1, T0=0.1, cooling=0.995,
                 segment=100, score_new_best=3.0, score_improve=1.0,
                 score_accept=0.5, score_reject=0.0, audit_interval=50,
                 operator_sets=None, disable_incremental=False):
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
        self.score_reject = score_reject
        self.audit_interval = audit_interval
        # Optional operator-set override (RQ5 ablation): a dict with any of
        # ``destroy`` / ``repair`` / ``local`` naming an operator dict.  Defaults
        # to the shipped operator sets.
        operator_sets = operator_sets or {}
        self._destroy = operator_sets.get("destroy") or destroy_operators
        self._repair = operator_sets.get("repair") or repair_operators
        self._local = operator_sets.get("local") or local_operators
        # RQ5 ablation: full replay per move instead of incremental resume.
        self.disable_incremental = disable_incremental
        self.checkpoint = None

    # ------------------------------------------------------------------ #

    def _pick(self, weights, rng):
        names = list(weights)
        w = np.array([weights[n] for n in names], dtype=float)
        w = np.clip(w, 1e-9, None)
        p = w / w.sum()
        return names[int(rng.choice(len(names), p=p))]

    def _draw_q(self, rng):
        return int(rng.randint(1, max(2, self.scenario.n_tasks // 2) + 1))

    def _fresh_state(self, initial, method_seed):
        rng = np.random.RandomState(method_seed)
        objective_spec = default_objective_spec(self.scenario, self.library)
        ctx = SearchContext(scenario=self.scenario, library=self.library,
                            field=self.field, objective_spec=objective_spec)
        cache = EvaluationCache.from_full(
            self.scenario, self.library, self.field, initial, objective_spec,
            audit_interval=self.audit_interval)
        cur = initial
        cur_res = cache.base_result
        ctx.cache = cache
        ctx.result = cur_res
        ctx.conflict = build_conflict_report(
            cur_res, self.scenario.limits.bandwidth_hz,
            self.scenario.limits.delta_t_s)
        wd = {n: 1.0 for n in self._destroy}
        wr = {n: 1.0 for n in self._repair}
        wl = {n: 1.0 for n in self._local}
        names = list(self._destroy) + list(self._repair) + list(self._local)
        scores = {n: 0.0 for n in names}
        uses = {n: 0 for n in names}
        best_feasible = initial if cur_res.feasible else None
        best_feasible_res = cur_res if cur_res.feasible else None
        return dict(rng=rng, ctx=ctx, cache=cache, objective_spec=objective_spec,
                    cur=cur, cur_res=cur_res, wd=wd, wr=wr, wl=wl, scores=scores,
                    uses=uses, T=self.T0, trace=[], best_feasible=best_feasible,
                    best_feasible_res=best_feasible_res, iterations_done=0)

    def _restore_state(self, cp):
        rng = np.random.RandomState()
        rng.set_state(cp.rng_state)
        ctx = SearchContext(scenario=self.scenario, library=self.library,
                            field=self.field, objective_spec=cp.objective_spec)
        ctx.cache = cp.cache
        ctx.result = cp.cur_res
        ctx.conflict = cp.conflict
        ctx.marginal_full_calls = cp.marginal_full_calls
        return dict(rng=rng, ctx=ctx, cache=cp.cache,
                    objective_spec=cp.objective_spec, cur=cp.cur,
                    cur_res=cp.cur_res, wd=dict(cp.wd), wr=dict(cp.wr),
                    wl=dict(cp.wl), scores=dict(cp.scores), uses=dict(cp.uses),
                    T=cp.T, trace=list(cp.trace), best_feasible=cp.best_feasible,
                    best_feasible_res=cp.best_feasible_res,
                    iterations_done=cp.iterations_done)

    def _checkpoint(self, st):
        return SolverCheckpoint(
            iterations_done=st["iterations_done"], rng_state=st["rng"].get_state(),
            cur=st["cur"], cur_res=st["cur_res"],
            best_feasible=st["best_feasible"], best_feasible_res=st["best_feasible_res"],
            wd=dict(st["wd"]), wr=dict(st["wr"]), wl=dict(st["wl"]),
            scores=dict(st["scores"]), uses=dict(st["uses"]), T=st["T"],
            trace=list(st["trace"]), cache=st["cache"],
            objective_spec=st["objective_spec"], conflict=st["ctx"].conflict,
            marginal_full_calls=st["ctx"].marginal_full_calls)

    # ------------------------------------------------------------------ #

    def solve(self, initial, iterations=1000, method_seed=42, checkpoint=None):
        if checkpoint is not None:
            st = self._restore_state(checkpoint)
        else:
            st = self._fresh_state(initial, method_seed)

        rng = st["rng"]
        ctx = st["ctx"]
        cache = st["cache"]
        cur = st["cur"]
        cur_res = st["cur_res"]
        wd, wr, wl = st["wd"], st["wr"], st["wl"]
        scores, uses = st["scores"], st["uses"]
        T = st["T"]
        trace = st["trace"]
        best_feasible = st["best_feasible"]
        best_feasible_res = st["best_feasible_res"]

        bandwidth = self.scenario.limits.bandwidth_hz
        delta_t = self.scenario.limits.delta_t_s
        n_uav = self.scenario.n_uav

        for _ in range(iterations):
            d = self._pick(wd, rng)
            r = self._pick(wr, rng)
            q = self._draw_q(rng)
            partial = self._destroy[d](cur, ctx, rng, q)
            new = self._repair[r](partial, ctx, rng)
            chosen = [d, r]
            local_op = None
            if rng.rand() < 0.5:
                local_op = self._pick(wl, rng)
                new = self._local[local_op](new, ctx, rng)
                chosen.append(local_op)

            if self.disable_incremental:
                new_res = evaluate_solution(
                    self.scenario, self.library, new, self.field,
                    st["objective_spec"])
                cache.full_calls += 1
            else:
                affected, per_uav_earliest = solution_diff(
                    cur, new, self.scenario.n_tasks)
                earliest = (min(per_uav_earliest.values())
                            if per_uav_earliest else 0)
                new_res = cache.evaluate_after_move(new, affected, earliest)

            # feasibility-first acceptance via lexicographic rank (spec §7.7)
            cur_rank = feasibility_rank(cur_res)[:5]
            new_rank = feasibility_rank(new_res)[:5]
            rank_improved = new_rank < cur_rank
            same_level = new_rank == cur_rank
            if rank_improved:
                accept = True
            elif same_level and cur_res.feasible:
                dj = new_res.objective - cur_res.objective
                improved = dj < 0.0
                accept = improved or rng.rand() < np.exp(-dj / max(T, 1e-9))
            else:
                accept = False
                improved = False

            new_conflict = build_conflict_report(new_res, bandwidth, delta_t)

            score = self.score_reject
            if accept:
                is_new_best = (new_res.feasible and
                               (best_feasible_res is None or
                                new_res.objective < best_feasible_res.objective))
                if is_new_best:
                    best_feasible, best_feasible_res = new, new_res
                    score = self.score_new_best
                elif rank_improved or improved:
                    score = self.score_improve
                else:
                    score = self.score_accept
                cur, cur_res = new, new_res
                ctx.result = cur_res
                ctx.conflict = new_conflict
                cache.commit(new, new_res)

            for op in chosen:
                scores[op] += score
                uses[op] += 1
            T *= self.cooling

            if len(trace) % self.segment == 0:
                for op in list(scores):
                    for table in (wd, wr, wl):
                        if op in table:
                            table[op] = ((1 - self.rho) * table[op]
                                         + self.rho * scores[op] / max(1, uses[op]))
                scores = {n: 0.0 for n in scores}
                uses = {n: 0 for n in uses}

            trace.append({
                "destroy": d, "repair": r, "local": local_op,
                "accepted": accept,
                "feasible": new_res.feasible,
                "feasibility_rank": feasibility_rank(new_res),
                "objective": new_res.objective,
                "conflict_loss": new_conflict.lost_bits,
                "weights": (dict(wd), dict(wr), dict(wl)),
                "full_calls": cache.full_calls,
                "incremental_calls": cache.incremental_calls,
            })

        st.update(rng=rng, ctx=ctx, cache=cache, cur=cur, cur_res=cur_res,
                  wd=wd, wr=wr, wl=wl, scores=scores, uses=uses, T=T, trace=trace,
                  best_feasible=best_feasible, best_feasible_res=best_feasible_res,
                  iterations_done=st["iterations_done"] + iterations)
        self.checkpoint = self._checkpoint(st)

        best = best_feasible if best_feasible is not None else cur
        best_res = best_feasible_res if best_feasible_res is not None else cur_res
        return SearchResult(best_solution=best, best_result=best_res,
                            search_trace=tuple(trace),
                            evaluator_calls={"full": cache.full_calls,
                                             "incremental": cache.incremental_calls,
                                             "recomputed_slots": cache.recomputed_slots,
                                             "audits": cache.audits,
                                             "marginal_full_calls": ctx.marginal_full_calls})
