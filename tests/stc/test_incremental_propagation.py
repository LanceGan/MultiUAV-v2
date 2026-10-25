"""Task 6 Step 4: exact affected-window incremental evaluation.

A tail move (reordering only the later edges of one UAV) changes the earliest
affected edge *after* slot zero, so the incremental evaluator must resume from
that later slot, recompute fewer slots than a full replay, and still equal the
full replay exactly.  A periodic full-replay audit must detect divergence.
"""

import dataclasses

import numpy as np
import pytest

from segment.stc.evaluator import evaluate_solution as real_evaluate_solution
from segment.stc.generators import generate_straight_hold
from segment.stc.incremental import (
    EvaluationCache,
    IncrementalDivergenceError,
    solution_diff,
)
from segment.stc.operators import _edges, _pick_candidate
from segment.stc.types import (
    ObjectiveSpec,
    PlanSolution,
    RadioSample,
    ReferenceValues,
    STCScenario,
    ScenarioLimits,
)


class _MockField:
    def __init__(self, n_sectors):
        self.n_sectors = n_sectors

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.full(self.n_sectors, 1.0),
                           g2a_outage=np.full(self.n_sectors, 0.1))


def _spec():
    return ObjectiveSpec(
        reference=ReferenceValues(t_ref_s=1.0, d_ref_s=1.0, source_hash="hand"),
        w_t=0.5, w_d=0.5)


def _setup(n_tasks=4):
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


def _sol(sc, lib, routes):
    cids = tuple(tuple(_pick_candidate(lib, s, t, sc.limits.delta_t_s)
                       for s, t in _edges(r, sc.n_tasks)) for r in routes)
    return PlanSolution(routes=tuple(tuple(r) for r in routes),
                        candidate_ids=cids)


def test_tail_move_resumes_late_and_equals_full_replay():
    sc, field, lib = _setup(4)
    base = _sol(sc, lib, [(0, 1, 2), (3,)])
    moved = _sol(sc, lib, [(0, 2, 1), (3,)])   # only UAV 0's tail reordered

    affected, per_uav = solution_diff(base, moved, sc.n_tasks)
    assert affected == frozenset({0})
    assert per_uav[0] == 1                      # change starts after the depot edge

    cache = EvaluationCache.from_full(sc, lib, field, base, _spec())
    inc = cache.evaluate_after_move(moved, affected, per_uav[0])
    full = real_evaluate_solution(sc, lib, moved, field, _spec())

    assert inc.makespan_s == pytest.approx(full.makespan_s)
    assert inc.total_delay_s == pytest.approx(full.total_delay_s)
    assert inc.feasible == full.feasible
    assert inc.objective == pytest.approx(full.objective)


def test_incremental_counts_slots_honestly():
    sc, field, lib = _setup(4)
    base = _sol(sc, lib, [(0, 1, 2), (3,)])
    moved = _sol(sc, lib, [(0, 2, 1), (3,)])

    cache = EvaluationCache.from_full(sc, lib, field, base, _spec())
    assert cache.full_calls == 1
    cache.evaluate_after_move(moved, frozenset({0}), 1)
    assert cache.full_calls == 1                # incremental does not full-replay
    assert cache.incremental_calls == 1
    # fewer slots recomputed than a full replay
    assert 0 < cache.recomputed_slots < cache.max_slots


def test_periodic_full_replay_audit_detects_divergence(monkeypatch):
    sc, field, lib = _setup(4)
    base = _sol(sc, lib, [(0, 1, 2), (3,)])
    moved = _sol(sc, lib, [(0, 2, 1), (3,)])

    cache = EvaluationCache.from_full(sc, lib, field, base, _spec(), audit_interval=1)

    import segment.stc.incremental as inc_module

    def wrong(sc_, lib_, sol_, field_, spec_, keep_timelines=True):
        r = real_evaluate_solution(sc_, lib_, sol_, field_, spec_, keep_timelines)
        return dataclasses.replace(r, makespan_s=r.makespan_s + 100.0)

    monkeypatch.setattr(inc_module, "evaluate_solution", wrong)
    with pytest.raises(IncrementalDivergenceError):
        cache.evaluate_after_move(moved, frozenset({0}), 1)
