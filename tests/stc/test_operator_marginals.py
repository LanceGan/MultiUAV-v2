"""Task 6 Step 3: real synchronous marginal evaluation in conflict-aware repair.

The fixture below is a genuine ranking reversal: inserting the missing task
into UAV 0 (option A) is cheaper by the *independent* single-UAV edge cost, but
inserting it into UAV 1 (option B) is better under the *synchronous*
shared-bandwidth evaluation (A makes UAV 0's upload collide with UAV 1 in the
same sector).  ``conflict_aware_insertion`` must choose B by evaluating real
candidate deltas through ``evaluate_solution``; a static
``1 + 0.1 * per_uav_loss`` multiplier would choose A and fail.
"""

import numpy as np

from segment.stc.evaluator import evaluate_solution
from segment.stc.slsqp_refine import candidate_cost
from segment.stc.stc_operators import SearchContext, conflict_aware_insertion
from segment.stc.types import (
    ObjectiveSpec,
    PlanSolution,
    RadioSample,
    ReferenceValues,
    STCScenario,
    ScenarioLimits,
    SegmentCandidate,
)


class _Field:
    def __init__(self, n_sectors):
        self.n_sectors = n_sectors

    def sample(self, pos):
        # uniform: every slot serves sector 0, so any two simultaneous active
        # uploaders share bandwidth
        return RadioSample(a2g_efficiency=np.full(self.n_sectors, 1.0),
                           g2a_outage=np.zeros(self.n_sectors))


def _spec():
    return ObjectiveSpec(
        reference=ReferenceValues(t_ref_s=1.0, d_ref_s=1.0, source_hash="hand"),
        w_t=0.5, w_d=0.5)


def _cand(cid, src, tgt, steps, data, n_sectors=3, rate=1e6):
    return SegmentCandidate(
        candidate_id=cid, source=src, target=tgt, generator="hand",
        positions_m=np.zeros((steps, 2)),
        full_rate_bps=np.full((steps, n_sectors), rate),
        sector_ids=np.zeros(steps, dtype=int),
        g2a_outage=np.zeros((steps, n_sectors)),
        flight_steps=steps, hold_position_m=(0.0, 0.0),
        source_data_bits=float(data), rankable=True)


def _fixture():
    limits = ScenarioLimits(delta_t_s=1.0, v_max_mps=10.0, uav_height_m=100.0,
                            arrive_radius_m=5.0, max_wait_s=10.0,
                            mission_deadline_s=1000.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    sc = STCScenario(n_uav=2, n_tasks=3,
                     task_positions_m=((10.0, 0.0), (50.0, 0.0), (20.0, 0.0)),
                     task_data_bits=(2e6, 4e6, 2e6), depot_m=(0.0, 0.0),
                     terminal_m=(60.0, 0.0), limits=limits, length_m=100.0,
                     width_m=100.0, n_bs=1, task_seed=42, map_seed=0)
    field = _Field(sc.n_sectors)
    data = {0: 2e6, 1: 4e6, 2: 2e6}
    depot_steps = {(-1, 0): 3, (-1, 1): 5, (-1, 2): 3}
    steps = {(0, 1): 5, (1, 0): 5, (0, 2): 3, (2, 0): 3,
             (1, 2): 5, (2, 1): 5, (0, 3): 5, (1, 3): 3, (2, 3): 3}
    lib = {}
    for (s, t), st in depot_steps.items():
        lib[(s, t)] = [_cand("d%d_%d" % (s, t), s, t, st, 0.0)]
    for (s, t), st in steps.items():
        lib[(s, t)] = [_cand("e%d_%d" % (s, t), s, t, st, data[s])]
    return sc, field, lib


def _plan(routes, lib, sc):
    cids = []
    for r in routes:
        edges = ([(-1, r[0])] + [(r[i], r[i + 1]) for i in range(len(r) - 1)]
                 + [(r[-1], sc.n_tasks)]) if r else []
        cids.append(tuple(lib[(s, t)][0].candidate_id for s, t in edges))
    return PlanSolution(routes=tuple(tuple(r) for r in routes),
                        candidate_ids=tuple(cids))


def _independent_cost(sol, lib, sc):
    total = 0.0
    for r, cids in zip(sol.routes, sol.candidate_ids):
        edges = ([(-1, r[0])] + [(r[i], r[i + 1]) for i in range(len(r) - 1)]
                 + [(r[-1], sc.n_tasks)]) if r else []
        for (s, t), cid in zip(edges, cids):
            c = next(c for c in lib[(s, t)] if c.candidate_id == cid)
            total += candidate_cost(c, sc.limits.delta_t_s)
    return total


def test_independent_prefers_a_but_synchronous_prefers_b():
    sc, field, lib = _fixture()
    A = _plan([(0, 2), (1,)], lib, sc)
    B = _plan([(0,), (1, 2)], lib, sc)
    # independent edge cost prefers A
    assert _independent_cost(A, lib, sc) < _independent_cost(B, lib, sc)
    # synchronous shared-bandwidth evaluation prefers B
    rA = evaluate_solution(sc, lib, A, field, _spec())
    rB = evaluate_solution(sc, lib, B, field, _spec())
    assert rA.objective > rB.objective


def test_conflict_aware_insertion_chooses_b():
    sc, field, lib = _fixture()
    destroyed = _plan([(0,), (1,)], lib, sc)
    ctx = SearchContext(scenario=sc, library=lib, field=field, objective_spec=_spec())
    rng = np.random.RandomState(0)
    repaired = conflict_aware_insertion(destroyed, ctx, rng)
    # the missing task (2) must land in UAV 1 after task 1 (option B), not UAV 0
    assert repaired.routes == ((0,), (1, 2))


def test_conflict_aware_insertion_counts_internal_evals():
    sc, field, lib = _fixture()
    destroyed = _plan([(0,), (1,)], lib, sc)
    ctx = SearchContext(scenario=sc, library=lib, field=field, objective_spec=_spec())
    rng = np.random.RandomState(0)
    conflict_aware_insertion(destroyed, ctx, rng)
    # the O(U·K) full evaluations inside the operator are counted (visible to
    # SearchResult.evaluator_calls via the solver), not silently bypassed
    assert ctx.marginal_full_calls > 0
