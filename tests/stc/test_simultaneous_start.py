"""Simultaneous-takeoff contract (Task 1).

Every non-idle UAV becomes airborne at global slot zero; there is no per-UAV
start offset and no departure dimension anywhere in the model.  An empty UAV
route is an idle (non-airborne) UAV: no outage, no bandwidth, complete at t=0.
"""

import inspect

import numpy as np

from segment.stc.evaluator import evaluate_solution
from segment.stc.generators import generate_straight_hold
from segment.stc.stc_operators import local_operators
from segment.stc.types import (
    ObjectiveSpec,
    PlanSolution,
    RadioSample,
    ReferenceValues,
    STCScenario,
    ScenarioLimits,
)


def _spec():
    return ObjectiveSpec(
        reference=ReferenceValues(t_ref_s=1.0, d_ref_s=1.0, source_hash="hand"),
        w_t=0.5, w_d=0.5)


class _MockField:
    def __init__(self, n_sectors):
        self.n_sectors = n_sectors

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.full(self.n_sectors, 1.0),
                           g2a_outage=np.zeros(self.n_sectors))


def _scenario(n_uav=2, n_tasks=2):
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    positions = tuple((1000.0 + i * 1000.0, 2000.0) for i in range(n_tasks))
    return STCScenario(n_uav=n_uav, n_tasks=n_tasks, task_positions_m=positions,
                       task_data_bits=tuple([1e6] * n_tasks), depot_m=(0.0, 2000.0),
                       terminal_m=(4000.0, 2000.0), limits=limits, length_m=4000.0,
                       width_m=4000.0, n_bs=4, task_seed=42, map_seed=0)


def _library(sc):
    field = _MockField(sc.n_sectors)
    lib = {}
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src != tgt:
                lib[(src, tgt)] = [generate_straight_hold(sc, field, src, tgt)]
    return field, lib


def test_plan_solution_has_no_departure_field():
    from dataclasses import fields
    sol = PlanSolution(routes=((0,), (1,)),
                       candidate_ids=(("straight_hold_-1_0", "straight_hold_0_2"),
                                      ("straight_hold_-1_1", "straight_hold_1_2")))
    assert [f.name for f in fields(PlanSolution)] == ["routes", "candidate_ids"]
    assert not hasattr(sol, "departure_steps")


def test_every_nonempty_uav_starts_at_global_slot_zero():
    sc = _scenario(2, 2)
    field, lib = _library(sc)
    sol = PlanSolution(routes=((0,), (1,)),
                       candidate_ids=(("straight_hold_-1_0", "straight_hold_0_2"),
                                      ("straight_hold_-1_1", "straight_hold_1_2")))
    res = evaluate_solution(sc, lib, sol, field, _spec())
    for tl in res.timelines:
        assert len(tl.slots) > 0
        assert tl.slots[0].time_start_s == 0.0
        assert tl.slots[0].airborne is True


def test_idle_uav_does_not_consume_bandwidth():
    sc = _scenario(2, 2)
    field, lib = _library(sc)
    # UAV 1 has an empty route: idle (non-airborne), complete at t=0.
    sol = PlanSolution(routes=((0, 1), ()),
                       candidate_ids=(("straight_hold_-1_0", "straight_hold_0_1",
                                       "straight_hold_1_2"),
                                      ()))
    res = evaluate_solution(sc, lib, sol, field, _spec())
    assert res.timelines[1].slots == ()            # idle: never airborne
    assert res.outage_ratio[1] == 0.0              # no airborne outage accrued
    # the lone active UAV gets the full bandwidth (idle UAV splits nothing)
    active = [s for s in res.timelines[0].slots if s.active_uploader]
    assert active
    assert all(s.allocated_bandwidth_hz == sc.limits.bandwidth_hz for s in active)


def test_exact_domain_contains_no_departure_dimension():
    from segment.stc.exact import solve_exact_tiny
    params = inspect.signature(solve_exact_tiny).parameters
    assert "departure_set" not in params
    assert "departure" not in " ".join(params)


def test_operator_registry_contains_no_departure_or_time_shift():
    assert "departure_shift" not in local_operators
    assert "time_shift" not in local_operators
