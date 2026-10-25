"""Regression tests for the final-review fix pass."""

import numpy as np

from segment.stc.generators import generate_straight_hold
from segment.stc.operators import build_initial_solution, _pick_candidate
from segment.stc.solver import STCALNS
from segment.stc.types import PlanSolution, RadioSample, STCScenario, ScenarioLimits


class _MockField:
    def __init__(self, n_sectors):
        self.n_sectors = n_sectors

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.full(self.n_sectors, 1.0),
                           g2a_outage=np.full(self.n_sectors, 0.1))


def _scenario(n_uav=2, n_tasks=6):
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, departure_step_s=1.0,
                            departure_max_s=60.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    return STCScenario(n_uav=n_uav, n_tasks=n_tasks,
                       task_positions_m=tuple((800.0 + i * 600.0, 2000.0) for i in range(n_tasks)),
                       task_data_bits=tuple([1e6] * n_tasks), depot_m=(0.0, 2000.0),
                       terminal_m=(4000.0, 2000.0), limits=limits, length_m=4000.0,
                       width_m=4000.0, n_bs=4, task_seed=42, map_seed=0)


def _lib(sc):
    field = _MockField(sc.n_sectors)
    lib = {}
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src != tgt:
                lib[(src, tgt)] = [generate_straight_hold(sc, field, src, tgt)]
    return field, lib


def test_solver_handles_infeasible_initial():
    sc = _scenario(2, 6)
    field, lib = _lib(sc)
    # infeasible initial: only task 0 served (tasks 1-5 unserved)
    routes = ((0,), ())
    cids = ((_pick_candidate(lib, -1, 0), _pick_candidate(lib, 0, 6)),
            (_pick_candidate(lib, -1, 6),))
    initial = PlanSolution(routes=routes, candidate_ids=cids, departure_steps=(0, 0))
    solver = STCALNS(sc, lib, field)
    res = solver.solve(initial, iterations=30, method_seed=42)
    assert res.best_result is not None
    assert len(res.search_trace) == 30
    assert not np.isnan(res.best_result.objective)


def test_initial_solution_handles_few_tasks():
    sc = _scenario(n_uav=3, n_tasks=2)  # fewer tasks than UAVs
    _, lib = _lib(sc)
    sol = build_initial_solution(sc, lib)
    assert len(sol.routes) == 3  # exactly n_uav routes (some empty)
    assert len(sol.departure_steps) == 3
