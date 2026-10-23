"""Task 6: synchronous shared-bandwidth evaluator."""

import numpy as np

from segment.stc.evaluator import evaluate_solution, select_serving_sector
from segment.stc.generators import generate_straight_hold
from segment.stc.types import PlanSolution, RadioSample, STCScenario, ScenarioLimits


def _sample(eff, outage):
    return RadioSample(a2g_efficiency=np.array(eff, dtype=float),
                       g2a_outage=np.array(outage, dtype=float))


def test_select_serving_sector_prefers_reliable_high_rate():
    s = _sample([1.0, 2.0, 0.5], [0.1, 0.4, 0.3])
    # sector 1 has the highest rate but outage 0.4 > 0.35 (unreliable)
    assert select_serving_sector(s, True, 0.35) == 0


def test_select_serving_sector_falls_back_to_min_outage():
    s = _sample([1.0, 2.0, 0.5], [0.5, 0.6, 0.7])
    assert select_serving_sector(s, True, 0.35) == 0


def test_select_serving_sector_no_payload_ignores_rate():
    s = _sample([0.0, 3.0, 1.0], [0.5, 0.6, 0.7])
    assert select_serving_sector(s, False, 0.35) == 0


class _MockField:
    def __init__(self, n_sectors, eff=1.0, outage=0.1):
        self.n_sectors = n_sectors
        self._eff = eff
        self._outage = outage

    def sample(self, pos):
        return _sample([self._eff] * self.n_sectors, [self._outage] * self.n_sectors)


def _scenario(n_tasks=2):
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, departure_step_s=1.0,
                            departure_max_s=60.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    positions = tuple((1000.0 + i * 1000.0, 2000.0) for i in range(n_tasks))
    return STCScenario(n_uav=2, n_tasks=n_tasks, task_positions_m=positions,
                       task_data_bits=tuple([1e6] * n_tasks), depot_m=(0.0, 2000.0),
                       terminal_m=(4000.0, 2000.0), limits=limits, length_m=4000.0,
                       width_m=4000.0, n_bs=4, task_seed=42, map_seed=0)


def test_evaluate_solution_returns_feasible_result():
    sc = _scenario(2)
    field = _MockField(sc.n_sectors, eff=1.0, outage=0.1)
    lib = {}
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src != tgt:
                lib[(src, tgt)] = [generate_straight_hold(sc, field, src, tgt)]
    sol = PlanSolution(routes=((0,), (1,)),
                       candidate_ids=(("straight_hold_-1_0", "straight_hold_0_2"),
                                      ("straight_hold_-1_1", "straight_hold_1_2")),
                       departure_steps=(0, 0))
    res = evaluate_solution(sc, lib, sol, field)
    assert res.feasible is True
    assert res.makespan_s > 0.0
    assert res.total_delay_s > 0.0
