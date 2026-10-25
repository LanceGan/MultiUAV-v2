"""Task 7: G2A airborne-outage accounting."""

import numpy as np

from segment.stc.evaluator import evaluate_solution
from segment.stc.generators import generate_straight_hold
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
                           g2a_outage=np.full(self.n_sectors, 0.1))


def _scenario():
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    return STCScenario(n_uav=2, n_tasks=2,
                       task_positions_m=((1000.0, 2000.0), (3000.0, 2000.0)),
                       task_data_bits=(1e6, 1e6), depot_m=(0.0, 2000.0),
                       terminal_m=(4000.0, 2000.0), limits=limits, length_m=4000.0,
                       width_m=4000.0, n_bs=4, task_seed=42, map_seed=0)


def _setup():
    sc = _scenario()
    field = _MockField(sc.n_sectors)
    lib = {}
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src != tgt:
                lib[(src, tgt)] = [generate_straight_hold(sc, field, src, tgt)]
    return sc, field, lib


def test_outage_ratio_normalized_by_airborne_time():
    sc, field, lib = _setup()
    base = dict(routes=((0,), (1,)),
                candidate_ids=(("straight_hold_-1_0", "straight_hold_0_2"),
                               ("straight_hold_-1_1", "straight_hold_1_2")))
    res = evaluate_solution(sc, lib, PlanSolution(**base), field, _spec())
    # constant 0.1 outage per airborne slot -> ratio exactly 0.1 per UAV
    np.testing.assert_allclose(res.outage_ratio, [0.1, 0.1])
