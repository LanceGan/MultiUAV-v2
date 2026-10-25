"""Task 10: resource-unaware baseline searches return a SearchResult."""

import numpy as np

from segment.stc.generators import generate_straight_hold
from segment.stc.operators import (
    run_cheapest_insertion,
    run_lns_sa,
    run_nearest_neighbor_2opt,
)
from segment.stc.types import RadioSample, STCScenario, ScenarioLimits


class _MockField:
    def __init__(self, n_sectors):
        self.n_sectors = n_sectors

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.full(self.n_sectors, 1.0),
                           g2a_outage=np.full(self.n_sectors, 0.1))


def _setup():
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    sc = STCScenario(n_uav=2, n_tasks=4,
                     task_positions_m=tuple((1000.0 + i * 800.0, 2000.0) for i in range(4)),
                     task_data_bits=tuple([1e6] * 4), depot_m=(0.0, 2000.0),
                     terminal_m=(4000.0, 2000.0), limits=limits, length_m=4000.0,
                     width_m=4000.0, n_bs=4, task_seed=42, map_seed=0)
    field = _MockField(sc.n_sectors)
    lib = {}
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src != tgt:
                lib[(src, tgt)] = [generate_straight_hold(sc, field, src, tgt)]
    return sc, field, lib


def test_baselines_return_search_result():
    sc, field, lib = _setup()
    for fn in (run_nearest_neighbor_2opt, run_cheapest_insertion, run_lns_sa):
        res = fn(sc, lib, field, seed=42)
        assert res.best_solution is not None
        assert res.best_result is not None
        assert res.best_result.makespan_s > 0.0
