"""Task 6 Step 5: STC-ALNS traces conflict feedback, adaptive weights, and
checkpoint/resume reproduces an uninterrupted run exactly."""

import numpy as np

from segment.stc.generators import generate_straight_hold
from segment.stc.operators import build_initial_solution
from segment.stc.solver import STCALNS
from segment.stc.types import RadioSample, STCScenario, ScenarioLimits


class _MockField:
    def __init__(self, n_sectors):
        self.n_sectors = n_sectors

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.full(self.n_sectors, 1.0),
                           g2a_outage=np.full(self.n_sectors, 0.1))


def _setup(n_tasks=6):
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


def test_solve_traces_conflict_feedback():
    sc, field, lib = _setup()
    solver = STCALNS(sc, lib, field)
    initial = build_initial_solution(sc, lib)
    res = solver.solve(initial, iterations=20, method_seed=5)
    assert len(res.search_trace) == 20
    entry = res.search_trace[0]
    for key in ("destroy", "repair", "local", "accepted", "feasible",
                "feasibility_rank", "objective", "conflict_loss", "weights",
                "full_calls", "incremental_calls"):
        assert key in entry
    # the search genuinely uses the incremental evaluator
    assert res.evaluator_calls["incremental"] > 0


def test_adaptive_weights_update():
    sc, field, lib = _setup()
    solver = STCALNS(sc, lib, field, segment=5, rho=0.5)
    initial = build_initial_solution(sc, lib)
    solver.solve(initial, iterations=25, method_seed=3)
    cp = solver.checkpoint
    assert cp is not None
    all_weights = list(cp.wd.values()) + list(cp.wr.values()) + list(cp.wl.values())
    assert any(abs(w - 1.0) > 1e-12 for w in all_weights)


def test_checkpoint_resume_matches_uninterrupted():
    sc, field, lib = _setup()
    initial = build_initial_solution(sc, lib)

    s_full = STCALNS(sc, lib, field)
    res_full = s_full.solve(initial, iterations=150, method_seed=7)

    s_part = STCALNS(sc, lib, field)
    s_part.solve(initial, iterations=80, method_seed=7)
    cp = s_part.checkpoint
    res_part = s_part.solve(initial, iterations=70, method_seed=7, checkpoint=cp)

    assert res_part.best_solution == res_full.best_solution
    assert res_part.best_result.objective == res_full.best_result.objective
    assert res_part.search_trace == res_full.search_trace
    assert res_part.evaluator_calls == res_full.evaluator_calls
