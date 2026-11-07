"""Focused coverage for the sealed exact-benchmark runner."""

import numpy as np
import pytest

from experiment import run_stc_exact
from segment.stc.types import (
    PlanSolution,
    RadioSample,
    STCScenario,
    ScenarioLimits,
)


class _MockField:
    def __init__(self, n_sectors=6):
        self.n_sectors = n_sectors

    def sample(self, pos):
        return RadioSample(
            a2g_efficiency=np.ones(self.n_sectors),
            g2a_outage=np.full(self.n_sectors, 0.1),
        )


def _scenario(n_tasks=6):
    limits = ScenarioLimits(
        delta_t_s=0.5,
        v_max_mps=50.0,
        uav_height_m=100.0,
        arrive_radius_m=15.0,
        max_wait_s=60.0,
        mission_deadline_s=7200.0,
        bandwidth_hz=1e6,
        uplink_power_w=0.2,
        eps_out=0.35,
        w_t=0.5,
        w_d=0.5,
    )
    return STCScenario(
        n_uav=2,
        n_tasks=n_tasks,
        task_positions_m=tuple(
            (500.0 + 300.0 * i, 900.0) for i in range(n_tasks)),
        task_data_bits=tuple(1e6 for _ in range(n_tasks)),
        depot_m=(0.0, 0.0),
        terminal_m=(4000.0, 4000.0),
        limits=limits,
        length_m=4000.0,
        width_m=4000.0,
        n_bs=2,
        task_seed=42,
        map_seed=42,
    )


def test_exact_library_has_one_straight_hold_per_legal_edge():
    scenario = _scenario(3)
    library = run_stc_exact._build_exact_library(scenario, _MockField())

    assert len(library) == 3 * (3 + 1)
    assert all(len(candidates) == 1 for candidates in library.values())
    assert all(candidates[0].generator == "straight_hold"
               for candidates in library.values())


def test_run_exact_scenario_rejects_task_count_beyond_base():
    with pytest.raises(ValueError, match="task_count"):
        run_stc_exact.run_exact_scenario(
            _scenario(4),
            5,
            spacing_m=1000.0,
            time_limit_s=1.0,
            max_leaves=100,
            iterations=1,
        )


def test_solution_payload_is_json_ready():
    solution = PlanSolution(
        routes=((0, 2), (1,)),
        candidate_ids=(("a", "b", "c"), ("d", "e")),
    )

    assert run_stc_exact._solution_payload(solution) == {
        "routes": [[0, 2], [1]],
        "candidate_ids": [["a", "b", "c"], ["d", "e"]],
    }


def test_smoke_main_checkpoints_exact_and_proposed(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        run_stc_exact, "load_stc_scenario",
        lambda *args, **kwargs: _scenario(6))

    def fake_exact(base, task_count, **kwargs):
        captured["task_count"] = task_count
        return (
            {"method": "exact_discrete_enumeration", "n_tasks": task_count,
             "proven_optimal": True, "discrete_search_complete": True,
             "timed_out": False},
            {"method": "proposed_stc_alns", "n_tasks": task_count},
        )

    monkeypatch.setattr(run_stc_exact, "run_exact_scenario", fake_exact)

    def fake_lifecycle(runner_name, cfg, run_id, matrix, per_scenario,
                       summary_fn, **kwargs):
        records = []
        per_scenario(*matrix[0], checkpoint=records.append)
        captured.update({
            "runner_name": runner_name,
            "cfg": cfg,
            "run_id": run_id,
            "records": records,
            "summary": summary_fn(records),
        })
        return 0

    monkeypatch.setattr(run_stc_exact, "run_lifecycle", fake_lifecycle)

    rc = run_stc_exact.main([
        "--smoke",
        "--run-id",
        "unit-smoke",
        "--iterations",
        "20",
        "--time-limit-s",
        "30",
    ])

    assert rc == 0
    assert captured["runner_name"] == "stc_exact"
    assert captured["run_id"] == "unit-smoke"
    assert captured["task_count"] == 2
    assert [record["method"] for record in captured["records"]] == [
        "exact_discrete_enumeration", "proposed_stc_alns"]
    assert captured["cfg"]["iterations"] == 20
    assert captured["summary"]["optimality_scope"] == (
        run_stc_exact.OPTIMALITY_SCOPE)
