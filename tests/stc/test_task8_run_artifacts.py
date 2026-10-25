"""Task 8: effective config, full seed matrix, and sealed replay records."""

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiment import run_stc_main as main
from segment.stc.types import RadioSample, ScenarioLimits, STCScenario


def strict_load(path):
    def reject(token):
        raise AssertionError("non-finite JSON token: " + token)
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    limits = ScenarioLimits(0.5, 50., 100., 15., 5., 100., 1e6, .2, .35, .5, .5)
    scenario = STCScenario(2, 2, ((25., 0.), (50., 0.)), (1e6, 1e6),
                           (0., 0.), (100., 0.), limits, 100., 100., 1, 3, 7)
    monkeypatch.setattr(main, "load_stc_scenario",
                        lambda u, task_seed, map_seed: replace(
                            scenario, n_uav=u, task_seed=task_seed, map_seed=map_seed))
    spacings = []

    class Field:
        world_hash = "test-map-world"
        n_sectors = 3
        @classmethod
        def from_scenario(cls, sc, spacing_m):
            spacings.append(spacing_m)
            return cls()
        def sample(self, pos):
            return RadioSample(np.ones(3), np.zeros(3))

    monkeypatch.setattr(main, "PerSectorRadioField", Field)
    # The legacy Jia adapter configures physical maps outside Field; isolate
    # those expensive external queries as well if a broken runner invokes it.
    from segment.stc import sequencing
    from sequence import radio_path
    monkeypatch.setattr(sequencing, "_configure_maps", lambda sc: "test-map-world")
    monkeypatch.setattr(radio_path.radio_map_A2G, "getPointDateRate", lambda loc: 10.)
    monkeypatch.setattr(radio_path.radio_map_G2A, "getPointMiniOutage",
                        lambda loc: ([.1], [10.]))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "iterations": 2, "spacing_m": 25., "smoke": True,
        "generator_config": {"straight_hold": True},
        "methods": ["region_distance_straight", "shortest_balanced_graph",
                    "gatsac_style", "proposed_stc_alns"],
        "uav_nums": [1, 2], "map_seeds": [7, 8], "task_seeds": [3, 4],
        "method_seeds": [11, 12],
    }), encoding="utf-8")
    return tmp_path, config, spacings


def test_config_drives_full_matrix_and_seals_replay_artifacts(tiny_run):
    root, config, spacings = tiny_run
    assert main.main(["--config", str(config), "--run-id", "matrix"]) == 0
    run = root / "results/runs/matrix"
    manifest = strict_load(run / "manifest.json")
    raw = strict_load(run / "raw_records.json")
    solutions = strict_load(run / "solutions.json")
    timelines = strict_load(run / "timelines.json")
    summary = strict_load(run / "summary.json")
    expected = {(u, m, t, s, method) for u in [1, 2] for m in [7, 8] for t in [3, 4]
                for s in [11, 12] for method in ["region_distance_straight",
                "shortest_balanced_graph", "gatsac_style", "proposed_stc_alns"]}
    keys = Counter((r["uav_num"], r["map_seed"], r["task_seed"], r["method_seed"],
                    r["method"]) for r in raw)
    assert set(keys) == expected and set(keys.values()) == {1}
    assert len(solutions) == len(timelines) == len(raw) == 64
    assert spacings == [25.] * 8
    cfg = manifest["config"]
    assert cfg["iterations"] == 2 and cfg["spacing_m"] == 25.
    assert cfg["smoke"] is True and cfg["generator_config"] == {"straight_hold": True}
    assert cfg["method_seeds"] == [11, 12] and len(cfg["methods"]) == 4
    assert manifest["inputs"]["config_file"] == hashlib.sha256(config.read_bytes()).hexdigest()
    for kind in ["resolved_config", "scenario", "task", "map", "library"]:
        assert any(key.startswith(kind) for key in manifest["inputs"])
    for name in ["raw_records.json", "solutions.json", "timelines.json", "summary.json"]:
        assert name in manifest["expected"]
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == manifest["hashes"][name]
    assert manifest["status"] == "success"
    assert summary["record_count"] == 64 and summary["unavailable_count"] == 32
    assert "scenarios" not in summary
    for r, sol, trace in zip(raw, solutions, timelines):
        assert r["record_id"] == sol["record_id"] == trace["record_id"]
        assert {"balance", "conflict_lost_bits", "conflict_duration_s", "waiting_s",
                "runtime_s", "evaluator_calls", "reference_hash", "library_hash",
                "failure_reason", "violations", "objective", "makespan_s", "total_delay_s"} <= r.keys()
        if r["method"] in ["gatsac_style", "shortest_balanced_graph"]:
            assert r["status"] == "unavailable" and r["failure_reason"]
            assert sol["solution"] is None and trace["timelines"] is None
        else:
            assert r["status"] == "evaluated"
            assert sol["solution"]["candidate_ids"]
            assert len(r["balance"]["completion_times_s"]) == r["uav_num"]
            assert r["reference_hash"] and r["library_hash"]
            assert trace["delta_t_s"] == .5 and trace["bandwidth_hz"] == 1e6
            for tl in trace["timelines"]:
                if tl["slots"]:
                    slot = tl["slots"][0]
                    assert slot["time_start_s"] == 0. and slot["airborne"] is True
                    assert {"pre_buffer_bits", "allocated_bandwidth_hz", "uploaded_bits",
                            "serving_sector", "candidate_id"} <= slot.keys()
            if r["method"] == "proposed_stc_alns":
                assert r["provenance"]["iterations_executed"] == 2
            else:
                assert r["provenance"]["trajectory_generators"] == ["straight_hold"]


@pytest.mark.parametrize("contents", [None, "{bad json", "[]", '{"iterations": NaN}'])
def test_invalid_config_rejected_before_run_creation(tmp_path, monkeypatch, contents):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "bad.json"
    if contents is not None:
        config.write_text(contents, encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main.main(["--config", str(config), "--run-id", "bad", "--dry-run"])
    assert error.value.code != 0
    assert not (tmp_path / "results/runs/bad").exists()


def test_strict_json_path_converts_nested_numpy_and_nonfinite_values(tmp_path):
    payload = {"objective": float("inf"), "balance": [float("nan"), -float("inf")],
               "trace": np.array([np.nan, 2.]), "counter": np.int64(3)}
    path = tmp_path / "strict.json"
    path.write_text(json.dumps(main._json_safe(payload), allow_nan=False), encoding="utf-8")
    parsed = strict_load(path)
    assert parsed["trace"][1] == 2. and parsed["counter"] == 3


def test_coverage_guard_rejects_missing_or_duplicate_records():
    config = {"uav_nums": [2], "map_seeds": [7], "task_seeds": [3],
              "method_seeds": [11, 12], "methods": ["region_distance_straight"]}
    guard = getattr(main, "_assert_coverage", None)
    assert callable(guard), "runner must check exact matrix coverage before sealing"
    row = dict(uav_num=2, map_seed=7, task_seed=3, method_seed=11,
               method="region_distance_straight", status="evaluated")
    with pytest.raises(RuntimeError, match="coverage"):
        guard([row], config)
    with pytest.raises(RuntimeError, match="coverage"):
        guard([row, row], config)


def test_config_changes_actual_generator_family_and_cli_overrides(tiny_run):
    root, config, spacings = tiny_run
    cfg = json.loads(config.read_text(encoding="utf-8"))
    cfg.update(generator_config={"comm_waypoints": {"grid_shape": [2, 2]}},
               graph_generators=["comm_direct"], methods=["shortest_balanced_graph"])
    config.write_text(json.dumps(cfg), encoding="utf-8")
    main.main(["--config", str(config), "--run-id", "override", "--iterations", "0",
               "--spacing-m", "20", "--method-seeds", "99", "--map-seeds", "7",
               "--task-seeds", "3", "--uav-nums", "2"])
    run = root / "results/runs/override"
    raw = strict_load(run / "raw_records.json")
    assert len(raw) == 1 and raw[0]["status"] == "evaluated"
    assert raw[0]["provenance"]["trajectory_generators"] == ["comm_direct"]
    assert spacings == [20.]
    manifest = strict_load(run / "manifest.json")
    assert manifest["config"]["iterations"] == 0
    assert manifest["config"]["method_seeds"] == [99]


def test_nonfinite_infeasible_result_artifacts_remain_strict_json(tiny_run, monkeypatch):
    root, config, _ = tiny_run
    original_load = main.load_stc_scenario
    def deadline_scenario(*args, **kwargs):
        sc = original_load(*args, **kwargs)
        return replace(sc, limits=replace(sc.limits, mission_deadline_s=.5))
    monkeypatch.setattr(main, "load_stc_scenario", deadline_scenario)
    main.main(["--config", str(config), "--run-id", "infeasible", "--iterations", "0",
               "--methods", "region_distance_straight", "--method-seeds", "11",
               "--map-seeds", "7", "--task-seeds", "3", "--uav-nums", "2"])
    run = root / "results/runs/infeasible"
    for path in run.glob("*.json"):
        strict_load(path)
    record = strict_load(run / "raw_records.json")[0]
    assert record["feasible"] is False and record["failure_reason"]
    assert record["objective"] is None
    assert None in record["balance"]["completion_times_s"]


def test_directory_config_is_rejected(tmp_path):
    with pytest.raises(SystemExit) as error:
        main.main(["--config", str(tmp_path), "--dry-run"])
    assert error.value.code != 0
