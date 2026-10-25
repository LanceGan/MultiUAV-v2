"""Task 2 Step 4: Pareto pruning and hashed/compatibility-checked persistence."""

import numpy as np
import pytest

from segment.stc.library import (
    generator_config_hash,
    load_library,
    pareto_prune,
    radio_params_hash,
    save_library,
    scenario_hash,
    task_content_hash,
)
from segment.stc.radio import PerSectorRadioField
from segment.stc.scenario import load_stc_scenario
from segment.stc.generators import generate_comm_waypoints, generate_straight_hold
from segment.stc.types import RadioSample, STCScenario, ScenarioLimits


def _cands():
    sc = load_stc_scenario(2, task_seed=42, map_seed=0)
    field = PerSectorRadioField.from_scenario(sc, spacing_m=2000.0)
    return sc, field, generate_comm_waypoints(sc, field, 0, 1, grid_shape=(4, 4))


def _objectives(c):
    flight = c.flight_steps * 0.5
    upload = c.source_data_bits / max(np.mean(c.full_rate_bps.max(axis=1)), 1e-9)
    outage = float(np.mean([c.g2a_outage[t, int(s)] for t, s in enumerate(c.sector_ids)]))
    return flight, upload, outage


def test_pareto_prune_removes_strictly_dominated():
    _, _, cands = _cands()
    pruned = pareto_prune(cands, max_candidates=10, min_sector_distance=0.0)
    objs = [_objectives(c) for c in pruned]
    # no candidate should be strictly dominated by another in the pruned set
    for i in range(len(objs)):
        for j in range(len(objs)):
            if i != j:
                dom = (objs[j][0] <= objs[i][0] and objs[j][1] <= objs[i][1]
                       and objs[j][2] <= objs[i][2]
                       and (objs[j][0] < objs[i][0] or objs[j][1] < objs[i][1]
                            or objs[j][2] < objs[i][2]))
                assert not dom
    assert len(pruned) <= 10


# ---- lightweight persistence fixtures (fast: no radio maps) ----

class _LightField:
    n_sectors = 4
    world_hash = "light-hash"

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.ones(self.n_sectors),
                           g2a_outage=np.zeros(self.n_sectors))


def _light():
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    sc = STCScenario(n_uav=2, n_tasks=2, task_positions_m=((1000.0, 1000.0), (2000.0, 1000.0)),
                     task_data_bits=(1e7, 2e7), depot_m=(0.0, 0.0),
                     terminal_m=(3000.0, 1000.0), limits=limits, length_m=4000.0,
                     width_m=4000.0, n_bs=4, task_seed=42, map_seed=0)
    field = _LightField()
    lib = {
        (-1, 0): (generate_straight_hold(sc, field, -1, 0),),
        (0, 1): (generate_straight_hold(sc, field, 0, 1),),
    }
    return sc, lib


def _path(tmp_path):
    return str(tmp_path / "lib.npz")


def test_round_trip_preserves_all_arrays_and_metadata(tmp_path):
    sc, lib = _light()
    path = _path(tmp_path)
    cfg = {"straight_hold": True}
    manifest = save_library(lib, sc, "light-hash", path, generator_config=cfg)
    loaded = load_library(path)
    assert loaded.world_hash == "light-hash"
    assert loaded.map_seed == sc.map_seed
    assert loaded.task_seed == sc.task_seed
    assert set(loaded.candidates.keys()) == set(lib.keys())
    for edge, cands in lib.items():
        back = loaded.candidates[edge]
        assert len(back) == len(cands)
        for orig, rt in zip(cands, back):
            assert rt.candidate_id == orig.candidate_id
            assert rt.generator == orig.generator
            assert rt.source == orig.source and rt.target == orig.target
            assert rt.flight_steps == orig.flight_steps
            assert rt.hold_position_m == tuple(float(x) for x in orig.hold_position_m)
            assert rt.source_data_bits == orig.source_data_bits
            assert rt.rankable == orig.rankable
            np.testing.assert_array_equal(rt.positions_m, orig.positions_m)
            np.testing.assert_array_equal(rt.full_rate_bps, orig.full_rate_bps)
            np.testing.assert_array_equal(rt.sector_ids, orig.sector_ids)
            np.testing.assert_array_equal(rt.g2a_outage, orig.g2a_outage)
    # manifest carries per-candidate content hashes and compatibility metadata
    assert len(manifest["candidate_hashes"]) == sum(len(v) for v in lib.values())
    assert manifest["generator_config_hash"] == generator_config_hash(cfg)
    assert manifest["scenario_hash"] == scenario_hash(sc)
    assert manifest["task_content_hash"] == task_content_hash(sc)
    assert manifest["radio_params_hash"] == radio_params_hash(sc)


def test_load_accepts_matching_expected(tmp_path):
    sc, lib = _light()
    path = _path(tmp_path)
    save_library(lib, sc, "light-hash", path)
    # no exception when all expected fields match
    load_library(path, expected={
        "world_hash": "light-hash",
        "scenario_hash": scenario_hash(sc),
        "task_content_hash": task_content_hash(sc),
        "radio_params_hash": radio_params_hash(sc),
        "si_schema": "si-v1",
    })


def test_load_rejects_wrong_world_hash(tmp_path):
    sc, lib = _light()
    save_library(lib, sc, "light-hash", _path(tmp_path))
    with pytest.raises(ValueError):
        load_library(_path(tmp_path), expected={"world_hash": "other"})


def test_load_rejects_wrong_scenario_hash(tmp_path):
    sc, lib = _light()
    save_library(lib, sc, "light-hash", _path(tmp_path))
    with pytest.raises(ValueError):
        load_library(_path(tmp_path), expected={"scenario_hash": "0" * 64})


def test_load_rejects_wrong_task_content_hash(tmp_path):
    sc, lib = _light()
    save_library(lib, sc, "light-hash", _path(tmp_path))
    with pytest.raises(ValueError):
        load_library(_path(tmp_path), expected={"task_content_hash": "0" * 64})


def test_load_rejects_wrong_radio_params_hash(tmp_path):
    sc, lib = _light()
    save_library(lib, sc, "light-hash", _path(tmp_path))
    with pytest.raises(ValueError):
        load_library(_path(tmp_path), expected={"radio_params_hash": "0" * 64})


def test_load_rejects_wrong_generator_config_hash(tmp_path):
    sc, lib = _light()
    save_library(lib, sc, "light-hash", _path(tmp_path), generator_config={"x": 1})
    with pytest.raises(ValueError):
        load_library(_path(tmp_path),
                     expected={"generator_config_hash": generator_config_hash({"x": 2})})


def test_load_rejects_wrong_checkpoint_hash(tmp_path):
    sc, lib = _light()
    save_library(lib, sc, "light-hash", _path(tmp_path), checkpoint_hashes={"td3": "h1"})
    with pytest.raises(ValueError):
        load_library(_path(tmp_path), expected={"checkpoint_hashes": {"td3": "h2"}})


def test_load_rejects_wrong_si_schema(tmp_path):
    sc, lib = _light()
    save_library(lib, sc, "light-hash", _path(tmp_path))
    with pytest.raises(ValueError):
        load_library(_path(tmp_path), expected={"si_schema": "legacy-units"})


def test_load_rejects_corrupted_candidate_content(tmp_path):
    sc, lib = _light()
    path = _path(tmp_path)
    save_library(lib, sc, "light-hash", path)
    with np.load(path) as arrs:
        data = {k: np.array(v) for k, v in arrs.items()}
    pos_key = [k for k in data if k.endswith("_pos")][0]
    data[pos_key] = data[pos_key] + 1.0
    np.savez(path, **data)
    with pytest.raises(ValueError):
        load_library(path)
