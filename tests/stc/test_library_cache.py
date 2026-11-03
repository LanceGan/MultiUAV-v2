"""Disk cache must be content-addressed and bit-for-bit reusable."""

import hashlib
from types import SimpleNamespace

import numpy as np

from segment.stc.types import RadioSample, STCScenario, ScenarioLimits


def _limits():
    return ScenarioLimits(
        delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
        arrive_radius_m=15.0, max_wait_s=60.0, mission_deadline_s=7200.0,
        bandwidth_hz=1e6, uplink_power_w=0.2, eps_out=0.35,
        w_t=0.5, w_d=0.5)


def _scenario():
    return STCScenario(
        n_uav=2, n_tasks=3,
        task_positions_m=((300.0, 300.0), (900.0, 300.0), (1500.0, 300.0)),
        task_data_bits=(1e7, 2e7, 3e7),
        depot_m=(0.0, 0.0), terminal_m=(2000.0, 300.0),
        limits=_limits(), length_m=2000.0, width_m=2000.0, n_bs=1,
        task_seed=42, map_seed=0)


class _Field:
    n_sectors = 1
    world_hash = "cache-test-world"

    def sample(self, pos):
        return RadioSample(np.array([4.0]), np.array([0.05]))


def _cfg():
    return {"straight_hold": True}


def test_base_library_cache_reuses_identical_content(tmp_path, monkeypatch):
    from segment.stc import cache
    from segment.stc.evaluator import library_content_hash

    calls = {"n": 0}
    original = cache.build_all_pairs_library

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cache, "build_all_pairs_library", counted)
    sc, field = _scenario(), _Field()
    first, hit1 = cache.load_or_build_base_library(
        sc, field, _cfg(), spacing_m=1000.0, workers=1,
        cache_root=tmp_path / "cache")
    second, hit2 = cache.load_or_build_base_library(
        sc, field, _cfg(), spacing_m=1000.0, workers=1,
        cache_root=tmp_path / "cache")
    assert not hit1 and hit2
    assert calls["n"] == 1
    assert library_content_hash(first.candidates) == library_content_hash(second.candidates)


def test_sac_cache_uses_checkpoint_hash(tmp_path, monkeypatch):
    from segment.stc import cache

    checkpoint = tmp_path / "sac.pth"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    calls = {"n": 0}
    candidate = SimpleNamespace(
        candidate_id="sac", source=-1, target=0, generator="drl:SAC:test",
        positions_m=np.array([[0.0, 0.0], [1.0, 1.0]]),
        full_rate_bps=np.ones((2, 1)), sector_ids=np.zeros((2, 1), dtype=int),
        g2a_outage=np.zeros((2, 1)), flight_steps=1,
        hold_position_m=(0.0, 0.0), source_data_bits=0.0, rankable=True)

    def fake_produce(path, scenario, field):
        calls["n"] += 1
        return cache.SacCandidateResult(
            "available", {(-1, 0): (candidate,)}, "", digest, str(path))

    monkeypatch.setattr(cache, "produce_sac_segment_candidates", fake_produce)
    sc, field = _scenario(), _Field()
    first, hit1 = cache.load_or_produce_sac_candidates(
        str(checkpoint), sc, field, spacing_m=1000.0,
        generator_config=_cfg(), cache_root=tmp_path / "cache")
    second, hit2 = cache.load_or_produce_sac_candidates(
        str(checkpoint), sc, field, spacing_m=1000.0,
        generator_config=_cfg(), cache_root=tmp_path / "cache")
    assert not hit1 and hit2
    assert calls["n"] == 1
    assert first.checkpoint_hash == second.checkpoint_hash == digest
    assert set(second.candidates) == {(-1, 0)}
