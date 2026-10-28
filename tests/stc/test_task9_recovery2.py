"""Task 9 recovery round 2: fixes for the second independent review.

Covers (in order): SAC non-arrival judged infeasible (not feasible), RQ1
segment feasibility computed from the segment trace (no virtual entry/exit
edges, T/D/O consistent with feasibility), per-edge + common-feasible IDs
written to records, main-runner partial persistence on failure, timeout /
KeyboardInterrupt sealed as failed, RQ5 method-seed / runtime / evaluator-calls
recording, task-scale extension that keeps existing tasks, and uplink power
actually entering the A2G radio propagation.
"""

import json
import time as _time

import numpy as np
import pytest

from segment.stc.types import RadioSample, ScenarioLimits, STCScenario


def _limits(**kw):
    base = dict(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                arrive_radius_m=15.0, max_wait_s=60.0, mission_deadline_s=7200.0,
                bandwidth_hz=1e6, uplink_power_w=0.2, eps_out=0.35,
                w_t=0.5, w_d=0.5)
    base.update(kw)
    return ScenarioLimits(**base)


def _scenario(n_uav=2, n_tasks=3, **kw):
    positions = tuple((50.0 + (i % 2) * 60.0, 50.0 + (i // 2) * 60.0)
                      for i in range(n_tasks))
    base = dict(n_uav=n_uav, n_tasks=n_tasks, task_positions_m=positions,
                task_data_bits=tuple([1e6] * n_tasks),
                depot_m=(10.0, 10.0), terminal_m=(390.0, 390.0),
                limits=_limits(), length_m=400.0, width_m=400.0, n_bs=1,
                task_seed=42, map_seed=0)
    base.update(kw)
    return STCScenario(**base)


class _ConstField:
    """Uniform field: every sector equal (no outage, full rate)."""

    def __init__(self, n_sectors=3, a2g=1.0, g2a=0.0):
        self.n_sectors = n_sectors
        self.world_hash = "const-field"
        self._a2g = a2g
        self._g2a = g2a

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.full(self.n_sectors, self._a2g),
                           g2a_outage=np.full(self.n_sectors, self._g2a))


def _field_for(sc):
    return _ConstField(n_sectors=sc.n_sectors)


def _dummy_spec():
    from segment.stc.types import ObjectiveSpec, ReferenceValues
    return ObjectiveSpec(reference=ReferenceValues(t_ref_s=1.0, d_ref_s=1.0,
                                                  source_hash="test"),
                         w_t=0.5, w_d=0.5)


# --------------------------------------------------------------------------- #
# 1. SAC non-arrival must NOT be judged feasible by RQ1
# --------------------------------------------------------------------------- #

def test_segment_non_rankable_candidate_infeasible():
    from experiment.stc_common import evaluate_segment
    from segment.stc.generators import generate_straight_relaxation

    sc = _scenario(1, 3)
    field = _field_for(sc)
    cand = generate_straight_relaxation(sc, field, 0, 1)  # rankable=False
    assert cand.rankable is False
    assert evaluate_segment(sc, cand, field, _dummy_spec())["feasible"] is False


def test_segment_sac_non_arrival_infeasible():
    from experiment.stc_common import evaluate_segment
    from segment.stc.sac_candidates import _rollout_drl_candidate

    class StaticPolicy:
        def select_action(self, state):
            return np.zeros(2)  # never moves -> never arrives

    sc = _scenario(2, 3)
    field = _field_for(sc)
    cand = _rollout_drl_candidate(sc, field, StaticPolicy(), 0, 1, 1e6,
                                  "drl:sac:test")
    assert cand.rankable is False
    assert evaluate_segment(sc, cand, field, _dummy_spec())["feasible"] is False


# --------------------------------------------------------------------------- #
# 2. Segment feasibility must come from the segment trace (no virtual-edge
#    dilution) and be consistent with the reported outage
# --------------------------------------------------------------------------- #

class _SourceOutageField:
    """High outage only at a single source point, zero elsewhere."""

    def __init__(self, n_sectors, source_pos, high=1.0):
        self.n_sectors = n_sectors
        self.world_hash = "source-outage-field"
        self._source = np.asarray(source_pos, dtype=float)
        self._high = high

    def sample(self, pos):
        pos = np.asarray(pos, dtype=float)
        g2a = np.zeros(self.n_sectors)
        if np.allclose(pos, self._source, atol=1.0):
            g2a[:] = self._high
        return RadioSample(np.ones(self.n_sectors), g2a)


def test_segment_feasibility_not_diluted_by_virtual_edges():
    from experiment.stc_common import evaluate_segment
    from segment.stc.generators import generate_straight_hold

    sc = _scenario(1, 3, limits=_limits(eps_out=.3))
    # The authoritative machine advances before sampling. The four-position
    # trace therefore takes three slots, with outage 1 only in the first.
    cand = generate_straight_hold(sc, _field_for(sc), 0, 1)
    field = _SourceOutageField(sc.n_sectors, cand.positions_m[1], high=1.)
    cand = generate_straight_hold(sc, field, 0, 1)
    result = evaluate_segment(sc, cand, field, _dummy_spec())
    assert result["feasible"] is False
    assert result["outage"] == pytest.approx(1 / 3, abs=1e-6)


def test_segment_outage_consistent_with_feasibility():
    from experiment.stc_common import evaluate_segment
    from segment.stc.generators import generate_straight_hold

    # uniform high outage -> outage 1.0 > eps_out -> infeasible, and the
    # reported outage equals exactly what the feasibility check used.
    sc = _scenario(1, 3)
    field = _ConstField(n_sectors=sc.n_sectors, a2g=1.0, g2a=1.0)
    cand = generate_straight_hold(sc, field, 0, 1)
    result = evaluate_segment(sc, cand, field, _dummy_spec())
    assert result["feasible"] is False
    assert result["outage"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 3. per-edge evaluation + common-feasible IDs must be written to records
# --------------------------------------------------------------------------- #

def test_segment_gate_records_include_per_edge_and_common_ids():
    from experiment import run_stc_segment_gate as gate

    sc = _scenario(1, 3)
    field = _field_for(sc)
    out = gate.run_segment_gate(sc, field)
    summary = out["summary"]
    common = {tuple(e) for e in summary["common_feasible_edges"]}
    for rec in out["records"]:
        if rec["status"] != "evaluated":
            continue
        assert "per_edge" in rec
        assert len(rec["per_edge"]) == rec["n_edges"]
        assert "common_feasible_edges" in rec
        for pe in rec["per_edge"]:
            assert "edge" in pe and "feasible" in pe
        # the common-feasible IDs embedded in the record match the summary
        rec_common = {tuple(e) for e in rec["common_feasible_edges"]}
        assert rec_common == common


# --------------------------------------------------------------------------- #
# 4. main runner must persist previously completed scenarios on failure
# --------------------------------------------------------------------------- #

def test_main_preserves_partial_records_on_failure(monkeypatch, tmp_path):
    from experiment import run_stc_main as main

    monkeypatch.chdir(tmp_path)
    sc = _scenario(2, 3)

    class Field:
        world_hash = "test-map-world"
        n_sectors = 3

        @classmethod
        def from_scenario(cls, scn, spacing_m):
            return cls()

        def sample(self, pos):
            return RadioSample(np.ones(3), np.zeros(3))

    monkeypatch.setattr(main, "PerSectorRadioField", Field)
    monkeypatch.setattr(main, "load_stc_scenario",
                        lambda u, task_seed, map_seed: _scenario(u, sc.n_tasks))

    class SacRes:
        status = "unavailable"
        candidates = {}
        reason = "no checkpoint"
        checkpoint_hash = ""
        checkpoint_path = ""

    class Lib:
        candidates = {}

    def fake_build(u, m, t, cfg):
        return sc, Field(), Lib(), None, SacRes()

    def fake_run(scn, field, library, reference, sac_result, u, m, t, cfg, rc,
                 checkpoint=None):
        if m == 8:
            raise RuntimeError("boom on scenario 2")
        rec = {"record_id": "x", "uav_num": u, "map_seed": m, "task_seed": t,
               "method_seed": 0, "method": "proposed_stc_alns",
               "status": "evaluated", "feasible": True, "objective": 1.0}
        checkpoint(rec, {"record_id": "x", "solution": None},
                   {"record_id": "x", "timelines": None})
        return [rec], [], []

    monkeypatch.setattr(main, "_build_context", fake_build)
    monkeypatch.setattr(main, "_run_scenario", fake_run)
    monkeypatch.setattr(main, "run_rq2",
                        lambda scn, field, lib, spec, seed=42:
                        {"assignment": [], "sequence": []})

    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "uav_nums": [2], "map_seeds": [7, 8], "task_seeds": [3],
        "method_seeds": [0], "methods": ["proposed_stc_alns"],
        "iterations": 2, "smoke": True, "smoke_tasks": 3,
        "generator_config": {"straight_hold": True},
    }))
    with pytest.raises(RuntimeError):
        main.main(["--config", str(config), "--run-id", "partial"])

    manifest = json.loads((tmp_path / "results/runs/partial/manifest.json").read_text())
    assert manifest["status"] == "failed"
    raw = json.loads((tmp_path / "results/runs/partial/raw_records.json").read_text())
    assert len(raw) == 1
    assert raw[0]["map_seed"] == 7


# --------------------------------------------------------------------------- #
# 5. timeout expiry during the last scenario + KeyboardInterrupt must seal
#    "failed", never "success"
# --------------------------------------------------------------------------- #

def test_lifecycle_timeout_expiry_during_last_scenario_seals_failed(tmp_path, monkeypatch):
    from experiment.stc_common import run_lifecycle

    monkeypatch.chdir(tmp_path)
    cfg = {"uav_nums": [2], "timeout_s": 0.05}

    def per_scenario(u, m, t):
        _time.sleep(0.3)  # exceeds the timeout *during* the only scenario
        return [{"u": u, "m": m, "t": t}]

    def summary_fn(records):
        return {"record_count": len(records)}

    with pytest.raises(TimeoutError):
        run_lifecycle("stc_test", cfg, "lifecycle_timeout", [(2, 7, 1)],
                      per_scenario, summary_fn)

    manifest = json.loads((tmp_path / "results/runs/lifecycle_timeout/manifest.json").read_text())
    assert manifest["status"] == "failed"


def test_lifecycle_keyboard_interrupt_seals_failed(tmp_path, monkeypatch):
    from experiment.stc_common import run_lifecycle

    monkeypatch.chdir(tmp_path)

    def per_scenario(u, m, t):
        raise KeyboardInterrupt()

    def summary_fn(records):
        return {"record_count": len(records)}

    with pytest.raises(KeyboardInterrupt):
        run_lifecycle("stc_test", {"uav_nums": [2]}, "lifecycle_kb", [(2, 7, 1)],
                      per_scenario, summary_fn)

    manifest = json.loads((tmp_path / "results/runs/lifecycle_kb/manifest.json").read_text())
    assert manifest["status"] == "failed"


# --------------------------------------------------------------------------- #
# 6. RQ5 scale must record method_seed / runtime / evaluator calls
# --------------------------------------------------------------------------- #

def test_scale_records_method_seed_runtime_evaluator_calls():
    from experiment import run_stc_scale as scale_mod

    base = _scenario(2, 3)
    out = scale_mod.run_scale(base, _field_for, seed=0)
    assert out["records"]
    for rec in out["records"]:
        assert "method_seed" in rec
        assert "runtime_s" in rec
        assert "evaluator_calls" in rec
        assert isinstance(rec["evaluator_calls"], dict)
        assert np.isfinite(rec["runtime_s"])


# --------------------------------------------------------------------------- #
# 7. task-scale "more tasks" must keep the existing tasks / depot / terminal
# --------------------------------------------------------------------------- #

def test_task_scale_more_tasks_keeps_existing_tasks():
    from experiment import run_stc_scale as scale_mod

    base = _scenario(2, 3)
    variants = dict(scale_mod._scale_scenarios(base, "task_scale"))
    k = base.n_tasks + 2
    ext = variants["tasks_%d" % k]
    assert ext.n_tasks == k
    assert ext.task_positions_m[:base.n_tasks] == base.task_positions_m
    assert ext.task_data_bits[:base.n_tasks] == base.task_data_bits
    assert ext.depot_m == base.depot_m
    assert ext.terminal_m == base.terminal_m


# --------------------------------------------------------------------------- #
# 8. uplink power must enter the A2G radio propagation
# --------------------------------------------------------------------------- #

def test_configure_radio_world_sets_a2g_uplink_power():
    from radio_map import radio_map_A2G
    from segment.stc.radio import configure_radio_world

    configure_radio_world(4000.0, 4, map_seed=0, uplink_power_w=0.7)
    assert radio_map_A2G.PB == pytest.approx(0.7)


def test_configure_radio_world_hashes_uplink_power():
    from segment.stc.radio import configure_radio_world

    h1 = configure_radio_world(4000.0, 4, map_seed=0, uplink_power_w=0.1)
    h2 = configure_radio_world(4000.0, 4, map_seed=0, uplink_power_w=2.0)
    assert h1 != h2
