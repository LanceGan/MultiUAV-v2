"""Task 9: RQ1-RQ5 experiment runners + SAC candidate production/loading.

The load-bearing requirements:
- every RQ produces REAL, nonempty finalized raw records on a tiny synthetic
  scenario (small K), all through the SAME scenario / library / ObjectiveSpec /
  synchronous evaluator;
- no runner reports success on missing inputs (config / library / checkpoint);
- no method/scenario pair is silently dropped;
- GATSAC is honestly "unavailable" without a compatible SAC checkpoint, and uses
  REAL SAC candidates (correct generator label, source payload, rankable) when a
  valid SI checkpoint exists.
"""

import hashlib
import json
import math
from dataclasses import replace

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
                depot_m=(0.0, 0.0), terminal_m=(200.0, 200.0),
                limits=_limits(), length_m=200.0, width_m=200.0, n_bs=1,
                task_seed=42, map_seed=0)
    base.update(kw)
    return STCScenario(**base)


class _ConstField:
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


@pytest.fixture(autouse=True)
def _fast_radio(monkeypatch):
    # Keep the legacy Jia time-energy / GA-EQTSP adapters deterministic and fast
    # by pinning the physical-map queries (mirrors test_system_baselines.py).
    from sequence import radio_path
    from segment.stc import sequencing
    monkeypatch.setattr(sequencing, "_configure_maps", lambda sc: "const-field")
    monkeypatch.setattr(radio_path.radio_map_A2G, "getPointDateRate",
                        lambda loc: 10.0)
    monkeypatch.setattr(radio_path.radio_map_G2A, "getPointMiniOutage",
                        lambda loc: ([0.1], [10.0]))


# --------------------------------------------------------------------------- #
# SAC candidate production / loading
# --------------------------------------------------------------------------- #

def _save_sac(path, v_max):
    from segment.sac import SAC
    model = SAC(7, 2,
                np.array([float(v_max), 2.0 * math.pi], dtype=np.float32),
                np.array([0.0, 0.0], dtype=np.float32))
    model.save(str(path))
    return model


def test_sac_loader_unavailable_for_missing_checkpoint():
    from segment.stc.sac_candidates import produce_sac_segment_candidates
    sc = _scenario()
    res = produce_sac_segment_candidates(str(sc) + "/missing.pth", sc, _field_for(sc))
    assert res.status == "unavailable"
    assert res.reason
    assert res.candidates == {}
    assert res.checkpoint_hash == ""


def test_sac_loader_unavailable_for_legacy_checkpoint(tmp_path):
    from segment.stc.sac_candidates import produce_sac_segment_candidates
    sc = _scenario()
    legacy = tmp_path / "legacy_sac.pth"
    _save_sac(legacy, v_max=0.5)  # 100-m-unit action bounds -> incompatible
    res = produce_sac_segment_candidates(str(legacy), sc, _field_for(sc))
    assert res.status == "unavailable"
    assert res.reason
    assert res.candidates == {}


def test_sac_loader_produces_real_candidates_for_si_checkpoint(tmp_path):
    from segment.stc.sac_candidates import produce_sac_segment_candidates
    sc = _scenario()
    si = tmp_path / "si_sac.pth"
    _save_sac(si, v_max=50.0)  # SI action bounds -> compatible
    digest = hashlib.sha256(si.read_bytes()).hexdigest()
    res = produce_sac_segment_candidates(str(si), sc, _field_for(sc))
    assert res.status == "available"
    assert res.checkpoint_hash == digest
    # every legal directed edge gets exactly one candidate with the SAC label
    from segment.stc.sac_candidates import legal_edges
    edges = legal_edges(sc.n_tasks)
    assert set(res.candidates) == set(edges)
    for (src, tgt), cands in res.candidates.items():
        assert len(cands) == 1
        c = cands[0]
        assert c.generator == "drl:sac:" + digest
        assert c.rankable is True
        assert c.source == src and c.target == tgt
        expected_data = 0.0 if src == -1 else sc.task_data_bits[src]
        assert c.source_data_bits == expected_data
        assert len(c.positions_m) == len(c.full_rate_bps) == len(c.g2a_outage)
        assert c.full_rate_bps.shape[1] == sc.n_sectors


def test_sac_loader_never_fabricates_when_incompatible(tmp_path):
    from segment.stc.sac_candidates import produce_sac_segment_candidates
    sc = _scenario()
    legacy = tmp_path / "legacy_sac.pth"
    _save_sac(legacy, v_max=0.5)
    res = produce_sac_segment_candidates(str(legacy), sc, _field_for(sc))
    assert res.status == "unavailable"
    assert all(not c.generator.startswith("drl:sac:") or False
               for cands in res.candidates.values() for c in cands)


# --------------------------------------------------------------------------- #
# RQ1 segment gate
# --------------------------------------------------------------------------- #

def test_segment_gate_produces_nonempty_records_for_all_generators(tmp_path):
    from experiment import run_stc_segment_gate as gate
    from segment.stc.generators import drl_generator_label
    sc = _scenario()
    field = _field_for(sc)
    si = tmp_path / "si_sac.pth"
    _save_sac(si, v_max=50.0)
    digest = hashlib.sha256(si.read_bytes()).hexdigest()
    out = gate.run_segment_gate(
        sc, field, drl_checkpoints={"SAC": str(si), "TD3": None, "DDPG": None})
    records = out["records"]
    by_gen = {r["generator"]: r for r in records}
    for gen in ("straight_hold", "comm_waypoint", "astar", "slsqp_refined"):
        assert gen in by_gen, "deterministic generator missing: %s" % gen
        r = by_gen[gen]
        assert r["status"] == "evaluated"
        assert r["n_edges"] == sc.n_tasks * (sc.n_tasks + 1)
        for key in ("feasible_frac", "t_mean_s", "d_mean_s", "o_mean",
                    "sector_diversity", "compute_time_s"):
            assert key in r
    # SAC available via real checkpoint; TD3/DDPG honestly unavailable
    assert by_gen[drl_generator_label("sac", digest)]["status"] == "evaluated"
    assert by_gen["TD3"]["status"] == "unavailable"
    assert by_gen["DDPG"]["status"] == "unavailable"
    assert out["summary"]["n_methods"] == len(records)


# --------------------------------------------------------------------------- #
# RQ2 assignment + sequence value
# --------------------------------------------------------------------------- #

def test_assignment_and_sequence_scan_produce_records():
    from experiment import run_stc_main as main
    from segment.stc.library import build_all_pairs_library
    from segment.stc.evaluator import default_objective_spec
    sc = _scenario(2, 3)
    field = _field_for(sc)
    library = build_all_pairs_library(
        sc, field, generator_config={"straight_hold": True})
    spec = default_objective_spec(sc, library.candidates)
    rq2 = main.run_rq2(sc, field, library.candidates, spec, seed=0)
    assign = rq2["assignment"]
    seq = rq2["sequence"]
    assert len(assign) >= 4 and len(seq) >= 3
    for r in assign + seq:
        assert r["status"] == "evaluated"
        assert r["feasible"] in (True, False)
        assert "objective" in r and "makespan_s" in r and "total_delay_s" in r


# --------------------------------------------------------------------------- #
# RQ3 evaluator audit
# --------------------------------------------------------------------------- #

def _plans():
    # Plan A: parallel (both UAVs upload near-simultaneously -> contention)
    # Plan B: serial (one UAV serves both tasks, no contention)
    a = {"routes": ((0,), (1,))}
    b = {"routes": ((0, 1), ())}
    return [a, b]


def _make_solutions(sc, library, plans):
    from segment.stc.operators import _plan
    sols = []
    for p in plans:
        sols.append(_plan(tuple(tuple(r) for r in p["routes"]), sc, library))
    return sols


def test_evaluator_audit_produces_metric_error_and_reversal_records():
    from experiment import run_stc_evaluator_audit as audit
    from segment.stc.library import build_all_pairs_library
    from segment.stc.evaluator import default_objective_spec
    sc = _scenario(2, 2)
    field = _field_for(sc)
    library = build_all_pairs_library(
        sc, field, generator_config={"straight_hold": True})
    spec = default_objective_spec(sc, library.candidates)
    plans = _make_solutions(sc, library.candidates, _plans())
    out = audit.run_evaluator_audit(sc, library.candidates, field, plans, spec)
    assert len(out["plans"]) == 2
    for p in out["plans"]:
        for mode in ("synchronous", "independent", "static_factor"):
            assert mode in p and p[mode]["makespan_s"] >= 0
    err = out["metric_error"]
    assert err["independent"]["makespan_s_abs"] >= 0
    assert err["static_factor"]["total_delay_s_abs"] >= 0
    # reversal fields are either an explicit list or the "none found" sentinel
    for key in ("feasibility_reversal", "ranking_reversal"):
        assert key in out
        v = out[key]
        assert (v == "none found") or (isinstance(v, list))


def test_evaluator_audit_reports_none_found_for_identical_plans():
    from experiment import run_stc_evaluator_audit as audit
    from segment.stc.library import build_all_pairs_library
    from segment.stc.evaluator import default_objective_spec
    sc = _scenario(1, 1)  # single UAV -> no contention possible
    field = _field_for(sc)
    library = build_all_pairs_library(
        sc, field, generator_config={"straight_hold": True})
    spec = default_objective_spec(sc, library.candidates)
    plans = _make_solutions(sc, library.candidates, [{"routes": ((0,),)}])
    out = audit.run_evaluator_audit(sc, library.candidates, field, plans, spec)
    assert out["feasibility_reversal"] == "none found"
    assert out["ranking_reversal"] == "none found"


# --------------------------------------------------------------------------- #
# RQ4 main + SAC wiring
# --------------------------------------------------------------------------- #

def _main_config(tmp_path, methods, sac_checkpoint=None):
    cfg = {
        "iterations": 2, "spacing_m": 25., "smoke": True,
        "generator_config": {"straight_hold": True},
        "methods": methods,
        "uav_nums": [2], "map_seeds": [7], "task_seeds": [3],
        "method_seeds": [11],
        "sac_checkpoint": sac_checkpoint,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _tiny_main(monkeypatch, tmp_path):
    from experiment import run_stc_main as main
    monkeypatch.chdir(tmp_path)
    sc = _scenario(2, 2)
    monkeypatch.setattr(main, "load_stc_scenario",
                        lambda u, task_seed, map_seed: replace(
                            sc, n_uav=u, task_seed=task_seed, map_seed=map_seed))

    class Field:
        world_hash = "test-map-world"
        n_sectors = 3

        @classmethod
        def from_scenario(cls, scn, spacing_m):
            return cls()

        def sample(self, pos):
            return RadioSample(np.ones(3), np.zeros(3))

    monkeypatch.setattr(main, "PerSectorRadioField", Field)
    return main


def test_main_gatsac_unavailable_without_sac(monkeypatch, tmp_path):
    main = _tiny_main(monkeypatch, tmp_path)
    config = _main_config(tmp_path, ["gatsac_style", "proposed_stc_alns"])
    assert main.main(["--config", str(config), "--run-id", "rq4"]) == 0
    raw = json.loads((tmp_path / "results/runs/rq4/raw_records.json").read_text())
    by_method = {r["method"]: r for r in raw}
    assert by_method["gatsac_style"]["status"] == "unavailable"
    assert by_method["gatsac_style"]["failure_reason"]
    assert by_method["proposed_stc_alns"]["status"] == "evaluated"


def test_main_gatsac_available_with_real_sac_checkpoint(monkeypatch, tmp_path):
    main = _tiny_main(monkeypatch, tmp_path)
    si = tmp_path / "si_sac.pth"
    _save_sac(si, v_max=50.0)
    config = _main_config(tmp_path, ["gatsac_style"], sac_checkpoint=str(si))
    assert main.main(["--config", str(config), "--run-id", "rq4sac"]) == 0
    raw = json.loads((tmp_path / "results/runs/rq4sac/raw_records.json").read_text())
    rec = raw[0]
    assert rec["method"] == "gatsac_style"
    assert rec["status"] == "evaluated"
    gens = rec["provenance"]["trajectory_generators"]
    assert gens and all(g.startswith("drl:sac:") for g in gens)


# --------------------------------------------------------------------------- #
# RQ5 ablation + scale
# --------------------------------------------------------------------------- #

def test_ablation_produces_records_for_each_mechanism():
    from experiment import run_stc_ablation as ab
    from segment.stc.library import build_all_pairs_library
    sc = _scenario(2, 3)
    field = _field_for(sc)
    library = build_all_pairs_library(
        sc, field, generator_config={"straight_hold": True,
                                     "comm_waypoints": {"grid_shape": (2, 2)}})
    out = ab.run_ablation(sc, field, library.candidates, seed=0, iterations=4)
    names = {r["ablation"] for r in out["records"]}
    assert "full" in names
    for mech in ab.ABLATIONS:
        assert mech in names
    for r in out["records"]:
        assert r["status"] in ("evaluated", "unavailable")


def test_scale_produces_records_for_each_family():
    from experiment import run_stc_scale as sc_mod
    base = _scenario(2, 3)

    def field_factory(sc):
        return _field_for(sc)

    out = sc_mod.run_scale(base, field_factory, seed=0)
    families = {r["family"] for r in out["records"]}
    for fam in sc_mod.SCALE_FAMILIES:
        assert fam in families
    assert all(r["status"] in ("evaluated", "unavailable")
               for r in out["records"])


# --------------------------------------------------------------------------- #
# Global constraints
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("module", ["run_stc_segment_gate", "run_stc_evaluator_audit",
                                    "run_stc_ablation", "run_stc_scale"])
def test_runner_no_success_on_missing_config(tmp_path, monkeypatch, module):
    import importlib
    runner = importlib.import_module("experiment." + module)
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit) as error:
        runner.main(["--config", str(missing), "--run-id", "missing"])
    assert error.value.code != 0
    assert not (tmp_path / "results/runs/missing").exists()


def test_main_no_success_on_missing_config(tmp_path, monkeypatch):
    from experiment import run_stc_main as main
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit) as error:
        main.main(["--config", str(missing), "--run-id", "missing"])
    assert error.value.code != 0
    assert not (tmp_path / "results/runs/missing").exists()
