"""Task 9 recovery: fixes for the independent-review Important findings.

Covers (in the prescribed order): SAC timeout teleport, RQ3 dict-library crash,
RQ3 independent-evaluation semantics, RQ1 authoritative segment evaluation +
common-feasible edges, RQ2 wiring + Shortest-Balanced assignment, RQ5 scale
running STC-ALNS, CLI flag application, and empty-matrix rejection.
"""

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
                depot_m=(0.0, 0.0), terminal_m=(400.0, 400.0),
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


# --------------------------------------------------------------------------- #
# 1. SAC timeout must not teleport / mark rankable
# --------------------------------------------------------------------------- #

def test_sac_timeout_candidate_not_rankable_and_not_teleported():
    from segment.stc.sac_candidates import _rollout_drl_candidate

    class StaticPolicy:
        def select_action(self, state):
            return np.zeros(2)  # v=0, theta=0 -> never moves

    sc = _scenario(2, 3)
    field = _field_for(sc)
    src, tgt = 0, 1
    cand = _rollout_drl_candidate(sc, field, StaticPolicy(), src, tgt,
                                  1e6, "drl:sac:test")
    assert cand.rankable is False
    # no teleport: the final recorded position is NOT snapped to the target
    target = np.asarray(sc.task_positions_m[tgt], dtype=float)
    last = np.asarray(cand.positions_m[-1], dtype=float)
    assert float(np.linalg.norm(last - target)) > sc.limits.arrive_radius_m


# --------------------------------------------------------------------------- #
# 2. RQ3 audit plans must accept a plain candidate dict
# --------------------------------------------------------------------------- #

def test_audit_plans_accepts_dict_library():
    from experiment import run_stc_evaluator_audit as audit
    from segment.stc.library import build_all_pairs_library

    sc = _scenario(2, 3)
    field = _field_for(sc)
    library = build_all_pairs_library(sc, field,
                                      generator_config={"straight_hold": True})
    plans = audit._audit_plans(sc, library.candidates, field, seed=0)
    assert len(plans) >= 2


# --------------------------------------------------------------------------- #
# 3. Independent evaluation must match synchronous feasibility
# --------------------------------------------------------------------------- #

def test_independent_eval_matches_sync_feasibility():
    from experiment.stc_common import evaluate_independent, evaluate_synchronous
    from segment.stc.library import build_all_pairs_library
    from segment.stc.operators import build_initial_solution
    from segment.stc.evaluator import default_objective_spec

    sc = _scenario(2, 4)
    field = _field_for(sc)
    library = build_all_pairs_library(sc, field,
                                      generator_config={"straight_hold": True})
    lib = library.candidates
    spec = default_objective_spec(sc, lib)
    sol = build_initial_solution(sc, lib, seed=0)

    sync = evaluate_synchronous(sc, lib, sol, field, spec)
    indep = evaluate_independent(sc, lib, sol, field, spec)
    assert indep["feasible"] == sync["feasible"]


# --------------------------------------------------------------------------- #
# 4. RQ1 authoritative segment evaluation: outage must matter
# --------------------------------------------------------------------------- #

def _dummy_spec():
    from segment.stc.types import ObjectiveSpec, ReferenceValues
    return ObjectiveSpec(reference=ReferenceValues(t_ref_s=1.0, d_ref_s=1.0,
                                                  source_hash="test"),
                         w_t=0.5, w_d=0.5)


def test_segment_authoritative_outage_is_infeasible():
    from experiment.stc_common import evaluate_segment
    from segment.stc.generators import generate_straight_hold

    # A field with G2A outage 1.0 everywhere: route-level outage exceeds eps_out.
    sc = _scenario(1, 3)
    field = _ConstField(n_sectors=sc.n_sectors, a2g=1.0, g2a=1.0)
    cand = generate_straight_hold(sc, field, 0, 1)
    result = evaluate_segment(sc, cand, field, _dummy_spec())
    assert result["feasible"] is False


def test_segment_authoritative_clean_field_is_feasible():
    from experiment.stc_common import evaluate_segment
    from segment.stc.generators import generate_straight_hold

    sc = _scenario(1, 3)
    field = _field_for(sc)  # outage 0 everywhere
    cand = generate_straight_hold(sc, field, 0, 1)
    result = evaluate_segment(sc, cand, field, _dummy_spec())
    assert result["feasible"] is True


# --------------------------------------------------------------------------- #
# 5. RQ1 common-feasible edge set
# --------------------------------------------------------------------------- #

def test_segment_gate_common_feasible_edges():
    from experiment import run_stc_segment_gate as gate

    sc = _scenario(1, 3)
    field = _field_for(sc)
    out = gate.run_segment_gate(sc, field)
    summary = out["summary"]
    assert "common_feasible_n" in summary
    assert "common_feasible_edges" in summary
    # T/D/O means are over the common-feasible subset only
    for rec in out["records"]:
        if rec["status"] == "evaluated":
            assert rec["t_mean_s"] is None or np.isfinite(rec["t_mean_s"])
            assert rec["common_feasible_n"] == summary["common_feasible_n"]


# --------------------------------------------------------------------------- #
# 6. RQ2 wiring + Shortest-Balanced assignment
# --------------------------------------------------------------------------- #

def test_rq2_includes_shortest_balanced():
    from experiment import run_stc_main as main
    from segment.stc.library import build_all_pairs_library
    from segment.stc.evaluator import default_objective_spec

    sc = _scenario(2, 4)
    field = _field_for(sc)
    library = build_all_pairs_library(sc, field,
                                      generator_config={"straight_hold": True})
    spec = default_objective_spec(sc, library.candidates)
    rq2 = main.run_rq2(sc, field, library.candidates, spec, seed=0)
    assign_methods = {r["method"] for r in rq2["assignment"]}
    assert "shortest_balanced" in assign_methods


def test_main_writes_rq2_artifact(monkeypatch, tmp_path):
    from experiment import run_stc_main as main
    from segment.stc.types import RadioSample

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

    monkeypatch.setattr(main, "load_stc_scenario",
                        lambda u, task_seed, map_seed: _scenario(u, sc.n_tasks))
    monkeypatch.setattr(main, "PerSectorRadioField", Field)

    import json
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "uav_nums": [2], "map_seeds": [7], "task_seeds": [3],
        "method_seeds": [11], "methods": ["proposed_stc_alns"],
        "iterations": 2, "smoke": True, "smoke_tasks": 3,
        "generator_config": {"straight_hold": True},
    }))
    assert main.main(["--config", str(config), "--run-id", "rq2wired"]) == 0
    rq2_path = tmp_path / "results/runs/rq2wired/rq2.json"
    assert rq2_path.exists()
    rq2 = json.loads(rq2_path.read_text())
    assert rq2 and rq2[0].get("assignment") and rq2[0].get("sequence")


# --------------------------------------------------------------------------- #
# 8. RQ5 scale must evaluate STC-ALNS (proposed), not only the open-loop
# --------------------------------------------------------------------------- #

def test_scale_runs_stc_alns():
    from experiment import run_stc_scale as scale_mod

    base = _scenario(2, 3)
    out = scale_mod.run_scale(base, _field_for, seed=0)
    methods = {r.get("method") for r in out["records"]}
    assert "proposed_stc_alns" in methods


# --------------------------------------------------------------------------- #
# 9. CLI flags must be applied
# --------------------------------------------------------------------------- #

def test_resolve_config_applies_smoke_and_tasks():
    from experiment.stc_common import resolve_runner_config
    import argparse

    defaults = {"uav_nums": [2], "smoke": False, "smoke_tasks": 6}
    args = argparse.Namespace(config=None, uav_nums=None, map_seeds=None,
                              task_seeds=None, method_seeds=None, iterations=None,
                              spacing_m=None, smoke=True, smoke_tasks=3)
    cfg = resolve_runner_config(args, defaults)
    assert cfg["smoke"] is True
    assert cfg["smoke_tasks"] == 3


# --------------------------------------------------------------------------- #
# 10. Empty matrix must be rejected
# --------------------------------------------------------------------------- #

def test_lifecycle_writes_failed_manifest_on_interruption(tmp_path, monkeypatch):
    from experiment.stc_common import run_lifecycle
    import json
    monkeypatch.chdir(tmp_path)

    def per_scenario(u, m, t):
        if t == 3:
            raise RuntimeError("boom")
        return [{"u": u, "m": m, "t": t}]

    def summary_fn(records):
        return {"record_count": len(records)}

    matrix = [(2, 7, 1), (2, 7, 2), (2, 7, 3)]
    with pytest.raises(RuntimeError):
        run_lifecycle("stc_test", {"uav_nums": [2]}, "lifecycle", matrix,
                      per_scenario, summary_fn)

    manifest = json.loads((tmp_path / "results/runs/lifecycle/manifest.json").read_text())
    assert manifest["status"] == "failed"
    # partial records from the two completed scenarios are preserved
    records = json.loads((tmp_path / "results/runs/lifecycle/records.json").read_text())
    assert len(records) == 2


def test_empty_matrix_rejected(tmp_path):
    from experiment.stc_common import resolve_runner_config
    import argparse

    defaults = {"uav_nums": [2], "map_seeds": [42], "task_seeds": [42],
                "method_seeds": [0]}

    # empty seed list declared in the config file must be rejected
    import json
    config = tmp_path / "empty.json"
    config.write_text(json.dumps({"uav_nums": [], "map_seeds": [42]}))
    args = argparse.Namespace(config=str(config), uav_nums=None, map_seeds=None,
                              task_seeds=None, method_seeds=None, iterations=None,
                              spacing_m=None, smoke=None, smoke_tasks=None)
    with pytest.raises(ValueError):
        resolve_runner_config(args, defaults)
