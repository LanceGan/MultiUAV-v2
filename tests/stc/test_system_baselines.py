"""Task 8: complete-system baseline adapters (six named end-to-end methods).

Every method must be evaluated by the SAME synchronous evaluator, start at
``t=0``, and use identical scenario/library/reference/failure rules.  Only the
Proposed STC-ALNS method may use synchronous conflict-feedback refinement; the
five baselines are isolated single-shot methods.

The isolation tests are the load-bearing part: they prove that no baseline
silently calls the proposed conflict-feedback entry point (``STCALNS.solve``)
and that Library-Open-Loop and Proposed STC-ALNS share the same initial
solution (only the conflict-feedback refinement differs).
"""

import numpy as np
import pytest
import hashlib
from dataclasses import replace
from pathlib import Path

from segment.stc.generators import generate_comm_waypoints, generate_straight_hold
from segment.stc.system_baselines import (
    SYSTEM_BASELINE_KEYS,
    SystemBaselineResult,
    run_system_baseline,
)
from segment.stc.types import RadioSample, STCScenario, ScenarioLimits


def _limits():
    return ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                          arrive_radius_m=15.0, max_wait_s=60.0,
                          mission_deadline_s=7200.0, bandwidth_hz=1e6,
                          uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)


def _scenario(n_uav=2, n_tasks=5):
    positions = tuple((60.0 + (i % 3) * 120.0, 60.0 + (i // 3) * 120.0)
                      for i in range(n_tasks))
    return STCScenario(n_uav=n_uav, n_tasks=n_tasks, task_positions_m=positions,
                       task_data_bits=tuple([1e6] * n_tasks),
                       depot_m=(0.0, 0.0), terminal_m=(400.0, 400.0),
                       limits=_limits(), length_m=400.0, width_m=400.0,
                       n_bs=4, task_seed=42, map_seed=0)


class _ConstField:
    def __init__(self, n_sectors):
        self.n_sectors = n_sectors
        self.world_hash = "const-field"

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.ones(self.n_sectors),
                           g2a_outage=np.zeros(self.n_sectors))


@pytest.fixture(autouse=True)
def _fast_radio(monkeypatch):
    # Mirror tests/stc/test_jia_sequence_adapter.py: keep the legacy Jia
    # time-energy adapter deterministic and fast by pinning the radio queries.
    from sequence import radio_path
    from segment.stc import sequencing
    monkeypatch.setattr(sequencing, "_configure_maps", lambda sc: "const-field")
    monkeypatch.setattr(radio_path.radio_map_A2G, "getPointDateRate",
                        lambda loc: 10.0)
    monkeypatch.setattr(radio_path.radio_map_G2A, "getPointMiniOutage",
                        lambda loc: ([0.1], [10.0]))


@pytest.fixture
def setup(tmp_path):
    sc = _scenario(2, 3)
    field = _ConstField(sc.n_sectors)
    # plain candidate dict (the evaluator/solver convention) with both a
    # Straight-Hold candidate and communication-aware relay candidates per edge.
    lib = {}
    checkpoint = tmp_path / "sac.pt"
    checkpoint.write_bytes(b"test SAC checkpoint fixture")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src == tgt:
                continue
            lib[(src, tgt)] = [generate_straight_hold(sc, field, src, tgt)]
            lib[(src, tgt)].extend(
                generate_comm_waypoints(sc, field, src, tgt, grid_shape=(2, 2)))
            straight = lib[(src, tgt)][0]
            lib[(src, tgt)].extend([
                replace(straight, generator="astar", candidate_id="graph_%d_%d" % (src, tgt)),
                replace(straight, generator="drl:SAC:" + checkpoint_hash,
                        candidate_id="sac_%d_%d" % (src, tgt)),
            ])
    field.checkpoint_path = str(checkpoint)
    return sc, lib, field


def _run(key, sc, lib, field, **kwargs):
    if key == "gatsac_style":
        kwargs["checkpoint_path"] = field.checkpoint_path
    return run_system_baseline(key, sc, lib, field, **kwargs)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def test_registry_lists_six_methods_in_order():
    assert SYSTEM_BASELINE_KEYS == (
        "region_distance_straight", "shortest_balanced_graph",
        "jia_ebtas_style", "gatsac_style",
        "library_open_loop", "proposed_stc_alns")


def test_unknown_method_raises(setup):
    sc, lib, field = setup
    with pytest.raises(ValueError):
        run_system_baseline("not_a_method", sc, lib, field)


# --------------------------------------------------------------------------- #
# Coverage + simultaneity + identical reference
# --------------------------------------------------------------------------- #

def test_every_method_covers_all_tasks_exactly_once(setup):
    sc, lib, field = setup
    for key in SYSTEM_BASELINE_KEYS:
        r = _run(key, sc, lib, field, iterations=5)
        assert isinstance(r, SystemBaselineResult)
        flat = sorted(t for route in r.solution.routes for t in route)
        assert flat == list(range(sc.n_tasks))


def test_every_method_starts_airborne_at_slot_zero(setup):
    sc, lib, field = setup
    for key in SYSTEM_BASELINE_KEYS:
        r = _run(key, sc, lib, field, iterations=5)
        for tl in r.result.timelines:
            if tl.slots:
                assert tl.slots[0].time_start_s == 0.0
                assert tl.slots[0].airborne is True


def test_all_methods_share_identical_reference(setup):
    sc, lib, field = setup
    results = {k: _run(k, sc, lib, field, iterations=5)
               for k in SYSTEM_BASELINE_KEYS}
    ref0 = results[SYSTEM_BASELINE_KEYS[0]].objective_spec.reference
    for k in SYSTEM_BASELINE_KEYS[1:]:
        ref = results[k].objective_spec.reference
        assert ref.t_ref_s == ref0.t_ref_s
        assert ref.d_ref_s == ref0.d_ref_s
        assert ref.source_hash == ref0.source_hash


def test_baselines_are_single_shot_full_evaluations(setup):
    sc, lib, field = setup
    for key in SYSTEM_BASELINE_KEYS:
        r = _run(key, sc, lib, field, iterations=5)
        if key == "proposed_stc_alns":
            assert r.evaluator_calls["full"] >= 1
            assert r.provenance["iterations_executed"] == 5
        else:
            assert r.evaluator_calls == {"full": 1}


# --------------------------------------------------------------------------- #
# Isolation (the load-bearing part)
# --------------------------------------------------------------------------- #

class _ProposedFeedbackCalled(Exception):
    pass


def test_baselines_never_call_proposed_feedback(setup, monkeypatch):
    sc, lib, field = setup
    from segment.stc.solver import STCALNS

    def boom_solve(self, *args, **kwargs):
        raise _ProposedFeedbackCalled()

    monkeypatch.setattr(STCALNS, "solve", boom_solve)

    # All five baselines must succeed without ever entering the proposed solver.
    for key in [k for k in SYSTEM_BASELINE_KEYS if k != "proposed_stc_alns"]:
        _run(key, sc, lib, field)

    # The proposed method is the only one that enters the conflict-feedback
    # refinement entry point.
    with pytest.raises(_ProposedFeedbackCalled):
        run_system_baseline("proposed_stc_alns", sc, lib, field, iterations=5)


def test_library_open_loop_and_proposed_share_initial_solution(setup):
    sc, lib, field = setup
    lo = run_system_baseline("library_open_loop", sc, lib, field)
    pr = run_system_baseline("proposed_stc_alns", sc, lib, field, iterations=5)
    assert lo.solution == pr.provenance["initial_solution"]


# --------------------------------------------------------------------------- #
# Claim integrity
# --------------------------------------------------------------------------- #

def test_gatsac_is_labeled_as_style_reproduction(setup):
    sc, lib, field = setup
    r = _run("gatsac_style", sc, lib, field)
    p = r.provenance
    assert p["trajectory_generators"] == [
        "drl:SAC:" + hashlib.sha256(Path(field.checkpoint_path).read_bytes()).hexdigest()]
    assert p["checkpoint"]["path"] == field.checkpoint_path
    assert p["checkpoint"]["sha256"] == p["trajectory_generators"][0].split(":")[-1]
    assert all(cid.startswith("sac_") for ids in r.solution.candidate_ids for cid in ids)
    assert "SAC" in p["trajectory_note"]
    assert "style reproduction" in p["trajectory_note"]
    assert p["proposed"] is False


def test_jia_is_labeled_as_style_reproduction(setup):
    sc, lib, field = setup
    r = run_system_baseline("jia_ebtas_style", sc, lib, field)
    p = r.provenance
    assert set(p["trajectory_generators"]) <= {"comm_direct", "comm_waypoint"}
    assert "style" in p["style_note"]
    assert p["proposed"] is False


@pytest.mark.parametrize("method,allowed", [
    ("region_distance_straight", {"straight_hold"}),
    ("shortest_balanced_graph", {"astar", "slsqp_refined"}),
    ("jia_ebtas_style", {"comm_direct", "comm_waypoint"}),
])
def test_required_family_never_falls_back(setup, method, allowed):
    sc, lib, field = setup
    missing = {edge: [c for c in cs if c.generator not in allowed]
               for edge, cs in lib.items()}
    with pytest.raises(RuntimeError, match="unavailable"):
        _run(method, sc, missing, field)


@pytest.mark.parametrize("method", ["region_distance_straight", "shortest_balanced_graph"])
def test_sac_only_library_cannot_impersonate_other_families(setup, method):
    sc, lib, field = setup
    sac_only = {edge: [c for c in cs if c.generator.startswith("drl:SAC:")]
                for edge, cs in lib.items()}
    with pytest.raises(RuntimeError, match="unavailable"):
        _run(method, sc, sac_only, field)


def test_gatsac_without_sac_is_explicitly_unavailable(setup):
    sc, lib, field = setup
    deterministic = {edge: [c for c in cs if not c.generator.startswith("drl:")]
                     for edge, cs in lib.items()}
    with pytest.raises(RuntimeError, match="unavailable"):
        _run("gatsac_style", sc, deterministic, field)


def test_gatsac_requires_matching_checkpoint(setup, tmp_path):
    sc, lib, field = setup
    with pytest.raises(RuntimeError, match="unavailable"):
        run_system_baseline("gatsac_style", sc, lib, field)
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"other checkpoint")
    with pytest.raises(RuntimeError, match="unavailable"):
        run_system_baseline("gatsac_style", sc, lib, field, checkpoint_path=str(wrong))


def test_graph_communication_family_requires_explicit_configuration(setup):
    sc, lib, field = setup
    comm_only = {edge: [c for c in cs if c.generator == "comm_direct"]
                 for edge, cs in lib.items()}
    r = _run("shortest_balanced_graph", sc, comm_only, field,
             graph_generators=["comm_direct"])
    assert r.provenance["trajectory_generators"] == ["comm_direct"]


def test_duplicate_candidate_ids_cannot_resolve_to_wrong_family(setup):
    sc, lib, field = setup
    colliding = {}
    for edge, candidates in lib.items():
        straight = candidates[0]
        sac = next(c for c in candidates if c.generator.startswith("drl:SAC:"))
        colliding[edge] = [straight, replace(sac, candidate_id=straight.candidate_id)]
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        _run("gatsac_style", sc, colliding, field)
