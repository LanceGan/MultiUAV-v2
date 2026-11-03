"""Determinism regression: parallel library build must equal sequential, bit-for-bit.

The all-pairs segment library is the dominant cost of a U=3 scenario.  It is
embarrassingly parallel (every directed edge is independent) and contains no
randomness (all generators are pure functions of scenario + field + edge).  The
parallel path therefore must reproduce the sequential result exactly; this test
locks that contract so a future change cannot silently perturb the numerical
world.
"""

import numpy as np

import experiment.run_stc_main as runner
from segment.stc.evaluator import default_objective_spec
from segment.stc.library import build_all_pairs_library
from segment.stc.radio import PerSectorRadioField
from segment.stc.types import ScenarioLimits, STCScenario


def _tiny_scenario():
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    return STCScenario(
        n_uav=2, n_tasks=3,
        task_positions_m=((600.0, 600.0), (1200.0, 600.0), (1800.0, 600.0)),
        task_data_bits=(1e7, 2e7, 3e7),
        depot_m=(0.0, 0.0), terminal_m=(2400.0, 600.0),
        limits=limits, length_m=4000.0, width_m=4000.0,
        n_bs=4, task_seed=42, map_seed=0,
    )


def _gen_cfg():
    return {"straight_hold": True, "straight_relaxation": True,
            "comm_waypoints": {"grid_shape": (4, 4), "max_waypoints": 1}}


def _assert_candidate_equal(x, y):
    assert x.candidate_id == y.candidate_id
    assert x.generator == y.generator
    assert x.source == y.source and x.target == y.target
    assert x.flight_steps == y.flight_steps
    assert tuple(x.hold_position_m) == tuple(y.hold_position_m)
    assert x.source_data_bits == y.source_data_bits
    assert x.rankable == y.rankable
    np.testing.assert_array_equal(x.positions_m, y.positions_m)
    np.testing.assert_array_equal(x.full_rate_bps, y.full_rate_bps)
    np.testing.assert_array_equal(x.sector_ids, y.sector_ids)
    np.testing.assert_array_equal(x.g2a_outage, y.g2a_outage)


def test_build_all_pairs_library_parallel_equals_sequential():
    sc = _tiny_scenario()
    field = PerSectorRadioField.from_scenario(sc, spacing_m=1000.0)
    cfg = _gen_cfg()

    seq = build_all_pairs_library(sc, field, generator_config=cfg, workers=1)
    par = build_all_pairs_library(sc, field, generator_config=cfg, workers=2)

    assert seq.map_seed == par.map_seed
    assert seq.task_seed == par.task_seed
    assert seq.world_hash == par.world_hash
    assert set(seq.candidates) == set(par.candidates)
    for edge in seq.candidates:
        ca, cb = seq.candidates[edge], par.candidates[edge]
        assert len(ca) == len(cb), "edge %r candidate count differs" % (edge,)
        for x, y in zip(ca, cb):
            _assert_candidate_equal(x, y)


def _grid_ctx():
    """A tiny method-grid context (2 methods x 2 seeds, few iterations)."""
    sc = _tiny_scenario()
    field = PerSectorRadioField.from_scenario(sc, spacing_m=1000.0)
    library = build_all_pairs_library(
        sc, field,
        generator_config={"straight_hold": True, "straight_relaxation": True},
        workers=1)
    reference = default_objective_spec(sc, library.candidates)
    cfg = {"methods": ["library_open_loop", "proposed_stc_alns"],
           "iterations": 5, "sac_checkpoint": None}
    solver_config = {"rho": 0.1, "T0": 0.1, "cooling": 0.995, "segment": 100,
                     "audit_interval": 50, "score_new_best": 3.0,
                     "score_improve": 1.0, "score_accept": 0.5, "score_reject": 0.0}
    return {
        "sc": sc, "field": field, "library": library, "reference": reference,
        "cfg": cfg, "unavailable": {}, "solver_config": solver_config,
        "options": {"checkpoint_path": None, "graph_generators": []},
        "inputs": {"library": "test-lib", "scenario": "test-sc",
                   "task": "test-task", "map": "test-map"},
        "prefix": "2_0_0", "uav_num": 2, "map_seed": 0, "task_seed": 0,
    }


def _normalized_records(records):
    """Serialize records with the runner's own encoder, normalizing wall-clock."""
    out = []
    for r in records:
        r2 = dict(r)
        r2["runtime_s"] = 0.0
        out.append(r2)
    return runner._json_bytes(out)


def test_run_method_grid_parallel_equals_sequential():
    ctx = _grid_ctx()
    seeds = [0, 1]
    seq_records, seq_sol, seq_tl, seq_fail = runner._run_method_grid(seeds, ctx, workers=1)
    par_records, par_sol, par_tl, par_fail = runner._run_method_grid(seeds, ctx, workers=2)
    assert seq_fail is None and par_fail is None
    assert _normalized_records(seq_records) == _normalized_records(par_records)
