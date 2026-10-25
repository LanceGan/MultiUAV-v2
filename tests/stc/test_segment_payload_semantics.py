"""Task 2 Step 1/2: source-payload semantics + candidate feasibility."""

import numpy as np

from segment.stc.generators import drl_generator_label
from segment.stc.library import build_all_pairs_library
from segment.stc.types import RadioSample, STCScenario, ScenarioLimits, SegmentCandidate


def _scenario(n_tasks=3):
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    positions = tuple((800.0 + i * 1000.0, 1500.0) for i in range(n_tasks))
    data_bits = tuple(float(2e7 + i * 1e6) for i in range(n_tasks))
    return STCScenario(n_uav=2, n_tasks=n_tasks, task_positions_m=positions,
                       task_data_bits=data_bits, depot_m=(0.0, 500.0),
                       terminal_m=(4000.0, 3000.0), limits=limits, length_m=4000.0,
                       width_m=4000.0, n_bs=4, task_seed=42, map_seed=0)


class _ConstField:
    def __init__(self, n_sectors=4):
        self.n_sectors = n_sectors
        self.world_hash = "const-field-hash"

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.ones(self.n_sectors),
                           g2a_outage=np.zeros(self.n_sectors))


def _by_gen(cands, gen):
    return [c for c in cands if c.generator == gen]


def _build(n_tasks=3, generator_config=None, field=None, drl=None, pareto_kwargs=None):
    sc = _scenario(n_tasks)
    field = field or _ConstField()
    lib = build_all_pairs_library(sc, field, generator_config=generator_config,
                                  drl_candidates=drl, pareto_kwargs=pareto_kwargs)
    return sc, lib


# ---- source-payload semantics ----

def test_depot_to_task_edges_carry_zero_payload():
    sc, lib = _build(3, generator_config={"straight_hold": True})
    k = sc.n_tasks
    for i in range(k):
        hold = _by_gen(lib.candidates[(-1, i)], "straight_hold")
        assert hold and all(c.source_data_bits == 0.0 for c in hold)


def test_task_to_terminal_edges_carry_source_payload():
    sc, lib = _build(3, generator_config={"straight_hold": True})
    k = sc.n_tasks
    for i in range(k):
        hold = _by_gen(lib.candidates[(i, k)], "straight_hold")
        assert hold and all(c.source_data_bits == sc.task_data_bits[i] for c in hold)


def test_task_to_task_edge_carries_source_not_target_payload():
    sc, lib = _build(3, generator_config={"straight_hold": True})
    i, j = 0, 2
    hold = _by_gen(lib.candidates[(i, j)], "straight_hold")[0]
    assert hold.source_data_bits == sc.task_data_bits[i]
    assert hold.source_data_bits != sc.task_data_bits[j]


def test_reversed_edges_carry_different_payloads():
    sc, lib = _build(3, generator_config={"straight_hold": True})
    i, j = 0, 1
    fwd = _by_gen(lib.candidates[(i, j)], "straight_hold")[0]
    rev = _by_gen(lib.candidates[(j, i)], "straight_hold")[0]
    assert fwd.source_data_bits == sc.task_data_bits[i]
    assert rev.source_data_bits == sc.task_data_bits[j]
    assert fwd.source_data_bits != rev.source_data_bits


# ---- candidate feasibility ----

def test_edge_candidates_include_all_generator_families():
    from segment.stc.library import _generate_edge_candidates
    cfg = {
        "straight_hold": True,
        "straight_relaxation": True,
        "comm_waypoints": {"grid_shape": (4, 4), "max_waypoints": 1},
        "astar": {"grid": 6},
        "slsqp_refine": {"max_waypoints": 1, "max_refinements": 3},
    }
    sc = _scenario(3)
    field = _ConstField()
    rankable, non_rankable = _generate_edge_candidates(sc, field, 0, 1, cfg)
    gens = {c.generator for c in rankable}
    assert "straight_hold" in gens
    assert "comm_direct" in gens
    assert "comm_waypoint" in gens
    assert "astar" in gens
    assert "slsqp_refined" in gens
    assert any(c.generator == "straight_relaxation" and c.rankable is False
               for c in non_rankable)


def test_relaxation_reference_is_non_rankable():
    cfg = {"straight_hold": True, "straight_relaxation": True}
    sc, lib = _build(3, generator_config=cfg)
    for edge, cands in lib.candidates.items():
        for c in cands:
            if c.generator == "straight_relaxation":
                assert c.rankable is False


def test_drl_candidates_are_labeled_by_model_and_checkpoint():
    label = drl_generator_label("td3", "abc123def")
    assert label.startswith("drl:")
    assert "td3" in label and "abc123def" in label

    sc = _scenario(3)
    field = _ConstField()
    T = 5
    drl = SegmentCandidate(
        candidate_id="td3_0_1_abc123def", source=0, target=1, generator=label,
        positions_m=np.zeros((T, 2)), full_rate_bps=np.ones((T, field.n_sectors)),
        sector_ids=np.zeros(T, dtype=int), g2a_outage=np.zeros((T, field.n_sectors)),
        flight_steps=T, hold_position_m=(0.0, 0.0),
        source_data_bits=float(sc.task_data_bits[0]), rankable=True)
    lib = build_all_pairs_library(sc, field, generator_config={"straight_hold": True},
                                  drl_candidates={(0, 1): [drl]})
    assert label in {c.generator for c in lib.candidates[(0, 1)]}
