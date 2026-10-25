"""Task 2 Step 3: Pareto/diversity pruning on flight time, upload delay,
outage, and sector signature."""

import numpy as np

from segment.stc.library import build_all_pairs_library, pareto_prune, _sector_signature
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


class _TwoZoneField:
    """Best sector switches at x=2000 (left -> sector 0, right -> sector 1).

    Max efficiency is identical on both sides, so a relay into the right zone is
    strictly slower (longer flight) at equal upload/outage — it can only survive
    Pareto pruning via sector diversity.
    """

    def __init__(self, n_sectors=4):
        self.n_sectors = n_sectors
        self.world_hash = "two-zone-hash"

    def sample(self, pos):
        x = float(pos[0])
        if x < 2000.0:
            a2g = np.array([3.0, 0.5, 0.5, 0.5])
        else:
            a2g = np.array([0.5, 3.0, 0.5, 0.5])
        outage = np.array([0.1, 0.1, 0.1, 0.1])
        return RadioSample(a2g_efficiency=a2g, g2a_outage=outage)


def _cand(cid, flight_steps, rate, data, outage, sectors, rankable=True):
    n_sectors = 4
    T = max(int(flight_steps), 1)
    return SegmentCandidate(
        candidate_id=cid, source=0, target=1, generator="synth",
        positions_m=np.zeros((T, 2)),
        full_rate_bps=np.full((T, n_sectors), float(rate)),
        sector_ids=np.asarray(sectors, dtype=int),
        g2a_outage=np.full((T, n_sectors), float(outage)),
        flight_steps=T, hold_position_m=(0.0, 0.0),
        source_data_bits=float(data), rankable=rankable)


def test_pareto_prune_removes_strictly_dominated():
    fast = _cand("fast", 10, 2.0, 20.0, 0.1, [0] * 10)
    slow = _cand("slow", 20, 2.0, 20.0, 0.1, [0] * 20)
    pruned = pareto_prune([slow, fast], max_candidates=10, min_sector_distance=0.0,
                          delta_t_s=0.5)
    ids = {c.candidate_id for c in pruned}
    assert ids == {"fast"}


def test_pareto_prune_prunes_on_outage_too():
    low_out = _cand("low_out", 10, 2.0, 20.0, 0.05, [0] * 10)
    high_out = _cand("high_out", 10, 2.0, 20.0, 0.4, [0] * 10)
    pruned = pareto_prune([high_out, low_out], max_candidates=10, min_sector_distance=0.0,
                          delta_t_s=0.5)
    ids = {c.candidate_id for c in pruned}
    assert ids == {"low_out"}


def test_pareto_prune_retains_sector_diverse_slower_candidate():
    fast = _cand("fast", 10, 2.0, 20.0, 0.1, [0] * 10)
    slow_other_sector = _cand("slow_s1", 20, 2.0, 20.0, 0.1, [1] * 20)
    pruned = pareto_prune([fast, slow_other_sector], max_candidates=10,
                          min_sector_distance=1.0, delta_t_s=0.5)
    ids = {c.candidate_id for c in pruned}
    assert ids == {"fast", "slow_s1"}


def test_pareto_prune_rejects_exact_duplicates():
    a = _cand("a", 10, 2.0, 20.0, 0.1, [0] * 10)
    b = _cand("b", 10, 2.0, 20.0, 0.1, [0] * 10)
    pruned = pareto_prune([a, b], max_candidates=10, min_sector_distance=0.0,
                          delta_t_s=0.5)
    assert len(pruned) == 1


def test_pareto_prune_keeps_nondominated_tradeoff():
    # faster but higher outage vs slower but lower outage -> both non-dominated
    a = _cand("a", 10, 2.0, 20.0, 0.4, [0] * 10)
    b = _cand("b", 20, 2.0, 20.0, 0.05, [0] * 20)
    pruned = pareto_prune([a, b], max_candidates=10, min_sector_distance=0.0,
                          delta_t_s=0.5)
    assert {c.candidate_id for c in pruned} == {"a", "b"}


def test_pareto_prune_respects_max_candidates():
    cands = [_cand("c%d" % i, 10 + i, 2.0, 20.0, 0.1, [i % 4] * (10 + i))
             for i in range(20)]
    pruned = pareto_prune(cands, max_candidates=5, min_sector_distance=0.0,
                          delta_t_s=0.5)
    assert len(pruned) <= 5


def test_build_default_path_retains_sector_diverse_candidate():
    """The shipped default (no explicit pareto_kwargs) keeps a slower candidate
    whose sector occupancy differs materially from the fast direct candidate."""
    sc = _scenario(3)
    field = _TwoZoneField()
    cfg = {
        "straight_hold": True,
        "straight_relaxation": True,
        "comm_waypoints": {"grid_shape": (4, 4), "max_waypoints": 1},
    }
    # default pareto_kwargs: max_candidates=5, min_sector_distance=0.5
    lib = build_all_pairs_library(sc, field, generator_config=cfg)
    rankable = [c for c in lib.candidates[(0, 1)] if c.rankable]
    fast = min(rankable, key=lambda c: c.flight_steps)
    # direct path is entirely left zone -> sector 0
    assert int(np.argmax(_sector_signature(fast))) == 0
    # a strictly slower relay candidate that reaches the right zone (sector 1)
    # must still be present thanks to sector diversity retention
    diverse = [c for c in rankable if int(np.argmax(_sector_signature(c))) != 0]
    assert diverse, "no sector-diverse candidate survived the default build path"
    assert all(c.flight_steps > fast.flight_steps for c in diverse)
