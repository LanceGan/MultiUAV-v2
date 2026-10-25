"""Task 2 Step 3: Pareto/diversity pruning on flight time, upload delay,
outage, and sector signature."""

import numpy as np

from segment.stc.library import pareto_prune
from segment.stc.types import SegmentCandidate


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
