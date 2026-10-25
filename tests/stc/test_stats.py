"""Task 14: statistical protocol."""

import numpy as np
import pytest

from segment.stc.stats import hierarchical_paired_summary, holm_adjust, paired_summary


def test_paired_summary_hand_calculation():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [0.0, 1.0, 2.0, 3.0]
    s = paired_summary(a, b, permutations=500, seed=0)
    assert s.mean_diff == pytest.approx(1.0)
    assert s.n == 4


def test_paired_summary_rejects_mismatched_pairs():
    with pytest.raises(ValueError):
        paired_summary([1.0, 2.0], [1.0])


def test_paired_summary_rejects_nan():
    with pytest.raises(ValueError):
        paired_summary([1.0, np.nan], [1.0, 2.0])


def test_paired_summary_deterministic():
    a = np.arange(10, dtype=float)
    b = np.arange(10, dtype=float)[::-1]
    s1 = paired_summary(a, b, permutations=500, seed=0)
    s2 = paired_summary(a, b, permutations=500, seed=0)
    assert s1 == s2


def test_holm_adjust_monotone():
    p = {"a": 0.01, "b": 0.03, "c": 0.05}
    adj = holm_adjust(p)
    assert adj["a"] <= adj["b"] <= adj["c"]
    assert all(0.0 <= v <= 1.0 for v in adj.values())


def test_hierarchical_aggregates_replicates():
    recs = []
    for scen in range(3):
        for rep in range(4):
            recs.append({"map_seed": 0, "task_seed": scen, "method_seed": rep,
                         "method": "A", "metric": 10.0 + scen})
            recs.append({"map_seed": 0, "task_seed": scen, "method_seed": rep,
                         "method": "B", "metric": 9.0 + scen})
    s = hierarchical_paired_summary(recs, "A", "B")
    assert s.mean_diff == pytest.approx(1.0)
