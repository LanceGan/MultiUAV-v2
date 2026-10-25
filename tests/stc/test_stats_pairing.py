"""Task 7: hierarchical paired statistics — explicit pairs, completeness,
finite metrics, and declared failure policies."""

import numpy as np
import pytest

from segment.stc.stats import (
    hierarchical_paired_summary,
    prespecified_contrasts,
)


def _rec(scen, method, rep, metric, feasible=True):
    return {"map_seed": 0, "task_seed": scen, "method_seed": rep,
            "method": method, "metric": metric, "feasible": feasible}


def test_requires_distinct_methods():
    with pytest.raises(ValueError):
        hierarchical_paired_summary([_rec(0, "A", 0, 1.0)], "A", "A")


def test_rejects_duplicate_replicate_keys():
    recs = [_rec(0, "A", 0, 1.0), _rec(0, "A", 0, 1.0),
            _rec(0, "B", 0, 2.0)]
    with pytest.raises(ValueError):
        hierarchical_paired_summary(recs, "A", "B")


def test_rejects_disjoint_replicate_sets():
    recs = [_rec(0, "A", 0, 1.0), _rec(0, "B", 10, 2.0)]
    with pytest.raises(ValueError):
        hierarchical_paired_summary(recs, "A", "B")


def test_rejects_incomplete_replicate_pairing():
    recs = [_rec(0, "A", 0, 1.0), _rec(0, "A", 1, 1.0), _rec(0, "A", 2, 1.0),
            _rec(0, "B", 0, 2.0), _rec(0, "B", 1, 2.0)]
    with pytest.raises(ValueError):
        hierarchical_paired_summary(recs, "A", "B")


def test_rejects_scenario_present_for_one_method_only():
    # scenario 1 has method A but no method B: must not silently narrow the
    # paired set to scenario 0.
    recs = [_rec(0, "A", 0, 1.0), _rec(1, "A", 0, 1.0),
            _rec(0, "B", 0, 2.0)]
    with pytest.raises(ValueError):
        hierarchical_paired_summary(recs, "A", "B")


def test_rejects_scenario_missing_for_method_a():
    recs = [_rec(0, "A", 0, 1.0),
            _rec(0, "B", 0, 2.0), _rec(1, "B", 0, 2.0)]
    with pytest.raises(ValueError):
        hierarchical_paired_summary(recs, "A", "B")


def test_rejects_nan_metric():
    recs = [_rec(0, "A", 0, np.nan), _rec(0, "B", 0, 2.0)]
    with pytest.raises(ValueError):
        hierarchical_paired_summary(recs, "A", "B")


def test_rejects_inf_metric():
    recs = [_rec(0, "A", 0, np.inf), _rec(0, "B", 0, 2.0)]
    with pytest.raises(ValueError):
        hierarchical_paired_summary(recs, "A", "B")


def test_success_rate_policy_over_all_instances():
    recs = []
    for scen in range(3):
        for rep in range(4):
            recs.append(_rec(scen, "A", rep, 5.0, feasible=(rep < 3)))
            recs.append(_rec(scen, "B", rep, 7.0, feasible=(rep < 1)))
    s = hierarchical_paired_summary(recs, "A", "B", failure_policy="success_rate")
    assert s.mean_diff == pytest.approx(0.5)
    assert s.n == 3


def test_success_rate_allows_disjoint_replicates():
    recs = [_rec(0, "A", 0, 1.0, feasible=True),
            _rec(0, "B", 10, 2.0, feasible=False)]
    s = hierarchical_paired_summary(recs, "A", "B", failure_policy="success_rate")
    assert s.mean_diff == pytest.approx(1.0)


def test_common_feasible_drops_failures_by_policy():
    recs = []
    for scen in range(2):
        for rep in range(4):
            recs.append(_rec(scen, "A", rep, 10.0, feasible=True))
            recs.append(_rec(scen, "B", rep, 8.0, feasible=(rep < 2)))
    s = hierarchical_paired_summary(recs, "A", "B", failure_policy="common_feasible")
    assert s.mean_diff == pytest.approx(2.0)
    assert s.n == 2


def test_penalized_all_instance_penalty():
    recs = []
    for scen in range(2):
        for rep in range(2):
            recs.append(_rec(scen, "A", rep, 10.0, feasible=True))
            recs.append(_rec(scen, "B", rep, 8.0, feasible=(rep < 1)))
    s = hierarchical_paired_summary(recs, "A", "B", failure_policy="penalized",
                                    penalty=100.0)
    assert s.mean_diff == pytest.approx(10.0 - (8.0 + 100.0) / 2.0)


def test_holm_applies_to_prespecified_contrasts():
    recs = []
    for scen in range(3):
        for rep in range(4):
            recs.append(_rec(scen, "A", rep, 10.0 + scen))
            recs.append(_rec(scen, "B", rep, 9.0 + scen))
            recs.append(_rec(scen, "C", rep, 11.0 + scen))
    summaries, adjusted = prespecified_contrasts(
        recs, {"A_vs_B": ("A", "B"), "A_vs_C": ("A", "C")})
    assert "A_vs_B" in summaries and "A_vs_C" in summaries
    assert set(adjusted) == {"A_vs_B", "A_vs_C"}
    assert all(0.0 <= v <= 1.0 for v in adjusted.values())
