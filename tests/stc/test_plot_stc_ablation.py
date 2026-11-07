"""Statistics and pairing checks for the RQ5 ablation figure."""

import numpy as np
import pytest

from plots.plot_stc_ablation import (
    ARM_ORDER,
    _exact_sign_flip_p,
    _holm_adjust,
    summarize,
)


def test_exact_sign_flip_reaches_one_over_two_to_n():
    assert _exact_sign_flip_p(np.ones(5)) == 1.0 / 32.0
    assert _exact_sign_flip_p(-np.ones(5)) == 1.0


def test_holm_adjust_is_monotone_and_clipped():
    adjusted = _holm_adjust([0.01, 0.04, 0.03, 1.0])

    np.testing.assert_allclose(adjusted, [0.04, 0.09, 0.09, 1.0])


def test_summarize_pairs_each_arm_with_same_seed_full():
    runs = []
    for seed in range(5):
        by_seed = {}
        records = []
        values = {"full": 0.8 + seed * 0.001}
        for arm in ARM_ORDER[1:]:
            values[arm] = values["full"] + (0.01 if arm == "path_switch" else 0.0)
        by_seed[seed] = values
        records.extend(
            {"method_seed": seed, "ablation": arm, "feasible": True}
            for arm in ARM_ORDER)
        runs.append({"records": records, "by_seed": by_seed})

    stats = summarize(runs)
    rows = {row["ablation"]: row for row in stats["rows"]}

    assert stats["n_method_seeds"] == 5
    assert rows["full"]["paired_delta_mean"] == 0.0
    assert rows["path_switch"]["paired_delta_mean"] == pytest.approx(0.01)
    assert rows["path_switch"]["worse_seeds"] == 5
    assert rows["ebtas_init"]["paired_delta_mean"] == 0.0
    assert rows["ebtas_init"]["tied_seeds"] == 5
