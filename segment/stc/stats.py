"""Paired statistical protocol (Task 14).

Paired bootstrap/permutation inference for method comparison, hierarchical
aggregation of replicate seeds within a scenario, and Holm-Bonferroni
correction for multiple comparisons.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PairedSummary:
    mean_diff: float
    median_diff: float
    std_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int


def paired_summary(a, b, confidence=0.95, permutations=10000, seed=0):
    """Paired summary of a-b (bootstrap CI + two-sided permutation p-value)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("a and b must have equal length")
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        raise ValueError("NaN values not allowed")
    n = len(a)
    d = a - b
    mean_diff = float(np.mean(d))
    median_diff = float(np.median(d))
    std_diff = float(np.std(d, ddof=1)) if n > 1 else 0.0

    rng = np.random.RandomState(seed)
    alpha = (1.0 - confidence) / 2.0
    boot = np.array([np.mean(rng.choice(d, size=n, replace=True))
                     for _ in range(permutations)])
    ci_low = float(np.percentile(boot, 100 * alpha))
    ci_high = float(np.percentile(boot, 100 * (1 - alpha)))

    count = 0
    for _ in range(permutations):
        perm_mean = np.mean(rng.choice([-1.0, 1.0], size=n) * d)
        if abs(perm_mean) >= abs(mean_diff):
            count += 1
    p_value = (count + 1.0) / (permutations + 1.0)

    return PairedSummary(mean_diff, median_diff, std_diff, ci_low, ci_high, p_value, n)


def hierarchical_paired_summary(records, scenario_keys=("map_seed", "task_seed"),
                                replicate_key="method_seed"):
    """Aggregate replicates within each scenario, then compare methods A/B."""
    # records: list of dicts with metric + scenario_keys + replicate_key + "method"
    groups = {}
    for r in records:
        key = tuple(r[k] for k in scenario_keys) + (r["method"],)
        groups.setdefault(key, []).append(r["metric"])

    scen_means = {}
    for key, vals in groups.items():
        scen_means.setdefault(key[:-1], {})[key[-1]] = float(np.mean(vals))

    a_vals, b_vals = [], []
    for scen in sorted(scen_means):
        methods = scen_means[scen]
        names = sorted(methods)
        if len(names) != 2:
            continue
        a_vals.append(methods[names[0]])
        b_vals.append(methods[names[1]])
    if not a_vals:
        return None
    return paired_summary(a_vals, b_vals)


def holm_adjust(p_values):
    """Holm-Bonferroni corrected p-values (monotone)."""
    names = list(p_values)
    ps = np.array([p_values[n] for n in names], dtype=float)
    order = np.argsort(ps)
    m = len(ps)
    adjusted = {}
    for rank, idx in enumerate(order):
        adjusted[names[idx]] = min(1.0, ps[idx] * (m - rank))
    prev = 0.0
    for idx in order:
        adjusted[names[idx]] = max(adjusted[names[idx]], prev)
        prev = adjusted[names[idx]]
    return adjusted
