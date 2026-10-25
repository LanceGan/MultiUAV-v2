"""Paired statistical protocol (Task 7).

Paired bootstrap/permutation inference for method comparison, hierarchical
aggregation of replicate seeds within a scenario, an explicit failure policy
for infeasible runs, and Holm-Bonferroni correction for multiple comparisons.

Validation is strict: method pairs must be explicit and distinct; scenario /
method / replicate keys must be unique; replicate sets must pair completely
(unless the declared policy aggregates independently); and every metric must be
finite.  Failures are never silently dropped — they are handled by the declared
``failure_policy``.
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


FAILURE_POLICIES = ("success_rate", "common_feasible", "penalized")


def paired_summary(a, b, confidence=0.95, permutations=10000, seed=0):
    """Paired summary of a-b (bootstrap CI + two-sided permutation p-value)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("a and b must have equal length")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("NaN/Inf values not allowed")
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


def _finite(value, where):
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError("metric must be a finite number (%s), got %r"
                         % (where, value))
    if not np.isfinite(f):
        raise ValueError("metric must be finite (%s), got %r" % (where, value))
    return f


def hierarchical_paired_summary(records, method_a, method_b, metric="metric",
                                scenario_keys=("map_seed", "task_seed"),
                                replicate_key="method_seed",
                                failure_policy="common_feasible",
                                penalty=None):
    """Hierarchical paired comparison of ``method_a`` vs ``method_b``.

    Steps:
    1. Validate: ``method_a != method_b``; every record carries the metric and
       the scenario/replicate keys; metric values are finite; no duplicate
       (scenario, method, replicate) key.
    2. Validate replicate pairing per scenario (for paired cost policies): the
       two methods must have the same replicate set (complete pairing); the
       ``success_rate`` policy aggregates independently and may accept disjoint
       replicate sets.
    3. Aggregate replicate seeds within each scenario into one observation per
       method, according to ``failure_policy``:
       * ``success_rate``     — per-scenario success rate (fraction of feasible
         replicates) over all paired instances.
       * ``common_feasible``  — mean metric over replicates feasible for *both*
         methods in the same replicate (failures dropped by declared policy).
       * ``penalized``        — mean metric over all replicates, with infeasible
         replicates replaced by ``penalty`` (a finite declared value).
    4. Run the paired summary on the two methods' scenario-level observations.

    A record is feasible unless it carries ``feasible: False``.  Returns ``None``
    when no scenario yields a pair.
    """
    if method_a == method_b:
        raise ValueError("method_a and method_b must differ (explicit method pairs)")
    if failure_policy not in FAILURE_POLICIES:
        raise ValueError("unknown failure_policy %r (expected one of %s)"
                         % (failure_policy, FAILURE_POLICIES))

    seen = set()
    keyed = []  # (scenario_tuple, method, replicate, feasible, metric)
    scen_reps = {}  # scenario -> method -> set(replicate)
    for idx, r in enumerate(records):
        m = r.get("method")
        if m not in (method_a, method_b):
            continue
        for k in scenario_keys:
            if k not in r:
                raise ValueError("record %d missing scenario key %r" % (idx, k))
        if replicate_key not in r:
            raise ValueError("record %d missing replicate key %r" % (idx, replicate_key))
        scen = tuple(r[k] for k in scenario_keys)
        rep = r[replicate_key]
        feasible = bool(r.get("feasible", True))
        val = _finite(r.get(metric), "record %d" % idx)
        key = (scen, m, rep)
        if key in seen:
            raise ValueError("duplicate (scenario, method, replicate) key %r" % (key,))
        seen.add(key)
        keyed.append((scen, m, rep, feasible, val))
        scen_reps.setdefault(scen, {}).setdefault(m, set()).add(rep)

    # scenario-coverage completeness: a scenario present for only one method is a
    # silent data-loss bug (e.g. an orchestrator failing to emit one method's
    # records for that scenario).  Reject it rather than narrowing the paired set.
    scenarios_a = {scen for scen, by_method in scen_reps.items() if method_a in by_method}
    scenarios_b = {scen for scen, by_method in scen_reps.items() if method_b in by_method}
    only_a = scenarios_a - scenarios_b
    only_b = scenarios_b - scenarios_a
    if only_a or only_b:
        raise ValueError(
            "incomplete scenario coverage: scenarios %r present for %s but not %s; "
            "scenarios %r present for %s but not %s"
            % (sorted(only_a), method_a, method_b,
               sorted(only_b), method_b, method_a))

    # complete replicate pairing (paired cost policies)
    if failure_policy in ("common_feasible", "penalized"):
        for scen, by_method in scen_reps.items():
            if method_a in by_method and method_b in by_method:
                sa, sb = by_method[method_a], by_method[method_b]
                if not (sa & sb):
                    raise ValueError(
                        "scenario %r has disjoint replicate sets between %s and %s"
                        % (scen, method_a, method_b))
                if sa != sb:
                    raise ValueError(
                        "scenario %r has incomplete replicate pairing: %s=%r vs %s=%r"
                        % (scen, method_a, sorted(sa), method_b, sorted(sb)))

    scen_means = _aggregate(keyed, method_a, method_b, failure_policy, penalty)

    scens = sorted(scen_means)
    a_vals = [scen_means[s][method_a] for s in scens
              if method_a in scen_means[s] and method_b in scen_means[s]]
    b_vals = [scen_means[s][method_b] for s in scens
              if method_a in scen_means[s] and method_b in scen_means[s]]
    if not a_vals:
        return None
    return paired_summary(a_vals, b_vals)


def _aggregate(keyed, method_a, method_b, failure_policy, penalty):
    scen_means = {}
    if failure_policy == "success_rate":
        rates = {}
        for scen, m, rep, feasible, val in keyed:
            rates.setdefault(scen, {}).setdefault(m, [0, 0])
            rates[scen][m][1] += 1
            if feasible:
                rates[scen][m][0] += 1
        for scen, by_method in rates.items():
            if method_a in by_method and method_b in by_method:
                scen_means[scen] = {
                    method_a: by_method[method_a][0] / by_method[method_a][1],
                    method_b: by_method[method_b][0] / by_method[method_b][1],
                }
        return scen_means

    if failure_policy == "common_feasible":
        by_scen_rep = {}
        for scen, m, rep, feasible, val in keyed:
            by_scen_rep.setdefault((scen, rep), {})[m] = (feasible, val)
        for (scen, rep), methods in by_scen_rep.items():
            if method_a not in methods or method_b not in methods:
                continue
            fa, va = methods[method_a]
            fb, vb = methods[method_b]
            if not (fa and fb):
                continue
            scen_means.setdefault(scen, {}).setdefault(method_a, []).append(va)
            scen_means.setdefault(scen, {}).setdefault(method_b, []).append(vb)
        return {scen: {m: float(np.mean(vals)) for m, vals in mm.items()}
                for scen, mm in scen_means.items()
                if method_a in mm and method_b in mm}

    # penalized
    if penalty is None:
        feasible_vals = [val for (scen, m, rep, feasible, val) in keyed if feasible]
        penalty = float(np.max(feasible_vals)) * 2.0 + 1.0 if feasible_vals else 1e9
    by_scen_method = {}
    for scen, m, rep, feasible, val in keyed:
        by_scen_method.setdefault((scen, m), []).append(val if feasible else penalty)
    out = {}
    for scen in sorted({s for (s, _m) in by_scen_method}):
        if (scen, method_a) in by_scen_method and (scen, method_b) in by_scen_method:
            out[scen] = {
                method_a: float(np.mean(by_scen_method[(scen, method_a)])),
                method_b: float(np.mean(by_scen_method[(scen, method_b)])),
            }
    return out


def prespecified_contrasts(records, contrasts, metric="metric",
                           scenario_keys=("map_seed", "task_seed"),
                           replicate_key="method_seed",
                           failure_policy="common_feasible", penalty=None):
    """Compute pre-specified method contrasts and Holm-correct their p-values.

    ``contrasts`` maps a name to a ``(method_a, method_b)`` pair.  Returns
    ``(summaries, adjusted_p_values)`` where ``summaries`` maps name ->
    :class:`PairedSummary` (``None`` when no scenario paired) and
    ``adjusted_p_values`` maps name -> Holm-adjusted p-value (1.0 for a missing
    contrast).
    """
    summaries = {}
    for name, (a, b) in contrasts.items():
        summaries[name] = hierarchical_paired_summary(
            records, a, b, metric, scenario_keys, replicate_key,
            failure_policy, penalty)
    pvals = {name: (s.p_value if s is not None else 1.0)
             for name, s in summaries.items()}
    return summaries, holm_adjust(pvals)


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
