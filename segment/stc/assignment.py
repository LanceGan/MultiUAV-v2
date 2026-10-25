"""EBTAS-inspired balanced task assignment (Task 3).

Four assignment methods over SI task positions/data:

- ``region_partition``: deterministic angular partition around the depot.
- ``spatial_kmeans``: spatial k-means + capacity-balanced post-processing.
- ``jia_ebtas``: Jia-2025 lineage feature assignment (geometry + payload +
  point communication-rate proxy) with explicit normalization and configured
  weights; a faithful adapter of the legacy ``jia_balanced`` criterion.
- ``library_aware_ebtas``: the same Jia-style balanced pipeline, but with the
  local rate proxy replaced by precomputed incident-edge summaries (median
  feasible flight time, upload delay, G2A outage, sector-diversity score).

Every method returns an :class:`AssignmentResult` and enforces balanced cluster
sizes (differing by at most one) when ``K >= U``; empty clusters appear only
when ``K < U``.  All quantities are SI (metres, seconds, bits, bit/s); unit
conversions happen only at the scenario/radio boundary.

"Balanced" here means *equal cluster counts* (at most one apart) plus
workload/communication-aware features guiding the grouping.  Payload-volume
balance is best-effort / equal-count only: neither the Jia-style baseline nor
the library-aware EBTAS imposes a hard payload-sum constraint, and the
library-aware features are communication-oriented (flight time, upload delay,
outage, sector diversity) rather than payload-balancing.  ``library_aware_ebtas``
differs from ``jia_ebtas`` only in its communication features (incident-edge
summaries vs the local rate proxy), not in the balancing rule.
"""

from dataclasses import dataclass

import numpy as np

from segment.stc.types import STCScenario

DEFAULT_JIA_WEIGHTS = {"spatial": 1.0, "payload": 1.0, "rate": 1.0}
DEFAULT_LIBRARY_WEIGHTS = {
    "spatial": 1.0, "payload": 1.0,
    "flight": 1.0, "delay": 1.0, "outage": 1.0, "diversity": 1.0,
}
INCIDENT_FEATURES = ("flight_s", "upload_delay_s", "outage", "sector_diversity")


@dataclass
class AssignmentResult:
    labels: np.ndarray        # (n_tasks,) int in 0..U-1
    clusters: tuple           # tuple of tuple of task ids, one per UAV
    feature_summary: dict     # resolved features, weights, seeds (provenance)
    balance_metrics: dict     # per-UAV payload/count/route-cost + inter-UAV std


# --------------------------------------------------------------------------- #
# Legacy-mirrored helpers (reproduced from assignment/baselines.py so the new
# module is SI-self-contained and does not import legacy 100-m unit assumptions)
# --------------------------------------------------------------------------- #

def _euclidean_dist_sq(a, b):
    """Squared distance matrix: a (n,d), b (m,d) -> (n,m)."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=2)


def _init_centers_kmeans_pp(points, k, rng):
    """k-means++ initialization (identical RNG call order to legacy)."""
    n_samples = points.shape[0]
    centers = np.empty((k, points.shape[1]), dtype=points.dtype)
    idx = rng.randint(0, n_samples)
    centers[0] = points[idx]
    closest_dist_sq = _euclidean_dist_sq(points, centers[0:1]).reshape(-1)
    for i in range(1, k):
        if closest_dist_sq.sum() <= 1e-12:
            # k > n_samples (or all points coincident): every point is already a
            # centre; the remaining centres are arbitrary, pick uniformly so the
            # index stays valid and the K < U empty-cluster case is clean.
            next_idx = rng.randint(0, n_samples)
        else:
            probs = closest_dist_sq / closest_dist_sq.sum()
            r = rng.rand()
            cumulative = np.cumsum(probs)
            next_idx = np.searchsorted(cumulative, r)
        centers[i] = points[next_idx]
        dist_sq_to_new = _euclidean_dist_sq(points, centers[i:i + 1]).reshape(-1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq_to_new)
    return centers


def _min_max_norm(arr):
    """Min-max normalization to [0, 1]; returns zeros for a constant array."""
    arr = np.asarray(arr, dtype=float)
    ptp = np.ptp(arr)
    if ptp < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr.min()) / ptp


def _balanced_assignment(dist_sq, k):
    """Capacity-constrained balanced post-processing (equal cluster sizes).

    Processes points in ascending minimum-cost order, assigning each to its
    cheapest cluster that still has capacity, so cluster sizes differ by at
    most one.  With ``k > n_samples`` the surplus clusters stay empty.
    """
    n_samples = dist_sq.shape[0]
    base = n_samples // k
    rest = n_samples % k
    capacities = [base + 1 if i < rest else base for i in range(k)]

    pref_order = np.argsort(dist_sq, axis=1)
    min_cost = np.min(dist_sq, axis=1)
    pts_order = np.argsort(min_cost)

    labels = np.full(n_samples, -1, dtype=int)
    cap = capacities.copy()
    for p in pts_order:
        for c in pref_order[p]:
            if cap[c] > 0:
                labels[p] = c
                cap[c] -= 1
                break
    for p in np.where(labels == -1)[0]:
        labels[p] = int(pref_order[p, 0])
    return labels


def _kmeans_balanced(features, n_clusters, seed, max_iters=100, tol=1e-4):
    """k-means++ over an augmented feature matrix + balanced post-processing."""
    n_samples = features.shape[0]
    rng = np.random.RandomState(seed)
    centers = _init_centers_kmeans_pp(features, n_clusters, rng)

    for _ in range(max_iters):
        dist_sq = _euclidean_dist_sq(features, centers)
        new_labels = np.argmin(dist_sq, axis=1)
        new_centers = np.zeros_like(centers)
        for idx in range(n_clusters):
            mask = (new_labels == idx)
            if mask.sum() > 0:
                new_centers[idx] = features[mask].mean(axis=0)
            else:
                new_centers[idx] = features[rng.randint(0, n_samples)]
        center_shift = np.sqrt(np.sum((centers - new_centers) ** 2, axis=1))
        centers = new_centers
        if np.max(center_shift) <= tol:
            break

    final_dist_sq = _euclidean_dist_sq(features, centers)
    return _balanced_assignment(final_dist_sq, n_clusters)


def _clusters_from_labels(labels, n_uav):
    return tuple(tuple(sorted(int(t) for t in np.where(labels == u)[0]))
                 for u in range(n_uav))


# --------------------------------------------------------------------------- #
# Balance metrics
# --------------------------------------------------------------------------- #

def _proxy_route_cost(scenario, task_ids):
    """Greedy nearest-neighbour route (depot -> tasks -> terminal) proxy cost.

    flight time + nominal upload time; used only for inter-UAV balance so it
    stays library-free and deterministic.
    """
    if not task_ids:
        return 0.0
    remaining = list(task_ids)
    cur = np.asarray(scenario.depot_m, dtype=float)
    length = 0.0
    while remaining:
        nxt = min(remaining, key=lambda j: float(np.linalg.norm(
            np.asarray(scenario.task_positions_m[j]) - cur)))
        length += float(np.linalg.norm(
            np.asarray(scenario.task_positions_m[nxt]) - cur))
        cur = np.asarray(scenario.task_positions_m[nxt], dtype=float)
        remaining.remove(nxt)
    length += float(np.linalg.norm(np.asarray(scenario.terminal_m) - cur))
    flight = length / scenario.limits.v_max_mps
    payload = sum(float(scenario.task_data_bits[t]) for t in task_ids)
    upload = payload / scenario.limits.bandwidth_hz  # nominal 1 bit/s/Hz
    return flight + upload


def _balance_metrics(scenario, clusters):
    counts = np.array([float(len(c)) for c in clusters])
    payloads = np.array([sum(float(scenario.task_data_bits[t]) for t in c)
                         for c in clusters])
    costs = np.array([_proxy_route_cost(scenario, c) for c in clusters])
    return {
        "payload_sum": tuple(float(x) for x in payloads),
        "task_count": tuple(int(x) for x in counts),
        "proxy_route_cost": tuple(float(x) for x in costs),
        "payload_std": float(np.std(payloads)),
        "count_std": float(np.std(counts)),
        "route_cost_std": float(np.std(costs)),
    }


def _finalize(scenario, labels, feature_summary):
    clusters = _clusters_from_labels(labels, scenario.n_uav)
    return AssignmentResult(labels=labels, clusters=clusters,
                            feature_summary=feature_summary,
                            balance_metrics=_balance_metrics(scenario, clusters))


# --------------------------------------------------------------------------- #
# Assignment methods
# --------------------------------------------------------------------------- #

def region_partition(scenario, seed=42):
    """Deterministic angular partition of tasks around the depot.

    ``seed`` is accepted for a uniform call signature but the method is purely
    geometric and seed-independent.
    """
    pts = np.asarray(scenario.task_positions_m, dtype=float)
    depot = np.asarray(scenario.depot_m, dtype=float)
    angles = np.arctan2(pts[:, 1] - depot[1], pts[:, 0] - depot[0])
    order = np.argsort(angles, kind="stable")
    groups = [list(g) for g in np.array_split(order, scenario.n_uav)]

    labels = np.empty(scenario.n_tasks, dtype=int)
    for u, g in enumerate(groups):
        labels[np.asarray(g, dtype=int)] = u

    return _finalize(scenario, labels, {
        "method": "region_partition",
        "seed": int(seed),
        "features": ("angular_sector",),
        "n_tasks": scenario.n_tasks,
        "n_uav": scenario.n_uav,
    })


def spatial_kmeans(scenario, seed=42):
    """Spatial k-means (geometry only) + capacity-balanced post-processing."""
    pts = np.asarray(scenario.task_positions_m, dtype=float)
    labels = _kmeans_balanced(pts, scenario.n_uav, seed)
    return _finalize(scenario, labels, {
        "method": "spatial_kmeans",
        "seed": int(seed),
        "features": ("x", "y"),
        "n_tasks": scenario.n_tasks,
        "n_uav": scenario.n_uav,
    })


def jia_ebtas(scenario, field=None, seed=42, weights=None):
    """Jia-2025 lineage balanced feature assignment (adapter of ``jia_balanced``).

    Features: geometry (legacy per-axis normalization), payload volume
    (min-max), and a point communication-rate proxy (max A2G spectral
    efficiency, min-max).  With ``field=None`` the rate feature is zeroed, so
    the method reproduces the legacy spatial+payload criterion exactly.
    """
    weights = dict(DEFAULT_JIA_WEIGHTS, **(weights or {}))
    n_tasks = scenario.n_tasks

    pts = np.asarray(scenario.task_positions_m, dtype=float)
    # legacy criterion divides by the per-axis range without subtracting the
    # minimum -- kept verbatim so the adapter matches ``jia_balanced``.
    spatial_feats = pts / (pts.max(axis=0) - pts.min(axis=0) + 1e-8)

    data = np.asarray(scenario.task_data_bits, dtype=float)
    vol_feats = ((data - data.min()) / (data.max() - data.min() + 1e-8)
                 ).reshape(-1, 1)

    rate = np.zeros(n_tasks)
    if field is not None:
        for i in range(n_tasks):
            rate[i] = float(np.max(
                field.sample(scenario.task_positions_m[i]).a2g_efficiency))
    rate_feats = _min_max_norm(rate).reshape(-1, 1)

    augmented = np.hstack([
        spatial_feats * weights["spatial"],
        vol_feats * weights["payload"],
        rate_feats * weights["rate"],
    ])

    labels = _kmeans_balanced(augmented, scenario.n_uav, seed)
    return _finalize(scenario, labels, {
        "method": "jia_ebtas",
        "seed": int(seed),
        "weights": dict(weights),
        "features": ("x", "y", "payload", "rate"),
        "rate_feature": "max_a2g_efficiency" if field is not None else None,
        "normalization": ("spatial=pts/range (legacy), "
                          "payload=minmax, rate=minmax"),
        "n_tasks": n_tasks,
        "n_uav": scenario.n_uav,
    })


# --------------------------------------------------------------------------- #
# Library-aware features
# --------------------------------------------------------------------------- #

def _mean_serving_outage(candidate):
    """Mean G2A outage along the preferred-sector path (matches library._outage)."""
    if len(candidate.sector_ids) == 0:
        return 0.0
    return float(np.mean([
        candidate.g2a_outage[t, int(s)]
        for t, s in enumerate(candidate.sector_ids)]))


def _sector_diversity(candidate):
    """Normalized Shannon entropy of the sector-occupancy signature in [0, 1].

    0 = always the same sector, 1 = uniform across sectors.  Higher means the
    trace spreads across sectors and is more flexible for contention avoidance.
    """
    if len(candidate.sector_ids) == 0:
        return 0.0
    n_sectors = candidate.full_rate_bps.shape[1]
    sig = np.bincount(np.asarray(candidate.sector_ids, dtype=int),
                      minlength=n_sectors) / len(candidate.sector_ids)
    sig = sig[sig > 0]
    entropy = -np.sum(sig * np.log(sig))
    max_entropy = np.log(n_sectors) if n_sectors > 1 else 1.0
    return float(entropy / max_entropy)


def _candidate_features(candidate, delta_t_s):
    flight = candidate.flight_steps * delta_t_s
    mean_rate = float(np.mean(candidate.full_rate_bps.max(axis=1)))
    delay = candidate.source_data_bits / max(mean_rate, 1e-9)
    outage = _mean_serving_outage(candidate)
    diversity = _sector_diversity(candidate)
    return (flight, delay, outage, diversity)


def incident_edge_summary(scenario, library):
    """Per-task median incident-edge summary ``(n_tasks, 4)``.

    Aggregates the four features (median feasible flight time, upload delay,
    G2A outage, sector-diversity score) over every rankable candidate of every
    incident edge carrying task ``i``'s payload (``i -> j`` for ``j != i`` and
    ``i -> terminal``).  Depot->task edges carry zero payload and are excluded.
    """
    dt = scenario.limits.delta_t_s
    n_tasks = scenario.n_tasks
    summary = np.zeros((n_tasks, 4), dtype=float)
    for i in range(n_tasks):
        feats = []
        for j in range(n_tasks):
            if j == i:
                continue
            for c in library.candidates.get((i, j), ()):
                if c.rankable:
                    feats.append(_candidate_features(c, dt))
        for c in library.candidates.get((i, n_tasks), ()):
            if c.rankable:
                feats.append(_candidate_features(c, dt))
        if feats:
            summary[i] = np.median(np.asarray(feats, dtype=float), axis=0)
    return summary


def library_aware_ebtas(scenario, library, seed=42, weights=None):
    """Library-aware EBTAS: Jia-style balanced pipeline over incident summaries.

    Replaces the local rate proxy with four precomputed incident-edge features
    (flight time, upload delay, outage, sector-diversity).  The Jia-style
    ``jia_ebtas`` is left unchanged as the baseline.

    Like ``jia_ebtas``, the method balances *cluster counts* (equal up to one)
    and uses communication-aware features to guide grouping; it does not impose
    a hard payload-sum constraint, so payload-volume balance is best-effort /
    equal-count only.
    """
    weights = dict(DEFAULT_LIBRARY_WEIGHTS, **(weights or {}))
    n_tasks = scenario.n_tasks

    pts = np.asarray(scenario.task_positions_m, dtype=float)
    spatial_feats = pts / (pts.max(axis=0) - pts.min(axis=0) + 1e-8)

    data = np.asarray(scenario.task_data_bits, dtype=float)
    vol_feats = ((data - data.min()) / (data.max() - data.min() + 1e-8)
                 ).reshape(-1, 1)

    incident = incident_edge_summary(scenario, library)
    incident_norm = np.zeros_like(incident)
    for k in range(4):
        incident_norm[:, k] = _min_max_norm(incident[:, k])

    augmented = np.hstack([
        spatial_feats * weights["spatial"],
        vol_feats * weights["payload"],
        incident_norm[:, 0:1] * weights["flight"],
        incident_norm[:, 1:2] * weights["delay"],
        incident_norm[:, 2:3] * weights["outage"],
        incident_norm[:, 3:4] * weights["diversity"],
    ])

    labels = _kmeans_balanced(augmented, scenario.n_uav, seed)
    return _finalize(scenario, labels, {
        "method": "library_aware_ebtas",
        "seed": int(seed),
        "weights": dict(weights),
        "features": ("x", "y", "payload") + INCIDENT_FEATURES,
        "incident_features": INCIDENT_FEATURES,
        "incident_summary": incident,
        "normalization": "spatial=pts/range, payload=minmax, incident=minmax",
        "n_tasks": n_tasks,
        "n_uav": scenario.n_uav,
    })
