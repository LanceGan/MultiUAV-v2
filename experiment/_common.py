"""Shared glue helpers for the Task 10 experiment-orchestration scripts.

Centralises the code that ``run_assign`` / ``run_seq`` / ``run_traj`` /
``run_end2end`` all need, so the four entry-points stay small and consistent:

* scenario data loading (points + per-point data sizes),
* the four layer-1 clustering methods (naive / balanced / jia / 4D),
* cluster splitting and per-UAV ``uav_traverse`` construction,
* per-point radio features (G2A outage / A2G SINR) for the balance metrics,
* comm-aware + Euclidean path-cost evaluation for a given ordered route,
* JSON serialisation.

All random sources use the scenario ``seed`` (``cfg['seed']``) so every run is
reproducible.  Radio-map queries (``radio_map_*``) draw from the global numpy
RNG, so callers are expected to seed ``np.random`` with ``cfg['seed']`` before
triggering any Monte-Carlo estimate.
"""

import json
import os

import numpy as np

from common.config import get_data_sizes, get_scenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "results", "datas")
LOG_DIR = os.path.join(ROOT, "results", "logs")
MODEL_DIR = os.path.join(ROOT, "results", "models")

# 4D feature weights (spatial, g2a, a2g, vol) — same convention as the tests.
W4D = (1.0, 1.0, 1.0, 0.5)
# Coordinate scale + UAV height for radio-map queries: env coords are 100 m
# units, radio maps are km (divide by 10), UAV flies at 100 m (0.1 km).
POINT_SCALE = 0.1
UAV_HEIGHT_KM = 0.1

# The four clustering methods compared in layer 1.
CLUSTER_METHODS = ("naive", "balanced", "jia", "4d")


def load_scenario(uav_num, seed=42):
    """Return ``(cfg, points, data_sizes)`` for a UAV count.

    ``points`` is ``(K, 2)`` in 100 m units (height column dropped) and
    ``data_sizes`` is the ``(K,)`` per-point offload volume (Mbit) from
    ``common.config.get_data_sizes``.
    """
    cfg = get_scenario(uav_num)
    if seed is not None:
        cfg["seed"] = int(seed)
    path = os.path.join(DATA_DIR, "Users_%d.txt" % int(cfg["user_num"]))
    points = np.loadtxt(path)[:, :2]
    data_sizes = np.asarray(
        get_data_sizes(int(cfg["user_num"]), seed=int(cfg["seed"])),
        dtype=float)
    return cfg, points, data_sizes


def cluster_labels(method, points, data_sizes, n, seed):
    """Run one clustering method and return the ``(K,)`` integer label array."""
    from assignment import baselines, clustering

    if method == "naive":
        return baselines.naive_kmeans(points, n, data_sizes, seed)[0]
    if method == "balanced":
        return baselines.balanced_kmeans(points, n, data_sizes, seed)[0]
    if method == "jia":
        return baselines.jia_balanced(points, n, data_sizes, seed)[0]
    if method == "4d":
        labels, _, _ = clustering.kmeans_4d(
            points, n, data_sizes, weights=W4D, uav_height=UAV_HEIGHT_KM,
            point_scale=POINT_SCALE, random_state=seed, balanced=True)
        return labels
    raise ValueError("unknown clustering method: %s" % method)


def split_clusters(labels, points, data_sizes, n):
    """Group points by cluster label.

    Returns a list (one entry per UAV/cluster) of ``(cluster_points,
    global_indices, cluster_data_sizes)`` so layer 2/3 can map back to the
    original point indices.
    """
    clusters = []
    for c in range(n):
        gidx = np.where(np.asarray(labels) == c)[0]
        clusters.append((points[gidx], gidx, data_sizes[gidx]))
    return clusters


def build_traverse(clusters, orders):
    """Map per-cluster visit orders to a global ``{uav_id: [global indices]}``."""
    traverse = {}
    for c, (_, gidx, _) in enumerate(clusters):
        traverse[c] = [int(gidx[i]) for i in orders[c]]
    return traverse


def per_point_comm(points, seed, uav_height_km=UAV_HEIGHT_KM):
    """Return per-point ``(g2a_outage, a2g_sinr)`` arrays (length K)."""
    from radio_map import radio_map_A2G, radio_map_G2A

    np.random.seed(int(seed))
    k = points.shape[0]
    g2a = np.zeros(k)
    a2g = np.zeros(k)
    for i in range(k):
        loc = np.array([[points[i, 0] / 10.0, points[i, 1] / 10.0,
                         uav_height_km]])
        g2a[i] = float(radio_map_G2A.getPointMiniOutage(loc)[0][0])
        a2g[i] = float(radio_map_A2G.getPointDateRate(loc))
    return g2a, a2g


def path_cost(ordered_points, data_sizes, cfg, seed):
    """Evaluate an ordered route ``[ini, p0..p_{K-1}, end]`` on both objectives.

    Returns ``(euclidean_length, comm_cost)``.  The comm-aware cost follows the
    layer-2 ``comm_dis_mat`` convention: the edge departing point ``i`` carries
    that point's data size (the ini/end anchor points carry none).
    """
    from sequence import radio_path

    np.random.seed(int(seed))
    n = ordered_points.shape[0]
    eucl = 0.0
    comm = 0.0
    for i in range(n - 1):
        a = ordered_points[i][:2]
        b = ordered_points[i + 1][:2]
        eucl += float(np.linalg.norm(b - a))
        ds = float(data_sizes[i - 1]) if 0 < i < n - 1 else 0.0
        comm += float(radio_path.edge_cost(a, b, ds, cfg)["cost"])
    return eucl, comm


def save_json(path, obj):
    """Pretty-print ``obj`` as UTF-8 JSON, creating parent directories."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return path
