"""Layer-1 communication-aware task assignment: 4D k-means clustering.

四维方差聚类：在增强特征空间 `[spatial*w_s, g2a*w_g, a2g*w_a, vol*w_v]`
中做 k-means，用 k-means++ 初始化，并以容量约束后处理实现簇规模均衡。

特征来源（逐点查询 radio map）：
- spatial: 归一化地理坐标
- G2A: 中断概率 outage probability（``radio_map_G2A.getPointMiniOutage``）
- A2G: 上行 SINR(dB)（``radio_map_A2G.getPointDateRate``），速率单调代理
- vol:  数据卸载量（Min-Max 归一化）

从旧 MultiUAV ``Clustering.py`` 迁移并保留算法，仅做了以下修正：
1. G2A 特征由 SINR 改为中断概率；
2. 为保证确定性，特征提取前按 ``random_state`` 种子化全局 numpy RNG
   （radio map 的蒙特卡洛快衰落估计依赖全局 RNG）。
"""
from typing import Tuple

import numpy as np

from radio_map import radio_map_A2G, radio_map_G2A


def _euclidean_dist_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """平方距离矩阵：a (n,d), b (m,d) -> (n,m)。"""
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=2)


def _init_centers_kmeans_pp(points: np.ndarray, k: int,
                            random_state: np.random.RandomState) -> np.ndarray:
    """k-means++ 初始化。"""
    n_samples = points.shape[0]
    centers = np.empty((k, points.shape[1]), dtype=points.dtype)
    idx = random_state.randint(0, n_samples)
    centers[0] = points[idx]
    closest_dist_sq = _euclidean_dist_sq(points, centers[0:1]).reshape(-1)
    for i in range(1, k):
        probs = closest_dist_sq / closest_dist_sq.sum()
        r = random_state.rand()
        cumulative = np.cumsum(probs)
        next_idx = np.searchsorted(cumulative, r)
        centers[i] = points[next_idx]
        dist_sq_to_new = _euclidean_dist_sq(points, centers[i:i + 1]).reshape(-1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq_to_new)
    return centers


def _min_max_norm(arr: np.ndarray) -> np.ndarray:
    """Min-Max 归一化到 [0, 1]；全等时返回零向量。"""
    ptp = np.ptp(arr)
    if ptp < 1e-8:
        return np.zeros_like(arr)
    return (arr - np.min(arr)) / ptp


def kmeans_4d(points: np.ndarray,
              k: int,
              offload_volumes: np.ndarray = None,
              weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
              max_iters: int = 100,
              tol: float = 1e-4,
              init: str = 'kmeans++',
              random_state: int = None,
              uav_height: float = 0.025,
              point_scale: float = 1.0,
              balanced: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """四维方差聚类算法。

    Args:
        points: 地理坐标 (n_samples, 2)。
        k: 无人机数量（聚类数）。
        offload_volumes: 每个巡航点的数据卸载量 (n_samples,)。
        weights: 四维特征权重 (w_spatial, w_g2a, w_a2g, w_vol)。
        uav_height: 无人机高度 (km)，用于 radio map 查询。
        point_scale: 坐标缩放系数。
        random_state: 随机种子（同时用于聚类与 radio map 特征）。
        balanced: 是否启用容量约束均衡后处理。

    Returns:
        (labels, spatial_centers, inertia)
    """
    if k <= 0:
        raise ValueError('k must be > 0')
    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError('points must be 2D array')

    rng = np.random.RandomState(random_state)
    n_samples, n_features = points.shape

    # 1. 地理空间特征 (Spatial)
    points_scaled = points * point_scale
    dataset_max_dist = np.sqrt(np.max(_euclidean_dist_sq(points_scaled, points_scaled))) + 1e-8
    spatial_feats = points_scaled / dataset_max_dist

    # 未提供卸载量时生成随机数据
    if offload_volumes is None:
        offload_volumes = rng.uniform(50, 150, n_samples)
    offload_volumes = np.asarray(offload_volumes, dtype=np.float64)
    vol_feats = _min_max_norm(offload_volumes).reshape(-1, 1)

    w_s, w_g, w_a, w_v = weights

    # 初始化 G2A 与 A2G 归一化特征
    g2a_feats = np.zeros((n_samples, 1))
    a2g_feats = np.zeros((n_samples, 1))

    # 2 & 3. 提取双地图通信特征 (G2A 中断概率 & A2G SINR)
    if w_g > 0.0 or w_a > 0.0:
        # radio map 的中断/SINR 是蒙特卡洛估计，内部使用全局 numpy RNG；
        # 按 random_state 种子化以保证固定种子下整条流水线可复现。
        if random_state is not None:
            np.random.seed(random_state)

        loc_vec = np.hstack([points_scaled, np.full((n_samples, 1), uav_height)])

        if w_g > 0.0:
            g2a_list = np.array([
                float(radio_map_G2A.getPointMiniOutage(loc_vec[i:i + 1, :])[0][0])
                for i in range(n_samples)
            ])
            g2a_feats = _min_max_norm(g2a_list).reshape(-1, 1)

        if w_a > 0.0:
            a2g_list = np.array([
                float(radio_map_A2G.getPointDateRate(loc_vec[i:i + 1, :]))
                for i in range(n_samples)
            ])
            a2g_feats = _min_max_norm(a2g_list).reshape(-1, 1)

    # 4. 构建增强特征空间
    augmented_points = np.hstack([
        spatial_feats * w_s,
        g2a_feats * w_g,
        a2g_feats * w_a,
        vol_feats * w_v
    ])

    # K-means 主循环（高维空间）
    if init == 'kmeans++':
        centers_aug = _init_centers_kmeans_pp(augmented_points, k, rng)
    elif init == 'random':
        indices = rng.choice(n_samples, size=k, replace=False)
        centers_aug = augmented_points[indices].copy()

    labels = np.full(n_samples, -1, dtype=int)

    for _ in range(max_iters):
        dist_sq = _euclidean_dist_sq(augmented_points, centers_aug)
        new_labels = np.argmin(dist_sq, axis=1)

        new_centers_aug = np.zeros_like(centers_aug)
        for idx in range(k):
            mask = (new_labels == idx)
            if mask.sum() > 0:
                new_centers_aug[idx] = augmented_points[mask].mean(axis=0)
            else:
                new_centers_aug[idx] = augmented_points[rng.randint(0, n_samples)]

        center_shift = np.sqrt(np.sum((centers_aug - new_centers_aug) ** 2, axis=1))
        centers_aug = new_centers_aug
        labels = new_labels

        if np.max(center_shift) <= tol:
            break

    final_dist_sq = _euclidean_dist_sq(augmented_points, centers_aug)

    # 后处理：容量/数量约束的均衡分配
    if balanced:
        cost = final_dist_sq  # 高维空间下的平方距离

        base = n_samples // k
        rest = n_samples % k
        capacities = [base + 1 if i < rest else base for i in range(k)]

        pref_order = np.argsort(cost, axis=1)
        min_cost = np.min(cost, axis=1)
        pts_order = np.argsort(min_cost)

        labels_bal = np.full(n_samples, -1, dtype=int)
        cap = capacities.copy()

        for p in pts_order:
            for c in pref_order[p]:
                if cap[c] > 0:
                    labels_bal[p] = c
                    cap[c] -= 1
                    break

        unassigned = np.where(labels_bal == -1)[0]
        for p in unassigned:
            labels_bal[p] = int(pref_order[p, 0])

        labels = labels_bal

    # 最终物理空间中心（还原为输入单位）
    spatial_centers = np.zeros((k, points.shape[1]))
    for idx in range(k):
        mask = (labels == idx)
        if mask.sum() > 0:
            spatial_centers[idx] = points[mask].mean(axis=0)
        else:
            spatial_centers[idx] = points[rng.randint(0, n_samples)]

    # 整体方差误差 (Inertia)
    inertia = float(np.sum(_euclidean_dist_sq(augmented_points, centers_aug)[np.arange(n_samples), labels]))

    return labels, spatial_centers, inertia
