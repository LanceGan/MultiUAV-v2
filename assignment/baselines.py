"""Layer-1 task-assignment baselines for paper comparison.

三个基线，签名统一为 ``fn(points, n_clusters, offload_volumes=None, random_state=42)``
并返回 ``(labels, centers)``：

- ``naive_kmeans``:  仅空间坐标的普通 k-means（忽略 offload_volumes）。
- ``balanced_kmeans``: 普通 k-means + 容量约束均衡后处理（忽略 offload_volumes）。
- ``jia_balanced``:   Jia2025 式空间 + 负载均衡聚类（使用 offload_volumes）。

由旧 MultiUAV ``baselines.py`` 迁移，函数名/签名对齐本任务测试接口。
"""
import numpy as np


# ---------------------------------------------------------------------------
#  辅助工具（自 Clustering.py 镜像）
# ---------------------------------------------------------------------------

def _euclidean_dist_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """平方距离矩阵：(n, d) x (m, d) -> (n, m)。"""
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=2)


def _init_centers_kmeans_pp(points: np.ndarray, k: int,
                            rng: np.random.RandomState) -> np.ndarray:
    """k-means++ 初始化。"""
    n_samples = points.shape[0]
    centers = np.empty((k, points.shape[1]), dtype=points.dtype)
    idx = rng.randint(0, n_samples)
    centers[0] = points[idx]
    closest_dist_sq = _euclidean_dist_sq(points, centers[0:1]).reshape(-1)
    for i in range(1, k):
        probs = closest_dist_sq / closest_dist_sq.sum()
        r = rng.rand()
        cumulative = np.cumsum(probs)
        next_idx = np.searchsorted(cumulative, r)
        centers[i] = points[next_idx]
        dist_sq_to_new = _euclidean_dist_sq(points, centers[i:i + 1]).reshape(-1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq_to_new)
    return centers


def _balanced_assignment(dist_sq: np.ndarray, k: int) -> np.ndarray:
    """容量约束均衡分配后处理。

    优先把置信度最高（最小代价）的点分配到其代价最小的簇，并使簇规模尽量均衡。
    """
    n_samples = dist_sq.shape[0]
    base = n_samples // k
    rest = n_samples % k
    capacities = [base + 1 if i < rest else base for i in range(k)]

    pref_order = np.argsort(dist_sq, axis=1)
    min_cost = np.min(dist_sq, axis=1)
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

    return labels_bal


def _spatial_centers(points_2d: np.ndarray, labels: np.ndarray, k: int,
                     rng: np.random.RandomState) -> np.ndarray:
    """按标签计算物理空间质心。"""
    n_samples = points_2d.shape[0]
    centers = np.zeros((k, points_2d.shape[1]))
    for idx in range(k):
        mask = (labels == idx)
        if mask.sum() > 0:
            centers[idx] = points_2d[mask].mean(axis=0)
        else:
            centers[idx] = points_2d[rng.randint(0, n_samples)]
    return centers


def _run_kmeans(points_2d: np.ndarray, k: int, rng: np.random.RandomState,
                max_iters: int = 100, tol: float = 1e-4) -> np.ndarray:
    """在给定特征空间上跑标准 k-means（k-means++ 初始化），返回簇中心。"""
    n_samples = points_2d.shape[0]
    centers = _init_centers_kmeans_pp(points_2d, k, rng)

    for _ in range(max_iters):
        dist_sq = _euclidean_dist_sq(points_2d, centers)
        new_labels = np.argmin(dist_sq, axis=1)

        new_centers = np.zeros_like(centers)
        for idx in range(k):
            mask = (new_labels == idx)
            if mask.sum() > 0:
                new_centers[idx] = points_2d[mask].mean(axis=0)
            else:
                new_centers[idx] = points_2d[rng.randint(0, n_samples)]

        center_shift = np.sqrt(np.sum((centers - new_centers) ** 2, axis=1))
        centers = new_centers

        if np.max(center_shift) <= tol:
            break

    return centers


# ---------------------------------------------------------------------------
#  聚类基线
# ---------------------------------------------------------------------------

def naive_kmeans(points, n_clusters, offload_volumes=None, random_state=42):
    """普通 k-means：仅使用空间坐标 (x, y)。

    Args:
        points: (N, D) 数组；仅使用前两列。
        n_clusters: 簇数（无人机数）。
        offload_volumes: 忽略（仅为统一签名）。
        random_state: 随机种子。

    Returns:
        labels: (N,) 簇标签数组。
        cluster_centers: (n_clusters, 2) 质心数组。
    """
    pts_2d = np.asarray(points[:, :2], dtype=np.float64)
    rng = np.random.RandomState(random_state)

    centers = _run_kmeans(pts_2d, n_clusters, rng)
    dist_sq = _euclidean_dist_sq(pts_2d, centers)
    labels = np.argmin(dist_sq, axis=1)

    cluster_centers = _spatial_centers(pts_2d, labels, n_clusters, rng)
    return labels, cluster_centers


def balanced_kmeans(points, n_clusters, offload_volumes=None, random_state=42):
    """普通 k-means + 容量约束均衡后处理（簇规模尽量相等）。

    Args:
        points: (N, 2) 或 (N, D) 空间坐标。
        n_clusters: 簇数（无人机数）。
        offload_volumes: 忽略（仅为统一签名）。
        random_state: 随机种子。

    Returns:
        labels: (N,) 均衡簇标签数组。
        cluster_centers: (n_clusters, 2) 空间质心。
    """
    pts_2d = np.asarray(points[:, :2], dtype=np.float64)
    rng = np.random.RandomState(random_state)

    centers = _run_kmeans(pts_2d, n_clusters, rng)
    final_dist_sq = _euclidean_dist_sq(pts_2d, centers)
    labels = _balanced_assignment(final_dist_sq, n_clusters)

    cluster_centers = _spatial_centers(pts_2d, labels, n_clusters, rng)
    return labels, cluster_centers


def jia_balanced(points, n_clusters, offload_volumes=None, random_state=42):
    """Jia2025 式均衡任务分配：空间 + 负载均衡（无 G2A/A2G 信息）。

    Args:
        points: (N, 2) 或 (N, D) 空间坐标。
        n_clusters: 簇数（无人机数）。
        offload_volumes: (N,) 每个巡检点的数据量。
        random_state: 随机种子。

    Returns:
        labels: (N,) 均衡簇标签数组。
        cluster_centers: (n_clusters, 2) 空间质心。
    """
    pts_2d = np.asarray(points[:, :2], dtype=np.float64)
    rng = np.random.RandomState(random_state)
    n_samples = pts_2d.shape[0]
    k = n_clusters

    if offload_volumes is None:
        offload_volumes = rng.uniform(50, 150, n_samples)
    offload_volumes = np.asarray(offload_volumes, dtype=np.float64)

    # 归一化空间坐标与负载
    spatial_max = np.max(pts_2d, axis=0) - np.min(pts_2d, axis=0) + 1e-8
    pts_norm = pts_2d / spatial_max

    vol_norm = (offload_volumes - np.min(offload_volumes)) / \
               (np.max(offload_volumes) - np.min(offload_volumes) + 1e-8)

    # 增强特征：[空间, 负载]
    augmented = np.hstack([pts_norm, vol_norm.reshape(-1, 1)])

    # k-means++ 初始化并迭代
    centers = _init_centers_kmeans_pp(augmented, k, rng)
    for _ in range(100):
        dist_sq = _euclidean_dist_sq(augmented, centers)
        new_labels = np.argmin(dist_sq, axis=1)

        new_centers = np.zeros_like(centers)
        for idx in range(k):
            mask = (new_labels == idx)
            if mask.sum() > 0:
                new_centers[idx] = augmented[mask].mean(axis=0)
            else:
                new_centers[idx] = augmented[rng.randint(0, n_samples)]

        center_shift = np.sqrt(np.sum((centers - new_centers) ** 2, axis=1))
        centers = new_centers
        if np.max(center_shift) <= 1e-4:
            break

    # 容量约束均衡后处理
    final_dist_sq = _euclidean_dist_sq(augmented, centers)
    labels = _balanced_assignment(final_dist_sq, k)

    cluster_centers = _spatial_centers(pts_2d, labels, k, rng)
    return labels, cluster_centers
