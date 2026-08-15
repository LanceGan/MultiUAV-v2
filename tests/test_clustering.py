import numpy as np
from assignment import clustering, baselines

def _pts(n=30):
    rng = np.random.RandomState(0)
    return rng.uniform(0, 40, (n, 2))

def test_kmeans_4d_balanced_and_deterministic():
    pts = _pts()
    vols = np.random.RandomState(1).uniform(50, 150, len(pts))
    l1, c1, _ = clustering.kmeans_4d(pts, 3, vols, (1.0,0.4,0.4,0.5),
                                     uav_height=0.1, point_scale=0.1, random_state=42)
    l2, c2, _ = clustering.kmeans_4d(pts, 3, vols, (1.0,0.4,0.4,0.5),
                                     uav_height=0.1, point_scale=0.1, random_state=42)
    assert np.array_equal(l1, l2)
    # 均衡：每簇大小差 <= 1
    sizes = np.bincount(l1)
    assert sizes.max() - sizes.min() <= 1
    assert set(l1) == {0,1,2}

def test_baselines_return_balanced_labels():
    pts = _pts()
    for fn in (baselines.naive_kmeans, baselines.balanced_kmeans, baselines.jia_balanced):
        lbl, cen = fn(pts, 3, offload_volumes=np.ones(len(pts))*100)
        assert set(lbl) == {0,1,2}
