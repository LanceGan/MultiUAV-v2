"""Task 10.1 — layer-1 task-assignment comparison.

Runs the four clustering methods (naive / balanced / jia / 4D) for N in
{2, 3, 4}, computes per-cluster spatial / communication / load balance metrics,
and writes ``results/logs/assign_<method>_uav<N>.json``.

Spatial balance is the per-cluster mean distance to the cluster centroid;
communication balance is the per-cluster mean G2A outage probability and mean
A2G SINR; load balance is the per-cluster total offload volume.  Across-cluster
balance is reported as the standard deviation of each per-cluster quantity.

CLI::

    python experiment/run_assign.py               # full 2/3/4 grid
    python experiment/run_assign.py --uav_nums 2  # N=2 only (smoke)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiment import _common


def cluster_metrics(labels, points, data_sizes, g2a, a2g, n):
    """Per-cluster spatial / comm / load statistics + across-cluster balance.

    The 4D clustering objective minimizes the *within-cluster* variance of
    [spatial, G2A outage, A2G SINR, data volume], so the correct quality metric
    is the mean within-cluster variance (homogeneity), NOT the between-cluster
    std of means.  Both are reported: the ``*_within_var`` keys are what the 4D
    method actually optimizes; the ``*_std`` keys are kept for reference.
    """
    per_cluster = []
    sizes, loads, spatial, comm_out, comm_sinr = [], [], [], [], []
    within_out, within_sinr, within_load = [], [], []
    for c in range(n):
        mask = np.asarray(labels) == c
        size = int(mask.sum())
        load = float(data_sizes[mask].sum())
        center = points[mask].mean(axis=0)
        spread = float(np.mean(np.linalg.norm(points[mask] - center, axis=1)))
        out_mean = float(g2a[mask].mean())
        sinr_mean = float(a2g[mask].mean())
        sizes.append(size)
        loads.append(load)
        spatial.append(spread)
        comm_out.append(out_mean)
        comm_sinr.append(sinr_mean)
        within_out.append(float(np.var(g2a[mask])))
        within_sinr.append(float(np.var(a2g[mask])))
        within_load.append(float(np.var(data_sizes[mask])))
        per_cluster.append({
            "cluster": c,
            "size": size,
            "load": load,
            "spatial_spread": spread,
            "g2a_outage_mean": out_mean,
            "a2g_sinr_mean": sinr_mean,
        })
    return {
        "per_cluster": per_cluster,
        "sizes": sizes,
        "loads": loads,
        "size_std": float(np.std(sizes)),
        "load_std": float(np.std(loads)),
        "spatial_spread_std": float(np.std(spatial)),
        "g2a_outage_std": float(np.std(comm_out)),
        "a2g_sinr_std": float(np.std(comm_sinr)),
        # 4D 目标：簇内方差（越小越同质）
        "g2a_outage_within_var": float(np.mean(within_out)),
        "a2g_sinr_within_var": float(np.mean(within_sinr)),
        "load_within_var": float(np.mean(within_load)),
    }


def run(uav_nums=(2, 3, 4), seed=42, log_dir=None):
    log_dir = log_dir or _common.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    for uav_num in uav_nums:
        cfg, points, data_sizes = _common.load_scenario(uav_num, seed=seed)
        seed = int(cfg["seed"])
        g2a, a2g = _common.per_point_comm(points, seed)
        for method in _common.CLUSTER_METHODS:
            labels = _common.cluster_labels(method, points, data_sizes,
                                            uav_num, seed)
            result = {
                "run_id": "assign_%s_uav%d" % (method, uav_num),
                "uav_num": uav_num,
                "method": method,
                "seed": seed,
                # Per-point cluster labels (and coordinates) so the layer-1
                # clustering figure in plots/plot_results.py can color points by
                # cluster. Coordinates are in 100m units (as loaded).
                "points": [[float(x), float(y)] for x, y in points],
                "labels": [int(l) for l in labels],
            }
            result.update(cluster_metrics(labels, points, data_sizes,
                                          g2a, a2g, uav_num))
            path = os.path.join(log_dir,
                                "assign_%s_uav%d.json" % (method, uav_num))
            _common.save_json(path, result)
            print("saved -> %s" % path)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Layer-1 clustering comparison.")
    parser.add_argument("--uav_nums", type=str, default="2,3,4",
                        help="comma-separated UAV counts (e.g. '2,3,4')")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args(argv)
    uav_nums = tuple(int(x) for x in args.uav_nums.split(",") if x.strip())
    return run(uav_nums=uav_nums, seed=args.seed, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
