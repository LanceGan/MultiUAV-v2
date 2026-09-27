"""Task 10.2 — layer-2 sequence-planning comparison (fixed 4D clustering).

For N in {2, 3, 4} and each of the five sequence methods (ga_euclidean / pso /
aco / nn_2opt / ga_eqtsp), plans each UAV's ordered route over its 4D-clustered
points, then evaluates that route on *both* objectives — the comm-aware path
cost (``sequence.radio_path.edge_cost``) and the pure Euclidean path length —
so every method is comparable on the same yardstick.  Writes
``results/logs/seq_<method>_uav<N>.json``.

CLI::

    python experiment/run_seq.py               # full 2/3/4 grid
    python experiment/run_seq.py --uav_nums 2  # N=2 only (smoke)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _common
from sequence import baselines, ga_eqtsp

SEQ_METHODS = {
    "ga_euclidean": baselines.ga_euclidean,
    "pso": baselines.pso,
    "aco": baselines.aco,
    "nn_2opt": baselines.nn_2opt,
    "ga_eqtsp": ga_eqtsp.ga_eqtsp,
}


def run(uav_nums=(2, 3, 4), seed=42, log_dir=None):
    log_dir = log_dir or _common.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    for uav_num in uav_nums:
        cfg, points, data_sizes = _common.load_scenario(uav_num, seed=seed)
        seed = int(cfg["seed"])
        labels = _common.cluster_labels("4d", points, data_sizes, uav_num, seed)
        clusters = _common.split_clusters(labels, points, data_sizes, uav_num)

        for name, fn in SEQ_METHODS.items():
            total_eucl = 0.0
            total_comm = 0.0
            per_uav = []
            for c, (cpts, gidx, cds) in enumerate(clusters):
                ordered, order_idx, method_cost = fn(
                    cpts, cfg["ini_loc"], cfg["end_loc"], cds, cfg)
                eucl, comm = _common.path_cost(ordered, cds, order_idx, cfg, seed)
                total_eucl += eucl
                total_comm += comm
                per_uav.append({
                    "uav": c,
                    "method_cost": float(method_cost),
                    "euclidean_length": float(eucl),
                    "comm_cost": float(comm),
                    "order_indices": [int(x) for x in order_idx],
                    # 轨迹图需要：访问顺序的全局点索引 + 完整顺序坐标（含起终点锚点）
                    "ordered_global_indices": [int(gidx[x]) for x in order_idx],
                    "ordered_points": [[float(x), float(y)] for x, y in ordered],
                })

            result = {
                "run_id": "seq_%s_uav%d" % (name, uav_num),
                "uav_num": uav_num,
                "method": name,
                "seed": seed,
                "total_euclidean_length": float(total_eucl),
                "total_comm_cost": float(total_comm),
                "per_uav": per_uav,
            }
            path = os.path.join(log_dir, "seq_%s_uav%d.json" % (name, uav_num))
            _common.save_json(path, result)
            print("saved -> %s" % path)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Layer-2 sequence comparison.")
    parser.add_argument("--uav_nums", type=str, default="2,3,4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args(argv)
    uav_nums = tuple(int(x) for x in args.uav_nums.split(",") if x.strip())
    return run(uav_nums=uav_nums, seed=args.seed, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
