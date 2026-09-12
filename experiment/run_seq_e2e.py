"""Layer-2 sequence end-to-end verification (consistent executor).

Question this answers: does the comm-aware GA_EQTSP sequence's advantage (a
*lower* comm_cost on the intermediate metric) translate into *better final
metrics* (shorter outage duration / transmission delay) when the trajectory is
executed by the SAME deterministic straight-line (greedy) policy?

The earlier ``run_end2end`` comparison confounded three variables at once
(clustering + sequence + executor) and let the suboptimal TD3 executor drag the
"proposed" arm down.  Here we hold clustering (4D) and the executor (greedy
straight-line) FIXED and vary only the per-UAV visit sequence: Euclidean GA vs
comm-aware GA_EQTSP.

CLI::

    python experiment/run_seq_e2e.py               # full 2/3/4 grid
    python experiment/run_seq_e2e.py --uav_nums 2  # N=2 only (smoke)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _common, _traj
from sequence import baselines, ga_eqtsp

# The two sequences compared on the SAME (4D-clustered, greedy-executed) setting.
SEQ_COMPARE = {
    "euclidean": baselines.ga_euclidean,
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

        greedy = _traj.GreedyPolicy(cfg["length"], cfg["width"], cfg["v_max"])
        for name, fn in SEQ_COMPARE.items():
            orders = []
            for cpts, _gidx, cds in clusters:
                _, order_idx, _ = fn(cpts, cfg["ini_loc"], cfg["end_loc"], cds, cfg)
                orders.append(order_idx)
            traverse = _common.build_traverse(clusters, orders)

            m = _traj.rollout_evaluate(
                cfg, traverse, greedy,
                pipeline_run_id="SeqE2E_%s_uav%d" % (name, uav_num),
                method="seq_%s" % name,
                brief_name="seq_e2e_%s_uav%d" % (name, uav_num),
                log_dir=log_dir)
            print("SeqE2E uav%d %-9s -> makespan %.1fs | outage %.1fs | delay %.1fs | rate %.2f"
                  % (uav_num, name, m["makespan"], m["outage_duration"],
                     m["total_delay"], m["avg_uplink_rate"]))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Layer-2 sequence end-to-end check (fixed 4D clustering + greedy executor).")
    parser.add_argument("--uav_nums", type=str, default="2,3,4",
                        help="comma-separated UAV counts (e.g. '2,3,4')")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args(argv)
    uav_nums = tuple(int(x) for x in args.uav_nums.split(",") if x.strip())
    return run(uav_nums=uav_nums, seed=args.seed, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
