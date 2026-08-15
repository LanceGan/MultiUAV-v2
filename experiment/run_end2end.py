"""Task 10.4 — full three-layer end-to-end comparison.

For N in {2, 3, 4} compares the *all-baseline* pipeline against the
*all-proposed* pipeline and writes end-to-end metrics:

* **all-baseline** = naive (spatial-only) clustering + Euclidean GA sequence +
  greedy non-comm-aware trajectory.
* **all-proposed** = 4D comm-aware clustering + comm-aware ga_eqtsp sequence +
  comm-aware TD3 trajectory.

The all-proposed leg is identical to ``run_traj``'s ``Proposed`` pipeline, so it
*reuses* that already-trained checkpoint (``--model_subdir``, default
``Proposed``) instead of retraining a duplicate TD3 — run
``experiment/run_traj.py`` first.  This keeps the bar chart from showing two
different-valued bars for the same algorithm.

Writes ``results/logs/end2end_<baseline|proposed>_uav<N>.json`` (plus the
plot-compatible ``UAV_<N>_<AllBaseline|Proposed>`` artifacts).

CLI::

    python experiment/run_end2end.py --uav_nums 2   # smoke (needs run_traj first)
    python experiment/run_end2end.py                # full grid
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _common, _traj
from sequence import baselines, ga_eqtsp


def build_traverse(uav_num, seed, cluster_method, seq_fn):
    """Cluster + sequence a scenario, return ``(cfg, traverse)``."""
    cfg, points, data_sizes = _common.load_scenario(uav_num, seed=seed)
    labels = _common.cluster_labels(cluster_method, points, data_sizes,
                                    uav_num, seed)
    clusters = _common.split_clusters(labels, points, data_sizes, uav_num)
    orders = []
    for (cpts, _gidx, cds) in clusters:
        _, order_idx, _ = seq_fn(cpts, cfg["ini_loc"], cfg["end_loc"], cds, cfg)
        orders.append(order_idx)
    return cfg, _common.build_traverse(clusters, orders)


def run(uav_nums=(2, 3, 4), seed=42, total_episode=300,
        model_subdir="Proposed", log_dir=None):
    # ``total_episode`` is retained for CLI / run_all compatibility only: the
    # all-proposed leg now REUSES run_traj's ``Proposed`` checkpoint (same 4D +
    # ga_eqtsp assignment) instead of retraining an identical TD3.
    _ = total_episode
    log_dir = log_dir or _common.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    results = []
    for uav_num in uav_nums:
        # --- all-baseline: naive + ga_euclidean + greedy ---
        cfg_base, traverse_base = build_traverse(
            uav_num, seed, "naive", baselines.ga_euclidean)
        greedy = _traj.GreedyPolicy(cfg_base["length"], cfg_base["width"],
                                    cfg_base["v_max"])
        base_metrics = _traj.rollout_evaluate(
            cfg_base, traverse_base, greedy,
            pipeline_run_id="UAV_%d_AllBaseline" % uav_num,
            method="AllBaseline",
            brief_name="end2end_baseline_uav%d" % uav_num,
            log_dir=log_dir)
        results.append(base_metrics)

        # --- all-proposed: 4d + ga_eqtsp + TD3 (reuse run_traj's model) ---
        cfg_prop, traverse_prop = build_traverse(
            uav_num, seed, "4d", ga_eqtsp.ga_eqtsp)
        model = _traj.load_td3(cfg_prop, traverse_prop,
                               model_subdir=model_subdir)
        prop_metrics = _traj.rollout_evaluate(
            cfg_prop, traverse_prop, model,
            pipeline_run_id="UAV_%d_%s" % (uav_num, model_subdir),
            method=model_subdir,
            brief_name="end2end_proposed_uav%d" % uav_num,
            log_dir=log_dir)
        results.append(prop_metrics)

        print("uav_num=%d baseline=%s proposed=%s"
              % (uav_num, base_metrics, prop_metrics))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Full-pipeline end-to-end comparison.")
    parser.add_argument("--uav_nums", type=str, default="2,3,4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_episode", type=int, default=300,
                        help="retained for run_all compatibility (unused: model is reused)")
    parser.add_argument("--model_subdir", type=str, default="Proposed",
                        help="name of the run_traj checkpoint to reuse")
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args(argv)
    uav_nums = tuple(int(x) for x in args.uav_nums.split(",") if x.strip())
    return run(uav_nums=uav_nums, seed=args.seed,
               total_episode=args.total_episode,
               model_subdir=args.model_subdir, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
