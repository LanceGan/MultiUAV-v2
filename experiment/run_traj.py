"""Task 10.3 — layer-3 trajectory comparison (fixed 4D clustering + ga_eqtsp).

For N in {2, 3, 4}, both trajectory approaches share the *same* clustered and
sequenced assignment (4D clustering + comm-aware ga_eqtsp):

* **Baseline** (non-comm-aware): a deterministic greedy waypoint follower that
  flies straight toward the current target at a fixed speed, with no radio-map
  awareness and no offload optimisation (``GreedyPolicy``).
* **Proposed** (comm-aware): the parameter-shared independent TD3 trained on the
  injected traverse, then rolled out deterministically.

Both produce the five real end-to-end metrics (makespan / total_delay /
avg_uplink_rate / outage_duration / total_energy) via ``experiment.evaluate``.
Writes ``results/logs/traj_<baseline|proposed>_uav<N>.json`` (plus the
plot-compatible ``UAV_<N>_<Baseline|Proposed>`` rollout/metrics/log artifacts).

CLI::

    python experiment/run_traj.py --uav_nums 2 --total_episode 200   # smoke
    python experiment/run_traj.py                                     # full grid
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _common, _traj
from sequence import ga_eqtsp


def fixed_traverse(uav_num, seed):
    """4D clustering + comm-aware ga_eqtsp -> ``{uav_id: [global indices]}``."""
    cfg, points, data_sizes = _common.load_scenario(uav_num, seed=seed)
    labels = _common.cluster_labels("4d", points, data_sizes, uav_num, seed)
    clusters = _common.split_clusters(labels, points, data_sizes, uav_num)
    orders = []
    for (cpts, _gidx, cds) in clusters:
        _, order_idx, _ = ga_eqtsp.ga_eqtsp(
            cpts, cfg["ini_loc"], cfg["end_loc"], cds, cfg)
        orders.append(order_idx)
    return cfg, points, data_sizes, _common.build_traverse(clusters, orders)


def run(uav_nums=(2, 3, 4), seed=42, total_episode=300,
        model_subdir="Proposed", log_dir=None):
    log_dir = log_dir or _common.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    results = []
    for uav_num in uav_nums:
        cfg, _points, _data_sizes, traverse = fixed_traverse(uav_num, seed)
        seed = int(cfg["seed"])

        # --- Baseline: greedy non-comm-aware waypoint follower ---
        greedy = _traj.GreedyPolicy(cfg["length"], cfg["width"], cfg["v_max"])
        base_metrics = _traj.rollout_evaluate(
            cfg, traverse, greedy,
            pipeline_run_id="UAV_%d_Baseline" % uav_num,
            method="Baseline",
            brief_name="traj_baseline_uav%d" % uav_num,
            log_dir=log_dir)
        results.append(base_metrics)

        # --- Proposed: comm-aware TD3 ---
        model = _traj.train_td3(
            cfg, traverse, total_episode=total_episode,
            model_subdir=model_subdir, log_dir=log_dir)
        prop_metrics = _traj.rollout_evaluate(
            cfg, traverse, model,
            pipeline_run_id="UAV_%d_%s" % (uav_num, model_subdir),
            method=model_subdir,
            brief_name="traj_proposed_uav%d" % uav_num,
            log_dir=log_dir)
        results.append(prop_metrics)

        print("uav_num=%d baseline=%s proposed=%s"
              % (uav_num, base_metrics, prop_metrics))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Layer-3 trajectory comparison.")
    parser.add_argument("--uav_nums", type=str, default="2,3,4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_episode", type=int, default=300)
    parser.add_argument("--model_subdir", type=str, default="Proposed")
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args(argv)
    uav_nums = tuple(int(x) for x in args.uav_nums.split(",") if x.strip())
    return run(uav_nums=uav_nums, seed=args.seed,
               total_episode=args.total_episode,
               model_subdir=args.model_subdir, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
