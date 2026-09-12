"""Layer-2 lambda_out sweep (consistent executor).

Scans the G2A-outage weight ``lambda_out`` in the sequence edge cost to find the
balance point where the comm-aware GA_EQTSP sequence still lowers the *final*
outage duration WITHOUT blowing up makespan (the current ``lambda_out = 3.0``
over-detours, trading a +35% makespan for a -23.6% outage at N=4).

For each candidate ``lambda_out`` we re-plan the per-UAV sequence with GA_EQTSP
(fixed 4D clustering) and execute it with the SAME deterministic greedy
straight-line policy, reporting makespan / outage / delay.  The Euclidean-GA
sequence (which does not use the radio map) is the lambda-independent baseline.

CLI::

    python experiment/run_lambda_sweep.py                     # N=2,4, lambda 0.5..2.0
    python experiment/run_lambda_sweep.py --uav_nums 4         # N=4 only
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _common, _traj
from sequence import baselines, ga_eqtsp

DEFAULT_LAMBDAS = (0.5, 1.0, 1.5, 2.0)


def run(uav_nums=(2, 4), lambda_outs=DEFAULT_LAMBDAS, seed=42, log_dir=None):
    log_dir = log_dir or _common.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    for uav_num in uav_nums:
        cfg, points, data_sizes = _common.load_scenario(uav_num, seed=seed)
        seed = int(cfg["seed"])
        labels = _common.cluster_labels("4d", points, data_sizes, uav_num, seed)
        clusters = _common.split_clusters(labels, points, data_sizes, uav_num)
        greedy = _traj.GreedyPolicy(cfg["length"], cfg["width"], cfg["v_max"])

        def _rollout(seq_fn, lam, name):
            cfg["lambda_out"] = lam
            orders = []
            for cpts, _gidx, cds in clusters:
                _, o, _ = seq_fn(cpts, cfg["ini_loc"], cfg["end_loc"], cds, cfg)
                orders.append(o)
            traverse = _common.build_traverse(clusters, orders)
            return _traj.rollout_evaluate(
                cfg, traverse, greedy,
                pipeline_run_id="LamSweep_%s_uav%d" % (name, uav_num),
                method="lam_%s" % name,
                brief_name="lam_sweep_%s_uav%d" % (name, uav_num),
                log_dir=log_dir)

        m = _rollout(baselines.ga_euclidean, cfg["lambda_out"], "euclidean")
        print("N=%d  euclidean      -> makespan %6.1fs | outage %6.1fs | delay %6.1fs"
              % (uav_num, m["makespan"], m["outage_duration"], m["total_delay"]))
        for lam in lambda_outs:
            m = _rollout(ga_eqtsp.ga_eqtsp, lam, "lam%.1f" % lam)
            print("N=%d  ga_eqtsp l=%.1f -> makespan %6.1fs | outage %6.1fs | delay %6.1fs"
                  % (uav_num, lam, m["makespan"], m["outage_duration"], m["total_delay"]))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sweep lambda_out for the layer-2 sequence.")
    parser.add_argument("--uav_nums", type=str, default="2,4")
    parser.add_argument("--lambda_outs", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args(argv)
    uav_nums = tuple(int(x) for x in args.uav_nums.split(",") if x.strip())
    lambda_outs = tuple(float(x) for x in args.lambda_outs.split(",") if x.strip())
    return run(uav_nums=uav_nums, lambda_outs=lambda_outs, seed=args.seed,
               log_dir=args.log_dir)


if __name__ == "__main__":
    main()
