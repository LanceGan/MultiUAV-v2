"""Task 11 — one-command reproduction of the full multi-UAV pipeline.

Runs the four experiment layers in order and then regenerates every paper
figure from the logs, so the entire paper result set can be reproduced with a
single command::

    python experiment/run_all.py                      # full 2/3/4 grid
    python experiment/run_all.py --uav_nums 2         # smoke (N=2 only)
    python experiment/run_all.py --uav_nums 2 --total_episode 200

Layers (each writes its own ``results/logs/*.json`` / ``*.npz``):
  1. ``run_assign``   — layer-1 task assignment (naive / balanced / jia / 4D)
  2. ``run_seq``      — layer-2 sequence planning (ga_euclidean / pso / aco /
                        nn_2opt / ga_eqtsp)
  3. ``run_traj``     — layer-3 trajectory (Baseline greedy vs Proposed TD3)
  4. ``run_end2end``  — full all-baseline vs all-proposed comparison
  5. ``plot_results`` — regenerate every paper figure from the logs
"""

import argparse
import os
import sys

# Make the project root importable when run as ``python experiment/run_all.py``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _common, run_assign, run_end2end, run_seq, run_traj
from plots import plot_results

FIGS_DIR = os.path.join(_common.ROOT, "results", "figs")


def run(uav_nums=(2, 3, 4), total_episode=300, seed=42,
        log_dir=None, figs_dir=None):
    """Execute the full pipeline in order and return a summary dict."""
    log_dir = log_dir or _common.LOG_DIR
    figs_dir = figs_dir or FIGS_DIR

    print("=" * 70)
    print("[1/5] layer-1 task assignment (naive / balanced / jia / 4D)")
    run_assign.run(uav_nums=uav_nums, seed=seed, log_dir=log_dir)

    print("=" * 70)
    print("[2/5] layer-2 sequence planning "
          "(ga_euclidean / pso / aco / nn_2opt / ga_eqtsp)")
    run_seq.run(uav_nums=uav_nums, seed=seed, log_dir=log_dir)

    print("=" * 70)
    print("[3/5] layer-3 trajectory (Baseline greedy vs Proposed TD3)")
    run_traj.run(uav_nums=uav_nums, seed=seed, total_episode=total_episode,
                 log_dir=log_dir)

    print("=" * 70)
    print("[4/5] end-to-end (AllBaseline vs AllProposed)")
    run_end2end.run(uav_nums=uav_nums, seed=seed, total_episode=total_episode,
                    log_dir=log_dir)

    print("=" * 70)
    print("[5/5] regenerate figures from logs")
    figures = plot_results.plot_all(logs_dir=log_dir, figs_dir=figs_dir)

    print("=" * 70)
    print("pipeline complete: %d figure(s) generated in %s"
          % (len(figures), figs_dir))
    return {"uav_nums": list(uav_nums), "total_episode": int(total_episode),
            "seed": int(seed), "figures": figures}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="One-command reproduction of the full multi-UAV pipeline.")
    parser.add_argument("--uav_nums", type=int, nargs="+", default=(2, 3, 4),
                        help="space-separated UAV counts (default: 2 3 4)")
    parser.add_argument("--total_episode", type=int, default=300,
                        help="TD3 training episodes (default: 300)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--figs_dir", type=str, default=None)
    args = parser.parse_args(argv)
    return run(uav_nums=tuple(args.uav_nums), total_episode=args.total_episode,
               seed=args.seed, log_dir=args.log_dir, figs_dir=args.figs_dir)


if __name__ == "__main__":
    main()
