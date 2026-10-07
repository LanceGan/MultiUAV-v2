"""MARL algorithm comparison for the trajectory layer.

Trains four multi-agent algorithms — the existing parameter-shared independent
TD3 (``ILD3``) plus the three newly implemented CTDE baselines ``MADDPG`` /
``MATD3`` / ``MAPPO`` — on the SAME environment, scenario (4D clustering +
GA_EQTSP sequence), seed, and episode budget, so their training reward curves
are directly comparable.

Each trainer writes ``results/logs/UAV_<N>_<Algo>_rewards.npz`` (``ep_rewards``),
which ``plots/plot_results.py`` already collects into ``training_rewards.png``.

CLI::

    python experiment/run_marl_compare.py                       # all 4, 500 eps
    python experiment/run_marl_compare.py --algorithms ILD3 MATD3
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _common, _traj
from sequence import ga_eqtsp
from trajectory import maddpg, matd3, mappo

ALGORITHMS = ("ILD3", "MADDPG", "MATD3", "MAPPO")


def run(uav_num=2, total_episode=500, seed=42, algorithms=ALGORITHMS, log_dir=None,
        all_a2g=False):
    log_dir = log_dir or _common.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    cfg, points, data_sizes = _common.load_scenario(uav_num, seed=seed)
    seed = int(cfg["seed"])
    labels = _common.cluster_labels("4d", points, data_sizes, uav_num, seed)
    clusters = _common.split_clusters(labels, points, data_sizes, uav_num)
    orders = []
    for cpts, _gidx, cds in clusters:
        _, o, _ = ga_eqtsp.ga_eqtsp(cpts, cfg["ini_loc"], cfg["end_loc"], cds, cfg)
        orders.append(o)
    traverse = _common.build_traverse(clusters, orders)

    for algo in algorithms:
        print("===== training %s =====" % algo)
        # 默认：提出方法 ILD3 用双地图（G2A 中断 + A2G 上行时延），其余 DRL 基线只用 G2A。
        # --all_a2g：所有算法都用双地图（架构对比实验 a，消掉"地图"变量）。
        use_a2g = True if all_a2g else (algo == "ILD3")
        if algo == "ILD3":
            _traj.train_td3(cfg, traverse, total_episode=total_episode,
                            model_subdir="ILD3", log_dir=log_dir, use_a2g=use_a2g)
        elif algo == "MADDPG":
            maddpg.train_maddpg(cfg, traverse, total_episode=total_episode,
                                model_subdir="MADDPG", log_dir=log_dir, use_a2g=use_a2g)
        elif algo == "MATD3":
            matd3.train_matd3(cfg, traverse, total_episode=total_episode,
                              model_subdir="MATD3", log_dir=log_dir, use_a2g=use_a2g)
        elif algo == "MAPPO":
            mappo.train_mappo(cfg, traverse, total_episode=total_episode,
                              model_subdir="MAPPO", log_dir=log_dir, use_a2g=use_a2g)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare MARL algorithms on the trajectory layer.")
    parser.add_argument("--uav_num", type=int, default=2)
    parser.add_argument("--total_episode", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithms", type=str, nargs="+", default=list(ALGORITHMS))
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--all_a2g", action="store_true",
                        help="all algorithms use the dual map (A2G+G2A), for the architecture ablation")
    args = parser.parse_args(argv)
    return run(uav_num=args.uav_num, total_episode=args.total_episode,
               seed=args.seed, algorithms=args.algorithms, log_dir=args.log_dir,
               all_a2g=args.all_a2g)


if __name__ == "__main__":
    main()
