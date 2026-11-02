"""SI 单段 DRL 训练入口（SAC / TD3 / DDPG，目标条件化，held-out 泛化评估）。

与正式首片同机运行：线程数钉死为 4（OMP + torch），不争抢正式实验的 CPU。
产物写入 ``results/models/stc_drl/``（checkpoint + 奖励/可行率 npz + provenance
JSON），不走 RunContext（那是正式实验目录的契约）。正式 3000 集示例::

    ./.venv/bin/python experiment/train_stc_drl.py --algo SAC --run-id si-v1

冒烟（粗场 20 集，验证端到端不崩）::

    ./.venv/bin/python experiment/train_stc_drl.py --algo SAC \
        --total-episodes 20 --spacing-m 1000 --run-id smoke
"""

import os

# 必须在任何 torch import 之前设置（numpy/torch 的 OpenMP 线程池按此初始化）
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.set_num_threads(4)

from segment.stc.drl_training import ALGOS, train
from segment.stc.radio import PerSectorRadioField
from segment.stc.scenario import load_stc_scenario


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", default="SAC", choices=list(ALGOS))
    parser.add_argument("--total-episodes", type=int, default=3000)
    parser.add_argument("--uav-num", type=int, default=2)
    parser.add_argument("--map-seed", type=int, default=42)
    parser.add_argument("--task-seed", type=int, default=42)
    parser.add_argument("--spacing-m", type=float, default=200.0)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--held-out-frac", type=float, default=0.2)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--run-id", default="si-v1")
    parser.add_argument("--out-dir", default="results/models/stc_drl")
    args = parser.parse_args(argv)

    scenario = load_stc_scenario(args.uav_num, task_seed=args.task_seed,
                                 map_seed=args.map_seed)
    print("[train_stc_drl] scenario u%d K=%d | map/task seed %d/%d | field %.0fm"
          % (scenario.n_uav, scenario.n_tasks, args.map_seed, args.task_seed,
             args.spacing_m))
    field = PerSectorRadioField.from_scenario(scenario, spacing_m=args.spacing_m)

    result = train(scenario, field, algo=args.algo,
                   total_episodes=args.total_episodes,
                   train_seed=args.train_seed,
                   held_out_frac=args.held_out_frac,
                   eval_interval=args.eval_interval,
                   out_dir=args.out_dir, run_id=args.run_id)
    print("=== done === best held-out feasible %.1f%% | best=%s"
          % (100.0 * result["best_heldout_feasible"], result["best_path"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
