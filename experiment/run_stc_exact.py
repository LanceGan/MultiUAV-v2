"""Tiny-instance exact benchmark entry point (Task 13)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment.stc.exact import solve_exact_tiny
from segment.stc.generators import generate_straight_hold
from segment.stc.radio import PerSectorRadioField
from segment.stc.scenario import load_stc_scenario


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav_num", type=int, default=2)
    parser.add_argument("--max_tasks", type=int, default=6)
    parser.add_argument("--map-seed", type=int, default=42)
    parser.add_argument("--task-seed", type=int, default=42)
    args = parser.parse_args(argv)

    sc = load_stc_scenario(args.uav_num, task_seed=args.task_seed, map_seed=args.map_seed)
    field = PerSectorRadioField.from_scenario(sc, spacing_m=1000.0)
    lib = {}
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src != tgt:
                lib[(src, tgt)] = [generate_straight_hold(sc, field, src, tgt)]

    res = solve_exact_tiny(sc, lib, field, max_tasks=args.max_tasks)
    print("exact: objective=%.4f proven_optimal=%s explored=%d runtime=%.1fs"
          % (res.objective, res.proven_optimal, res.explored, res.runtime_s))
    return 0


if __name__ == "__main__":
    main()
