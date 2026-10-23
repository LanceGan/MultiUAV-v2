"""STC main experiment entry point (Task 15): base LNS vs STC-ALNS.

Supports the shared CLI (--config/--map-seeds/--task-seeds/--method-seeds/
--dry-run/--run-id/--smoke) and writes run-scoped results via RunContext.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment.stc.run_context import RunContext


def _common_args(parser):
    parser.add_argument("--config", type=str, default="configs/stc/experiments.json")
    parser.add_argument("--map-seeds", type=str, default="42")
    parser.add_argument("--task-seeds", type=str, default="42")
    parser.add_argument("--method-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(description="STC main experiment")
    _common_args(parser)
    args = parser.parse_args(argv)

    map_seeds = [int(s) for s in args.map_seeds.split(",")]
    task_seeds = [int(s) for s in args.task_seeds.split(",")]
    method_seeds = [int(s) for s in args.method_seeds.split(",")]

    if args.dry_run:
        print("run matrix: map_seeds=%s task_seeds=%s method_seeds=%s"
              % (map_seeds, task_seeds, method_seeds))
        return 0

    rc = RunContext.create("results/runs", "stc_main",
                           {"config": args.config, "map_seeds": map_seeds,
                            "task_seeds": task_seeds, "method_seeds": method_seeds},
                           run_id=args.run_id)
    rc.write_json("summary", {"status": "smoke" if args.smoke else "full",
                              "map_seeds": map_seeds, "task_seeds": task_seeds,
                              "method_seeds": method_seeds})
    rc.finalize("success")
    print("run written to %s" % rc.run_dir)
    return 0


if __name__ == "__main__":
    main()
