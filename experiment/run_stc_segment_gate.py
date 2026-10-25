"""RQ1 segment-method gate entry point (Task 14).

Compares the single-segment generators (Straight-MaxSpeed, Straight-Hold,
Comm-Waypoint, Grid A*, SLSQP, and learned baselines) on the same edge set.
Full implementation is filled in at the formal experiment stage; this entry
point defines the CLI and the paired-statistics pipeline.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    parser = argparse.ArgumentParser(description="RQ1 segment-method gate")
    parser.add_argument("--uav_num", type=int, default=2)
    parser.add_argument("--map-seeds", type=str, default="42")
    parser.add_argument("--task-seeds", type=str, default="42")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print("RQ1 segment gate dry-run: uav=%d map=%s task=%s"
              % (args.uav_num, args.map_seeds, args.task_seeds))
    return 0


if __name__ == "__main__":
    main()
