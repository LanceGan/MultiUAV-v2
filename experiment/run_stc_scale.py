"""STC controlled-scale entry point (Task 15)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.run_stc_main import _common_args


def main(argv=None):
    parser = argparse.ArgumentParser(description="STC controlled-scale families")
    _common_args(parser)
    args = parser.parse_args(argv)
    if args.dry_run:
        print("scale families: vary-UAV-count (fixed area/BS/tasks), vary-task-scale "
              "(area/task/BS density), comm-load, map-seed, fast-fading replay")
        return 0
    return 0


if __name__ == "__main__":
    main()
