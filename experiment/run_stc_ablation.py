"""STC mechanism ablation entry point (Task 15)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.run_stc_main import _common_args


def main(argv=None):
    parser = argparse.ArgumentParser(description="STC mechanism ablation")
    _common_args(parser)
    args = parser.parse_args(argv)
    if args.dry_run:
        print("ablation matrix: single-mechanism removals (fixed-weight, no-hotspot, "
              "no-conflict-repair, no-path-switch, no-departure-shift, single-candidate)")
        return 0
    return 0


if __name__ == "__main__":
    main()
