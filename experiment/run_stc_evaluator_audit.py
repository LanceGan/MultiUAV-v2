"""RQ2 evaluator audit entry point (Task 14).

Compares exclusive-bandwidth, static average-factor, and synchronous evaluation
on identical solutions.  Full implementation at the formal experiment stage.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    parser = argparse.ArgumentParser(description="RQ2 evaluator audit")
    parser.add_argument("--uav_num", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print("RQ2 evaluator audit dry-run: uav=%d" % args.uav_num)
    return 0


if __name__ == "__main__":
    main()
