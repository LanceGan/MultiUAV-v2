"""Regenerate STC figures from frozen run directories (Task 15)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot STC results from frozen runs")
    parser.add_argument("--runs", type=str, nargs="+", required=True)
    args = parser.parse_args(argv)
    print("would plot from runs: %s" % ", ".join(args.runs))
    return 0


if __name__ == "__main__":
    main()
