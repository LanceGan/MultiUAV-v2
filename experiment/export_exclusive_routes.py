"""Export the fixed 4D + GA-EQTSP routes used by formal experiments."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment.run_exclusive_marl_pilot import _fixed_routes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uav-num", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    routes = _fixed_routes(args.uav_num, args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"routes": {str(k): v for k, v in routes.items()}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print("saved routes -> %s" % path.resolve())
    return routes


if __name__ == "__main__":
    main()
