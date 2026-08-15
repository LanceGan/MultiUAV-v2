"""Unified evaluation (Task 8).

Reads a raw rollout log ``results/logs/<run_id>.npz`` (written by
``trajectory/test.py``) and computes the five real end-to-end metrics directly
from the recorded data — no manual adjustment (spec §7.2 / §7.3):

* ``makespan`` (s)      : max over UAV completion times (unfinished -> horizon).
* ``total_delay`` (s)   : Σ_k τ_k, the per-point offload delay.
* ``avg_uplink_rate``   : mean instantaneous A2G uplink rate (Mbps).
* ``outage_duration``   : Σ outage·Δt, G2A downlink outage (s).
* ``total_energy``      : Σ_i E_i, rotary-wing flight energy (J).

Writes ``results/logs/<run_id>.metrics.json``.

CLI::

    python experiment/evaluate.py --run_id UAV_2_Ours_best
"""

import argparse
import json
import os
import sys

# Make the project root importable when run as ``python experiment/evaluate.py``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def evaluate_log(log):
    """Compute the five metrics from a loaded rollout log (npz NpzFile/dict)."""
    uav_num = int(log["uav_num"])
    delta_t = float(log["delta_t"])
    T = int(log["T"])

    times = np.asarray(log["times"], dtype=np.float64)
    energies = np.asarray(log["energies"], dtype=np.float64)
    outage_duration = np.asarray(log["outage_duration"], dtype=np.float64)

    # total_delay = Σ_k τ_k across all UAVs and all visited points.
    total_delay = 0.0
    for i in range(uav_num):
        tau = np.asarray(log["tau_%d" % i], dtype=np.float64)
        total_delay += float(np.sum(tau))

    # avg_uplink_rate = Σ rate / Σ active steps (mean instantaneous rate).
    rate_sum = float(np.sum(np.asarray(log["rate_sum"], dtype=np.float64)))
    rate_steps = int(np.sum(np.asarray(log["rate_steps"], dtype=np.int64)))
    avg_uplink_rate = (rate_sum / rate_steps) if rate_steps > 0 else 0.0

    # makespan (s): completion step of the slowest UAV; unfinished -> horizon T.
    completion = np.where(times < 0.0, float(T), times)
    makespan_steps = float(np.max(completion))
    makespan = makespan_steps * delta_t

    return {
        "makespan": float(makespan),
        "makespan_steps": float(makespan_steps),
        "total_delay": float(total_delay),
        "avg_uplink_rate": float(avg_uplink_rate),
        "outage_duration": float(np.sum(outage_duration)),
        "total_energy": float(np.sum(energies)),
    }


def main(run_id, log_dir="results/logs"):
    npz_path = os.path.join(log_dir, "%s.npz" % run_id)
    if not os.path.exists(npz_path):
        raise FileNotFoundError("rollout log not found: %s" % npz_path)

    log = np.load(npz_path)
    metrics = evaluate_log(log)
    metrics["run_id"] = run_id
    metrics["uav_num"] = int(log["uav_num"])

    out_path = os.path.join(log_dir, "%s.metrics.json" % run_id)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(json.dumps(metrics, indent=2))
    print("saved metrics -> %s" % out_path)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the five real end-to-end metrics (Task 8).")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="results/logs")
    args = parser.parse_args()
    main(args.run_id, log_dir=args.log_dir)
