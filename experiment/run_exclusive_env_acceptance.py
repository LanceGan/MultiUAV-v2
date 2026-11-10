"""Deterministic acceptance run for the exclusive-sector environment.

This is an environment-integration test, not a paper performance experiment.
It uses the fixed U=2/s42 scenario, a region-partition distance route, and a
deterministic buffered waypoint follower.  The run passes only when every task
is served, every payload is uploaded, exclusive sector assignment remains
one-to-one, and all UAVs reach the terminal.
"""

import argparse
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segment.stc.assignment import region_partition
from segment.stc.e2e_exclusive import ExclusiveSectorFixedRouteEnv
from segment.stc.radio import PerSectorRadioField
from segment.stc.run_context import RunContext
from segment.stc.scenario import load_stc_scenario
from segment.stc.sequencing import distance_nn_2opt


EXPECTED = ("summary.json", "records.json", "routes.json", "trajectory.npz")


def _action(env, uav_id, outside_upload_m):
    target = env._target_position(uav_id)
    delta = np.asarray(target, dtype=float) - env.positions[uav_id]
    distance = float(np.linalg.norm(delta))
    theta = math.atan2(delta[1], delta[0])
    reserve = outside_upload_m if env.buffers[uav_id] > 1e-9 else 5.0
    speed = min(env.v_max, max(0.0, (distance - reserve) / env.dt))
    guide_delta = env.guide_target(uav_id) - env.positions[uav_id]
    guide_bearing = math.atan2(guide_delta[1], guide_delta[0])
    return [speed, env._wrap_angle(theta - guide_bearing)]


def run(args):
    start = time.perf_counter()
    scenario = load_stc_scenario(
        args.uav_num, task_seed=args.task_seed, map_seed=args.map_seed
    )
    field = PerSectorRadioField.from_scenario(scenario, spacing_m=args.spacing_m)
    if args.routes_json:
        payload = json.loads(Path(args.routes_json).read_text(encoding="utf-8"))
        raw_routes = payload.get("routes", payload)
        routes = {int(k): [int(t) for t in v] for k, v in raw_routes.items()}
        route_source = str(Path(args.routes_json).resolve())
    else:
        assignment = region_partition(scenario, seed=args.task_seed)
        routes = {}
        for uav_id, cluster in enumerate(assignment.clusters):
            sequence = distance_nn_2opt(scenario, cluster, seed=args.task_seed)
            routes[uav_id] = list(sequence.route)
        route_source = "region_partition + distance_nn_2opt"

    env = ExclusiveSectorFixedRouteEnv(
        scenario,
        field,
        routes,
        horizon_steps=args.horizon_steps,
        arrival_radius_m=args.arrival_radius_m,
        safe_distance_m=args.safe_distance_m,
        outage_probability_threshold=args.outage_probability_threshold,
        outage_max_s=args.outage_max_s,
    )

    config = {
        "experiment": "exclusive environment acceptance",
        "scenario_label": "s42",
        "uav_num": int(args.uav_num),
        "n_tasks": int(scenario.n_tasks),
        "task_seed": int(args.task_seed),
        "map_seed": int(args.map_seed),
        "policy": args.policy,
        "route_source": route_source,
        "arrival_radius_m": float(args.arrival_radius_m),
        "safe_distance_m": float(args.safe_distance_m),
        "outage_probability_threshold": float(args.outage_probability_threshold),
        "outage_max_s": float(args.outage_max_s),
        "horizon_steps": int(args.horizon_steps),
        "outside_upload_m": float(args.outside_upload_m),
        "spacing_m": float(args.spacing_m),
    }
    rc = RunContext.create(
        args.run_root, "exclusive_env_acceptance", config,
        run_id=args.run_id, expected=EXPECTED
    )
    rc.register_input("task_positions_m", data=scenario.task_positions_m)
    rc.register_input("task_data_bits", data=scenario.task_data_bits)
    rc.register_input("routes", data=routes)

    records = []
    obs = env.reset()
    while True:
        if args.policy == "mapguide":
            actions = [env.guide_action(uav_id) for uav_id in range(env.n_uav)]
        else:
            actions = [
                _action(env, uav_id, args.outside_upload_m)
                for uav_id in range(env.n_uav)
            ]
        obs, rewards, terminal, info = env.step(actions)
        records.append({
            "t": int(env.t),
            "actions": [[float(x) for x in row] for row in actions],
            "positions_m": env.positions.tolist(),
            "buffers_bits": env.buffers.tolist(),
            "sectors": info["sectors"].tolist(),
            "visited_tasks": int(np.sum(info["visited"])),
            "done": info["done"].tolist(),
            "reward_sum": float(np.sum(rewards)),
        })
        if terminal or info["truncated"]:
            break

    metrics = env.metrics()
    objective = {
        "t_ref_s": metrics["makespan_s"],
        "d_ref_s": metrics["total_delay_s"],
        "source": "acceptance policy: %s" % args.policy,
        "is_formal_reference": bool(args.policy == "buffered_waypoint"),
    }
    summary = {
        "run_id": rc.run_id,
        "status": "accepted" if metrics["feasible"] else "rejected",
        "elapsed_s": float(time.perf_counter() - start),
        "metrics": metrics,
        "reference": objective,
        "sector_count": int(field.n_sectors),
        "route_lengths": {str(k): len(v) for k, v in routes.items()},
    }

    trajectory_bytes = io.BytesIO()
    arrays = {
        "traj_%d" % i: np.asarray(env.trajectories[i], dtype=float)
        for i in range(env.n_uav)
    }
    np.savez(trajectory_bytes, **arrays)

    rc.write_json("summary", summary)
    rc.write_json("records", records)
    rc.write_json("routes", {str(k): v for k, v in routes.items()})
    rc.write_bytes("trajectory.npz", trajectory_bytes.getvalue())
    manifest = rc.finalize("success" if metrics["feasible"] else "failed")

    print("run_id=%s" % rc.run_id)
    print("status=%s" % summary["status"])
    print("visited=%d/%d uploaded=%d/%d" % (
        metrics["visited_tasks"], scenario.n_tasks,
        metrics["uploaded_tasks"], scenario.n_tasks
    ))
    print("sector_conflicts=%d early_service=%d early_terminal=%d" % (
        metrics["sector_conflicts"],
        metrics["early_service_arrivals"],
        metrics["early_terminal_arrivals"],
    ))
    print("T_s=%.3f D_s=%.3f feasible=%s" % (
        metrics["makespan_s"], metrics["total_delay_s"], metrics["feasible"]
    ))
    print("manifest=%s" % manifest)
    if not metrics["feasible"]:
        raise RuntimeError("exclusive environment acceptance failed: %r"
                           % metrics["failure_reasons"])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uav-num", type=int, default=2)
    parser.add_argument("--task-seed", type=int, default=42)
    parser.add_argument("--map-seed", type=int, default=42)
    parser.add_argument("--arrival-radius-m", type=float, default=10.0)
    parser.add_argument("--outside-upload-m", type=float, default=12.0)
    parser.add_argument("--safe-distance-m", type=float, default=30.0)
    parser.add_argument("--outage-probability-threshold", type=float, default=0.5)
    parser.add_argument("--outage-max-s", type=float, default=20.0)
    parser.add_argument("--horizon-steps", type=int, default=2000)
    parser.add_argument("--spacing-m", type=float, default=200.0)
    parser.add_argument("--run-root", default="results/runs")
    parser.add_argument("--run-id", default="acceptance-exclusive-u2-s42-r1")
    parser.add_argument("--routes-json", default=None)
    parser.add_argument(
        "--policy", choices=("buffered_waypoint", "mapguide"),
        default="buffered_waypoint",
    )
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
