"""RQ2 assignment and RQ3 sequencing under one full-route executor.

The current common executor is MapGuide. Swap-in MCO-A* can replace
``_execute_mapguide`` without changing the component comparisons.
"""

import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import _common
from segment.stc.assignment import region_partition
from segment.stc.e2e_exclusive import ExclusiveSectorFixedRouteEnv
from segment.stc.radio import PerSectorRadioField
from segment.stc.run_context import RunContext
from segment.stc.scenario import load_stc_scenario
from segment.stc.sequencing import (
    distance_nn_2opt,
    ga_eqtsp_sequence,
    jia_time_energy_tsp,
)


RQ2_METHODS = ("region", "spatial", "balanced", "jia", "4d")
RQ3_METHODS = ("nn_2opt", "euclidean_ga", "jia_time_energy", "ga_eqtsp")
EXPECTED = ("summary.json", "records.json", "routes.json", "trajectory.npz")


def _reference(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ref = payload.get("reference", {})
    return float(ref["t_ref_s"]), float(ref["d_ref_s"])


def _sequence(method, scenario, task_ids, seed):
    if method == "nn_2opt":
        return list(distance_nn_2opt(scenario, task_ids, seed=seed).route)
    if method == "jia_time_energy":
        return list(jia_time_energy_tsp(scenario, task_ids, seed=seed).route)
    if method == "ga_eqtsp":
        return list(ga_eqtsp_sequence(scenario, task_ids, seed=seed).route)
    if method == "euclidean_ga":
        from sequence import baselines

        cfg, points, data_sizes = _common.load_scenario(
            scenario.n_uav, seed=seed
        )
        _, order, _ = baselines.ga_euclidean(
            points[np.asarray(task_ids)],
            cfg["ini_loc"],
            cfg["end_loc"],
            data_sizes[np.asarray(task_ids)],
            cfg,
        )
        return [int(task_ids[i]) for i in order]
    raise ValueError("unknown sequencing method: %s" % method)


def _assignment_labels(method, uav_num, seed):
    cfg, points, data_sizes = _common.load_scenario(uav_num, seed=seed)
    if method == "region":
        scenario = load_stc_scenario(
            uav_num, task_seed=seed, map_seed=seed
        )
        result = region_partition(scenario, seed=seed)
        labels = np.empty(scenario.n_tasks, dtype=int)
        for cluster_id, cluster in enumerate(result.clusters):
            labels[np.asarray(cluster, dtype=int)] = cluster_id
        return labels, points, data_sizes
    baseline_method = "naive" if method == "spatial" else method
    return (
        _common.cluster_labels(
            baseline_method, points, data_sizes, uav_num, seed
        ),
        points,
        data_sizes,
    )


def _build_rq2_routes(method, scenario, seed):
    labels, _points, _data = _assignment_labels(
        method, scenario.n_uav, seed
    )
    routes = {}
    for uav_id in range(scenario.n_uav):
        task_ids = tuple(int(x) for x in np.flatnonzero(labels == uav_id))
        routes[uav_id] = list(ga_eqtsp_sequence(
            scenario, task_ids, seed=seed
        ).route)
    return routes


def _build_rq3_routes(method, scenario, seed):
    labels, _points, _data = _assignment_labels(
        "4d", scenario.n_uav, seed
    )
    routes = {}
    for uav_id in range(scenario.n_uav):
        task_ids = tuple(int(x) for x in np.flatnonzero(labels == uav_id))
        routes[uav_id] = _sequence(method, scenario, task_ids, seed)
    return routes


def _execute_mapguide(scenario, field, routes, args):
    env = ExclusiveSectorFixedRouteEnv(
        scenario, field, routes,
        horizon_steps=args.horizon_steps,
        arrival_radius_m=args.arrival_radius_m,
        safe_distance_m=args.safe_distance_m,
        outage_probability_threshold=args.outage_probability_threshold,
        outage_max_s=args.outage_max_s,
    )
    records = []
    env.reset()
    while True:
        actions = [env.guide_action(i) for i in range(env.n_uav)]
        actions = env.project_safe_actions(
            actions, outage_threshold=args.safety_outage_threshold
        )
        _, rewards, terminal, info = env.step(actions)
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
    return env, records


def _run_one(kind, method, args, reference):
    scenario = load_stc_scenario(
        args.uav_num, task_seed=args.seed, map_seed=args.seed
    )
    field = PerSectorRadioField.from_scenario(
        scenario, spacing_m=args.spacing_m
    )
    if kind == "rq2":
        routes = _build_rq2_routes(method, scenario, args.seed)
        fixed = "GA-EQTSP + MapGuide"
        run_id = "mac-rq2-u%d-s42-%s" % (args.uav_num, method)
    else:
        routes = _build_rq3_routes(method, scenario, args.seed)
        fixed = "4D + MapGuide"
        run_id = "mac-rq3-u%d-s42-%s" % (args.uav_num, method)

    config = {
        "experiment": kind,
        "method": method,
        "uav_num": int(args.uav_num),
        "scenario_label": "s42",
        "seed": int(args.seed),
        "route_source": fixed,
        "executor": "MapGuide + one-step safety projection",
        "safety_outage_threshold": float(args.safety_outage_threshold),
        "arrival_radius_m": float(args.arrival_radius_m),
        "outage_max_s": float(args.outage_max_s),
        "horizon_steps": int(args.horizon_steps),
        "routes": {str(k): v for k, v in routes.items()},
    }
    rc = RunContext.create(
        args.run_root, "exclusive_%s" % kind, config,
        run_id=run_id, expected=EXPECTED,
    )
    rc.write_json("resolved_config", config)
    rc.register_input("task_positions_m", data=scenario.task_positions_m)
    rc.register_input("task_data_bits", data=scenario.task_data_bits)
    rc.register_input("routes", data=routes)

    env, records = _execute_mapguide(
        scenario, field, routes, args
    )
    metrics = env.metrics()
    if metrics["feasible"]:
        t_ref, d_ref = reference
        objective = (
            0.5 * metrics["makespan_s"] / t_ref
            + 0.5 * metrics["total_delay_s"] / d_ref
        )
    else:
        objective = None
    summary = {
        "run_id": run_id,
        "suite": kind,
        "method": method,
        "status": "accepted" if metrics["feasible"] else "rejected",
        "metrics": metrics,
        "objective": objective,
        "t_ref_s": reference[0],
        "d_ref_s": reference[1],
        "routes": {str(k): v for k, v in routes.items()},
    }

    trajectory_bytes = io.BytesIO()
    np.savez(
        trajectory_bytes,
        **{
            "traj_%d" % i: np.asarray(env.trajectories[i], dtype=float)
            for i in range(env.n_uav)
        }
    )
    rc.write_json("summary", summary)
    rc.write_json("records", records)
    rc.write_json("routes", {str(k): v for k, v in routes.items()})
    rc.write_bytes("trajectory.npz", trajectory_bytes.getvalue())
    rc.finalize("success" if metrics["feasible"] else "failed")
    print(
        "%s %-16s feasible=%s T=%.1f D=%.1f J=%s"
        % (
            kind, method, metrics["feasible"], metrics["makespan_s"],
            metrics["total_delay_s"],
            "inf" if objective is None else "%.4f" % objective,
        ),
        flush=True,
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("rq2", "rq3", "both"), default="both")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--uav-num", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-radius-m", type=float, default=10.0)
    parser.add_argument("--safe-distance-m", type=float, default=30.0)
    parser.add_argument("--outage-probability-threshold", type=float, default=0.5)
    parser.add_argument("--outage-max-s", type=float, default=20.0)
    parser.add_argument("--safety-outage-threshold", type=float, default=0.2)
    parser.add_argument("--horizon-steps", type=int, default=2000)
    parser.add_argument("--spacing-m", type=float, default=200.0)
    parser.add_argument("--run-root", default="results/runs")
    parser.add_argument(
        "--reference-summary",
        default="results/runs/acceptance-exclusive-u2-s42-current-straight-r7/summary.json",
    )
    args = parser.parse_args(argv)
    reference = _reference(args.reference_summary)
    summaries = []
    selected = set(args.methods) if args.methods else None
    if args.suite in ("rq2", "both"):
        for method in RQ2_METHODS:
            if selected is not None and method not in selected:
                continue
            summaries.append(_run_one("rq2", method, args, reference))
    if args.suite in ("rq3", "both"):
        for method in RQ3_METHODS:
            if selected is not None and method not in selected:
                continue
            summaries.append(_run_one("rq3", method, args, reference))
    return summaries


if __name__ == "__main__":
    main()
