"""STC main experiment entry point (Task 8 / Task 15).

Runs the six complete-system methods (Region-Distance-Straight,
Shortest-Balanced-Graph, Jia-EBTAS-style, GATSAC-style, Library-Open-Loop,
Proposed STC-ALNS) through the SAME synchronous evaluator from ``t=0``, sharing
one scenario / library / reference / failure rule set, and writes a run-scoped
summary via :class:`RunContext`.

Supports the shared CLI (``--config/--map-seeds/--task-seeds/--method-seeds/
--dry-run/--run-id/--smoke``) plus ``--uav-nums/--iterations/--spacing-m``.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment.stc.library import build_all_pairs_library
from segment.stc.radio import PerSectorRadioField
from segment.stc.run_context import RunContext
from segment.stc.scenario import load_stc_scenario
from segment.stc.system_baselines import (
    METHOD_SPECS,
    SYSTEM_BASELINE_KEYS,
    run_system_baseline,
)
from segment.stc.types import PlanSolution, STCScenario

# Reduced generator config + tiny scenario for fast smoke runs.  A* and SLSQP
# refinement are omitted so a smoke run stays tractable; both the Straight-Hold
# and communication-aware (comm-waypoint) candidate families remain present.
SMOKE_GENERATOR_CONFIG = {
    "straight_hold": True,
    "comm_waypoints": {"grid_shape": (3, 3), "max_waypoints": 1},
}


def _common_args(parser):
    parser.add_argument("--config", type=str, default="configs/stc/experiments.json")
    parser.add_argument("--map-seeds", type=str, default="42")
    parser.add_argument("--task-seeds", type=str, default="42")
    parser.add_argument("--method-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")


def _small_scenario(base, n_tasks=6):
    """Smoke scenario reusing the real map (same n_bs/length/map_seed)."""
    return STCScenario(
        n_uav=base.n_uav, n_tasks=n_tasks,
        task_positions_m=base.task_positions_m[:n_tasks],
        task_data_bits=base.task_data_bits[:n_tasks],
        depot_m=base.depot_m, terminal_m=base.terminal_m,
        limits=base.limits, length_m=base.length_m, width_m=base.width_m,
        n_bs=base.n_bs, task_seed=base.task_seed, map_seed=base.map_seed)


def _json_safe(value):
    """Recursively convert a payload to a JSON-serializable form.

    ``PlanSolution`` provenance entries (e.g. the Proposed STC-ALNS
    ``initial_solution``) become plain ``routes``/``candidate_ids`` lists.
    """
    if isinstance(value, PlanSolution):
        return {"routes": [list(rt) for rt in value.routes],
                "candidate_ids": [list(c) for c in value.candidate_ids]}
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)  # "inf" / "-inf" / "nan" (valid JSON string)
    return value


def _result_payload(r):
    v = r.result.violations
    return {
        "method": r.method,
        "label": METHOD_SPECS[r.method].label,
        "proposed": METHOD_SPECS[r.method].proposed,
        "objective": r.result.objective,
        "feasible": r.result.feasible,
        "makespan_s": r.result.makespan_s,
        "total_delay_s": r.result.total_delay_s,
        "outage_ratio": list(r.result.outage_ratio),
        "violations": {
            "unserved_tasks": v.unserved_tasks,
            "upload_failures": v.upload_failures,
            "outage_excess": v.outage_excess,
            "deadline_exceeded": v.deadline_exceeded,
            "wait_exceeded": v.wait_exceeded,
        },
        "evaluator_calls": r.evaluator_calls,
        "provenance": _json_safe(r.provenance),
    }


def _run_scenario(uav_num, map_seed, task_seed, method_seed, args):
    base = load_stc_scenario(uav_num, task_seed=task_seed, map_seed=map_seed)
    sc = _small_scenario(base) if args.smoke else base
    spacing_m = 1000.0 if args.smoke else args.spacing_m
    field = PerSectorRadioField.from_scenario(sc, spacing_m=spacing_m)

    generator_config = SMOKE_GENERATOR_CONFIG if args.smoke else None
    library = build_all_pairs_library(sc, field, generator_config=generator_config)

    objective_spec = None
    methods = {}
    for key in SYSTEM_BASELINE_KEYS:
        r = run_system_baseline(key, sc, library, field, seed=task_seed,
                                method_seed=method_seed, iterations=args.iterations)
        objective_spec = r.objective_spec
        methods[key] = _result_payload(r)

    return {
        "uav_num": uav_num, "map_seed": map_seed, "task_seed": task_seed,
        "method_seed": method_seed,
        "n_tasks": sc.n_tasks, "n_uav": sc.n_uav, "n_bs": sc.n_bs,
        "length_m": sc.length_m, "width_m": sc.width_m,
        "n_edges": len(library.candidates),
        "n_candidates": sum(len(cs) for cs in library.candidates.values()),
        "world_hash": field.world_hash,
        "reference": {
            "t_ref_s": objective_spec.reference.t_ref_s,
            "d_ref_s": objective_spec.reference.d_ref_s,
            "source_hash": objective_spec.reference.source_hash,
        },
        "methods": methods,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="STC main experiment")
    _common_args(parser)
    parser.add_argument("--uav-nums", type=str, default="2")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--spacing-m", type=float, default=200.0)
    args = parser.parse_args(argv)

    map_seeds = [int(s) for s in args.map_seeds.split(",")]
    task_seeds = [int(s) for s in args.task_seeds.split(",")]
    method_seeds = [int(s) for s in args.method_seeds.split(",")]
    uav_nums = [int(s) for s in args.uav_nums.split(",")]

    if args.dry_run:
        print("run matrix: uav_nums=%s map_seeds=%s task_seeds=%s method_seeds=%s"
              % (uav_nums, map_seeds, task_seeds, method_seeds))
        return 0

    rc = RunContext.create("results/runs", "stc_main",
                           {"config": args.config, "uav_nums": uav_nums,
                            "map_seeds": map_seeds, "task_seeds": task_seeds,
                            "method_seeds": method_seeds},
                           run_id=args.run_id)

    scenarios = []
    for uav_num in uav_nums:
        for map_seed in map_seeds:
            for task_seed in task_seeds:
                method_seed = method_seeds[0] if method_seeds else 42
                scenarios.append(_run_scenario(
                    uav_num, map_seed, task_seed, method_seed, args))

    summary = {
        "status": "smoke" if args.smoke else "full",
        "methods": list(SYSTEM_BASELINE_KEYS),
        "scenarios": scenarios,
    }
    rc.write_json("summary", summary)
    rc.finalize("success")
    print("run written to %s" % rc.run_dir)

    # Human-readable one-line table.
    for s in scenarios:
        line = "uav=%d map=%d task=%d: " % (s["uav_num"], s["map_seed"], s["task_seed"])
        line += "  ".join("%s=%.3f" % (k, s["methods"][k]["objective"])
                          for k in SYSTEM_BASELINE_KEYS)
        print(line)
    return 0


if __name__ == "__main__":
    main()
