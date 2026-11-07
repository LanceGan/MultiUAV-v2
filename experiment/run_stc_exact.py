"""Sealed tiny-instance exact benchmark for K=4/6/8.

The exact result is optimal only within the declared discrete candidate space:
task assignment x visit order x one straight-hold candidate on every directed
edge. It is not a continuous-space global optimum.
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.stc_common import (
    add_common_args,
    input_provenance,
    legal_edges,
    resolve_int_list,
    resolve_runner_config,
    run_lifecycle,
    seg_view,
    small_scenario,
)
from segment.stc.evaluator import default_objective_spec, evaluate_solution
from segment.stc.exact import solve_exact_tiny
from segment.stc.generators import generate_straight_hold
from segment.stc.operators import build_ebtas_initial_solution
from segment.stc.radio import PerSectorRadioField
from segment.stc.scenario import load_stc_scenario
from segment.stc.solver import STCALNS


DEFAULT_CONFIG = {
    "uav_nums": [2],
    "map_seeds": [42],
    "task_seeds": [42],
    "method_seeds": [0],
    "exact_tasks": [4, 6, 8],
    "smoke": False,
    "smoke_tasks": 2,
    "spacing_m": 1000.0,
    "time_limit_s": 3600.0,
    "max_leaves": 2_000_000,
    "iterations": 1000,
}

EXPECTED_ARTIFACTS = (
    "resolved_config.json", "records.json", "summary.json", "progress.json")
OPTIMALITY_SCOPE = "optimal within the discrete candidate space"
SEARCH_DOMAIN = "assignment x visit_order x straight_hold_candidate"


def _build_exact_library(scenario, field):
    """One persisted straight-hold candidate on every legal directed edge."""
    library = {}
    for source, target in legal_edges(scenario.n_tasks):
        library[(source, target)] = (
            generate_straight_hold(scenario, field, source, target),
        )
    return library


def _solution_payload(solution):
    return {
        "routes": [list(route) for route in solution.routes],
        "candidate_ids": [
            list(candidate_ids) for candidate_ids in solution.candidate_ids],
    }


def _evaluated_metrics(scenario, library, solution, field, spec):
    if not any(solution.routes):
        return {
            "feasible": False,
            "makespan_s": None,
            "total_delay_s": None,
            "objective": None,
        }
    result = evaluate_solution(
        scenario, library, solution, field, spec, keep_timelines=False)
    return {
        "feasible": bool(result.feasible),
        "makespan_s": float(result.makespan_s),
        "total_delay_s": float(result.total_delay_s),
        "objective": (None if result.objective == float("inf")
                      else float(result.objective)),
    }


def run_exact_scenario(base_scenario, task_count, *, spacing_m,
                       time_limit_s, max_leaves, iterations,
                       checkpoint=None):
    """Evaluate exact enumeration and Proposed on the same K-task subproblem."""
    if task_count < 1 or task_count > base_scenario.n_tasks:
        raise ValueError(
            "task_count must be in [1, %d], got %d"
            % (base_scenario.n_tasks, task_count))
    scenario = small_scenario(base_scenario, task_count)
    field = PerSectorRadioField.from_scenario(scenario, spacing_m=spacing_m)
    library = _build_exact_library(scenario, field)
    spec = default_objective_spec(scenario, library)
    provenance = input_provenance(scenario, library, field, spec)
    common = {
        **provenance,
        "n_uav": scenario.n_uav,
        "n_tasks": scenario.n_tasks,
        "map_seed": scenario.map_seed,
        "task_seed": scenario.task_seed,
        "method_seed": 0,
        "candidate_generators": ["straight_hold"],
        "search_domain": SEARCH_DOMAIN,
        "optimality_scope": OPTIMALITY_SCOPE,
    }

    start = time.perf_counter()
    exact = solve_exact_tiny(
        scenario,
        library,
        field,
        max_tasks=scenario.n_tasks,
        time_limit_s=time_limit_s,
        objective_spec=spec,
        max_leaves=max_leaves,
    )
    exact_runtime = time.perf_counter() - start
    exact_metrics = _evaluated_metrics(
        scenario, library, exact.incumbent, field, spec)
    search_complete = (
        bool(exact.proven_optimal)
        or (not any(exact.incumbent.routes)
            and math.isinf(exact.lower_bound)))
    exact_record = {
        **common,
        "method": "exact_discrete_enumeration",
        "status": "evaluated",
        **exact_metrics,
        "proven_optimal": bool(exact.proven_optimal),
        "discrete_search_complete": bool(search_complete),
        "timed_out": not bool(search_complete),
        "lower_bound": (None if exact.lower_bound == float("inf")
                        else float(exact.lower_bound)),
        "gap": (None if exact.gap == float("inf") else float(exact.gap)),
        "explored": int(exact.explored),
        "pruned": int(exact.pruned),
        "runtime_s": float(exact.runtime_s),
        "wall_runtime_s": float(exact_runtime),
        "time_limit_s": float(time_limit_s),
        "max_leaves": int(max_leaves),
        "solution": _solution_payload(exact.incumbent),
    }
    if checkpoint is not None:
        checkpoint(exact_record)

    initial = build_ebtas_initial_solution(
        scenario, seg_view(library, scenario, field), seed=0)
    start = time.perf_counter()
    proposed = STCALNS(scenario, library, field, objective_spec=spec).solve(
        initial, iterations=iterations, method_seed=0)
    proposed_runtime = time.perf_counter() - start
    proposed_record = {
        **common,
        "method": "proposed_stc_alns",
        "status": "evaluated",
        "feasible": bool(proposed.best_result.feasible),
        "objective": (None if proposed.best_result.objective == float("inf")
                      else float(proposed.best_result.objective)),
        "makespan_s": float(proposed.best_result.makespan_s),
        "total_delay_s": float(proposed.best_result.total_delay_s),
        "runtime_s": float(proposed_runtime),
        "iterations": int(iterations),
        "iterations_executed": len(proposed.search_trace),
        "evaluator_calls": proposed.evaluator_calls,
        "solution": _solution_payload(proposed.best_solution),
    }
    if checkpoint is not None:
        checkpoint(proposed_record)
    return exact_record, proposed_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--tasks", default=None,
        help="comma-separated task counts, e.g. 4,6,8")
    parser.add_argument("--time-limit-s", type=float, default=None)
    parser.add_argument("--max-leaves", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        cfg = resolve_runner_config(args, DEFAULT_CONFIG)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.tasks is not None:
        cfg["exact_tasks"] = resolve_int_list(args.tasks)
    if args.time_limit_s is not None:
        cfg["time_limit_s"] = args.time_limit_s
    if args.max_leaves is not None:
        cfg["max_leaves"] = args.max_leaves
    if not cfg["exact_tasks"] or any(k <= 0 for k in cfg["exact_tasks"]):
        parser.error("--tasks must contain positive integers")
    if cfg["time_limit_s"] <= 0.0:
        parser.error("--time-limit-s must be positive")
    if cfg["max_leaves"] <= 0:
        parser.error("--max-leaves must be positive")

    if args.dry_run:
        print("exact benchmark: tasks=%s uav_nums=%s map_seeds=%s task_seeds=%s"
              % (cfg["exact_tasks"], cfg["uav_nums"], cfg["map_seeds"],
                 cfg["task_seeds"]))
        return 0

    def per_scenario(uav_num, map_seed, task_seed, checkpoint):
        base = load_stc_scenario(
            uav_num, task_seed=task_seed, map_seed=map_seed)
        task_counts = (
            [min(base.n_tasks, cfg["smoke_tasks"])]
            if cfg["smoke"] else cfg["exact_tasks"])

        def emit(record):
            checkpoint({**record, "base_uav_num": uav_num,
                        "base_map_seed": map_seed,
                        "base_task_seed": task_seed})

        for task_count in task_counts:
            run_exact_scenario(
                base,
                task_count,
                spacing_m=cfg["spacing_m"],
                time_limit_s=cfg["time_limit_s"],
                max_leaves=cfg["max_leaves"],
                iterations=cfg["iterations"],
                checkpoint=emit,
            )

    def summary_fn(records):
        exact = [r for r in records
                 if r["method"] == "exact_discrete_enumeration"]
        return {
            "record_count": len(records),
            "task_counts": sorted({r["n_tasks"] for r in records}),
            "exact_runs": len(exact),
            "proven_optimal": sum(bool(r["proven_optimal"]) for r in exact),
            "complete_discrete_searches": sum(
                bool(r["discrete_search_complete"]) for r in exact),
            "timed_out": sum(bool(r["timed_out"]) for r in exact),
            "optimality_scope": OPTIMALITY_SCOPE,
        }

    matrix = [
        (uav_num, map_seed, task_seed)
        for uav_num in cfg["uav_nums"]
        for map_seed in cfg["map_seeds"]
        for task_seed in cfg["task_seeds"]
    ]
    return run_lifecycle(
        "stc_exact",
        cfg,
        args.run_id,
        matrix,
        per_scenario,
        summary_fn,
        incremental_records=True,
    )


if __name__ == "__main__":
    sys.exit(main())
