"""RQ5a: single-mechanism ablation of STC-ALNS (Task 9 Step 5).

Each run disables exactly one mechanism while keeping every other mechanism and
the SAME scenario / library / ObjectiveSpec / synchronous evaluator:

- ``ebtas_init``        greedy NN split instead of EBTAS initial solution
- ``multi_candidate``   single best candidate per edge (no Pareto/diversity)
- ``hotspot_destroy``   no conflict/sector hotspot destroy operators
- ``conflict_repair``   no conflict-aware insertion (proxy-only repair)
- ``path_switch``       no path-switch local operator
- ``adaptive_weights``  fixed uniform operator weights (rho=0)
- ``incremental_eval``  full replay every move (no incremental resume)

The ``full`` record runs the unmodified solver; every ablation is scored through
the same :class:`STCALNS` entry point and the authoritative synchronous evaluator.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.stc_common import (
    add_common_args,
    json_bytes,
    resolve_runner_config,
    seg_view,
    small_scenario,
)
from segment.stc.library import build_all_pairs_library
from segment.stc.operators import build_ebtas_initial_solution, build_initial_solution
from segment.stc.radio import PerSectorRadioField
from segment.stc.run_context import RunContext
from segment.stc.scenario import load_stc_scenario
from segment.stc.slsqp_refine import candidate_cost
from segment.stc.solver import STCALNS
from segment.stc.stc_operators import (
    destroy_operators,
    local_operators,
    repair_operators,
)

ABLATIONS = {
    "ebtas_init": "greedy NN split initial solution instead of EBTAS",
    "multi_candidate": "single best candidate per edge (no Pareto/diversity)",
    "hotspot_destroy": "no conflict/sector hotspot destroy operators",
    "conflict_repair": "no conflict-aware insertion (proxy-only repair)",
    "path_switch": "no path-switch local operator",
    "adaptive_weights": "fixed uniform operator weights (rho=0)",
    "incremental_eval": "full replay every move (no incremental resume)",
}

DEFAULT_CONFIG = {
    "uav_nums": [2], "map_seeds": [42], "task_seeds": [42],
    "smoke": False, "smoke_tasks": 6, "spacing_m": 1000.0,
    "iterations": 1000,
    "generator_config": {"straight_hold": True,
                         "comm_waypoints": {"grid_shape": (5, 5)}},
    "solver": {"rho": 0.1, "T0": 0.1, "cooling": 0.995, "segment": 100,
               "audit_interval": 50},
}
EXPECTED_ARTIFACTS = ("resolved_config.json", "records.json", "summary.json")


def _reduced_library(scenario, library):
    dt = scenario.limits.delta_t_s
    reduced = {}
    for edge, cands in library.items():
        rankable = [c for c in cands if c.rankable]
        best = min(rankable, key=lambda c: candidate_cost(c, dt))
        reduced[edge] = [best] + [c for c in cands if not c.rankable]
    return reduced


def run_ablation(scenario, field, library, seed=42, iterations=1000,
                 solver_config=None):
    """Run the full method and each single-mechanism ablation, return records."""
    lib = library.candidates if hasattr(library, "candidates") else dict(library)
    base_config = dict(solver_config or {})

    def _initial(cand):
        return build_ebtas_initial_solution(
            scenario, seg_view(cand, scenario, field), seed=seed)

    def _run_one(name, initial, cand, kwargs):
        solver = STCALNS(scenario, cand, field, **dict(base_config, **kwargs))
        sr = solver.solve(initial, iterations=iterations, method_seed=seed)
        r = sr.best_result
        return {
            "ablation": name, "status": "evaluated",
            "feasible": bool(r.feasible),
            "objective": (None if r.objective == float("inf")
                          else float(r.objective)),
            "makespan_s": float(r.makespan_s),
            "total_delay_s": float(r.total_delay_s),
            "evaluator_calls": sr.evaluator_calls,
        }

    records = [_run_one("full", _initial(lib), lib, {})]

    for name in ABLATIONS:
        initial = _initial(lib)
        cand = lib
        kwargs = {}
        if name == "ebtas_init":
            initial = build_initial_solution(scenario, lib, seed=seed)
        elif name == "multi_candidate":
            cand = _reduced_library(scenario, lib)
            initial = _initial(cand)
        elif name == "hotspot_destroy":
            kwargs["operator_sets"] = {
                "destroy": {k: v for k, v in destroy_operators.items()
                            if k not in ("conflict_hotspot_removal",
                                         "sector_related_removal")}}
        elif name == "conflict_repair":
            kwargs["operator_sets"] = {
                "repair": {k: v for k, v in repair_operators.items()
                           if k not in ("conflict_aware_insertion",
                                        "sector_diversity_insertion")}}
        elif name == "path_switch":
            kwargs["operator_sets"] = {
                "local": {k: v for k, v in local_operators.items()
                          if k != "path_switch"}}
        elif name == "adaptive_weights":
            kwargs["rho"] = 0.0
        elif name == "incremental_eval":
            kwargs["disable_incremental"] = True
        try:
            records.append(_run_one(name, initial, cand, kwargs))
        except Exception as exc:  # a mechanism removal must not crash the runner
            records.append({"ablation": name, "status": "unavailable",
                            "feasible": None, "objective": None,
                            "makespan_s": None, "total_delay_s": None,
                            "evaluator_calls": None,
                            "failure_reason": repr(exc)})

    return {"records": records,
            "summary": {"n_ablations": len(ABLATIONS),
                        "methods": ["full"] + list(ABLATIONS)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--smoke-tasks", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        cfg = resolve_runner_config(args, DEFAULT_CONFIG)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        print("RQ5 ablation: mechanisms=%s" % (["full"] + list(ABLATIONS)))
        return 0

    records_all = []
    for uav_num, map_seed, task_seed in _matrix(cfg):
        base = load_stc_scenario(uav_num, task_seed=task_seed, map_seed=map_seed)
        sc = small_scenario(base, cfg["smoke_tasks"]) if cfg["smoke"] else base
        field = PerSectorRadioField.from_scenario(sc, spacing_m=cfg["spacing_m"])
        library = build_all_pairs_library(sc, field,
                                          generator_config=cfg["generator_config"])
        out = run_ablation(sc, field, library.candidates, seed=task_seed,
                           iterations=cfg["iterations"],
                           solver_config=cfg["solver"])
        for r in out["records"]:
            r = dict(r)
            r.update({"uav_num": uav_num, "map_seed": map_seed,
                      "task_seed": task_seed})
            records_all.append(r)

    rc = RunContext.create("results/runs", "stc_ablation", cfg,
                           run_id=args.run_id, expected=EXPECTED_ARTIFACTS)
    rc.register_input("resolved_config", data=json_bytes(cfg))
    summary = {"record_count": len(records_all),
               "n_ablations": len(ABLATIONS)}
    for name, value in (("resolved_config.json", cfg),
                        ("records.json", records_all),
                        ("summary.json", summary)):
        rc.write_bytes(name, json_bytes(value))
    rc.finalize("success")
    print("run written to %s (%d records)" % (rc.run_dir, len(records_all)))
    return 0


def _matrix(cfg):
    import itertools
    return list(itertools.product(cfg["uav_nums"], cfg["map_seeds"],
                                  cfg["task_seeds"]))


if __name__ == "__main__":
    sys.exit(main())
