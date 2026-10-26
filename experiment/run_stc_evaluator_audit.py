"""RQ3: synchronous shared-bandwidth necessity audit (Task 9 Step 3).

On IDENTICAL complete plans, compare three evaluation modes:

- independent full bandwidth (each UAV alone, no sharing),
- legacy static congestion factor (a constant multiplier on independent delay),
- slot-synchronous shared bandwidth (the authoritative evaluator).

Reports metric error (absolute T/D deviation), feasibility reversal, and ranking
reversal, with an explicit ``none found`` sentinel when a comparison shows no
reversal.  The authoritative physics is always
:func:`segment.stc.evaluator.evaluate_solution`.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.stc_common import (
    add_common_args,
    evaluate_independent,
    evaluate_static_factor,
    evaluate_synchronous,
    json_bytes,
    resolve_runner_config,
    run_lifecycle,
    small_scenario,
)
from segment.stc.evaluator import default_objective_spec
from segment.stc.library import build_all_pairs_library
from segment.stc.operators import build_ebtas_initial_solution, build_initial_solution
from segment.stc.radio import PerSectorRadioField
from segment.stc.run_context import RunContext
from segment.stc.scenario import load_stc_scenario

DEFAULT_CONFIG = {
    "uav_nums": [2], "map_seeds": [42], "task_seeds": [42],
    "smoke": False, "smoke_tasks": 6, "spacing_m": 1000.0,
    "congestion_factor": 1.0,
    "generator_config": {"straight_hold": True},
}
EXPECTED_ARTIFACTS = ("resolved_config.json", "records.json", "summary.json")

APPROX_MODES = ("independent", "static_factor")


def run_evaluator_audit(scenario, library, field, plans, objective_spec=None,
                        congestion_factor=1.0):
    """Compare the three evaluation modes on identical ``plans``.

    ``plans`` is an iterable of :class:`~segment.stc.types.PlanSolution`.
    """
    spec = objective_spec or default_objective_spec(scenario, library)
    plan_records = []
    for i, plan in enumerate(plans):
        sync = evaluate_synchronous(scenario, library, plan, field, spec)
        indep = evaluate_independent(scenario, library, plan, field, spec)
        static = evaluate_static_factor(scenario, library, plan, field, spec,
                                        congestion_factor)
        plan_records.append({"plan_id": i, "synchronous": sync,
                             "independent": indep, "static_factor": static})

    err = {}
    for mode in APPROX_MODES:
        t_err = np.mean([abs(p[mode]["makespan_s"]
                             - p["synchronous"]["makespan_s"])
                         for p in plan_records])
        d_err = np.mean([abs(p[mode]["total_delay_s"]
                             - p["synchronous"]["total_delay_s"])
                         for p in plan_records])
        err[mode] = {"makespan_s_abs": float(t_err),
                     "total_delay_s_abs": float(d_err)}

    feasibility_reversal = []
    for p in plan_records:
        for mode in APPROX_MODES:
            if p[mode]["feasible"] != p["synchronous"]["feasible"]:
                feasibility_reversal.append({
                    "plan_id": p["plan_id"], "mode": mode,
                    "synchronous_feasible": p["synchronous"]["feasible"],
                    "approximation_feasible": p[mode]["feasible"],
                })

    ranking_reversal = []
    for mode in APPROX_MODES:
        for i in range(len(plan_records)):
            for j in range(i + 1, len(plan_records)):
                oi_s, oj_s = (plan_records[i]["synchronous"]["objective"],
                              plan_records[j]["synchronous"]["objective"])
                oi_a, oj_a = (plan_records[i][mode]["objective"],
                              plan_records[j][mode]["objective"])
                if None in (oi_s, oj_s, oi_a, oj_a):
                    continue
                sync_order = (oi_s > oj_s) - (oi_s < oj_s)
                app_order = (oi_a > oj_a) - (oi_a < oj_a)
                if sync_order and app_order and sync_order != app_order:
                    ranking_reversal.append({"pair": [i, j], "mode": mode})

    return {
        "plans": plan_records,
        "metric_error": err,
        "feasibility_reversal": (feasibility_reversal if feasibility_reversal
                                 else "none found"),
        "ranking_reversal": (ranking_reversal if ranking_reversal
                             else "none found"),
    }


def _audit_plans(scenario, library, field, seed):
    from experiment.stc_common import seg_view
    lib_view = seg_view(library, scenario, field)
    plans = [build_initial_solution(scenario, library, seed),
             build_ebtas_initial_solution(scenario, lib_view, seed)]
    try:
        from segment.stc.system_baselines import run_system_baseline
        for method in ("region_distance_straight", "library_open_loop"):
            r = run_system_baseline(method, scenario, library, field, seed=seed)
            plans.append(r.solution)
    except Exception:
        pass
    return plans


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--congestion-factor", type=float, default=None)
    args = parser.parse_args(argv)
    try:
        cfg = resolve_runner_config(args, DEFAULT_CONFIG)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.congestion_factor is not None:
        cfg["congestion_factor"] = args.congestion_factor

    if args.dry_run:
        print("RQ3 evaluator audit: modes=%s factor=%.3f" % (
            list(APPROX_MODES), cfg["congestion_factor"]))
        return 0

    def per_scenario(uav_num, map_seed, task_seed):
        base = load_stc_scenario(uav_num, task_seed=task_seed, map_seed=map_seed)
        sc = small_scenario(base, cfg["smoke_tasks"]) if cfg["smoke"] else base
        field = PerSectorRadioField.from_scenario(sc, spacing_m=cfg["spacing_m"])
        library = build_all_pairs_library(
            sc, field, generator_config=cfg["generator_config"])
        spec = default_objective_spec(sc, library.candidates)
        plans = _audit_plans(sc, library.candidates, field, map_seed)
        out = run_evaluator_audit(sc, library.candidates, field, plans, spec,
                                  cfg["congestion_factor"])
        return [{"uav_num": uav_num, "map_seed": map_seed, "task_seed": task_seed,
                 "n_plans": len(out["plans"]),
                 "metric_error": out["metric_error"],
                 "feasibility_reversal": out["feasibility_reversal"],
                 "ranking_reversal": out["ranking_reversal"]}]

    def summary_fn(records_all):
        return {"record_count": len(records_all)}

    return run_lifecycle("stc_evaluator_audit", cfg, args.run_id, _matrix(cfg),
                         per_scenario, summary_fn)


def _matrix(cfg):
    import itertools
    return list(itertools.product(cfg["uav_nums"], cfg["map_seeds"],
                                  cfg["task_seeds"]))


if __name__ == "__main__":
    sys.exit(main())
