"""RQ5b: controlled-scale applicability families (Task 9 Step 5).

Varies one dimension at a time while holding everything else fixed, evaluating
the EBTAS initial solution (library-open-loop) through the authoritative
synchronous evaluator:

- ``uav_count``   number of UAVs
- ``task_scale``  number of tasks
- ``payload``     per-point payload volume
- ``bandwidth``   shared uplink bandwidth
- ``uplink_power``  UAV uplink power
- ``map_seed``    radio map seed

Each variant rebuilds its own radio field and segment library from the SAME
scenario/objective/physics path.
"""

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.stc_common import (
    add_common_args,
    json_bytes,
    resolve_runner_config,
    seg_view,
    small_scenario,
)
from segment.stc.evaluator import default_objective_spec, evaluate_solution
from segment.stc.library import build_all_pairs_library
from segment.stc.operators import build_ebtas_initial_solution
from segment.stc.radio import PerSectorRadioField
from segment.stc.run_context import RunContext
from segment.stc.scenario import load_stc_scenario

SCALE_FAMILIES = ("uav_count", "task_scale", "payload", "bandwidth",
                  "uplink_power", "map_seed")

DEFAULT_CONFIG = {
    "uav_nums": [2], "map_seeds": [42], "task_seeds": [42],
    "smoke": False, "smoke_tasks": 6, "spacing_m": 1000.0,
    "generator_config": {"straight_hold": True},
}
EXPECTED_ARTIFACTS = ("resolved_config.json", "records.json", "summary.json")


def _scale_scenarios(base, family):
    variants = []
    if family == "uav_count":
        for u in sorted({max(1, base.n_uav - 1), base.n_uav, base.n_uav + 1}):
            variants.append(("uav_%d" % u, replace(base, n_uav=u)))
    elif family == "task_scale":
        for k in sorted({min(2, base.n_tasks), base.n_tasks,
                         min(base.n_tasks + 1, base.n_tasks)}):
            variants.append(("tasks_%d" % k, small_scenario(base, k)))
    elif family == "payload":
        for f in (0.5, 2.0):
            variants.append(("payload_x%.1f" % f,
                             replace(base, task_data_bits=tuple(
                                 b * f for b in base.task_data_bits))))
    elif family == "bandwidth":
        for b in (base.limits.bandwidth_hz * 0.5, base.limits.bandwidth_hz * 2.0):
            variants.append(("bandwidth_%.0e" % b,
                             replace(base, limits=replace(base.limits,
                                                          bandwidth_hz=b))))
    elif family == "uplink_power":
        for p in (base.limits.uplink_power_w * 0.5, base.limits.uplink_power_w * 2.0):
            variants.append(("power_%.2f" % p,
                             replace(base, limits=replace(base.limits,
                                                          uplink_power_w=p))))
    elif family == "map_seed":
        for ms in sorted({base.map_seed + 1, base.map_seed + 2}):
            variants.append(("map_seed_%d" % ms, replace(base, map_seed=ms)))
    return variants


def run_scale(base_scenario, field_factory, seed=42, generator_config=None):
    """Run every scale family, evaluating the library-open-loop initial solution.

    ``field_factory`` is ``(scenario) -> field`` so each variant gets its own
    radio field and segment library through the shared build path.
    """
    gen_cfg = generator_config or {"straight_hold": True}
    records = []
    for family in SCALE_FAMILIES:
        for label, sc in _scale_scenarios(base_scenario, family):
            field = field_factory(sc)
            library = build_all_pairs_library(sc, field, generator_config=gen_cfg)
            spec = default_objective_spec(sc, library.candidates)
            solution = build_ebtas_initial_solution(
                sc, seg_view(library, sc, field), seed=seed)
            result = evaluate_solution(sc, library.candidates, solution, field,
                                       spec, keep_timelines=False)
            records.append({
                "family": family, "variant": label,
                "status": "evaluated", "feasible": bool(result.feasible),
                "objective": (None if result.objective == float("inf")
                              else float(result.objective)),
                "makespan_s": float(result.makespan_s),
                "total_delay_s": float(result.total_delay_s),
                "n_uav": sc.n_uav, "n_tasks": sc.n_tasks,
                "bandwidth_hz": sc.limits.bandwidth_hz,
                "uplink_power_w": sc.limits.uplink_power_w,
                "map_seed": sc.map_seed,
            })
    return {"records": records,
            "summary": {"n_families": len(SCALE_FAMILIES),
                        "record_count": len(records)}}


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
        print("RQ5 scale families: %s" % list(SCALE_FAMILIES))
        return 0

    def field_factory(sc):
        return PerSectorRadioField.from_scenario(sc, spacing_m=cfg["spacing_m"])

    records_all = []
    for uav_num, map_seed, task_seed in _matrix(cfg):
        base = load_stc_scenario(uav_num, task_seed=task_seed, map_seed=map_seed)
        base = (small_scenario(base, cfg["smoke_tasks"]) if cfg["smoke"]
                else base)
        out = run_scale(base, field_factory, seed=task_seed,
                        generator_config=cfg["generator_config"])
        for r in out["records"]:
            r = dict(r)
            r.update({"base_uav_num": uav_num, "base_map_seed": map_seed,
                      "base_task_seed": task_seed})
            records_all.append(r)

    rc = RunContext.create("results/runs", "stc_scale", cfg,
                           run_id=args.run_id, expected=EXPECTED_ARTIFACTS)
    rc.register_input("resolved_config", data=json_bytes(cfg))
    summary = {"record_count": len(records_all),
               "n_families": len(SCALE_FAMILIES)}
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
