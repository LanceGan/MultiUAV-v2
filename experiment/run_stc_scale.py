"""RQ5b: controlled-scale applicability families (Task 9 Step 5 + recovery).

Each single-dimension family varies one quantity while holding everything else
fixed, evaluating BOTH the library-open-loop initial solution AND the Proposed
STC-ALNS through the authoritative synchronous evaluator.  A separate
``density`` family scales task count + region + base-station count together
(system-scale density, plan §6.3), and ``task_scale`` varies the task count in
both directions (fewer and more tasks) within the same region.
"""

import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.stc_common import (
    add_common_args,
    json_bytes,
    resolve_runner_config,
    run_lifecycle,
    seg_view,
    small_scenario,
)
from segment.stc.evaluator import default_objective_spec, evaluate_solution
from segment.stc.library import build_all_pairs_library
from segment.stc.operators import build_ebtas_initial_solution
from segment.stc.radio import PerSectorRadioField
from segment.stc.run_context import RunContext
from segment.stc.scenario import load_stc_scenario
from segment.stc.solver import STCALNS
from segment.stc.types import STCScenario

SCALE_FAMILIES = ("uav_count", "task_scale", "payload", "bandwidth",
                  "uplink_power", "map_seed")
# (n_uav, n_tasks, region_m, n_bs) — system-scale density families (plan §6.3)
DENSITY_FAMILIES = ((2, 20, 4000.0, 4), (3, 30, 5000.0, 5), (4, 40, 6000.0, 6))
METHODS = ("library_open_loop", "proposed_stc_alns")

DEFAULT_CONFIG = {
    "uav_nums": [2], "map_seeds": [42], "task_seeds": [42],
    "smoke": False, "smoke_tasks": 6, "spacing_m": 1000.0,
    "iterations": 1000,
    "generator_config": {"straight_hold": True},
}
EXPECTED_ARTIFACTS = ("resolved_config.json", "records.json", "summary.json")


def _synthetic_scenario(n_uav, n_tasks, length_m, n_bs, task_seed, map_seed,
                        limits):
    """Deterministic synthetic task layout within a ``length_m`` square region."""
    rng = np.random.RandomState((task_seed * 1000003 + n_tasks * 7919)
                                & 0x7fffffff)
    positions = tuple((float(rng.uniform(0.05 * length_m, 0.95 * length_m)),
                       float(rng.uniform(0.05 * length_m, 0.95 * length_m)))
                      for _ in range(n_tasks))
    data = tuple(float(1e6 * (10.0 + 20.0 * rng.rand())) for _ in range(n_tasks))
    return STCScenario(
        n_uav=n_uav, n_tasks=n_tasks, task_positions_m=positions,
        task_data_bits=data, depot_m=(0.0, 0.0),
        terminal_m=(length_m, length_m), limits=limits,
        length_m=length_m, width_m=length_m, n_bs=n_bs,
        task_seed=task_seed, map_seed=map_seed)


def _extend_scenario(base, n_tasks):
    """Extend ``base`` to ``n_tasks`` by KEEPING every existing task / payload /
    depot / terminal and appending deterministic extra tasks within the SAME
    region.

    ``task_scale`` must only add tasks; it must never regenerate the original
    tasks, payloads or depot (which would silently change the comparison point).
    """
    n_new = n_tasks - base.n_tasks
    rng = np.random.RandomState((base.task_seed * 1000003 + n_tasks * 7919)
                                & 0x7fffffff)
    positions = base.task_positions_m + tuple(
        (float(rng.uniform(0.05 * base.length_m, 0.95 * base.length_m)),
         float(rng.uniform(0.05 * base.width_m, 0.95 * base.width_m)))
        for _ in range(n_new))
    data = base.task_data_bits + tuple(
        float(1e6 * (10.0 + 20.0 * rng.rand())) for _ in range(n_new))
    return STCScenario(
        n_uav=base.n_uav, n_tasks=n_tasks, task_positions_m=positions,
        task_data_bits=data, depot_m=base.depot_m, terminal_m=base.terminal_m,
        limits=base.limits, length_m=base.length_m, width_m=base.width_m,
        n_bs=base.n_bs, task_seed=base.task_seed, map_seed=base.map_seed)


def _scale_scenarios(base, family):
    variants = []
    if family == "uav_count":
        for u in sorted({max(1, base.n_uav - 1), base.n_uav, base.n_uav + 1}):
            variants.append(("uav_%d" % u, replace(base, n_uav=u)))
    elif family == "task_scale":
        for k in sorted({max(1, base.n_tasks - 2), base.n_tasks,
                         base.n_tasks + 2}):
            if k <= base.n_tasks:
                variants.append(("tasks_%d" % k, small_scenario(base, k)))
            else:
                # more tasks within the SAME region/BS, keeping the original
                # tasks/depot/terminal (density within a fixed area)
                variants.append(("tasks_%d" % k, _extend_scenario(base, k)))
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


def _make_record(method, result, sc, method_seed=None, runtime_s=None,
                 evaluator_calls=None):
    return {
        "method": method, "status": "evaluated",
        "feasible": bool(result.feasible),
        "objective": (None if result.objective == float("inf")
                      else float(result.objective)),
        "makespan_s": float(result.makespan_s),
        "total_delay_s": float(result.total_delay_s),
        "method_seed": method_seed,
        "runtime_s": runtime_s,
        "evaluator_calls": evaluator_calls,
        "n_uav": sc.n_uav, "n_tasks": sc.n_tasks,
        "bandwidth_hz": sc.limits.bandwidth_hz,
        "uplink_power_w": sc.limits.uplink_power_w,
        "map_seed": sc.map_seed,
    }


def _eval_methods(sc, field, gen_cfg, seed, iterations):
    """Evaluate both library-open-loop and Proposed STC-ALNS on ``sc``.

    Each method records its own method seed, wall-clock runtime, and evaluator
    call count so RQ5 is auditable.  The two methods use independent method
    seeds (open-loop ``seed``, Proposed ``seed + 1``) to avoid coupling their
    randomness.
    """
    library = build_all_pairs_library(sc, field, generator_config=gen_cfg)
    lib = library.candidates
    spec = default_objective_spec(sc, lib)
    seg = seg_view(lib, sc, field)
    initial = build_ebtas_initial_solution(sc, seg, seed=seed)

    records = []
    t0 = time.perf_counter()
    res = evaluate_solution(sc, lib, initial, field, spec, keep_timelines=False)
    records.append(_make_record(
        "library_open_loop", res, sc, method_seed=seed,
        runtime_s=time.perf_counter() - t0,
        evaluator_calls={"full": 1, "incremental": 0}))

    t0 = time.perf_counter()
    sr = STCALNS(sc, lib, field).solve(initial, iterations=iterations,
                                       method_seed=seed + 1)
    records.append(_make_record(
        "proposed_stc_alns", sr.best_result, sc, method_seed=seed + 1,
        runtime_s=time.perf_counter() - t0,
        evaluator_calls=sr.evaluator_calls))
    return records


def run_scale(base_scenario, field_factory, seed=42, generator_config=None,
              iterations=30):
    """Run every single-dimension scale family for both methods."""
    gen_cfg = generator_config or {"straight_hold": True}
    records = []
    for family in SCALE_FAMILIES:
        for label, sc in _scale_scenarios(base_scenario, family):
            field = field_factory(sc)
            for rec in _eval_methods(sc, field, gen_cfg, seed, iterations):
                rec["family"] = family
                rec["variant"] = label
                records.append(rec)
    return {"records": records,
            "summary": {"n_families": len(SCALE_FAMILIES),
                        "methods": list(METHODS),
                        "record_count": len(records)}}


def run_density_scale(base_scenario, field_factory, seed=42, generator_config=None,
                      iterations=30):
    """System-scale density: task count + region + BS count scale together."""
    gen_cfg = generator_config or {"straight_hold": True}
    limits = base_scenario.limits
    records = []
    for n_uav, n_tasks, length_m, n_bs in DENSITY_FAMILIES:
        sc = _synthetic_scenario(n_uav, n_tasks, length_m, n_bs,
                                 base_scenario.task_seed, base_scenario.map_seed,
                                 limits)
        field = field_factory(sc)
        for rec in _eval_methods(sc, field, gen_cfg, seed, iterations):
            rec["family"] = "density"
            rec["variant"] = "U%d_K%d_%.0fkm_BS%d" % (n_uav, n_tasks,
                                                      length_m / 1000.0, n_bs)
            records.append(rec)
    return {"records": records,
            "summary": {"n_families": 1, "methods": list(METHODS),
                        "record_count": len(records)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args(argv)
    try:
        cfg = resolve_runner_config(args, DEFAULT_CONFIG)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        print("RQ5 scale families: %s + density" % list(SCALE_FAMILIES))
        return 0

    def field_factory(sc):
        return PerSectorRadioField.from_scenario(sc, spacing_m=cfg["spacing_m"])

    def per_scenario(uav_num, map_seed, task_seed):
        base = load_stc_scenario(uav_num, task_seed=task_seed, map_seed=map_seed)
        base = (small_scenario(base, cfg["smoke_tasks"]) if cfg["smoke"]
                else base)
        recs = []
        for run_fn in (run_scale, run_density_scale):
            out = run_fn(base, field_factory, seed=task_seed,
                         generator_config=cfg["generator_config"],
                         iterations=cfg["iterations"])
            recs.extend({**r, "base_uav_num": uav_num, "base_map_seed": map_seed,
                         "base_task_seed": task_seed} for r in out["records"])
        return recs

    def summary_fn(records_all):
        return {"record_count": len(records_all),
                "n_families": len(SCALE_FAMILIES) + 1,
                "methods": list(METHODS)}

    return run_lifecycle("stc_scale", cfg, args.run_id, _matrix(cfg),
                         per_scenario, summary_fn)


def _matrix(cfg):
    import itertools
    return list(itertools.product(cfg["uav_nums"], cfg["map_seeds"],
                                  cfg["task_seeds"]))


if __name__ == "__main__":
    sys.exit(main())
