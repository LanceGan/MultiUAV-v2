"""RQ1: pairwise single-segment method gate (Task 9 Step 1).

Compares Straight-Hold, Comm-Waypoint, state-expanded graph (A*), SLSQP
refinement, and the learned baselines (TD3 / DDPG / SAC) on IDENTICAL directed
edges.  Reports full-set feasibility, common-feasible paired T/D/O, sector
diversity, and compute time.  Learned baselines use real checkpoint-bound
trajectories only (the SAC production path is validated in
``segment.stc.sac_candidates``); a missing/incompatible checkpoint is recorded
as an explicit ``unavailable``, never a fabricated candidate.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.stc_common import (
    add_common_args,
    json_bytes,
    legal_edges,
    resolve_runner_config,
    segment_metrics,
    small_scenario,
)
from segment.stc.generators import (
    drl_generator_label,
    generate_comm_waypoints,
    generate_straight_hold,
)
from segment.stc.graph_search import astar_candidates
from segment.stc.radio import PerSectorRadioField
from segment.stc.sac_candidates import produce_sac_segment_candidates
from segment.stc.scenario import load_stc_scenario
from segment.stc.slsqp_refine import candidate_cost, refine_candidate

DETERMINISTIC_GENERATORS = ("straight_hold", "comm_waypoint", "astar",
                            "slsqp_refined")
DRL_MODELS = ("TD3", "DDPG", "SAC")

DEFAULT_GATE_CONFIG = {
    "straight_hold": True,
    "comm_waypoints": {"grid_shape": (5, 5), "max_waypoints": 1},
    "astar": {"grid": 8},
    "slsqp_refine": {"max_waypoints": 1, "max_refinements": 1},
}

DEFAULT_CONFIG = {
    "uav_nums": [2], "map_seeds": [42], "task_seeds": [42],
    "smoke": False, "smoke_tasks": 6, "spacing_m": 1000.0,
    "generator_config": DEFAULT_GATE_CONFIG,
    "drl_checkpoints": {"SAC": None, "TD3": None, "DDPG": None},
}
EXPECTED_ARTIFACTS = ("resolved_config.json", "records.json", "summary.json")


def _generate_one(generator, scenario, field, source, target, generator_config):
    cfg = generator_config or {}
    if generator == "straight_hold":
        return [generate_straight_hold(scenario, field, source, target)]
    if generator == "comm_waypoint":
        cw = cfg.get("comm_waypoints") or {"grid_shape": (10, 10),
                                           "max_waypoints": 1}
        return list(generate_comm_waypoints(
            scenario, field, source, target,
            grid_shape=cw.get("grid_shape", (10, 10)),
            max_waypoints=cw.get("max_waypoints", 1)))
    if generator == "astar":
        astar = cfg.get("astar") or {"grid": 10}
        return [astar_candidates(scenario, field, source, target,
                                 grid=astar.get("grid", 10))]
    if generator == "slsqp_refined":
        slsqp = cfg.get("slsqp_refine") or {"max_waypoints": 1,
                                            "max_refinements": 3}
        relays = list(generate_comm_waypoints(scenario, field, source, target))
        dt = scenario.limits.delta_t_s
        relays.sort(key=lambda c: candidate_cost(c, dt))
        return [refine_candidate(relays[0], scenario, field,
                                 max_waypoints=slsqp.get("max_waypoints", 1))]
    raise ValueError("unknown generator %r" % generator)


def _deterministic_records(scenario, field, generator_config):
    dt = scenario.limits.delta_t_s
    edges = legal_edges(scenario.n_tasks)
    records = []
    for gen in DETERMINISTIC_GENERATORS:
        start = time.perf_counter()
        best_metrics = []
        n_candidates = 0
        for (src, tgt) in edges:
            cands = _generate_one(gen, scenario, field, src, tgt,
                                  generator_config)
            n_candidates += len(cands)
            best = min(cands, key=lambda c: candidate_cost(c, dt))
            best_metrics.append(segment_metrics(best, scenario))
        compute = time.perf_counter() - start
        records.append(_aggregate(gen, "evaluated", edges, best_metrics,
                                  n_candidates, compute, None))
    return records


def _aggregate(generator, status, edges, best_metrics, n_candidates, compute,
               failure_reason):
    arr = np.asarray([[m["flight_time_s"], m["upload_delay_s"], m["outage"],
                       m["sector_diversity"], float(m["feasible"])]
                      for m in best_metrics])
    return {
        "generator": generator,
        "status": status,
        "n_edges": len(edges),
        "n_candidates": int(n_candidates),
        "feasible_frac": float(np.mean(arr[:, 4])) if len(arr) else 0.0,
        "t_mean_s": float(np.mean(arr[:, 0])) if len(arr) else 0.0,
        "d_mean_s": float(np.mean(arr[:, 1])) if len(arr) else 0.0,
        "o_mean": float(np.mean(arr[:, 2])) if len(arr) else 0.0,
        "sector_diversity": float(np.mean(arr[:, 3])) if len(arr) else 0.0,
        "compute_time_s": float(compute),
        "failure_reason": failure_reason,
    }


def run_segment_gate(scenario, field, generator_config=None, drl_checkpoints=None):
    """Run the RQ1 pairwise segment gate and return ``{"records", "summary"}``.

    ``drl_checkpoints`` maps a model name (TD3/DDPG/SAC) to a checkpoint path
    (``None`` = unavailable).  Deterministic generators are always evaluated;
    SAC is produced only from a validated SI checkpoint (see
    :func:`segment.stc.sac_candidates.produce_sac_segment_candidates`); TD3 and
    DDPG record an explicit ``unavailable`` (no compatible production path).
    """
    drl_checkpoints = dict(drl_checkpoints or {})
    generator_config = generator_config or DEFAULT_GATE_CONFIG
    records = _deterministic_records(scenario, field, generator_config)

    sac_result = produce_sac_segment_candidates(
        drl_checkpoints.get("SAC"), scenario, field)
    if sac_result.status == "available":
        edges = legal_edges(scenario.n_tasks)
        best_metrics = []
        n_candidates = 0
        start = time.perf_counter()
        for (src, tgt) in edges:
            cands = sac_result.candidates[(src, tgt)]
            n_candidates += len(cands)
            best = min(cands, key=lambda c: candidate_cost(
                c, scenario.limits.delta_t_s))
            best_metrics.append(segment_metrics(best, scenario))
        compute = time.perf_counter() - start
        records.append(_aggregate(
            drl_generator_label("sac", sac_result.checkpoint_hash), "evaluated",
            edges, best_metrics, n_candidates, compute, None))
    else:
        records.append({"generator": "SAC", "status": "unavailable",
                        "n_edges": 0, "n_candidates": 0, "feasible_frac": None,
                        "t_mean_s": None, "d_mean_s": None, "o_mean": None,
                        "sector_diversity": None, "compute_time_s": None,
                        "failure_reason": sac_result.reason})

    for model in ("TD3", "DDPG"):
        cp = drl_checkpoints.get(model)
        if cp:
            records.append({"generator": model, "status": "unavailable",
                            "n_edges": 0, "n_candidates": 0,
                            "feasible_frac": None, "t_mean_s": None,
                            "d_mean_s": None, "o_mean": None,
                            "sector_diversity": None, "compute_time_s": None,
                            "failure_reason":
                                "%s checkpoint production is not available" % model})
        else:
            records.append({"generator": model, "status": "unavailable",
                            "n_edges": 0, "n_candidates": 0,
                            "feasible_frac": None, "t_mean_s": None,
                            "d_mean_s": None, "o_mean": None,
                            "sector_diversity": None, "compute_time_s": None,
                            "failure_reason":
                                "%s checkpoint is not configured (no path)" % model})

    summary = {
        "n_methods": len(records),
        "n_edges": len(legal_edges(scenario.n_tasks)),
        "evaluated": sorted(r["generator"] for r in records
                            if r["status"] == "evaluated"),
        "unavailable": sorted(r["generator"] for r in records
                              if r["status"] == "unavailable"),
    }
    return {"records": records, "summary": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--sac-checkpoint", default=None)
    parser.add_argument("--smoke-tasks", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        cfg = resolve_runner_config(args, DEFAULT_CONFIG)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        print("RQ1 segment gate matrix: generators=%s drl=%s" % (
            list(DETERMINISTIC_GENERATORS), sorted(DRL_MODELS)))
        return 0

    from segment.stc.run_context import RunContext
    records_all = []
    for uav_num, map_seed, task_seed in _matrix(cfg):
        base = load_stc_scenario(uav_num, task_seed=task_seed, map_seed=map_seed)
        sc = small_scenario(base, cfg["smoke_tasks"]) if cfg["smoke"] else base
        field = PerSectorRadioField.from_scenario(sc, spacing_m=cfg["spacing_m"])
        out = run_segment_gate(sc, field,
                               generator_config=cfg["generator_config"],
                               drl_checkpoints=cfg["drl_checkpoints"])
        for r in out["records"]:
            r = dict(r)
            r.update({"uav_num": uav_num, "map_seed": map_seed,
                      "task_seed": task_seed})
            records_all.append(r)

    rc = RunContext.create("results/runs", "stc_segment_gate", cfg,
                           run_id=args.run_id, expected=EXPECTED_ARTIFACTS)
    rc.register_input("resolved_config", data=json_bytes(cfg))
    summary = {"record_count": len(records_all),
               "n_methods": len(DETERMINISTIC_GENERATORS) + len(DRL_MODELS)}
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
