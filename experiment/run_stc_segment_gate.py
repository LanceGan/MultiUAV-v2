"""RQ1: pairwise single-segment method gate (Task 9 Step 1 + recovery).

Compares Straight-Hold, Comm-Waypoint, state-expanded graph (A*), SLSQP
refinement, and the learned baselines (TD3 / DDPG / SAC) on IDENTICAL directed
edges.  Segment feasibility is authoritative (see
:func:`experiment.stc_common.evaluate_segment`): the G2A outage constraint,
reliable-then-rate sector selection, slot upload, and every completion rule are
enforced by the synchronous evaluator on a minimal isolating plan.

T/D/O are averaged over the *common-feasible* edge subset only; feasibility is
reported over the full edge set.  Learned baselines use real checkpoint-bound
trajectories only; a missing/incompatible checkpoint is an explicit
``unavailable``, never a fabricated candidate.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.stc_common import (
    add_common_args,
    evaluate_segment,
    json_bytes,
    legal_edges,
    resolve_runner_config,
    run_lifecycle,
    small_scenario,
)
from segment.stc.evaluator import default_objective_spec
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


def _reference_spec(scenario, field):
    """A valid ObjectiveSpec over a straight-hold library (references only; the
    reference values do not affect per-segment feasibility)."""
    lib = {}
    for (src, tgt) in legal_edges(scenario.n_tasks):
        lib[(src, tgt)] = (generate_straight_hold(scenario, field, src, tgt),)
    return default_objective_spec(scenario, lib)


def _evaluate_generator(gen, scenario, field, spec, generator_config):
    edges = legal_edges(scenario.n_tasks)
    per_edge = {}
    n_candidates = 0
    start = time.perf_counter()
    for (src, tgt) in edges:
        cands = _generate_one(gen, scenario, field, src, tgt, generator_config)
        n_candidates += len(cands)
        best = min(cands, key=lambda c: candidate_cost(c, scenario.limits.delta_t_s))
        per_edge[(src, tgt)] = evaluate_segment(scenario, best, field, spec)
    return per_edge, n_candidates, time.perf_counter() - start


def _aggregate(gen, per_edge, n_candidates, compute, common_edges, world_hash,
               checkpoint_hash=None):
    edges = list(per_edge)
    feas = [per_edge[e]["feasible"] for e in edges]
    feasible_frac = float(np.mean(feas)) if feas else 0.0
    div = float(np.mean([per_edge[e]["sector_diversity"] for e in edges])) \
        if edges else 0.0
    cf = [per_edge[e] for e in edges if e in common_edges]
    t = float(np.mean([m["flight_time_s"] for m in cf])) if cf else None
    d = float(np.mean([m["upload_delay_s"] for m in cf])) if cf else None
    o = float(np.mean([m["outage"] for m in cf])) if cf else None
    # Per-edge evaluation carries the candidate id + trajectory content hash and
    # the world hash, so swapping a candidate or mixing a different radio world
    # is detectable in the persisted evidence.
    per_edge_records = [{
        **per_edge[e],
        "edge": [int(e[0]), int(e[1])],
        "generator": per_edge[e].get("generator", gen),
        "world_hash": world_hash,
        "checkpoint_hash": checkpoint_hash,
        "candidate_id": per_edge[e].get("candidate_id"),
        "trajectory_hash": per_edge[e].get("trajectory_hash"),
        "feasible": bool(per_edge[e]["feasible"]),
        "flight_time_s": per_edge[e].get("flight_time_s"),
        "upload_delay_s": per_edge[e].get("upload_delay_s"),
        "outage": per_edge[e].get("outage"),
    } for e in edges]
    return {
        "generator": gen, "status": "evaluated",
        "n_edges": len(edges), "n_candidates": int(n_candidates),
        "feasible_frac": feasible_frac, "common_feasible_n": len(cf),
        "t_mean_s": t, "d_mean_s": d, "o_mean": o,
        "sector_diversity": div, "compute_time_s": float(compute),
        "world_hash": world_hash,
        "checkpoint_hash": checkpoint_hash,
        "scenario_hash": next((m["scenario_hash"] for m in per_edge.values()), None),
        "per_edge": per_edge_records,
        "common_feasible_edges": sorted([list(e) for e in common_edges]),
        "failure_reason": None,
    }


def _unavailable_record(gen, reason):
    return {"generator": gen, "status": "unavailable", "n_edges": 0,
            "n_candidates": 0, "feasible_frac": None, "common_feasible_n": None,
            "t_mean_s": None, "d_mean_s": None, "o_mean": None,
            "sector_diversity": None, "compute_time_s": None,
            "failure_reason": reason}


def run_segment_gate(scenario, field, generator_config=None, drl_checkpoints=None):
    """Run the RQ1 pairwise segment gate and return ``{"records", "summary"}``.

    Deterministic generators are always evaluated; SAC is produced only from a
    validated SI checkpoint; TD3/DDPG record an explicit ``unavailable`` (no
    compatible production path).
    """
    drl_checkpoints = dict(drl_checkpoints or {})
    generator_config = generator_config or DEFAULT_GATE_CONFIG
    spec = _reference_spec(scenario, field)
    edges = legal_edges(scenario.n_tasks)

    evaluated = {}
    for gen in DETERMINISTIC_GENERATORS:
        per_edge, n_candidates, compute = _evaluate_generator(
            gen, scenario, field, spec, generator_config)
        evaluated[gen] = (per_edge, n_candidates, compute)

    # SAC timing includes checkpoint load + policy rollout (recovery #5)
    sac_start = time.perf_counter()
    sac_result = produce_sac_segment_candidates(drl_checkpoints.get("SAC"),
                                                scenario, field)
    sac_per_edge = {}
    sac_n_candidates = 0
    if sac_result.status == "available":
        for (src, tgt) in edges:
            cands = sac_result.candidates[(src, tgt)]
            sac_n_candidates += len(cands)
            best = min(cands, key=lambda c: candidate_cost(
                c, scenario.limits.delta_t_s))
            sac_per_edge[(src, tgt)] = evaluate_segment(scenario, best, field, spec)
        evaluated[drl_generator_label("sac", sac_result.checkpoint_hash)] = (
            sac_per_edge, sac_n_candidates, time.perf_counter() - sac_start)

    # common-feasible edge intersection across every evaluated generator
    common = None
    for gen, (per_edge, _n, _c) in evaluated.items():
        feasible_edges = {e for e, m in per_edge.items() if m["feasible"]}
        common = feasible_edges if common is None else (common & feasible_edges)
    common = common or set()

    records = []
    for gen in DETERMINISTIC_GENERATORS:
        per_edge, n_candidates, compute = evaluated[gen]
        records.append(_aggregate(gen, per_edge, n_candidates, compute, common,
                                  field.world_hash))

    if sac_result.status == "available":
        label = drl_generator_label("sac", sac_result.checkpoint_hash)
        per_edge, n_candidates, compute = evaluated[label]
        records.append(_aggregate(label, per_edge, n_candidates, compute, common,
                                  field.world_hash,
                                  checkpoint_hash=sac_result.checkpoint_hash))
    else:
        records.append(_unavailable_record("SAC", sac_result.reason))

    for model in ("TD3", "DDPG"):
        records.append(_unavailable_record(
            model, "%s checkpoint production is not available" % model))

    summary = {
        "n_methods": len(records),
        "n_edges": len(edges),
        "common_feasible_n": len(common),
        "common_feasible_edges": sorted([list(e) for e in common]),
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
    args = parser.parse_args(argv)
    try:
        cfg = resolve_runner_config(args, DEFAULT_CONFIG)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    if args.sac_checkpoint is not None:
        cfg["drl_checkpoints"] = dict(cfg["drl_checkpoints"])
        cfg["drl_checkpoints"]["SAC"] = args.sac_checkpoint

    if args.dry_run:
        print("RQ1 segment gate matrix: generators=%s drl=%s" % (
            list(DETERMINISTIC_GENERATORS), sorted(DRL_MODELS)))
        return 0

    def per_scenario(uav_num, map_seed, task_seed):
        base = load_stc_scenario(uav_num, task_seed=task_seed, map_seed=map_seed)
        sc = small_scenario(base, cfg["smoke_tasks"]) if cfg["smoke"] else base
        field = PerSectorRadioField.from_scenario(sc, spacing_m=cfg["spacing_m"])
        out = run_segment_gate(sc, field,
                               generator_config=cfg["generator_config"],
                               drl_checkpoints=cfg["drl_checkpoints"])
        return [{**r, "uav_num": uav_num, "map_seed": map_seed,
                 "task_seed": task_seed} for r in out["records"]]

    def summary_fn(records_all):
        return {"record_count": len(records_all),
                "n_methods": len(DETERMINISTIC_GENERATORS) + len(DRL_MODELS)}

    return run_lifecycle("stc_segment_gate", cfg, args.run_id, _matrix(cfg),
                         per_scenario, summary_fn)


def _matrix(cfg):
    import itertools
    return list(itertools.product(cfg["uav_nums"], cfg["map_seeds"],
                                  cfg["task_seeds"]))


if __name__ == "__main__":
    sys.exit(main())
