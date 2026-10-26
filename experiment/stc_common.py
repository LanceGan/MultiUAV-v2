"""Shared plumbing for the RQ1-RQ5 STC experiment runners (Task 9 + recovery).

Every runner uses the SAME scenario loading, tiny-scenario shrinking, field
construction, library building, single-UAV segment metrics, evaluation
shortcuts, and sealed run-output primitives so no runner invents its own physics
or objective path.  The authoritative objective is always
:class:`segment.stc.types.ObjectiveSpec` and the authoritative physics is always
:func:`segment.stc.evaluator.evaluate_solution`.

Segment-level feasibility (RQ1) is authoritative too: it runs the synchronous
evaluator on a minimal single-UAV plan that isolates the segment under test, so
a segment whose G2A outage exceeds ``eps_out`` (or that fails any other
completion rule) is correctly reported infeasible.
"""

import argparse
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from segment.stc.evaluator import evaluate_solution, select_serving_sector
from segment.stc.types import (
    PlanSolution,
    RadioSample,
    STCScenario,
    SegmentCandidate,
    SegmentLibrary,
)

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #

def json_safe(value):
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_bytes(value):
    return json.dumps(json_safe(value), sort_keys=True, allow_nan=False,
                      ensure_ascii=False).encode("utf-8")


def strict_json_load(path):
    def reject(token):
        raise ValueError("non-finite JSON token: " + token)
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject)


# --------------------------------------------------------------------------- #
# Scenario + args
# --------------------------------------------------------------------------- #

def small_scenario(base, n_tasks):
    n_tasks = min(n_tasks, base.n_tasks)
    return STCScenario(
        n_uav=base.n_uav, n_tasks=n_tasks,
        task_positions_m=base.task_positions_m[:n_tasks],
        task_data_bits=base.task_data_bits[:n_tasks], depot_m=base.depot_m,
        terminal_m=base.terminal_m, limits=base.limits, length_m=base.length_m,
        width_m=base.width_m, n_bs=base.n_bs, task_seed=base.task_seed,
        map_seed=base.map_seed)


def legal_edges(n_tasks):
    edges = []
    for i in range(n_tasks):
        edges.append((-1, i))
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i != j:
                edges.append((i, j))
    for i in range(n_tasks):
        edges.append((i, n_tasks))
    return edges


def add_common_args(parser):
    parser.add_argument("--config", default=None)
    for flag in ("map-seeds", "task-seeds", "uav-nums", "method-seeds"):
        parser.add_argument("--" + flag, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke", action="store_true", default=None)
    parser.add_argument("--smoke-tasks", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--spacing-m", type=float, default=None)
    parser.add_argument("--timeout-s", type=float, default=None)


def resolve_int_list(value):
    return [int(x) for x in value.split(",")] if value is not None else None


def resolve_runner_config(args, defaults):
    """Resolve a runner config: defaults <- config file <- CLI overrides.

    Raises ``ValueError`` when ``--config`` is given but the file is missing or
    invalid, or when a resulting experiment matrix is empty, so a runner never
    reaches run creation for a missing/empty input.
    """
    def reject(token):
        raise ValueError("non-finite JSON constant %s" % token)

    cfg = dict(defaults)
    if getattr(args, "config", None):
        path = Path(args.config)
        if not path.is_file():
            raise ValueError("config file not found: %s" % args.config)
        declared = json.loads(path.read_text(encoding="utf-8"),
                              parse_constant=reject)
        if not isinstance(declared, dict):
            raise ValueError("config must be a JSON object")
        cfg.update(declared)
    for key in ("uav_nums", "map_seeds", "task_seeds", "method_seeds"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = [int(x) for x in val.split(",")]
    if getattr(args, "iterations", None) is not None:
        cfg["iterations"] = args.iterations
    if getattr(args, "spacing_m", None) is not None:
        cfg["spacing_m"] = args.spacing_m
    # CLI smoke flags must actually apply (recovery #9)
    if getattr(args, "smoke", None) is not None:
        cfg["smoke"] = args.smoke
    if getattr(args, "smoke_tasks", None) is not None:
        cfg["smoke_tasks"] = args.smoke_tasks
    if getattr(args, "timeout_s", None) is not None:
        cfg["timeout_s"] = args.timeout_s
    # reject empty experiment matrices (recovery #10)
    for key in ("uav_nums", "map_seeds", "task_seeds", "method_seeds"):
        if key in cfg and (not isinstance(cfg[key], list) or not cfg[key]):
            raise ValueError("%s must be a nonempty list" % key)
    return cfg


# --------------------------------------------------------------------------- #
# Library view
# --------------------------------------------------------------------------- #

def seg_view(library, scenario, field=None):
    """Wrap a plain candidate dict as a ``SegmentLibrary`` (assignment/sequencing
    read ``.candidates``).  Accepts an existing ``SegmentLibrary`` unchanged."""
    if isinstance(library, SegmentLibrary):
        return library
    world_hash = getattr(field, "world_hash", None) or "runner"
    return SegmentLibrary(candidates=dict(library), map_seed=scenario.map_seed,
                          task_seed=scenario.task_seed, world_hash=world_hash)


# --------------------------------------------------------------------------- #
# Authoritative single-segment metrics (RQ1)
# --------------------------------------------------------------------------- #

def _sector_entropy(candidate):
    if len(candidate.sector_ids) == 0:
        return 0.0
    n_sectors = candidate.full_rate_bps.shape[1]
    sig = np.bincount(np.asarray(candidate.sector_ids, dtype=int),
                      minlength=n_sectors) / len(candidate.sector_ids)
    sig = sig[sig > 0]
    if n_sectors <= 1:
        return 0.0
    return float(-np.sum(sig * np.log(sig)) / np.log(n_sectors))


def _relabel_candidate(cand, new_source, new_target):
    return SegmentCandidate(
        candidate_id=cand.candidate_id, source=new_source, target=new_target,
        generator=cand.generator, positions_m=cand.positions_m,
        full_rate_bps=cand.full_rate_bps, sector_ids=cand.sector_ids,
        g2a_outage=cand.g2a_outage, flight_steps=cand.flight_steps,
        hold_position_m=cand.hold_position_m,
        source_data_bits=cand.source_data_bits, rankable=cand.rankable)


def _segment_tdo(candidate, scenario):
    """Segment-level T/D/O via the authoritative per-segment rules.

    Uses the reliable-then-rate serving sector, a slot-by-slot upload of the
    source payload, and the G2A outage along the serving-sector path, with any
    residual upload draining at the hold boundary (bounded by ``max_wait_s``).
    """
    dt = scenario.limits.delta_t_s
    eps_out = scenario.limits.eps_out
    P = float(candidate.source_data_bits)
    flight = float(candidate.flight_steps * dt)
    n = len(candidate.positions_m)

    remaining = P
    upload_time = 0.0
    outage_time = 0.0
    air_time = 0.0
    for t in range(n):
        a2g = candidate.full_rate_bps[t] / scenario.limits.bandwidth_hz
        g2a = candidate.g2a_outage[t]
        sec = int(select_serving_sector(RadioSample(a2g, g2a),
                                        remaining > 0, eps_out))
        outage_time += float(g2a[sec]) * dt
        air_time += dt
        if remaining > 0:
            rate = float(candidate.full_rate_bps[t][sec])
            remaining = max(0.0, remaining - rate * dt)
            upload_time += dt
    if remaining > 0 and n >= 1:
        # residual drains at the hold boundary (final anchor) rate, bounded by
        # the maximum pre-arrival waiting budget
        t = n - 1
        a2g = candidate.full_rate_bps[t] / scenario.limits.bandwidth_hz
        g2a = candidate.g2a_outage[t]
        sec = int(select_serving_sector(RadioSample(a2g, g2a), True, eps_out))
        rate = float(candidate.full_rate_bps[t][sec])
        while remaining > 0 and upload_time <= scenario.limits.max_wait_s:
            remaining = max(0.0, remaining - rate * dt)
            upload_time += dt
    return {
        "flight_time_s": flight,
        "upload_delay_s": float(upload_time),
        "outage": float(outage_time / max(air_time, 1e-9)),
        "sector_diversity": _sector_entropy(candidate),
    }


def evaluate_segment(scenario, candidate, field, objective_spec):
    """Authoritative per-segment metrics.

    ``feasible`` comes from the synchronous evaluator on a minimal single-UAV
    plan that isolates the segment (the only payload-carrying edge), so the G2A
    outage constraint and every other completion rule are enforced.  ``T/D/O``
    come from the per-segment rules in :func:`_segment_tdo`.
    """
    from segment.stc.generators import generate_straight_hold

    src, tgt = candidate.source, candidate.target
    K = scenario.n_tasks
    A = np.asarray(scenario.depot_m if src == -1
                   else scenario.task_positions_m[src], dtype=float)
    B = np.asarray(scenario.terminal_m if tgt == K
                   else scenario.task_positions_m[tgt], dtype=float)
    P = float(candidate.source_data_bits)

    sc = STCScenario(
        n_uav=1, n_tasks=2,
        task_positions_m=(tuple(A), tuple(B)),
        task_data_bits=(P, 0.0),
        depot_m=tuple(A), terminal_m=tuple(B),
        limits=scenario.limits, length_m=scenario.length_m,
        width_m=scenario.width_m, n_bs=scenario.n_bs,
        task_seed=scenario.task_seed, map_seed=scenario.map_seed)

    d0 = generate_straight_hold(sc, field, -1, 0)   # depot(A) -> task0(A), zero length
    seg = _relabel_candidate(candidate, 0, 1)       # task0(A) -> task1(B): the segment
    d1 = generate_straight_hold(sc, field, 1, 2)    # task1(B) -> terminal(B), zero length
    lib = {(-1, 0): (d0,), (0, 1): (seg,), (1, 2): (d1,)}
    sol = PlanSolution(routes=((0, 1),),
                       candidate_ids=((d0.candidate_id, seg.candidate_id,
                                       d1.candidate_id),))

    result = evaluate_solution(sc, lib, sol, field, objective_spec,
                               keep_timelines=False)
    metrics = _segment_tdo(candidate, scenario)
    metrics["feasible"] = bool(result.feasible)
    return metrics


# --------------------------------------------------------------------------- #
# Evaluation shortcuts (RQ3)
# --------------------------------------------------------------------------- #

def _sync_result_dict(result):
    return {"feasible": bool(result.feasible),
            "makespan_s": float(result.makespan_s),
            "total_delay_s": float(result.total_delay_s),
            "objective": (None if result.objective == float("inf")
                          else float(result.objective))}


def evaluate_synchronous(scenario, library, solution, field, objective_spec):
    result = evaluate_solution(scenario, library, solution, field,
                               objective_spec, keep_timelines=False)
    return _sync_result_dict(result)


def _reduce_to_single_uav(scenario, library, route, cids):
    """Reduce a full multi-UAV scenario/library/solution to a single-UAV plan
    containing only ``route``'s tasks (re-indexed ``0..k-1``)."""
    K = scenario.n_tasks
    k = len(route)
    orig_edges = (([(-1, route[0])]
                   + [(route[i], route[i + 1]) for i in range(k - 1)]
                   + [(route[-1], K)]) if k else [])
    sc_red = STCScenario(
        n_uav=1, n_tasks=k,
        task_positions_m=tuple(scenario.task_positions_m[t] for t in route),
        task_data_bits=tuple(scenario.task_data_bits[t] for t in route),
        depot_m=scenario.depot_m, terminal_m=scenario.terminal_m,
        limits=scenario.limits, length_m=scenario.length_m,
        width_m=scenario.width_m, n_bs=scenario.n_bs,
        task_seed=scenario.task_seed, map_seed=scenario.map_seed)
    lib_red = {}
    for e, ((os, ot), cid) in enumerate(zip(orig_edges, cids)):
        cand = next(c for c in library[(os, ot)] if c.candidate_id == cid)
        lib_red[(e - 1, e)] = (_relabel_candidate(cand, e - 1, e),)
    sol_red = PlanSolution(routes=(tuple(range(k)),),
                           candidate_ids=(tuple(cids),))
    return sc_red, lib_red, sol_red


def evaluate_independent(scenario, library, solution, field, objective_spec):
    """Independent full-bandwidth: each UAV alone with only its own tasks.

    Each UAV is evaluated in a reduced single-UAV scenario containing only its
    assigned tasks, so tasks assigned to other UAVs are never counted as
    unserved (no false infeasibility).  Makespan is the max single-UAV
    completion; delay is the sum.
    """
    per = []
    for u in range(scenario.n_uav):
        route = solution.routes[u]
        if not route:
            continue  # idle UAV contributes nothing
        sc_red, lib_red, sol_red = _reduce_to_single_uav(
            scenario, library, route, solution.candidate_ids[u])
        res = evaluate_solution(sc_red, lib_red, sol_red, field, objective_spec,
                                keep_timelines=False)
        per.append(res)
    feasible = all(r.feasible for r in per)
    makespan = max((r.makespan_s for r in per), default=0.0)
    delay = sum(r.total_delay_s for r in per)
    objective = (objective_spec.score(makespan, delay) if feasible
                 else float("inf"))
    return {"feasible": bool(feasible), "makespan_s": float(makespan),
            "total_delay_s": float(delay),
            "objective": (None if objective == float("inf")
                          else float(objective))}


def evaluate_static_factor(scenario, library, solution, field, objective_spec,
                           congestion_factor=1.0):
    """Legacy static congestion factor: independent delay scaled by a constant."""
    indep = evaluate_independent(scenario, library, solution, field, objective_spec)
    delay = indep["total_delay_s"] * float(congestion_factor)
    feasible = indep["feasible"]
    objective = (objective_spec.score(indep["makespan_s"], delay) if feasible
                 else float("inf"))
    return {"feasible": bool(feasible), "makespan_s": indep["makespan_s"],
            "total_delay_s": float(delay),
            "objective": (None if objective == float("inf")
                          else float(objective))}


# --------------------------------------------------------------------------- #
# Lifecycle-protected matrix execution (recovery #7)
# --------------------------------------------------------------------------- #

def run_lifecycle(runner_name, cfg, run_id, matrix, per_scenario, summary_fn,
                  records_name="records.json"):
    """Run a matrix with RunContext-first, incremental writes, and a failed
    manifest on interruption.

    Creates the :class:`~segment.stc.run_context.RunContext` *before* computing,
    writes ``resolved_config.json`` immediately, writes raw records after each
    scenario (plus a ``progress.json`` checkpoint), prints per-scenario progress,
    enforces an optional wall-clock ``timeout_s``, and finalizes ``failed`` (with
    the partial records) on any error — so a long run never loses all results on
    a crash.
    """
    import time as _time
    from segment.stc.run_context import RunContext

    expected = ("resolved_config.json", records_name, "summary.json",
                "progress.json")
    rc = RunContext.create("results/runs", runner_name, cfg, run_id=run_id,
                           expected=expected)
    rc.register_input("resolved_config", data=json_bytes(cfg))
    rc.write_bytes("resolved_config.json", json_bytes(cfg))

    records_all = []
    completed = []
    start = _time.perf_counter()
    timeout = cfg.get("timeout_s")
    try:
        for idx, key in enumerate(matrix):
            if timeout is not None and _time.perf_counter() - start > timeout:
                raise TimeoutError("%s exceeded timeout %.0fs"
                                   % (runner_name, timeout))
            print("[%s] scenario %d/%d %s" % (runner_name, idx + 1, len(matrix),
                                              tuple(key)))
            records_all.extend(per_scenario(*key))
            completed.append(list(key))
            rc.write_bytes(records_name, json_bytes(records_all))
            rc.write_bytes("progress.json", json_bytes(
                {"completed": completed, "total": len(matrix)}))
        rc.write_bytes("summary.json", json_bytes(summary_fn(records_all)))
        rc.finalize("success")
    except Exception:
        try:
            rc.write_bytes(records_name, json_bytes(records_all))
        except Exception:
            pass
        rc.finalize("failed")
        raise
    print("run written to %s (%d records)" % (rc.run_dir, len(records_all)))
    return 0
