"""Shared plumbing for the RQ1-RQ5 STC experiment runners (Task 9).

Every runner uses the SAME scenario loading, tiny-scenario shrinking, field
construction, library building, single-UAV segment metrics, evaluation
shortcuts, and sealed run-output primitives so no runner invents its own physics
or objective path.  The authoritative objective is always
:class:`segment.stc.types.ObjectiveSpec` and the authoritative physics is always
:func:`segment.stc.evaluator.evaluate_solution`.
"""

import argparse
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from segment.stc.evaluator import evaluate_solution
from segment.stc.types import PlanSolution, STCScenario, SegmentLibrary

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
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--spacing-m", type=float, default=None)


def resolve_int_list(value):
    return [int(x) for x in value.split(",")] if value is not None else None


def resolve_runner_config(args, defaults):
    """Resolve a runner config: defaults <- config file <- CLI overrides.

    Raises ``ValueError`` when ``--config`` is given but the file is missing or
    invalid, so a missing-input runner never reaches run creation.
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
# Single-UAV segment metrics (RQ1)
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


def segment_metrics(candidate, scenario):
    """Single-UAV segment metrics (flight time, upload delay, outage, diversity).

    ``feasible`` follows the single-segment convention: the source payload can be
    uploaded within the flight time plus the arrival-boundary hold budget.
    """
    dt = scenario.limits.delta_t_s
    flight = candidate.flight_steps * dt
    mean_rate = float(np.mean(candidate.full_rate_bps.max(axis=1)))
    delay = candidate.source_data_bits / max(mean_rate, 1e-9)
    if len(candidate.sector_ids) == 0:
        outage = 0.0
    else:
        outage = float(np.mean([
            candidate.g2a_outage[t, int(s)]
            for t, s in enumerate(candidate.sector_ids)]))
    feasible = (candidate.source_data_bits <= 1e-9
                or delay <= flight + scenario.limits.max_wait_s)
    return {
        "flight_time_s": float(flight),
        "upload_delay_s": float(delay),
        "outage": float(outage),
        "sector_diversity": _sector_entropy(candidate),
        "feasible": bool(feasible),
    }


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


def evaluate_independent(scenario, library, solution, field, objective_spec):
    """Independent full-bandwidth: each UAV alone (no sharing), then combine.

    Makespan is the max single-UAV completion; delay is the sum.  A UAV is
    evaluated by building a plan where only that UAV is airborne, so the
    authoritative synchronous slot machine runs with a single active uploader
    (full bandwidth throughout).
    """
    n_uav = scenario.n_uav
    per = []
    for u in range(n_uav):
        routes = [()] * n_uav
        cids = [()] * n_uav
        routes[u] = solution.routes[u]
        cids[u] = solution.candidate_ids[u]
        sol = PlanSolution(routes=tuple(routes), candidate_ids=tuple(cids))
        res = evaluate_solution(scenario, library, sol, field, objective_spec,
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
