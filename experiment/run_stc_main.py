"""Complete-CKM, simultaneous-start experiments with sealed replay data.

DEFAULT_CONFIG documents defaults. Declared top-level JSON values override
them, then explicit CLI flags override the file. ``solver.iterations`` aliases
``iterations``. Generator settings completely declare enabled generators;
omitted families resolve to false, enabled objects receive explicit defaults.
Straight flags require booleans; other families accept boolean enable/disable
or a parameter object. Only the implemented single-relay mode is accepted.
Smoke defaults: six tasks, spacing 1000m, reduced generators; explicit settings
take precedence. The deterministic builder cannot train or invent SAC data.
"""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment.stc.conflicts import build_conflict_report
from segment.stc.evaluator import default_objective_spec
from segment.stc.library import DEFAULT_GENERATOR_CONFIG, build_all_pairs_library
from segment.stc.radio import PerSectorRadioField
from segment.stc.run_context import RunContext
from segment.stc.scenario import load_stc_scenario
from segment.stc.system_baselines import (
    BaselineUnavailable, COMM_AWARE_GENS, GRAPH_GENS, METHOD_SPECS, SYSTEM_BASELINE_KEYS,
    run_system_baseline, validate_baseline_availability,
)
from segment.stc.types import STCScenario

SMOKE_GENERATOR_CONFIG = {
    "straight_hold": True,
    "comm_waypoints": {"grid_shape": [3, 3], "max_waypoints": 1},
}
DEFAULT_CONFIG = {
    "uav_nums": [2], "map_seeds": [42], "task_seeds": [42],
    "method_seeds": [0, 1, 2, 3, 4], "methods": list(SYSTEM_BASELINE_KEYS),
    "iterations": 1000, "spacing_m": 200., "smoke": False, "smoke_tasks": 6,
    "generator_config": DEFAULT_GENERATOR_CONFIG,
    "graph_generators": sorted(GRAPH_GENS), "sac_checkpoint": None,
    "solver": {"rho": .1, "T0": .1, "cooling": .995, "segment": 100,
               "audit_interval": 50},
    "score": {"new_best": 3., "improve": 1., "accept": .5, "reject": 0.},
}
EXPECTED_ARTIFACTS = ("resolved_config.json", "raw_records.json", "solutions.json",
                      "timelines.json", "summary.json")


def _common_args(parser):
    parser.add_argument("--config", default="configs/stc/experiments.json")
    for flag in ("map-seeds", "task-seeds", "method-seeds", "uav-nums", "methods"):
        parser.add_argument("--" + flag, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke", action="store_true", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--spacing-m", type=float, default=None)


def _reject_constant(token):
    raise ValueError("non-finite JSON constant %s" % token)


def _config_integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))


def _config_number(value, name, minimum=0, maximum=None, exclusive_min=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= minimum if exclusive_min else value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError("%s has an invalid numeric type or range" % name)


def _resolved_generators(declared):
    """Validate every family/parameter and materialize enabled defaults."""
    if not isinstance(declared, dict):
        raise ValueError("generator_config must be an object")
    if set(declared) - set(DEFAULT_GENERATOR_CONFIG):
        raise ValueError("unsupported generator_config families")
    resolved = {}
    for family, defaults in DEFAULT_GENERATOR_CONFIG.items():
        value = declared.get(family, False)
        name = "generator_config." + family
        if isinstance(defaults, bool):
            if type(value) is not bool:
                raise ValueError("%s must be boolean" % name)
            resolved[family] = value
            continue
        if type(value) is bool:
            if not value:
                resolved[family] = False
                continue
            value = {}
        if not isinstance(value, dict) or set(value) - set(defaults):
            raise ValueError("unsupported %s parameters" % name)
        params = deepcopy(defaults)
        params.update(value)
        if family == "comm_waypoints":
            shape = params["grid_shape"]
            if not isinstance(shape, (list, tuple)) or len(shape) != 2:
                raise ValueError(name + ".grid_shape must contain two integers")
            for extent in shape:
                _config_integer(extent, name + ".grid_shape", 1)
            params["grid_shape"] = list(shape)
        if family == "astar":
            _config_integer(params["grid"], name + ".grid", 2)
        if "max_waypoints" in params:
            if type(params["max_waypoints"]) is not int or params["max_waypoints"] != 1:
                raise ValueError(name + ".max_waypoints must be 1 (implemented single-relay mode)")
        if "max_refinements" in params:
            _config_integer(params["max_refinements"], name + ".max_refinements", 0)
        resolved[family] = params
    return resolved


def _resolve_config(args):
    path = Path(args.config)
    if not path.is_file():
        raise ValueError("config must be an existing file: %s" % path)
    declared = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(declared, dict):
        raise ValueError("config must be a JSON object")
    unknown = set(declared) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError("unknown config fields: %s" % sorted(unknown))
    cfg = deepcopy(DEFAULT_CONFIG)
    for key, value in declared.items():
        if key in {"solver", "score"}:
            if not isinstance(value, dict):
                raise ValueError("%s must be an object" % key)
            allowed = set(DEFAULT_CONFIG[key]) | ({"iterations"} if key == "solver" else set())
            if set(value) - allowed:
                raise ValueError("unsupported %s settings" % key)
            cfg[key].update(value)
        else:
            cfg[key] = value
    if "iterations" in cfg["solver"]:
        _config_integer(cfg["solver"]["iterations"], "solver.iterations", 0)
        if "iterations" not in declared:
            cfg["iterations"] = cfg["solver"]["iterations"]
    cfg["solver"].pop("iterations", None)
    for key in ("iterations", "spacing_m", "smoke", "methods", "uav_nums",
                "map_seeds", "task_seeds", "method_seeds"):
        value = getattr(args, key)
        if value is not None:
            if key in {"methods", "uav_nums", "map_seeds", "task_seeds", "method_seeds"}:
                value = value.split(",")
                if key != "methods":
                    value = [int(x) for x in value]
            cfg[key] = value
    if not isinstance(cfg["smoke"], bool):
        raise ValueError("smoke must be boolean")
    if cfg["smoke"]:
        if "spacing_m" not in declared and args.spacing_m is None:
            cfg["spacing_m"] = 1000.
        if "generator_config" not in declared:
            cfg["generator_config"] = deepcopy(SMOKE_GENERATOR_CONFIG)
    for key in ("uav_nums", "map_seeds", "task_seeds", "method_seeds"):
        values = cfg[key]
        if (not isinstance(values, list) or not values
                or any(type(x) is not int for x in values)
                or len(values) != len(set(values))):
            raise ValueError("%s must be a nonempty list of distinct integers" % key)
    if any(x <= 0 for x in cfg["uav_nums"]):
        raise ValueError("uav_nums must be positive")
    if (not isinstance(cfg["methods"], list) or not cfg["methods"]
            or any(x not in METHOD_SPECS for x in cfg["methods"])
            or len(cfg["methods"]) != len(set(cfg["methods"]))):
        raise ValueError("methods must be distinct registered method names")
    for key, minimum in (("iterations", 0), ("smoke_tasks", 1)):
        _config_integer(cfg[key], key, minimum)
    _config_number(cfg["spacing_m"], "spacing_m", exclusive_min=True)
    _config_number(cfg["solver"]["rho"], "solver.rho", maximum=1)
    _config_number(cfg["solver"]["cooling"], "solver.cooling", maximum=1, exclusive_min=True)
    _config_number(cfg["solver"]["T0"], "solver.T0", exclusive_min=True)
    _config_integer(cfg["solver"]["segment"], "solver.segment", 1)
    _config_integer(cfg["solver"]["audit_interval"], "solver.audit_interval", 0)
    for key, value in cfg["score"].items():
        _config_number(value, "score." + key)
    cfg["generator_config"] = _resolved_generators(cfg["generator_config"])
    if (not isinstance(cfg["graph_generators"], list)
            or not cfg["graph_generators"]
            or any(type(x) is not str or x not in GRAPH_GENS | COMM_AWARE_GENS
                   for x in cfg["graph_generators"])
            or len(cfg["graph_generators"]) != len(set(cfg["graph_generators"]))):
        raise ValueError("graph_generators must declare distinct graph/SLSQP or communication families")
    if cfg["sac_checkpoint"] is not None:
        if not isinstance(cfg["sac_checkpoint"], str):
            raise ValueError("sac_checkpoint must be a path or null")
        checkpoint = Path(cfg["sac_checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = path.resolve().parent / checkpoint
        cfg["sac_checkpoint"] = str(checkpoint.resolve())
    json.dumps(cfg, allow_nan=False)
    return cfg


def _small_scenario(base, n_tasks=6):
    n_tasks = min(n_tasks, base.n_tasks)
    return STCScenario(
        n_uav=base.n_uav, n_tasks=n_tasks,
        task_positions_m=base.task_positions_m[:n_tasks],
        task_data_bits=base.task_data_bits[:n_tasks], depot_m=base.depot_m,
        terminal_m=base.terminal_m, limits=base.limits, length_m=base.length_m,
        width_m=base.width_m, n_bs=base.n_bs, task_seed=base.task_seed,
        map_seed=base.map_seed)


def _json_safe(value):
    """One strict output path: typed arrays/dataclasses and non-finite -> null."""
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_bytes(value):
    return json.dumps(_json_safe(value), sort_keys=True, allow_nan=False,
                      ensure_ascii=False).encode("utf-8")


def _result_payload(r, scenario):
    result = r.result
    conflict = build_conflict_report(result, scenario.limits.bandwidth_hz,
                                     scenario.limits.delta_t_s)
    waiting = [sum(s.time_end_s - s.time_start_s for s in tl.slots if s.holding)
               for tl in result.timelines]
    return {
        "method": r.method, "label": METHOD_SPECS[r.method].label,
        "proposed": METHOD_SPECS[r.method].proposed, "status": "evaluated",
        "feasible": result.feasible, "objective": result.objective,
        "makespan_s": result.makespan_s, "total_delay_s": result.total_delay_s,
        "outage_ratio": result.outage_ratio, "violations": result.violations,
        "balance": result.balance, "conflict_lost_bits": conflict.lost_bits,
        "conflict_duration_s": conflict.total_overlap_s,
        "conflict_per_uav_lost_bits": conflict.per_uav_loss,
        "waiting_s": sum(waiting), "per_uav_waiting_s": waiting,
        "evaluator_calls": r.evaluator_calls, "provenance": r.provenance,
        "failure_reason": None if result.feasible else "synchronous evaluation infeasible",
    }


def _run_scenario(uav_num, map_seed, task_seed, cfg, rc):
    base = load_stc_scenario(uav_num, task_seed=task_seed, map_seed=map_seed)
    sc = _small_scenario(base, cfg["smoke_tasks"]) if cfg["smoke"] else base
    field = PerSectorRadioField.from_scenario(sc, spacing_m=cfg["spacing_m"])
    library = build_all_pairs_library(sc, field, generator_config=cfg["generator_config"])
    reference = default_objective_spec(sc, library.candidates)
    prefix = "%d_%d_%d" % (uav_num, map_seed, task_seed)
    inputs = {}
    for kind, data in (
        ("scenario", sc),
        ("task", {"positions_m": sc.task_positions_m, "data_bits": sc.task_data_bits}),
        ("map", {"world_hash": field.world_hash}),
        ("library", [[src, tgt, list(cs)]
                     for (src, tgt), cs in sorted(library.candidates.items())]),
    ):
        inputs[kind] = rc.register_input(kind + "_" + prefix, data=_json_bytes(data))
    options = {"checkpoint_path": cfg["sac_checkpoint"],
               "graph_generators": cfg["graph_generators"]}
    unavailable = {}
    for method in cfg["methods"]:
        try:
            validate_baseline_availability(method, sc, library, **options)
        except BaselineUnavailable as exc:
            unavailable[method] = str(exc)
    records, solutions, timelines = [], [], []
    solver_config = dict(cfg["solver"])
    solver_config.update({"score_" + key: val for key, val in cfg["score"].items()})
    for method_seed in cfg["method_seeds"]:
        for method in cfg["methods"]:
            record_id = "%s_%d_%s" % (prefix, method_seed, method)
            common = {"record_id": record_id, "uav_num": uav_num,
                      "map_seed": map_seed, "task_seed": task_seed,
                      "method_seed": method_seed, "method": method,
                      "reference": reference, "reference_hash": reference.reference.source_hash,
                      "library_hash": inputs["library"], "input_hashes": inputs,
                      "world_hash": field.world_hash}
            start = time.perf_counter()
            if method in unavailable:
                payload = dict(status="unavailable", feasible=None, objective=None,
                               makespan_s=None, total_delay_s=None, outage_ratio=None,
                               violations=None, balance=None, conflict_lost_bits=None,
                               conflict_duration_s=None, waiting_s=None,
                               evaluator_calls={"full": 0, "incremental": 0},
                               failure_reason=unavailable[method], availability_declared=True,
                               label=METHOD_SPECS[method].label, proposed=False,
                               provenance={"required_method": method,
                                           "checkpoint_path": cfg["sac_checkpoint"]})
                solution, trace = None, None
            else:
                r = run_system_baseline(method, sc, library, field, seed=method_seed,
                                        method_seed=method_seed, iterations=cfg["iterations"],
                                        solver_config=solver_config, **options)
                payload = _result_payload(r, sc)
                solution, trace = r.solution, r.result.timelines
            payload["runtime_s"] = time.perf_counter() - start
            records.append(dict(common, **payload))
            solutions.append({"record_id": record_id, "solution": solution})
            timelines.append({"record_id": record_id, "delta_t_s": sc.limits.delta_t_s,
                              "bandwidth_hz": sc.limits.bandwidth_hz, "timelines": trace})
    return records, solutions, timelines


def _assert_coverage(records, cfg):
    fields = ("uav_num", "map_seed", "task_seed", "method_seed", "method")
    expected = set(itertools.product(*(cfg[key] for key in
                   ("uav_nums", "map_seeds", "task_seeds", "method_seeds", "methods"))))
    actual = Counter(tuple(r[key] for key in fields) for r in records)
    if (set(actual) != expected or any(n != 1 for n in actual.values())
            or any(r["status"] not in {"evaluated", "unavailable"} for r in records)
            or any(r["status"] == "unavailable" and
                   not (r.get("availability_declared") and r.get("failure_reason"))
                   for r in records)):
        raise RuntimeError("method-seed matrix coverage mismatch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    _common_args(parser)
    args = parser.parse_args(argv)
    try:
        cfg = _resolve_config(args)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print("run matrix: " + json.dumps(cfg, sort_keys=True, allow_nan=False))
        return 0
    rc = RunContext.create("results/runs", "stc_main", cfg, run_id=args.run_id,
                           expected=EXPECTED_ARTIFACTS)
    rc.register_input("config_file", path=args.config)
    rc.register_input("resolved_config", data=_json_bytes(cfg))
    if cfg["sac_checkpoint"] and Path(cfg["sac_checkpoint"]).is_file():
        rc.register_input("checkpoint_sac", path=cfg["sac_checkpoint"])
    records, solutions, timelines = [], [], []
    try:
        for uav_num, map_seed, task_seed in itertools.product(
                cfg["uav_nums"], cfg["map_seeds"], cfg["task_seeds"]):
            raw, sol, trace = _run_scenario(uav_num, map_seed, task_seed, cfg, rc)
            records.extend(raw)
            solutions.extend(sol)
            timelines.extend(trace)
        _assert_coverage(records, cfg)
        counts = Counter((r["method"], r["status"]) for r in records)
        summary = {"mode": "smoke" if cfg["smoke"] else "full",
                   "record_count": len(records),
                   "unavailable_count": sum(r["status"] == "unavailable" for r in records),
                   "inventory": [{"method": method, "status": status, "count": count}
                                 for (method, status), count in sorted(counts.items())]}
        for name, value in (("resolved_config.json", cfg), ("raw_records.json", records),
                            ("solutions.json", solutions), ("timelines.json", timelines),
                            ("summary.json", summary)):
            rc.write_bytes(name, _json_bytes(value))
        rc.finalize("success")
    except Exception:
        rc.finalize("failed")
        raise
    print("run written to %s (%d records, %d unavailable)" % (
        rc.run_dir, len(records), summary["unavailable_count"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
