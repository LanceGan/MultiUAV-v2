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
from segment.stc.evaluator import default_objective_spec, evaluate_solution
from segment.stc.library import DEFAULT_GENERATOR_CONFIG, build_all_pairs_library
from segment.stc.radio import PerSectorRadioField
from segment.stc.run_context import RunContext
from segment.stc.sac_candidates import produce_sac_segment_candidates
from segment.stc.scenario import load_stc_scenario
from segment.stc.system_baselines import (
    BaselineUnavailable, COMM_AWARE_GENS, GRAPH_GENS, METHOD_SPECS, SYSTEM_BASELINE_KEYS,
    run_system_baseline, validate_baseline_availability,
)
from segment.stc.types import PlanSolution, STCScenario

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
                      "timelines.json", "summary.json", "rq2.json")


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


def _unavailable_payload(method, reason, cfg):
    """A method declared unavailable before any execution begins."""
    return dict(status="unavailable", feasible=None, objective=None,
                makespan_s=None, total_delay_s=None, outage_ratio=None,
                violations=None, balance=None, conflict_lost_bits=None,
                conflict_duration_s=None, waiting_s=None,
                evaluator_calls={"full": 0, "incremental": 0},
                failure_reason=reason, availability_declared=True,
                label=METHOD_SPECS[method].label, proposed=METHOD_SPECS[method].proposed,
                provenance={"required_method": method,
                            "checkpoint_path": cfg["sac_checkpoint"]})


def _failed_payload(method, exc, cfg):
    payload = _unavailable_payload(method, "execution failure: %r" % exc, cfg)
    payload.update(status="failed", availability_declared=False,
                   evaluator_calls=None, failure_type=type(exc).__name__)
    return payload


def _build_context(uav_num, map_seed, task_seed, cfg):
    """Build the shared scenario/field/library/reference for one scenario.

    RQ4 and RQ2 both consume this same context, so the field and library are
    constructed exactly once per scenario (no duplicate radio-field build).
    """
    base = load_stc_scenario(uav_num, task_seed=task_seed, map_seed=map_seed)
    sc = _small_scenario(base, cfg["smoke_tasks"]) if cfg["smoke"] else base
    field = PerSectorRadioField.from_scenario(sc, spacing_m=cfg["spacing_m"])
    # Produce real checkpoint-bound SAC candidates; an incompatible/missing
    # checkpoint is an explicit unavailable, never a deterministic substitution.
    sac_result = produce_sac_segment_candidates(cfg["sac_checkpoint"], sc, field)
    drl_candidates = (sac_result.candidates if sac_result.status == "available"
                      else {})
    library = build_all_pairs_library(sc, field,
                                      generator_config=cfg["generator_config"],
                                      drl_candidates=drl_candidates)
    reference = default_objective_spec(sc, library.candidates)
    return sc, field, library, reference, sac_result


def _run_scenario(sc, field, library, reference, sac_result,
                  uav_num, map_seed, task_seed, cfg, rc, checkpoint=None):
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
    if sac_result.status == "available":
        inputs["checkpoint_sac"] = rc.register_input(
            "checkpoint_sac_" + prefix, path=sac_result.checkpoint_path)
    options = {"checkpoint_path": cfg["sac_checkpoint"],
               "graph_generators": cfg["graph_generators"]}
    unavailable = {}
    if sac_result.status == "unavailable" and "gatsac_style" in cfg["methods"]:
        unavailable["gatsac_style"] = sac_result.reason
    for method in cfg["methods"]:
        if method in unavailable:
            continue
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
            failure = None
            if method in unavailable:
                payload = _unavailable_payload(method, unavailable[method], cfg)
                solution, trace = None, None
            else:
                try:
                    r = run_system_baseline(method, sc, library, field,
                                            seed=method_seed,
                                            method_seed=method_seed,
                                            iterations=cfg["iterations"],
                                            solver_config=solver_config,
                                            **options)
                    payload = _result_payload(r, sc)
                    solution, trace = r.solution, r.result.timelines
                except (Exception, KeyboardInterrupt) as exc:
                    payload = _failed_payload(method, exc, cfg)
                    failure = exc
                    solution, trace = None, None
            payload["runtime_s"] = time.perf_counter() - start
            records.append(dict(common, **payload))
            solutions.append({"record_id": record_id, "solution": solution})
            timelines.append({"record_id": record_id, "delta_t_s": sc.limits.delta_t_s,
                              "bandwidth_hz": sc.limits.bandwidth_hz, "timelines": trace})
            if checkpoint is not None:
                checkpoint(records[-1], solutions[-1], timelines[-1])
            else:
                for name, value in (("raw_records.json", records),
                                    ("solutions.json", solutions),
                                    ("timelines.json", timelines)):
                    rc.write_bytes(name, _json_bytes(value))
            if failure is not None:
                raise failure
    return records, solutions, timelines


def run_rq2(scenario, field, library, objective_spec, seed=42, checkpoint=None):
    """RQ2: assignment-method and sequence-method value under a fixed
    segment/evaluator path (Task 9 Step 2).

    Assignment methods (region, spatial k-means, Jia-EBTAS, library-aware
    EBTAS) are compared with a fixed library-aware sequence; sequence methods
    (distance, Jia time-energy, GA-EQTSP, library-aware) are compared with a
    fixed library-aware-EBTAS assignment.  Every plan is scored through the
    SAME :class:`ObjectiveSpec` and the authoritative synchronous evaluator.
    ``checkpoint`` receives the partial payload after every completed or failed
    method, before another method runs or an execution exception propagates.
    """
    from experiment.stc_common import seg_view
    from segment.stc.assignment import (
        jia_ebtas,
        library_aware_ebtas,
        region_partition,
        shortest_balanced,
        spatial_kmeans,
    )
    from segment.stc.sequencing import (
        distance_nn_2opt,
        ga_eqtsp_sequence,
        jia_time_energy_tsp,
        library_aware_sequence,
    )
    from segment.stc.slsqp_refine import candidate_cost

    lib = library.candidates if hasattr(library, "candidates") else dict(library)
    seg = seg_view(lib, scenario, field)
    dt = scenario.limits.delta_t_s
    K = scenario.n_tasks

    def _edges(route):
        if not route:
            return []
        return ([(-1, route[0])]
                + [(route[i], route[i + 1]) for i in range(len(route) - 1)]
                + [(route[-1], K)])

    def _candidate_ids(edges):
        out = []
        for s, t in edges:
            rankable = [c for c in lib[(s, t)] if c.rankable]
            out.append(min(rankable, key=lambda c: candidate_cost(c, dt)).candidate_id)
        return tuple(out)

    def _evaluate(routes):
        cids = tuple(_candidate_ids(_edges(r)) for r in routes)
        sol = PlanSolution(routes=tuple(tuple(r) for r in routes),
                           candidate_ids=cids)
        res = evaluate_solution(scenario, lib, sol, field, objective_spec,
                                keep_timelines=False)
        return {"status": "evaluated", "feasible": bool(res.feasible),
                "objective": (None if res.objective == float("inf")
                              else float(res.objective)),
                "makespan_s": float(res.makespan_s),
                "total_delay_s": float(res.total_delay_s)}

    def _failed(method, layer, exc):
        return {"method": method, "layer": layer, "status": "failed",
                "failure_reason": repr(exc), "failure_type": type(exc).__name__,
                "interrupted": isinstance(exc, KeyboardInterrupt),
                "availability_declared": False, "evaluator_calls": None,
                "seed": seed, "uav_num": scenario.n_uav,
                "map_seed": scenario.map_seed, "task_seed": scenario.task_seed,
                "feasible": None, "objective": None, "makespan_s": None,
                "total_delay_s": None}

    payload = {"assignment": [], "sequence": [],
               "summary": {"n_assignment": 0, "n_sequence": 0}}

    def _record(rec):
        layer = rec["layer"]
        payload[layer].append(rec)
        payload["summary"]["n_" + layer] = len(payload[layer])
        if checkpoint is not None:
            checkpoint(payload)

    assign_methods = {
        "region_partition": lambda: region_partition(scenario, seed=seed),
        "spatial_kmeans": lambda: spatial_kmeans(scenario, seed=seed),
        "shortest_balanced": lambda: shortest_balanced(scenario, seed=seed),
        "jia_ebtas": lambda: jia_ebtas(scenario, field=field, seed=seed),
        "library_aware_ebtas": lambda: library_aware_ebtas(scenario, seg, seed=seed),
    }
    for name, fn in assign_methods.items():
        try:
            assign = fn()
            routes = [library_aware_sequence(scenario, seg, cluster, seed=seed).route
                      for cluster in assign.clusters]
            rec = _evaluate(routes)
            rec.update({"method": name, "layer": "assignment"})
        except (Exception, KeyboardInterrupt) as exc:
            _record(_failed(name, "assignment", exc))
            raise
        _record(rec)

    try:
        base_assign = library_aware_ebtas(scenario, seg, seed=seed)
    except (Exception, KeyboardInterrupt) as exc:
        rec = _failed("library_aware_ebtas", "sequence", exc)
        rec["phase"] = "base_assignment"
        _record(rec)
        raise
    seq_methods = {
        "distance_nn_2opt": lambda ids: distance_nn_2opt(scenario, ids, seed=seed),
        "jia_time_energy_tsp": lambda ids: jia_time_energy_tsp(scenario, ids, seed=seed),
        "ga_eqtsp_sequence": lambda ids: ga_eqtsp_sequence(scenario, ids, seed=seed),
        "library_aware_sequence": lambda ids: library_aware_sequence(scenario, seg, ids, seed=seed),
    }
    for name, fn in seq_methods.items():
        try:
            routes = [fn(cluster).route for cluster in base_assign.clusters]
            rec = _evaluate(routes)
            rec.update({"method": name, "layer": "sequence"})
        except (Exception, KeyboardInterrupt) as exc:
            _record(_failed(name, "sequence", exc))
            raise
        _record(rec)

    return payload


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
    rc.write_bytes("resolved_config.json", _json_bytes(cfg))
    records, solutions, timelines = [], [], []
    rq2_records = []

    def _persist_partial():
        # completed scenarios survive a crash: write the partial records now so
        # finalize("failed") seals them instead of an empty run directory.
        for name, value in (("raw_records.json", records),
                            ("solutions.json", solutions),
                            ("timelines.json", timelines),
                            ("rq2.json", rq2_records)):
            try:
                rc.write_bytes(name, _json_bytes(value))
            except Exception:
                pass

    summary = None

    def _checkpoint_method(record, solution, timeline):
        records.append(record)
        solutions.append(solution)
        timelines.append(timeline)
        # Do not swallow checkpoint errors during execution: a method is only
        # complete once its evidence has reached the owning run artifacts.
        for name, value in (("raw_records.json", records),
                            ("solutions.json", solutions),
                            ("timelines.json", timelines)):
            rc.write_bytes(name, _json_bytes(value))

    try:
        for uav_num, map_seed, task_seed in itertools.product(
                cfg["uav_nums"], cfg["map_seeds"], cfg["task_seeds"]):
            sc, field, library, reference, sac_result = _build_context(
                uav_num, map_seed, task_seed, cfg)
            _run_scenario(sc, field, library, reference,
                                            sac_result, uav_num, map_seed,
                                            task_seed, cfg, rc,
                                            checkpoint=_checkpoint_method)
            # RQ2: assignment/sequence value on the SAME context (recovery #6).
            rq2 = {"uav_num": uav_num, "map_seed": map_seed,
                   "task_seed": task_seed}
            rq2_records.append(rq2)

            def _checkpoint_rq2(partial):
                rq2.update(partial)
                rc.write_bytes("rq2.json", _json_bytes(rq2_records))

            rq2.update(run_rq2(sc, field, library.candidates, reference,
                               seed=map_seed, checkpoint=_checkpoint_rq2))
            _persist_partial()   # incremental checkpoint after each scenario
        _assert_coverage(records, cfg)

        counts = Counter((r["method"], r["status"]) for r in records)
        summary = {"mode": "smoke" if cfg["smoke"] else "full",
                   "record_count": len(records),
                   "unavailable_count": sum(r["status"] == "unavailable" for r in records),
                   "inventory": [{"method": method, "status": status, "count": count}
                                 for (method, status), count in sorted(counts.items())]}
        rc.write_bytes("summary.json", _json_bytes(summary))
        rc.finalize("success")
    except KeyboardInterrupt:
        _persist_partial()
        rc.finalize("failed")
        raise
    except Exception:
        _persist_partial()
        rc.finalize("failed")
        raise
    print("run written to %s (%d records, %d unavailable)" % (
        rc.run_dir, len(records), summary["unavailable_count"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
