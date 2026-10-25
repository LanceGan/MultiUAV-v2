"""Pareto/diversity pruning and the frozen all-pairs segment library.

``build_all_pairs_library`` produces exactly ``K(K+1)`` directed task edges
(``K`` depot->task with zero payload, ``K(K-1)`` task->task carrying the source
payload, ``K`` task->terminal).  Per edge it runs the configured generators
(Straight-Hold, Comm-Waypoint, state-expanded A*, SLSQP refinement, plus the
non-rankable Straight-MaxSpeed relaxation reference), then Pareto-prunes the
rankable candidates on flight time, upload delay, G2A outage and sector
signature.

Persistence is hashed and compatibility-checked: the manifest records content
hashes for every candidate plus scenario / task-content / map-world / radio
parameter / generator-config / checkpoint hashes, and ``load_library`` rejects
any mismatch.
"""

import hashlib
import json
import os

import numpy as np

from segment.stc.generators import (
    generate_comm_waypoints,
    generate_straight_hold,
    generate_straight_relaxation,
)
from segment.stc.graph_search import astar_candidates
from segment.stc.slsqp_refine import candidate_cost, refine_candidate
from segment.stc.types import SegmentCandidate, SegmentLibrary

# Persistence schema versions.  ``LIBRARY_SCHEMA`` gates deserialisation;
# ``SI_SCHEMA`` pins the unit convention (metres / seconds / bits / hertz / watt).
LIBRARY_SCHEMA = 2
SI_SCHEMA = "si-v1"

DEFAULT_GENERATOR_CONFIG = {
    "straight_hold": True,
    "straight_relaxation": True,
    "comm_waypoints": {"grid_shape": (10, 10), "max_waypoints": 1},
    "astar": {"grid": 10},
    "slsqp_refine": {"max_waypoints": 1, "max_refinements": 3},
}


# --------------------------------------------------------------------------- #
# Pareto / diversity pruning
# --------------------------------------------------------------------------- #

def _flight_time(c, delta_t_s):
    return c.flight_steps * delta_t_s


def _upload_delay(c):
    mean_rate = float(np.mean(c.full_rate_bps.max(axis=1)))
    return c.source_data_bits / max(mean_rate, 1e-9)


def _outage(c):
    """Mean serving-sector G2A outage along the preferred-sector path."""
    if len(c.sector_ids) == 0:
        return 0.0
    return float(np.mean([c.g2a_outage[t, int(s)] for t, s in enumerate(c.sector_ids)]))


def _sector_signature(c):
    """Fraction of flight steps spent serving each sector (len n_sectors)."""
    n_sectors = int(c.full_rate_bps.shape[1])
    counts = np.zeros(n_sectors, dtype=float)
    for s in c.sector_ids:
        counts[int(s)] += 1.0
    return counts / max(len(c.sector_ids), 1)


def _sector_l1_distance(a, b):
    return float(np.sum(np.abs(np.asarray(a) - np.asarray(b))))


def pareto_prune(candidates, *, max_candidates, min_sector_distance=0.0,
                 delta_t_s=0.5):
    """Keep a bounded, non-dominated, sector-diverse candidate set.

    A candidate is dominated if another is no worse in flight time AND upload
    delay AND G2A outage and strictly better in at least one.  When
    ``min_sector_distance > 0``, a dominated candidate is retained anyway if its
    sector occupancy differs materially (L1 distance >= ``min_sector_distance``)
    from *every* candidate that dominates it.  Exact duplicates (identical
    objectives and identical sector signature) are collapsed to one.
    """
    if not candidates:
        return ()

    objs = np.array([[_flight_time(c, delta_t_s), _upload_delay(c), _outage(c)]
                     for c in candidates])
    sigs = [_sector_signature(c) for c in candidates]

    def _dup_key(i):
        return (tuple(np.round(objs[i], 9)), tuple(np.round(sigs[i], 9)))

    keep = []
    seen = set()
    for i in range(len(candidates)):
        # reject exact duplicates (identical objectives AND identical signature),
        # keeping the first occurrence only
        key = _dup_key(i)
        if key in seen:
            continue
        seen.add(key)

        dominated_by = [j for j in range(len(candidates)) if j != i and (
            objs[j, 0] <= objs[i, 0] and objs[j, 1] <= objs[i, 1]
            and objs[j, 2] <= objs[i, 2]
            and (objs[j, 0] < objs[i, 0] or objs[j, 1] < objs[i, 1]
                 or objs[j, 2] < objs[i, 2]))]
        if dominated_by:
            # retain only when materially different sector occupancy from all dominators
            if min_sector_distance > 0 and all(
                    _sector_l1_distance(sigs[i], sigs[j]) >= min_sector_distance
                    for j in dominated_by):
                keep.append(i)
        else:
            keep.append(i)

    kept = [candidates[i] for i in keep]
    kept.sort(key=lambda c: candidate_cost(c, delta_t_s))
    return tuple(kept[:max_candidates])


# --------------------------------------------------------------------------- #
# Library assembly
# --------------------------------------------------------------------------- #

def _legal_edges(n_tasks):
    """Exactly ``K(K+1)`` legal directed task edges (no self, no depot->terminal)."""
    edges = []
    for i in range(n_tasks):
        edges.append((-1, i))              # depot -> task (zero payload)
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i != j:
                edges.append((i, j))       # task -> task (source payload)
    for i in range(n_tasks):
        edges.append((i, n_tasks))         # task -> terminal (source payload)
    return edges


def _generate_edge_candidates(scenario, field, source, target, generator_config):
    """Run the configured generators for one directed edge.

    Returns ``(rankable, non_rankable)`` lists of :class:`SegmentCandidate`.
    """
    rankable = []
    non_rankable = []

    if generator_config.get("straight_hold", False):
        rankable.append(generate_straight_hold(scenario, field, source, target))

    if generator_config.get("straight_relaxation", False):
        non_rankable.append(
            generate_straight_relaxation(scenario, field, source, target))

    cw_cfg = generator_config.get("comm_waypoints")
    if cw_cfg:
        rankable.extend(generate_comm_waypoints(
            scenario, field, source, target,
            grid_shape=cw_cfg.get("grid_shape", (10, 10)),
            max_waypoints=cw_cfg.get("max_waypoints", 1)))

    astar_cfg = generator_config.get("astar")
    if astar_cfg:
        rankable.append(astar_candidates(
            scenario, field, source, target, grid=astar_cfg.get("grid", 10)))

    slsqp_cfg = generator_config.get("slsqp_refine")
    if slsqp_cfg:
        relays = [c for c in rankable if c.generator == "comm_waypoint"]
        relays.sort(key=lambda c: candidate_cost(c, scenario.limits.delta_t_s))
        max_refinements = slsqp_cfg.get("max_refinements", 3)
        max_waypoints = slsqp_cfg.get("max_waypoints", 1)
        for c in relays[:max_refinements]:
            rankable.append(refine_candidate(
                c, scenario, field, max_waypoints=max_waypoints))

    return rankable, non_rankable


def build_all_pairs_library(scenario, field, *, generator_config=None,
                            pareto_kwargs=None, drl_candidates=None):
    """Build the frozen all-pairs directed segment library.

    Produces exactly ``K(K+1)`` task edges.  ``drl_candidates`` is an optional
    mapping ``{(source, target): [SegmentCandidate, ...]}`` of DRL baseline
    entries (labeled with :func:`drl_generator_label`); they are stored verbatim
    after pruning and never dropped.
    """
    generator_config = (generator_config if generator_config is not None
                        else DEFAULT_GENERATOR_CONFIG)
    pareto_kwargs = dict(pareto_kwargs or {"max_candidates": 5})
    pareto_kwargs.setdefault("delta_t_s", scenario.limits.delta_t_s)
    drl_candidates = drl_candidates or {}

    candidates = {}
    for (source, target) in _legal_edges(scenario.n_tasks):
        rankable, non_rankable = _generate_edge_candidates(
            scenario, field, source, target, generator_config)
        pruned = pareto_prune(rankable, **pareto_kwargs)
        extra = list(non_rankable) + list(drl_candidates.get((source, target), ()))
        candidates[(source, target)] = tuple(pruned) + tuple(extra)

    return SegmentLibrary(candidates=candidates, map_seed=scenario.map_seed,
                          task_seed=scenario.task_seed, world_hash=field.world_hash)


def build_library(scenario, field, generators, pareto_kwargs=None):
    """Backward-compatible alias (deprecated): delegates to ``build_all_pairs_library``.

    Kept for any legacy caller that passed an explicit generator list; the list
    is ignored in favour of the default generator configuration.
    """
    return build_all_pairs_library(scenario, field, pareto_kwargs=pareto_kwargs)


# --------------------------------------------------------------------------- #
# Hashing helpers
# --------------------------------------------------------------------------- #

def _sha(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def _float_bytes(value):
    return np.ascontiguousarray(np.asarray(value, dtype=float)).tobytes()


def scenario_hash(scenario):
    """Content hash of the full scenario (positions, data, limits, region, BS)."""
    parts = [b"stc-scenario-v1",
             _float_bytes(scenario.task_positions_m),
             _float_bytes(scenario.task_data_bits),
             _float_bytes(scenario.depot_m),
             _float_bytes(scenario.terminal_m),
             _float_bytes([scenario.length_m, scenario.width_m, scenario.n_bs]),
             np.asarray(scenario.n_tasks, dtype=np.int64).tobytes(),
             np.asarray(scenario.n_bs, dtype=np.int64).tobytes()]
    for name in sorted(scenario.limits.__dataclass_fields__):
        parts.append(repr(getattr(scenario.limits, name)).encode("utf-8"))
    return _sha(*parts)


def task_content_hash(scenario):
    """Content hash of the task positions and payloads only (map-independent)."""
    return _sha(b"stc-task-content-v1",
                _float_bytes(scenario.task_positions_m),
                _float_bytes(scenario.task_data_bits))


def radio_params(scenario):
    """Radio-relevant parameters recorded for compatibility checks."""
    return {
        "bandwidth_hz": scenario.limits.bandwidth_hz,
        "uplink_power_w": scenario.limits.uplink_power_w,
        "eps_out": scenario.limits.eps_out,
        "uav_height_m": scenario.limits.uav_height_m,
        "n_bs": scenario.n_bs,
        "n_sectors": scenario.n_sectors,
    }


def radio_params_hash(scenario):
    return _sha(b"stc-radio-params-v1",
                json.dumps(radio_params(scenario), sort_keys=True).encode("utf-8"))


def generator_config_hash(generator_config):
    return _sha(b"stc-generator-config-v1",
                json.dumps(generator_config, sort_keys=True).encode("utf-8"))


def _candidate_content_hash(positions_m, full_rate_bps, sector_ids, g2a_outage,
                            candidate_id, source, target, generator,
                            flight_steps, hold_position_m, source_data_bits, rankable):
    h = hashlib.sha256()
    scalar = (str(candidate_id), int(source), int(target), str(generator),
              int(flight_steps), tuple(float(x) for x in hold_position_m),
              float(source_data_bits), bool(rankable))
    h.update(repr(scalar).encode("utf-8"))
    for arr in (positions_m, full_rate_bps, sector_ids, g2a_outage):
        h.update(np.ascontiguousarray(np.asarray(arr)).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_library(library, scenario, world_hash, path, *, generator_config=None,
                 checkpoint_hashes=None):
    """Persist the library (metadata JSON + candidate-array NPZ), return the manifest."""
    generator_config = generator_config if generator_config is not None \
        else DEFAULT_GENERATOR_CONFIG
    checkpoint_hashes = dict(checkpoint_hashes or {})
    os.makedirs(os.path.dirname(path), exist_ok=True)

    keys = []
    arrays = {}
    candidate_hashes = {}
    for (src, tgt), cands in library.items():
        for c in cands:
            key = "%d_%d_%s" % (src, tgt, c.candidate_id)
            cid, source, target, gen = c.candidate_id, c.source, c.target, c.generator
            flight_steps = int(c.flight_steps)
            hold = [float(x) for x in c.hold_position_m]
            bits = float(c.source_data_bits)
            rankable = bool(c.rankable)
            keys.append([src, tgt, cid, gen, source, target, flight_steps,
                         hold, bits, rankable])
            arrays[key + "_pos"] = np.asarray(c.positions_m)
            arrays[key + "_rate"] = np.asarray(c.full_rate_bps)
            arrays[key + "_sector"] = np.asarray(c.sector_ids)
            arrays[key + "_outage"] = np.asarray(c.g2a_outage)
            candidate_hashes[key] = _candidate_content_hash(
                c.positions_m, c.full_rate_bps, c.sector_ids, c.g2a_outage,
                cid, source, target, gen, flight_steps, hold, bits, rankable)

    meta = {
        "schema": LIBRARY_SCHEMA,
        "si_schema": SI_SCHEMA,
        "world_hash": world_hash,
        "scenario_hash": scenario_hash(scenario),
        "task_content_hash": task_content_hash(scenario),
        "radio_params": radio_params(scenario),
        "radio_params_hash": radio_params_hash(scenario),
        "generator_config": generator_config,
        "generator_config_hash": generator_config_hash(generator_config),
        "checkpoint_hashes": checkpoint_hashes,
        "map_seed": scenario.map_seed,
        "task_seed": scenario.task_seed,
        "n_edges": len(library),
        "n_candidates": len(keys),
        "keys": keys,
        "candidate_hashes": candidate_hashes,
    }

    if arrays:
        np.savez(path, **arrays)
    else:
        np.savez(path)
    meta_path = path + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_library(path, *, expected=None):
    """Load a persisted library, verifying hashes and an optional compatibility set.

    ``expected`` is a dict of compatibility fields (any of ``world_hash``,
    ``scenario_hash``, ``task_content_hash``, ``radio_params_hash``,
    ``generator_config_hash``, ``checkpoint_hashes``, ``si_schema``, ``schema``).
    A present-but-different value raises :class:`ValueError`.  Per-candidate
    content hashes are always verified against the manifest.
    """
    with open(path + ".json", encoding="utf-8") as f:
        meta = json.load(f)

    if meta.get("schema") != LIBRARY_SCHEMA:
        raise ValueError("library schema %r != expected %d"
                         % (meta.get("schema"), LIBRARY_SCHEMA))
    if meta.get("si_schema") != SI_SCHEMA:
        raise ValueError("SI schema %r != expected %r"
                         % (meta.get("si_schema"), SI_SCHEMA))

    if expected:
        for key, want in expected.items():
            if key == "schema":
                actual = meta.get("schema")
            elif key == "si_schema":
                actual = meta.get("si_schema")
            elif key == "world_hash":
                actual = meta.get("world_hash")
            elif key == "scenario_hash":
                actual = meta.get("scenario_hash")
            elif key == "task_content_hash":
                actual = meta.get("task_content_hash")
            elif key == "radio_params_hash":
                actual = meta.get("radio_params_hash")
            elif key == "generator_config_hash":
                actual = meta.get("generator_config_hash")
            elif key == "checkpoint_hashes":
                actual = meta.get("checkpoint_hashes", {})
            else:
                raise ValueError("unknown expected compatibility field %r" % key)
            if actual != want:
                raise ValueError(
                    "library compatibility mismatch on %r: stored %r != expected %r"
                    % (key, actual, want))

    arrs = np.load(path)
    library = {}
    for (src, tgt, cid, gen, source, target, flight_steps, hold, bits, rankable) in meta["keys"]:
        key = "%d_%d_%s" % (src, tgt, cid)
        cand = SegmentCandidate(
            candidate_id=cid, source=source, target=target, generator=gen,
            positions_m=arrs[key + "_pos"], full_rate_bps=arrs[key + "_rate"],
            sector_ids=arrs[key + "_sector"], g2a_outage=arrs[key + "_outage"],
            flight_steps=flight_steps, hold_position_m=tuple(hold),
            source_data_bits=bits, rankable=rankable)
        if _candidate_content_hash(
                cand.positions_m, cand.full_rate_bps, cand.sector_ids, cand.g2a_outage,
                cid, source, target, gen, flight_steps, hold, bits, rankable) \
                != meta.get("candidate_hashes", {}).get(key):
            raise ValueError("content hash mismatch for candidate %r" % key)
        library.setdefault((src, tgt), []).append(cand)

    return SegmentLibrary(candidates=library, map_seed=meta["map_seed"],
                          task_seed=meta["task_seed"], world_hash=meta["world_hash"])
