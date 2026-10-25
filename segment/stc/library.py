"""Pareto/diversity pruning and segment library persistence (Task 5)."""

import json
import os

import numpy as np

from segment.stc.slsqp_refine import candidate_cost


def pareto_prune(candidates, *, max_candidates, min_sector_distance=0.0):
    """Keep non-dominated candidates, bounded to ``max_candidates``.

    A candidate is dominated if another is no worse in flight time AND upload
    delay and strictly better in at least one.  ``min_sector_distance`` is
    reserved for future sector-diversity filtering.
    """
    costs = [candidate_cost(c, 0.5) for c in candidates]  # 0.5s default slot
    flight = [c.flight_steps * 0.5 for c in candidates]
    upload = [c.source_data_bits / max(float(np.mean(c.full_rate_bps.max(axis=1))), 1e-9)
              for c in candidates]

    keep = []
    for i in range(len(candidates)):
        dominated = False
        for j in range(len(candidates)):
            if i == j:
                continue
            if (flight[j] <= flight[i] and upload[j] <= upload[i]
                    and (flight[j] < flight[i] or upload[j] < upload[i])):
                dominated = True
                break
        if not dominated:
            keep.append(candidates[i])

    keep.sort(key=lambda c: candidate_cost(c, 0.5))
    return tuple(keep[:max_candidates])


def build_library(scenario, field, generators, pareto_kwargs=None):
    """Build the per-edge candidate library (placeholder; expanded later)."""
    pareto_kwargs = pareto_kwargs or {"max_candidates": 5}
    library = {}
    for src in range(-1, scenario.n_tasks + 1):
        for tgt in range(scenario.n_tasks + 1):
            if src == tgt:
                continue
            cands = []
            for gen in generators:
                out = gen(scenario, field, src, tgt)
                cands.extend(out if isinstance(out, list) else [out])
            library[(src, tgt)] = pareto_prune(cands, **pareto_kwargs)
    return library


def save_library(library, scenario, world_hash, path):
    """Persist the library: metadata JSON + per-candidate array NPZ."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {"world_hash": world_hash, "task_seed": scenario.task_seed,
            "map_seed": scenario.map_seed, "n_edges": len(library), "schema": 1}
    keys = []
    arrays = {}
    for (src, tgt), cands in library.items():
        for c in cands:
            key = "%d_%d_%s" % (src, tgt, c.candidate_id)
            keys.append([src, tgt, c.candidate_id, c.generator, c.source, c.target,
                         c.flight_steps, list(c.hold_position_m),
                         float(c.source_data_bits), bool(c.rankable)])
            arrays[key + "_pos"] = np.asarray(c.positions_m)
            arrays[key + "_rate"] = np.asarray(c.full_rate_bps)
            arrays[key + "_sector"] = np.asarray(c.sector_ids)
            arrays[key + "_outage"] = np.asarray(c.g2a_outage)
    if arrays:
        np.savez(path, **arrays)
    meta["keys"] = keys
    meta_path = path + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def load_library(path):
    """Load a persisted library back into a ``SegmentLibrary``."""
    from segment.stc.types import SegmentCandidate, SegmentLibrary
    with open(path + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    arrs = np.load(path)
    library = {}
    for (src, tgt, cid, gen, source, target, flight_steps, hold, bits, rankable) in meta["keys"]:
        key = "%d_%d_%s" % (src, tgt, cid)
        library.setdefault((src, tgt), []).append(SegmentCandidate(
            candidate_id=cid, source=source, target=target, generator=gen,
            positions_m=arrs[key + "_pos"], full_rate_bps=arrs[key + "_rate"],
            sector_ids=arrs[key + "_sector"], g2a_outage=arrs[key + "_outage"],
            flight_steps=flight_steps, hold_position_m=tuple(hold),
            source_data_bits=bits, rankable=rankable))
    return SegmentLibrary(candidates=library, map_seed=meta["map_seed"],
                          task_seed=meta["task_seed"], world_hash=meta["world_hash"])
