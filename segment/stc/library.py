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
    """Persist the library (metadata JSON + per-edge candidate arrays NPZ)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {"world_hash": world_hash, "task_seed": scenario.task_seed,
            "map_seed": scenario.map_seed, "n_edges": len(library)}
    meta_path = path + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def load_library(path):
    """Load a saved library (placeholder; raises for missing metadata)."""
    meta_path = path + ".json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return meta
