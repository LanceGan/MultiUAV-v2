"""Content-addressed cache for expensive STC segment generation.

The cache stores only candidate-generation products.  Every load validates the
scenario, radio world, generator configuration, checkpoint hashes and a source
fingerprint; a mismatch rebuilds rather than reusing stale numerical data.
"""

import hashlib
import json
import os
from pathlib import Path
import platform
from functools import lru_cache

import numpy as np

from segment.stc.library import (
    build_all_pairs_library,
    generator_config_hash,
    load_library,
    radio_params_hash,
    save_library,
    scenario_hash,
)
from segment.stc.sac_candidates import (
    SacCandidateResult,
    produce_sac_segment_candidates,
)


ROOT = Path(__file__).resolve().parents[2]
CACHE_SCHEMA = "stc-library-cache-v1"


@lru_cache(maxsize=1)
def _source_fingerprint():
    """Hash numerical source files so algorithm edits invalidate cached data."""
    files = sorted((ROOT / "segment" / "stc").glob("*.py"))
    files += sorted((ROOT / "radio_map").glob("*.py"))
    files.append(ROOT / "common" / "config.py")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cache_entry(cache_root, kind, scenario, field, spacing_m, generator_config,
                 checkpoint_hashes=None):
    payload = {
        "schema": CACHE_SCHEMA,
        "kind": kind,
        "scenario_hash": scenario_hash(scenario),
        "world_hash": field.world_hash,
        "radio_params_hash": radio_params_hash(scenario),
        "generator_config_hash": generator_config_hash(generator_config),
        "spacing_m": float(spacing_m),
        "checkpoint_hashes": dict(sorted((checkpoint_hashes or {}).items())),
        "source_fingerprint": _source_fingerprint(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return Path(cache_root) / CACHE_SCHEMA / kind / key / "library.npz", payload


def _expected(payload):
    return {
        "world_hash": payload["world_hash"],
        "scenario_hash": payload["scenario_hash"],
        "radio_params_hash": payload["radio_params_hash"],
        "generator_config_hash": payload["generator_config_hash"],
        "checkpoint_hashes": payload["checkpoint_hashes"],
    }


def _save_atomic(library, scenario, field, path, generator_config,
                 checkpoint_hashes=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp-%d" % os.getpid()
    tmp_path = path.with_name(path.stem + suffix + path.suffix)
    tmp_meta = Path(str(tmp_path) + ".json")
    meta_path = Path(str(path) + ".json")
    save_library(
        library.candidates,
        scenario,
        field.world_hash,
        str(tmp_path),
        generator_config=generator_config,
        checkpoint_hashes=checkpoint_hashes,
    )
    os.replace(tmp_path, path)
    os.replace(tmp_meta, meta_path)


def load_or_build_base_library(scenario, field, generator_config, *,
                               spacing_m, workers=1, cache_root=None):
    """Return ``(library, cache_hit)`` for the deterministic base library."""
    if cache_root is None:
        return build_all_pairs_library(
            scenario, field, generator_config=generator_config,
            workers=workers), False
    path, payload = _cache_entry(
        cache_root, "base", scenario, field, spacing_m, generator_config)
    if path.is_file():
        try:
            return load_library(str(path), expected=_expected(payload)), True
        except (OSError, ValueError, KeyError):
            pass
    library = build_all_pairs_library(
        scenario, field, generator_config=generator_config, workers=workers)
    _save_atomic(library, scenario, field, path, generator_config)
    return library, False


def load_or_produce_sac_candidates(checkpoint_path, scenario, field, *,
                                   spacing_m, generator_config,
                                   cache_root=None):
    """Return ``(SacCandidateResult, cache_hit)`` for checkpoint-bound SAC."""
    if not checkpoint_path or cache_root is None:
        return produce_sac_segment_candidates(checkpoint_path, scenario, field), False
    path_obj = Path(checkpoint_path)
    if not path_obj.is_file():
        return produce_sac_segment_candidates(checkpoint_path, scenario, field), False
    digest = hashlib.sha256(path_obj.read_bytes()).hexdigest()
    checkpoints = {"SAC": digest}
    path, payload = _cache_entry(
        cache_root, "sac", scenario, field, spacing_m, generator_config,
        checkpoints)
    if path.is_file():
        try:
            library = load_library(str(path), expected=_expected(payload))
            return SacCandidateResult(
                "available", library.candidates, "", digest, str(path_obj)), True
        except (OSError, ValueError, KeyError):
            pass
    result = produce_sac_segment_candidates(checkpoint_path, scenario, field)
    if result.status == "available":
        _save_atomic(
            type("_LibraryView", (), {"candidates": result.candidates})(),
            scenario, field, path, generator_config, checkpoints)
    return result, False
