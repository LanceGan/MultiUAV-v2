"""Run-scoped provenance and sealed atomic artifact writes (Task 7).

Every formal run writes into ``results/runs/<run_id>/`` and records a sealed
manifest with git state (including the dirty diff and untracked-source content
hashes), command, times, dependency versions, resolved config (contents, not
just paths), registered input hashes (map/task arrays, library/checkpoint
hashes), and SHA-256 hashes of every written artifact.

A run declares its *expected artifacts* up front.  ``finalize("success")``
raises if any declared expected artifact is missing; a finalized context rejects
further writes and re-finalization; and the manifest write is atomic.  A
repeated explicit ``run_id`` is rejected so no run ever overwrites another.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _hash_file(path):
    return _hash_bytes(Path(path).read_bytes())


def _canonical_sha256(obj):
    return _hash_bytes(json.dumps(obj, sort_keys=True).encode("utf-8"))


def _to_bytes(obj):
    if isinstance(obj, bytes):
        return obj
    if isinstance(obj, str):
        return obj.encode("utf-8")
    if isinstance(obj, (dict, list, tuple)):
        return json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return np.ascontiguousarray(obj).tobytes()
        return np.ascontiguousarray(np.asarray(obj)).tobytes()
    except Exception:
        return repr(obj).encode("utf-8")


def _resolve_value(v):
    if isinstance(v, dict):
        return {k: _resolve_value(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_resolve_value(x) for x in v]
    if isinstance(v, str):
        p = Path(v)
        if p.is_file():
            try:
                return {"path": v, "sha256": _hash_file(p)}
            except Exception:
                return v
        return v
    return v


def _resolve_config(config):
    """Resolve file paths inside the config to content hashes (freeze contents)."""
    return _resolve_value(dict(config))


def _run_git(args):
    try:
        out = subprocess.run(["git"] + args, capture_output=True).stdout
        return out.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _collect_source_hashes(path, cap=1000):
    """SHA-256 of a file or, recursively, of every file under a directory."""
    out = {}
    path = Path(path)
    if path.is_file():
        try:
            out[str(path)] = _hash_file(path)
        except Exception:
            out[str(path)] = "unreadable"
        return out
    if path.is_dir():
        for child in path.rglob("*"):
            if len(out) >= cap:
                break
            if child.is_file():
                try:
                    out[str(child)] = _hash_file(child)
                except Exception:
                    out[str(child)] = "unreadable"
    return out


def _git_provenance():
    info = {"head": "unknown", "status": "unknown", "diff": "",
            "diff_staged": "", "untracked_hashes": {}}
    head = _run_git(["rev-parse", "HEAD"]).strip()
    if head:
        info["head"] = head
    status = _run_git(["status", "--porcelain"]).strip()
    if status:
        info["status"] = status
    diff = _run_git(["diff"])
    if diff:
        info["diff"] = diff
    diff_staged = _run_git(["diff", "--staged"])
    if diff_staged:
        info["diff_staged"] = diff_staged
    untracked = {}
    try:
        for line in info["status"].splitlines():
            if line.startswith("??"):
                p = Path(line[3:].strip())
                for fp, h in _collect_source_hashes(p).items():
                    untracked[fp] = h
                    if len(untracked) >= 200:
                        break
            if len(untracked) >= 200:
                break
    except Exception:
        pass
    info["untracked_hashes"] = untracked
    return info


def _versions():
    import importlib.metadata as md
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy", "torch", "pytest"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = "unknown"
    return out


class RunContext:
    """A single, non-overwritable, sealable experiment run directory."""

    def __init__(self, run_dir, experiment, config, run_id, start_time, expected):
        self.run_dir = Path(run_dir)
        self.experiment = experiment
        self.config = dict(config)
        self.run_id = run_id
        self.start_time = start_time
        self._hashes = {}
        self._input_hashes = {}
        self._expected = set(expected or [])
        self._sealed = False
        # Freeze the execution environment at creation time so the manifest
        # records the ACTUAL start state, not whatever changed by the end.
        self._frozen_config = dict(config)
        self._frozen_config_sha = _canonical_sha256(dict(config))
        self._frozen_config_resolved = _resolve_config(dict(config))
        self._start_git = _git_provenance()
        self._start_versions = _versions()

    @classmethod
    def create(cls, root, experiment, config, run_id=None, expected=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_dir = root / run_id
        run_dir.mkdir(exist_ok=False)  # raises FileExistsError on a repeated id
        return cls(run_dir, experiment, config, run_id,
                   datetime.now().isoformat(timespec="seconds"), expected)

    def _check_open(self):
        if self._sealed:
            raise RuntimeError("run %r is sealed; no further writes" % self.run_id)

    def write_json(self, name, payload):
        """Atomically write ``<name>.json`` (tmp file + rename), record hash."""
        self._check_open()
        if not name.endswith(".json"):
            name = name + ".json"
        path = self.run_dir / name
        tmp = self.run_dir / (".tmp-" + name)
        tmp.write_bytes(json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"))
        tmp.replace(path)
        self._hashes[name] = _hash_file(path)
        return path

    def write_bytes(self, name, data):
        """Atomically write a raw artifact (figures, traces, ...), record hash."""
        self._check_open()
        path = self.run_dir / name
        tmp = self.run_dir / (".tmp-" + name)
        tmp.write_bytes(data)
        tmp.replace(path)
        self._hashes[name] = _hash_file(path)
        return path

    def register_input(self, name, data=None, path=None):
        """Hash an input (array/bytes/file) without writing an artifact.

        Used to freeze map/task arrays, candidate libraries, and checkpoint
        hashes into the manifest's ``inputs`` section.
        """
        self._check_open()
        if path is not None:
            self._input_hashes[name] = _hash_file(Path(path))
        else:
            self._input_hashes[name] = _hash_bytes(_to_bytes(data))
        return self._input_hashes[name]

    def finalize(self, status):
        """Write the sealed manifest and return its path.

        ``status="success"`` requires every declared expected artifact to be
        present; otherwise a ``RuntimeError`` is raised.  The manifest is
        written atomically and the context is sealed afterwards (further writes
        and re-finalization are rejected).
        """
        if self._sealed:
            raise RuntimeError("run %r is already finalized" % self.run_id)
        if status == "success":
            missing = [n for n in sorted(self._expected) if n not in self._hashes]
            if missing:
                raise RuntimeError(
                    "cannot finalize success: missing expected artifacts %r"
                    % missing)
            # Re-verify each expected artifact is still on disk and unchanged.
            for name in sorted(self._expected):
                path = self.run_dir / name
                if not path.is_file():
                    raise RuntimeError(
                        "expected artifact %r is missing on disk at seal time" % name)
                if _hash_file(path) != self._hashes[name]:
                    raise RuntimeError(
                        "expected artifact %r hash mismatch at seal time" % name)

        manifest = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "status": status,
            "git": self._start_git,
            "git_end": _git_provenance(),
            "command": " ".join(sys.argv),
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(timespec="seconds"),
            "versions": self._start_versions,
            "config": self._frozen_config,
            "config_sha256": self._frozen_config_sha,
            "config_resolved": self._frozen_config_resolved,
            "inputs": dict(self._input_hashes),
            "hashes": dict(self._hashes),
            "expected": sorted(self._expected),
        }
        path = self.run_dir / "manifest.json"
        tmp = self.run_dir / ".tmp-manifest.json"
        tmp.write_bytes(json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
        tmp.replace(path)
        self._sealed = True
        return path
