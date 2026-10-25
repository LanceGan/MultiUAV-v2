"""Run-scoped provenance and atomic artifact writes (Task 1).

Every formal run writes into ``results/runs/<run_id>/`` and records a manifest
with git state, command, times, dependency versions, resolved config, and
SHA-256 hashes of every written artifact.  A repeated explicit ``run_id`` is
rejected so no run ever overwrites another.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _git_info():
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        head = "unknown"
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        status = "unknown"
    return head, status


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
    """A single, non-overwritable experiment run directory."""

    def __init__(self, run_dir, experiment, config, run_id, start_time):
        self.run_dir = Path(run_dir)
        self.experiment = experiment
        self.config = dict(config)
        self.run_id = run_id
        self.start_time = start_time
        self._hashes = {}

    @classmethod
    def create(cls, root, experiment, config, run_id=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_dir = root / run_id
        run_dir.mkdir(exist_ok=False)  # raises FileExistsError on a repeated id
        return cls(run_dir, experiment, config, run_id,
                   datetime.now().isoformat(timespec="seconds"))

    def write_json(self, name, payload):
        """Atomically write ``<name>.json`` (tmp file + rename), record hash."""
        if not name.endswith(".json"):
            name = name + ".json"
        path = self.run_dir / name
        tmp = self.run_dir / (".tmp-" + name)
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(path)
        self._hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def finalize(self, status):
        """Write the final manifest and return its path."""
        head, git_status = _git_info()
        manifest = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "status": status,
            "git": {"head": head, "status": git_status},
            "command": " ".join(sys.argv),
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(timespec="seconds"),
            "versions": _versions(),
            "config": self.config,
            "hashes": dict(self._hashes),
        }
        path = self.run_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path
