"""Task 15: experiment CLI/config behaviour."""

import os
import subprocess
import sys

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(mod, *args):
    return subprocess.run([PY, "-m", "experiment.%s" % mod, *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_main_dry_run_prints_matrix():
    r = _run("run_stc_main", "--dry-run")
    assert r.returncode == 0
    assert "run matrix" in r.stdout


def test_main_supports_common_flags():
    r = _run("run_stc_main", "--dry-run", "--map-seeds", "1,2", "--task-seeds", "3",
             "--method-seeds", "4,5", "--run-id", "x")
    assert r.returncode == 0


def test_ablation_and_scale_dry_run():
    for mod in ("run_stc_ablation", "run_stc_scale"):
        r = _run(mod, "--dry-run")
        assert r.returncode == 0
