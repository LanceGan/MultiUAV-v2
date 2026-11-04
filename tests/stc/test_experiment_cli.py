"""Task 15: experiment CLI/config behaviour."""

import os
import subprocess
import sys
from pathlib import Path

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
             "--method-seeds", "4,5", "--run-id", "x",
             "--sac-checkpoint", "results/models/stc_drl/sac_u2_si-v1_best.pth")
    assert r.returncode == 0


def test_main_sac_checkpoint_cli_resolves_from_working_directory():
    from experiment.run_stc_main import _common_args, _resolve_config
    import argparse

    parser = argparse.ArgumentParser()
    _common_args(parser)
    args = parser.parse_args([
        "--sac-checkpoint", "results/models/stc_drl/sac_u2_si-v1_best.pth"])
    cfg = _resolve_config(args)
    assert Path(cfg["sac_checkpoint"]) == (
        Path(ROOT) / "results/models/stc_drl/sac_u2_si-v1_best.pth").resolve()


def test_ablation_and_scale_dry_run():
    for mod in ("run_stc_ablation", "run_stc_scale"):
        r = _run(mod, "--dry-run")
        assert r.returncode == 0
