"""Task 15: plotting provenance (minimal)."""

import subprocess
import sys

PY = sys.executable


def test_plot_cli_requires_runs():
    r = subprocess.run([PY, "plots/plot_stc_results.py"], capture_output=True, text=True)
    assert r.returncode != 0  # --runs is required
