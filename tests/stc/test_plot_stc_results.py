"""Task 10: formal plotting (plots/plot_stc_results.py).

The plotting module reads ONLY finalized (sealed, status ``success``) runs and
derives every plotted value from the run artifacts — there are no hard-coded
result values anywhere.
"""

import json
import subprocess
import sys

PY = sys.executable


def _record(method, status="evaluated", feasible=True, makespan=7.0, delay=2.0,
            proposed=False, t_ref=8.0, d_ref=2.0):
    rec = {"method": method, "label": method, "proposed": proposed,
           "status": status, "feasible": feasible, "makespan_s": makespan,
           "total_delay_s": delay,
           "reference": {"reference": {"t_ref_s": t_ref, "d_ref_s": d_ref},
                         "w_t": 0.5, "w_d": 0.5}}
    if status == "unavailable":
        rec.update({"feasible": None, "makespan_s": None, "total_delay_s": None,
                    "objective": None})
    return rec


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_plot_cli_requires_runs():
    r = subprocess.run([PY, "plots/plot_stc_results.py"], capture_output=True, text=True)
    assert r.returncode != 0  # --runs / --runs-dir is required


# --------------------------------------------------------------------------- #
# Pure extraction
# --------------------------------------------------------------------------- #

def test_extract_td_points_normalizes_and_skips():
    from plots import plot_stc_results as plots
    records = [
        _record("proposed_stc_alns", proposed=True, makespan=7.0, delay=2.0),
        _record("jia_ebtas_style", makespan=4.0, delay=1.0),
        _record("gatsac_style", status="unavailable"),
        _record("infeasible", feasible=False, makespan=100.0, delay=50.0),
        _record("no_ref", makespan=7.0, delay=2.0),
    ]
    records[-1]["reference"] = None
    points = plots.extract_td_points(records)
    by_method = {p["method"]: p for p in points}
    assert set(by_method) == {"proposed_stc_alns", "jia_ebtas_style"}
    # normalization is derived from the record, not hard-coded
    assert by_method["proposed_stc_alns"]["t_norm"] == 7.0 / 8.0
    assert by_method["proposed_stc_alns"]["d_norm"] == 2.0 / 2.0
    assert by_method["jia_ebtas_style"]["t_norm"] == 4.0 / 8.0


def test_extract_feasibility_success_rate():
    from plots import plot_stc_results as plots
    records = [
        _record("a", feasible=True),
        _record("a", feasible=True),
        _record("a", feasible=False),
        _record("b", feasible=True),
        _record("b", status="unavailable"),
    ]
    out = plots.extract_feasibility(records)
    assert out["a"] == {"evaluated": 3, "feasible": 2, "success_rate": 2 / 3}
    assert out["b"] == {"evaluated": 1, "feasible": 1, "success_rate": 1.0}


def test_extract_ablation_skips_unavailable():
    from plots import plot_stc_results as plots
    records = [
        {"ablation": "full", "status": "evaluated", "objective": 0.80},
        {"ablation": "ebtas_init", "status": "evaluated", "objective": 0.90},
        {"ablation": "path_switch", "status": "unavailable", "objective": None},
    ]
    out = plots.extract_ablation(records)
    assert out == [{"ablation": "full", "objective": 0.80},
                   {"ablation": "ebtas_init", "objective": 0.90}]


def test_extract_scale_skips_unavailable():
    from plots import plot_stc_results as plots
    records = [
        {"family": "uav", "variant": "U=2", "status": "evaluated", "objective": 0.5},
        {"family": "uav", "variant": "U=3", "status": "evaluated", "objective": 0.6},
        {"family": "uav", "variant": "U=4", "status": "unavailable", "objective": None},
    ]
    out = plots.extract_scale(records)
    assert out == [{"variant": "U=2", "objective": 0.5},
                   {"variant": "U=3", "objective": 0.6}]


# --------------------------------------------------------------------------- #
# Finalized-run loading
# --------------------------------------------------------------------------- #

def test_load_valid_runs_skips_invalid(tmp_path):
    from plots import plot_stc_results as plots
    import hashlib

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    good = tmp_path / "good"
    good.mkdir()
    data = json.dumps([_record("proposed_stc_alns", proposed=True)]).encode()
    (good / "raw_records.json").write_bytes(data)
    (good / "manifest.json").write_text(json.dumps({
        "run_id": "good", "experiment": "stc_main", "status": "success",
        "git": {"head": "abc"}, "inputs": {"x": "y"},
        "expected": ["raw_records.json"],
        "hashes": {"raw_records.json": sha(data)}}))

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "manifest.json").write_text(json.dumps({
        "run_id": "bad", "experiment": "stc_main", "status": "failed",
        "git": {"head": "abc"}, "inputs": {},
        "expected": [], "hashes": {}}))

    valid = plots.load_valid_runs([good, bad])
    assert [r.run_id for r in valid] == ["good"]


# --------------------------------------------------------------------------- #
# Rendering (headless)
# --------------------------------------------------------------------------- #

def test_render_writes_figures(tmp_path):
    from plots import plot_stc_results as plots
    import hashlib

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    run = tmp_path / "rq4"
    run.mkdir()
    data = json.dumps([_record("proposed_stc_alns", proposed=True),
                       _record("jia_ebtas_style"),
                       _record("library_open_loop")]).encode()
    (run / "raw_records.json").write_bytes(data)
    (run / "manifest.json").write_text(json.dumps({
        "run_id": "rq4", "experiment": "stc_main", "status": "success",
        "git": {"head": "abc"}, "inputs": {"x": "y"},
        "expected": ["raw_records.json"],
        "hashes": {"raw_records.json": sha(data)}}))
    out_dir = tmp_path / "figs"
    written = plots.render_figures({"stc_main": [plots.load_run(run)]},
                                   out_dir=out_dir)
    assert written
    for path in written:
        assert path.exists()
