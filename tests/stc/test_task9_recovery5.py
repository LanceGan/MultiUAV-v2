"""RQ2 execution failures must survive method checkpoints and fail the run."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from test_rq_pipelines import _fast_radio, _field_for, _scenario


ASSIGNMENTS = ["region_partition", "spatial_kmeans", "shortest_balanced",
               "jia_ebtas", "library_aware_ebtas"]
SEQUENCES = ["distance_nn_2opt", "jia_time_energy_tsp", "ga_eqtsp_sequence",
             "library_aware_sequence"]


@pytest.fixture
def rq2_run(monkeypatch, tmp_path):
    from experiment import run_stc_main as main
    from segment.stc.evaluator import default_objective_spec
    from segment.stc.library import build_all_pairs_library

    sc = _scenario(1, 2, map_seed=7, task_seed=23)
    field = _field_for(sc)
    library = build_all_pairs_library(sc, field, generator_config={"straight_hold": True})
    spec = default_objective_spec(sc, library.candidates)
    sac = SimpleNamespace(status="unavailable", reason="missing", checkpoint_path="")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_build_context", lambda *a: (sc, field, library, spec, sac))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "uav_nums": [1], "map_seeds": [7], "task_seeds": [23],
        "method_seeds": [11], "methods": ["library_open_loop"],
        "iterations": 0, "generator_config": {"straight_hold": True},
    }))
    return main, config, tmp_path / "results/runs/rq2", (sc, field, library, spec)


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("stage", ["assignment", "sequence", "base_assignment"])
def test_rq2_failure_checkpoints_completed_methods_and_fails_manifest(
        monkeypatch, rq2_run, stage, failure_type):
    from segment.stc import assignment, sequencing

    main, config, run, _ = rq2_run
    failure = failure_type("RQ2 injected failure")
    before_failure = []
    if stage == "assignment":
        module, method, completed_assignment, completed_sequence = (
            assignment, "spatial_kmeans", ASSIGNMENTS[:1], [])
    elif stage == "sequence":
        module, method, completed_assignment, completed_sequence = (
            sequencing, "jia_time_energy_tsp", ASSIGNMENTS, SEQUENCES[:1])
    else:
        module, method, completed_assignment, completed_sequence = (
            assignment, "library_aware_ebtas", ASSIGNMENTS, [])
    original = getattr(module, method)
    calls = 0

    def fail_method(*args, **kwargs):
        nonlocal calls
        calls += 1
        if stage == "base_assignment" and calls == 1:
            return original(*args, **kwargs)
        path = run / "rq2.json"
        before_failure.append(json.loads(path.read_text()) if path.exists() else None)
        raise failure

    real_rq2 = main.run_rq2

    def run_with_failure(*args, **kwargs):
        # Inject only inside RQ2: RQ4 also uses library_aware_ebtas beforehand.
        monkeypatch.setattr(module, method, fail_method)
        return real_rq2(*args, **kwargs)

    monkeypatch.setattr(main, "run_rq2", run_with_failure)
    caught = None
    try:
        main.main(["--config", str(config), "--run-id", "rq2"])
    except (Exception, KeyboardInterrupt) as exc:
        caught = exc

    # Check disk and lifecycle even when the original runner swallowed the error.
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    saved = json.loads((run / "rq2.json").read_text())
    assert len(saved) == 1
    payload = saved[0]
    assert (payload["uav_num"], payload["map_seed"], payload["task_seed"]) == (1, 7, 23)
    assert before_failure and before_failure[0] is not None
    prior = before_failure[0][0]
    assert [r["method"] for r in prior["assignment"]] == completed_assignment
    assert [r["method"] for r in prior["sequence"]] == completed_sequence
    assert all(r["status"] == "evaluated" for layer in ("assignment", "sequence")
               for r in prior[layer])
    layer = "assignment" if stage == "assignment" else "sequence"
    assert len(payload[layer]) == len(prior[layer]) + 1
    for key in ("assignment", "sequence"):
        completed = payload[key][:-1] if key == layer else payload[key]
        assert completed == prior[key]
    bad = payload[layer][-1]
    assert bad["method"] == method and bad["layer"] == layer
    assert bad["status"] == "failed" and bad["availability_declared"] is False
    assert bad["failure_type"] == failure_type.__name__
    assert "RQ2 injected failure" in bad["failure_reason"]
    assert bad["interrupted"] is (failure_type is KeyboardInterrupt)
    assert (bad["seed"], bad["uav_num"], bad["map_seed"], bad["task_seed"]) == (7, 1, 7, 23)
    for metric in ("feasible", "objective", "makespan_s", "total_delay_s", "evaluator_calls"):
        assert bad[metric] is None
    if stage == "base_assignment":
        assert bad["phase"] == "base_assignment"
    assert payload["summary"] == {
        "n_assignment": len(payload["assignment"]), "n_sequence": len(payload["sequence"])}
    assert manifest["hashes"]["rq2.json"] == hashlib.sha256((run / "rq2.json").read_bytes()).hexdigest()
    assert caught is failure


def test_successful_rq2_keeps_schema_inventory_and_main_artifact(rq2_run):
    main, config, run, (sc, field, library, spec) = rq2_run
    direct = main.run_rq2(sc, field, library.candidates, spec, seed=7)
    assert set(direct) == {"assignment", "sequence", "summary"}
    assert direct["summary"] == {"n_assignment": 5, "n_sequence": 4}
    for layer, methods in (("assignment", ASSIGNMENTS), ("sequence", SEQUENCES)):
        assert [r["method"] for r in direct[layer]] == methods
        for rec in direct[layer]:
            assert set(rec) == {"method", "layer", "status", "feasible", "objective",
                                "makespan_s", "total_delay_s"}
            assert rec["status"] == "evaluated" and rec["layer"] == layer
    assert main.main(["--config", str(config), "--run-id", "rq2"]) == 0
    saved = json.loads((run / "rq2.json").read_text())
    assert saved == [dict(direct, uav_num=1, map_seed=7, task_seed=23)]
    assert json.loads((run / "manifest.json").read_text())["status"] == "success"
