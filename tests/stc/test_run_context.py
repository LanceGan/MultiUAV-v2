"""Task 1: run-scoped provenance and atomic artifacts."""

import json

import pytest

from segment.stc.run_context import RunContext


def test_run_context_never_overwrites_existing_run(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    RunContext.create(root, "test", {"a": 1}, run_id="run-1")
    with pytest.raises(FileExistsError):
        RunContext.create(root, "test", {"a": 1}, run_id="run-1")


def test_failed_run_has_non_success_status(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    rc = RunContext.create(root, "test", {}, run_id="run-2")
    manifest = rc.finalize("failed")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["status"] == "failed"


def test_finalize_records_git_config_versions_and_hashes(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    rc = RunContext.create(root, "test", {"k": "v"}, run_id="run-3")
    rc.write_json("summary", {"a": 1})
    manifest = rc.finalize("success")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["status"] == "success"
    assert "head" in data["git"] and "status" in data["git"]
    assert data["config"]["k"] == "v"
    assert "python" in data["versions"] and "numpy" in data["versions"]
    assert "summary.json" in data["hashes"]
    assert "start_time" in data and "end_time" in data
    assert "command" in data
