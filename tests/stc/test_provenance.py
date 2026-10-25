"""Task 7: sealed run provenance — expected artifacts, atomic manifest,
input/config freezing, and post-finalize rejection."""

import hashlib
import json

import numpy as np
import pytest

from segment.stc.run_context import RunContext


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def test_finalize_success_requires_declared_artifacts(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    rc = RunContext.create(root, "test", {"k": "v"}, run_id="r1",
                           expected=["summary.json", "figures.png"])
    rc.write_json("summary", {"a": 1})
    with pytest.raises(RuntimeError):
        rc.finalize("success")
    rc.write_bytes("figures.png", b"\x89PNG-fake")
    manifest = rc.finalize("success")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["status"] == "success"


def test_failed_finalize_does_not_require_artifacts(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    rc = RunContext.create(root, "test", {}, run_id="r2",
                           expected=["summary.json"])
    manifest = rc.finalize("failed")
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "failed"


def test_finalized_context_rejects_writes_and_refinalize(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    rc = RunContext.create(root, "test", {}, run_id="r3")
    rc.finalize("success")
    with pytest.raises(RuntimeError):
        rc.write_json("late", {"a": 1})
    with pytest.raises(RuntimeError):
        rc.write_bytes("late.png", b"x")
    with pytest.raises(RuntimeError):
        rc.finalize("success")


def test_manifest_is_atomic_and_has_artifact_hashes(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    rc = RunContext.create(root, "test", {"k": "v"}, run_id="r4")
    rc.write_json("summary", {"a": 1})
    rc.write_bytes("fig.png", b"PNGDATA")
    manifest = rc.finalize("success")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["hashes"]["summary.json"] == _sha(
        json.dumps({"a": 1}, indent=2, ensure_ascii=False).encode("utf-8"))
    assert data["hashes"]["fig.png"] == _sha(b"PNGDATA")
    leftovers = [p.name for p in (root / "r4").iterdir() if p.name.startswith(".tmp")]
    assert leftovers == []


def test_config_contents_frozen_with_content_hash(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    rc = RunContext.create(root, "test", {"k": "v", "n": 3}, run_id="r5")
    rc.write_json("summary", {})
    manifest = rc.finalize("success")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["config"]["k"] == "v"
    assert data["config_sha256"] == _sha(
        json.dumps({"k": "v", "n": 3}, sort_keys=True).encode("utf-8"))


def test_config_file_paths_resolved_to_content_hashes(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    cfgfile = tmp_path / "tasks.json"
    cfgfile.write_text("[1, 2, 3]", encoding="utf-8")
    rc = RunContext.create(root, "test", {"tasks_path": str(cfgfile)}, run_id="r6")
    rc.write_json("summary", {})
    manifest = rc.finalize("success")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    resolved = data["config_resolved"]["tasks_path"]
    assert resolved["path"] == str(cfgfile)
    assert resolved["sha256"] == _sha(b"[1, 2, 3]")


def test_register_input_hashes_arrays_and_files(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    rc = RunContext.create(root, "test", {}, run_id="r7")
    arr = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    rc.register_input("task_positions", arr)
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"checkpoint-bytes")
    rc.register_input("checkpoint", path=ckpt)
    rc.write_json("summary", {})
    manifest = rc.finalize("success")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["inputs"]["task_positions"] == _sha(np.ascontiguousarray(arr).tobytes())
    assert data["inputs"]["checkpoint"] == _sha(b"checkpoint-bytes")


def test_manifest_records_git_provenance_keys(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    rc = RunContext.create(root, "test", {}, run_id="r8")
    rc.write_json("summary", {})
    manifest = rc.finalize("success")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert "head" in data["git"]
    assert "status" in data["git"]
    assert "diff" in data["git"]
    assert "untracked_hashes" in data["git"]
