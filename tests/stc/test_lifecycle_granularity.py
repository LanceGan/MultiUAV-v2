"""D1/D2 修复（pilot 发现的 run_lifecycle 粒度缺陷）。

pilot 的 scale 步在单场景矩阵里跑了 12.3 小时：--timeout-s 7200 从未触发
（D1：超时只在场景边界检查），且 records.json 零字节落盘（D2：scale 走
非增量路径，记录只在场景边界写出）。本文件钉死两项修复：

- D1：记录级检查点 ``_checkpoint`` 在**持久化该记录之后**检查超时——超预算
  立即中止并写 failed manifest，已完成记录一条不丢；
- D2：scale 的 run_scale / run_density_scale 支持逐记录 ``emit`` 回调，
  main 以 incremental_records=True 接入 run_lifecycle。
"""

import json
import time

import pytest


# --------------------------------------------------------------------------- #
# D1：记录级超时检查
# --------------------------------------------------------------------------- #

def test_record_checkpoint_enforces_timeout(tmp_path, monkeypatch):
    # 单场景矩阵 + 逐记录 checkpoint：第二条记录落盘后时钟已超预算，
    # 必须在该 checkpoint 处中止（而不是等整个场景跑完），failed manifest
    # 保全两条已写记录。
    from experiment.stc_common import run_lifecycle

    monkeypatch.chdir(tmp_path)
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "perf_counter", lambda: clock["t"])

    emitted = []

    def per_scenario(uav, mseed, tseed, checkpoint):
        emitted.append(1)
        checkpoint({"record": 1})
        clock["t"] = 100.0  # 模拟第二条记录耗时远超预算
        emitted.append(2)
        checkpoint({"record": 2})  # 持久化后在此中止
        emitted.append(3)  # 不应执行到
        checkpoint({"record": 3})

    with pytest.raises(TimeoutError):
        run_lifecycle("stc_test", {"timeout_s": 10}, "d1_record_timeout",
                      [(2, 42, 42)], per_scenario, lambda rs: {},
                      incremental_records=True)

    assert emitted == [1, 2], "超时必须在第二条记录的 checkpoint 处中止"
    run_dir = tmp_path / "results/runs/d1_record_timeout"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    records = json.loads((run_dir / "records.json").read_text())
    assert records == [{"record": 1}, {"record": 2}], \
        "中止前已完成的记录必须全部保全"


def test_record_checkpoint_without_timeout_unaffected(tmp_path, monkeypatch):
    from experiment.stc_common import run_lifecycle

    monkeypatch.chdir(tmp_path)

    def per_scenario(uav, mseed, tseed, checkpoint):
        checkpoint({"record": 1})
        checkpoint({"record": 2})

    rc = run_lifecycle("stc_test", {}, "d1_no_timeout", [(2, 42, 42)],
                       per_scenario, lambda rs: {"n": len(rs)},
                       incremental_records=True)
    assert rc == 0
    manifest = json.loads(
        (tmp_path / "results/runs/d1_no_timeout/manifest.json").read_text())
    assert manifest["status"] == "success"


# --------------------------------------------------------------------------- #
# D2：scale 逐记录 emit
# --------------------------------------------------------------------------- #

def _fake_eval_methods(sc, field, gen_cfg, seed, iterations):
    return [{"method": "library_open_loop", "status": "evaluated"},
            {"method": "proposed_stc_alns", "status": "evaluated"}]


def test_run_scale_emits_each_record(monkeypatch):
    import experiment.run_stc_scale as scale

    monkeypatch.setattr(scale, "_eval_methods", _fake_eval_methods)
    sc = scale._synthetic_scenario(2, 3, 4000.0, 4, 42, 42, _limits())
    seen = []
    out = scale.run_scale(sc, lambda s: object(), seed=0,
                          generator_config={"straight_hold": True},
                          iterations=1, emit=seen.append)
    assert len(seen) == len(out["records"]) > 0
    assert all(r.get("family") and r.get("variant") for r in seen), \
        "emit 必须发生在 family/variant 标注之后"


def test_run_density_scale_emits_each_record(monkeypatch):
    import experiment.run_stc_scale as scale

    monkeypatch.setattr(scale, "_eval_methods", _fake_eval_methods)
    sc = scale._synthetic_scenario(2, 3, 4000.0, 4, 42, 42, _limits())
    seen = []
    out = scale.run_density_scale(sc, lambda s: object(), seed=0,
                                  generator_config={"straight_hold": True},
                                  iterations=1, emit=seen.append)
    assert len(seen) == len(out["records"]) > 0
    assert all(r.get("family") == "density" for r in seen)


def _limits():
    from segment.stc.types import ScenarioLimits
    return ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                          arrive_radius_m=15.0, max_wait_s=60.0,
                          mission_deadline_s=7200.0, bandwidth_hz=1e6,
                          uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
