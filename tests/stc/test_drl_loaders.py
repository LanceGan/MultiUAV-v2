"""TD3/DDPG 加载器接线（收官序列第 4 项）。

TD3 与 DDPG 共用 segment/td3.py 的确定性 Actor（l1..l4 + action_scale/
action_bias buffer），_load_actor_policy 一个加载器校验两者；
produce_drl_segment_candidates 统一三模型入口；segment_gate 的 DRL 块
改为三模型循环。本文件钉死：缺失/legacy 单位 checkpoint 诚实 unavailable、
SI checkpoint 产出带 drl:<model>:<hash> 标签的全边候选。
"""

import math

import numpy as np
import pytest

from segment.stc.types import RadioSample, STCScenario, ScenarioLimits


def _scenario(n_tasks=2):
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    positions = tuple((60.0 + i * 70.0, 60.0) for i in range(n_tasks))
    return STCScenario(n_uav=1, n_tasks=n_tasks, task_positions_m=positions,
                       task_data_bits=tuple(1e6 for _ in range(n_tasks)),
                       depot_m=(0.0, 0.0), terminal_m=(200.0, 200.0),
                       limits=limits, length_m=200.0, width_m=200.0, n_bs=1,
                       task_seed=42, map_seed=0)


class _FlatField:
    world_hash = "flat" * 16
    n_sectors = 1

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.array([4.0]),
                           g2a_outage=np.array([0.05]))


def _save_actor_ckpt(tmp_path, model, v_max):
    max_action = np.array([float(v_max), 2.0 * math.pi], dtype=np.float32)
    min_action = np.array([0.0, 0.0], dtype=np.float32)
    if model == "td3":
        from segment.td3 import TD3
        policy = TD3(7, 2, max_action, min_action)
    else:
        from segment.ddpg import DDPG
        policy = DDPG(7, 2, max_action, min_action)
    path = tmp_path / ("%s_test.pth" % model)
    policy.save(str(path))
    return str(path)


@pytest.mark.parametrize("model", ["td3", "ddpg"])
def test_missing_checkpoint_is_unavailable(model):
    from segment.stc.sac_candidates import produce_drl_segment_candidates
    res = produce_drl_segment_candidates(model, "/no/such/file.pth",
                                         _scenario(), _FlatField())
    assert res.status == "unavailable"
    assert res.reason and res.candidates == {} and res.checkpoint_hash == ""


@pytest.mark.parametrize("model", ["td3", "ddpg"])
def test_legacy_unit_checkpoint_is_unavailable(tmp_path, model):
    # 旧 100m 单位制：v_max=0.5 → action_scale/bias 与 SI 期望相差百倍
    from segment.stc.sac_candidates import produce_drl_segment_candidates
    path = _save_actor_ckpt(tmp_path, model, v_max=0.5)
    res = produce_drl_segment_candidates(model, path, _scenario(), _FlatField())
    assert res.status == "unavailable"
    assert "not the STC SI" in res.reason


@pytest.mark.parametrize("model", ["td3", "ddpg"])
def test_si_checkpoint_produces_labeled_candidates(tmp_path, model):
    from segment.stc.sac_candidates import produce_drl_segment_candidates
    from segment.stc.sac_candidates import legal_edges
    sc = _scenario()
    path = _save_actor_ckpt(tmp_path, model, v_max=sc.limits.v_max_mps)
    res = produce_drl_segment_candidates(model, path, sc, _FlatField())
    assert res.status == "available"
    assert len(res.checkpoint_hash) == 64
    edges = set(legal_edges(sc.n_tasks))
    assert set(res.candidates) == edges
    for cands in res.candidates.values():
        for c in cands:
            assert c.generator == "drl:%s:%s" % (model, res.checkpoint_hash)


def test_unknown_model_rejected():
    from segment.stc.sac_candidates import produce_drl_segment_candidates
    with pytest.raises(ValueError):
        produce_drl_segment_candidates("ppo", None, _scenario(), _FlatField())


def test_gate_reports_td3_unavailable_reason_honestly():
    # 未配置路径时，gate 的 TD3/DDPG 记录必须带加载器给出的真实原因
    from experiment.run_stc_segment_gate import run_segment_gate
    out = run_segment_gate(_scenario(), _FlatField(),
                           generator_config={"straight_hold": True},
                           drl_checkpoints={})
    by_gen = {r["generator"]: r for r in out["records"]}
    for model in ("SAC", "TD3", "DDPG"):
        assert by_gen[model]["status"] == "unavailable"
        assert "not configured" in by_gen[model]["failure_reason"]
