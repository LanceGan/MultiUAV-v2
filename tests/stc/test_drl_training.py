"""SI 单段 DRL 训练入口的回归测试（segment/stc/drl_training.py）。

钉住三件事：
1. 奖励包装层的符号与终局逻辑（对称前进、到达且传完 +400、失败 −1000）；
2. held-out 边集切分确定性、不相交、无遗漏；
3. 训练产出的 SAC checkpoint 通过 ``_load_sac_policy`` 的兼容性校验
   （这是 GATSAC-style 基线能否"available"的硬条件）。
"""

import math

import numpy as np
import pytest

from segment.stc.types import RadioSample, ScenarioLimits, STCScenario
from segment.stc.drl_training import (
    R_FEASIBLE,
    R_INFEASIBLE,
    SingleSegmentRewardEnv,
    edge_env,
    split_edges,
    train,
)
from segment.stc.sac_candidates import legal_edges


def _limits(**kw):
    base = dict(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                arrive_radius_m=15.0, max_wait_s=60.0, mission_deadline_s=7200.0,
                bandwidth_hz=1e6, uplink_power_w=0.2, eps_out=0.35,
                w_t=0.5, w_d=0.5)
    base.update(kw)
    return ScenarioLimits(**base)


def _scenario(n_tasks=2, **kw):
    positions = tuple((60.0 + i * 70.0, 60.0) for i in range(n_tasks))
    base = dict(n_uav=2, n_tasks=n_tasks, task_positions_m=positions,
                task_data_bits=tuple([1e6] * n_tasks),
                depot_m=(0.0, 0.0), terminal_m=(200.0, 200.0),
                limits=_limits(), length_m=200.0, width_m=200.0, n_bs=1,
                task_seed=42, map_seed=0)
    base.update(kw)
    return STCScenario(**base)


class _ConstField:
    def __init__(self, n_sectors=3, a2g=1.0, g2a=0.0):
        self.n_sectors = n_sectors
        self.world_hash = "const-field"
        self._a2g = a2g
        self._g2a = g2a

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.full(self.n_sectors, self._a2g),
                           g2a_outage=np.full(self.n_sectors, self._g2a))


# --------------------------------------------------------------------------- #
# 奖励包装层
# --------------------------------------------------------------------------- #

def test_reward_symmetric_progress():
    # 零数据边、起点在场内（远离边界钳位）：朝目标飞一步奖励为正，
    # 背向飞一步为负（对称、无套利）
    sc = _scenario()  # task0=(60,60) -> task1=(130,60)，东西向，域内可退
    env = SingleSegmentRewardEnv(sc, _ConstField(), 0, 1, 0.0)
    env.reset()
    goal = env.env.goal
    theta_to = math.atan2(goal[1] - env.env.pos[1], goal[0] - env.env.pos[0])
    _, r_fwd, _, _ = env.step(np.array([50.0, theta_to]))
    env.reset()
    _, r_back, _, _ = env.step(np.array([50.0, theta_to + math.pi]))
    assert r_fwd > 0.0 > r_back
    # 对称：前进/后退的位移分量相消，剩两次时间惩罚（共 −2）
    assert r_fwd + r_back == pytest.approx(-2.0, abs=0.5)


def test_reward_terminal_feasible_bonus():
    # 起点一步即到、零数据 → 终局可行 +400 主导
    sc = _scenario(task_positions_m=((20.0, 0.0), (130.0, 60.0)))
    env = SingleSegmentRewardEnv(sc, _ConstField(), -1, 0, 0.0)
    env.reset()
    _, r, done, info = env.step(np.array([50.0, 0.0]))
    assert done and info["feasible"] and info["arrived"] and info["drained"]
    assert r > R_FEASIBLE - 20.0


def test_reward_terminal_infeasible_penalty():
    # 一步到达但数据排不完 → −1000 主导，feasible=False
    sc = _scenario(task_positions_m=((20.0, 0.0), (130.0, 60.0)),
                   task_data_bits=(1e12, 1e12))
    env = SingleSegmentRewardEnv(sc, _ConstField(), 0, 1, 1e12)
    env.reset()
    goal = env.env.goal
    theta = math.atan2(goal[1] - env.env.pos[1], goal[0] - env.env.pos[0])
    r, done, info = 0.0, False, {}
    while not done:
        _, r, done, info = env.step(np.array([50.0, theta]))
    assert info["arrived"] and not info["drained"] and not info["feasible"]
    assert r < R_INFEASIBLE + 200.0


def test_edge_env_payload_semantics():
    # depot 边数据为 0；i→j 边携带源点 i 的数据
    sc = _scenario(task_data_bits=(3e6, 7e6))
    assert edge_env(sc, _ConstField(), (-1, 1)).env.data_size == 0.0
    assert edge_env(sc, _ConstField(), (0, 1)).env.data_size == 3e6
    assert edge_env(sc, _ConstField(), (1, 2)).env.data_size == 7e6


# --------------------------------------------------------------------------- #
# held-out 切分
# --------------------------------------------------------------------------- #

def test_split_edges_deterministic_disjoint_complete():
    n_tasks = 4
    t1, h1 = split_edges(n_tasks, 0.2, seed=7)
    t2, h2 = split_edges(n_tasks, 0.2, seed=7)
    assert t1 == t2 and h1 == h2                      # 同 seed 可复现
    assert not (set(t1) & set(h1))                    # 不相交
    assert sorted(set(t1) | set(h1)) == sorted(legal_edges(n_tasks))  # 无遗漏
    assert len(h1) == round(0.2 * len(legal_edges(n_tasks)))
    t3, h3 = split_edges(n_tasks, 0.2, seed=8)
    assert h3 != h1                                   # 换 seed 切分不同


def test_split_edges_zero_frac():
    t, h = split_edges(3, 0.0, seed=1)
    assert h == [] and sorted(t) == sorted(legal_edges(3))


# --------------------------------------------------------------------------- #
# 训练冒烟 + checkpoint 兼容性（GATSAC 基线可用性的硬条件）
# --------------------------------------------------------------------------- #

def test_smoke_training_checkpoint_passes_loader(tmp_path):
    from segment.stc.sac_candidates import _load_sac_policy
    sc = _scenario(n_tasks=2)
    field = _ConstField()
    result = train(sc, field, algo="SAC", total_episodes=2, train_seed=0,
                   held_out_frac=0.2, eval_interval=1,
                   out_dir=str(tmp_path), run_id="smoke")
    policy, digest, reason = _load_sac_policy(result["best_path"], sc)
    assert policy is not None, reason
    assert reason == "" and len(digest) == 64
    # 曲线产物齐全
    data = np.load(result["npz_path"])
    assert len(data["ep_rewards"]) == 2
    assert len(data["ep_feasible"]) == 2
    assert data["eval_heldout_feasible"].shape[1] == 2  # (ep, rate) 列
