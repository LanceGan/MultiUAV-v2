import numpy as np

from trajectory.env import MultiUAVInspectionEnv
from common.config import get_scenario


def _make_env(uav_num=2):
    cfg = get_scenario(uav_num)
    return MultiUAVInspectionEnv(cfg)


def test_data_offload_decreases_buffer():
    env = _make_env()
    env.reset()
    u = 0
    env._data_buffer[u] = 100.0          # 初始缓存
    env._offload(u, env.delta_t)         # 卸载一步
    assert env._data_buffer[u] < 100.0   # 卸载后缓存减少（SINR 非 -inf 时速率 > 0）


def test_collision_penalty_applied():
    env = _make_env()
    env.reset()
    env.UAVs[0].x, env.UAVs[0].y = 10.0, 10.0
    env.UAVs[1].x, env.UAVs[1].y = 10.0, 10.05  # 小于 safe_distance
    r = env._collision_penalty(0)
    assert r < 0


def test_obs_and_action_dim():
    env = _make_env()
    env.reset()
    # 自身位置(2) + 目标相对(2) + 数据缓存(1) + 剩余时间(1) + 其他机相对位置(2*(N-1))
    assert env.local_obs_dim == 6 + 2 * (env.uav_num - 1)
    assert env.action_dim == 2


def test_reset_returns_obs_list():
    env = _make_env(3)
    obs = env.reset()
    assert isinstance(obs, list)
    assert len(obs) == env.uav_num
    for o in obs:
        assert isinstance(o, np.ndarray)
        assert o.shape == (env.local_obs_dim,)


def test_step_returns_expected_structure():
    env = _make_env()
    env.reset()
    actions = [[0.1, 0.0], [0.1, 1.0]]
    obs, rewards, dones, info = env.step(actions)
    assert isinstance(obs, list) and len(obs) == env.uav_num
    assert isinstance(rewards, list) and len(rewards) == env.uav_num
    assert isinstance(dones, list) and len(dones) == env.uav_num
    assert isinstance(info, dict) and "terminal" in info


def test_empty_traverse_terminates_immediately():
    env = _make_env()
    env.set_uav_traverse({0: [], 1: []})
    env.reset()
    obs, rewards, dones, info = env.step([[0.0, 0.0], [0.0, 0.0]])
    assert info["terminal"] is True
