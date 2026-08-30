"""Smoke tests for the per-agent independent TD3 architecture (IL MARL).

The trajectory layer was changed from a single parameter-shared ``SharedTD3``
(one actor + one twin critic training on all UAVs' pooled transitions) to one
``SharedTD3`` + one replay buffer *per UAV*. These tests pin that each UAV owns
independent weights (distinct deterministic actions on the same observation)
and trains on its own replay only.
"""

import numpy as np

from common.config import get_scenario
from trajectory.env import MultiUAVInspectionEnv
from trajectory.replay import ReplayBuffer
from trajectory.td3 import SharedTD3


def _make_agents(uav_num=2):
    cfg = get_scenario(uav_num)
    env = MultiUAVInspectionEnv(cfg, safe_distance=0.1)
    max_action = np.array([float(cfg["v_max"]), 2.0 * np.pi], dtype=np.float32)
    min_action = np.array([0.0, 0.0], dtype=np.float32)
    agents = [SharedTD3(env.local_obs_dim, env.action_dim, max_action, min_action)
              for _ in range(uav_num)]
    return env, agents


def test_two_agents_produce_distinct_actions():
    env, agents = _make_agents(2)
    obs = env.reset()

    # Same observation -> different deterministic action, because each agent has
    # its own independently-initialised actor weights.
    a0 = agents[0].select_action(obs[0])
    a1 = agents[1].select_action(obs[0])
    assert a0.shape == (2,)
    assert a1.shape == (2,)
    assert not np.allclose(a0, a1)


def test_per_agent_replays_train_independently():
    env, agents = _make_agents(2)
    obs = env.reset()
    batch_size = agents[0].batch_size
    replays = [ReplayBuffer(env.local_obs_dim, env.action_dim, max_size=1000)
               for _ in range(2)]

    for i in range(2):
        # Fill only agent i's buffer, then take one TD3 gradient step on it.
        for _ in range(batch_size):
            a = agents[i].select_action(obs[i])
            replays[i].add(obs[i], a, 0.0, obs[i], 0.0)
        loss = agents[i].train(replays[i])
        assert loss["critic_loss"] is not None
        # Each agent's buffer holds exactly its own batch_size transitions —
        # no pooling across UAVs.
        assert len(replays[i]) == batch_size
