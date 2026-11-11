"""Tests for parameter-shared TD3 behavior cloning."""

import numpy as np
import torch

from trajectory import td3
from trajectory.td3 import SharedTD3
from trajectory.td3_bc import (
    DemoReplayBuffer,
    SharedTD3BC,
    behavior_cloning_loss,
    linear_decay,
)


def test_linear_decay_reaches_persistent_floor():
    assert linear_decay(1.0, 0, 600, 0.25) == 1.0
    assert np.isclose(linear_decay(1.0, 600, 600, 0.25), 0.25)
    assert np.isclose(linear_decay(1.0, 1000, 600, 0.25), 0.25)


def test_behavior_cloning_loss_masks_online_samples():
    predicted = torch.tensor([[0.0, 1.0], [1.0, 2.0]])
    target = torch.tensor([[0.5, 1.5], [0.0, 0.0]])
    mask = torch.tensor([[1.0], [0.0]])

    loss = behavior_cloning_loss(predicted, target, mask, max_speed=5.0)
    expected = (1.0 - np.cos(-0.5)) + ((1.0 - 1.5) / 5.0) ** 2
    assert np.isclose(float(loss), expected)


def test_behavior_cloning_loss_uses_speed_then_heading_order():
    predicted = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[3.0, 0.5]])
    mask = torch.ones((1, 1))

    loss = behavior_cloning_loss(predicted, target, mask, max_speed=10.0)
    expected = ((1.0 - 3.0) / 10.0) ** 2 + (1.0 - np.cos(-0.5))
    assert np.isclose(float(loss), expected)


def test_demo_replay_keeps_fixed_demo_fraction():
    buffer = DemoReplayBuffer(3, 2, max_size=64, demo_fraction=0.25)
    for index in range(8):
        buffer.add(
            np.full(3, index, dtype=float),
            np.zeros(2),
            -1.0,
            np.full(3, index + 1, dtype=float),
            False,
            demo=index < 4,
        )
    online_count = buffer.size
    for index in range(online_count, 8):
        buffer.add(
            np.full(3, index, dtype=float),
            np.zeros(2),
            -1.0,
            np.full(3, index + 1, dtype=float),
            False,
            demo=True,
        )

    _, _, _, _, _, _, demo_mask = buffer.sample(32)
    assert int(demo_mask.sum().item()) == 8


def test_shared_td3_bc_train_uses_demo_replay():
    td3.device = torch.device("cpu")
    buffer = DemoReplayBuffer(3, 2, max_size=64, demo_fraction=0.25)
    for index in range(16):
        state = np.asarray([index / 10.0, 0.1, 0.2], dtype=float)
        action = np.asarray([0.5, 0.0], dtype=float)
        buffer.add(
            state, action, -0.1, state + 0.01, False,
            demo=index < 8,
        )

    model = SharedTD3BC(
        3, 2,
        np.asarray([1.0, np.pi], dtype=np.float32),
        np.asarray([0.0, -np.pi], dtype=np.float32),
        batch_size=16,
        teacher_anchor_weight=1.0,
    )
    model.bc_weight = 1.0
    model.train(buffer)
    result = model.train(buffer)

    assert np.isfinite(result["critic_loss"])
    assert np.isfinite(result["bc_loss"])
    assert result["bc_loss"] >= 0.0
    assert np.isfinite(result["teacher_loss"])
    assert result["teacher_loss"] >= 0.0


def test_demo_replay_preserves_teacher_action_for_online_samples():
    buffer = DemoReplayBuffer(2, 2, max_size=16, demo_fraction=0.0)
    state = np.asarray([0.1, 0.2], dtype=float)
    executed = np.asarray([1.0, 0.4], dtype=float)
    teacher = np.asarray([0.2, -0.1], dtype=float)
    buffer.add(state, executed, 0.0, state, False, teacher_action=teacher)

    _, action, teacher_action, _, _, _, demo_mask = buffer.sample(1)
    np.testing.assert_allclose(action.numpy()[0], executed)
    np.testing.assert_allclose(teacher_action.numpy()[0], teacher)
    assert int(demo_mask.sum().item()) == 0
