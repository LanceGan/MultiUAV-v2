"""Tests for parameter-shared MA-D3QN components."""

import numpy as np
import torch

from trajectory import ma_d3qn


def test_discrete_action_codec_has_hover_and_relative_headings():
    codec = ma_d3qn.DiscreteActionCodec(v_max=50.0)

    assert len(codec) == 33
    np.testing.assert_allclose(codec.decode(0), [0.0, 0.0])
    np.testing.assert_allclose(codec.decode(1), [25.0, 0.0])
    assert codec.encode([24.0, 0.01]) == 1
    assert codec.encode([1.0, 1.0]) == 0


def test_nstep_assembler_emits_nstep_target_fields():
    assembler = ma_d3qn.NStepAssembler(n=3, gamma=0.5)
    records = []
    for index in range(3):
        records.extend(assembler.push(
            np.asarray([index], dtype=float),
            index,
            1.0,
            np.asarray([index + 1], dtype=float),
            False,
        ))

    assert len(records) == 1
    record = records[0]
    assert record[5] == 1.0 + 0.5 + 0.25
    assert record[7] is False
    assert record[8] == 0.125


def test_ma_d3qn_replay_mixes_demo_partition():
    ma_d3qn.device = torch.device("cpu")
    buffer = ma_d3qn.DiscreteReplayBuffer(
        3, capacity=64, demo_fraction=0.25
    )
    record = (
        np.zeros(3), 0, 1.0, np.zeros(3), False,
        1.0, np.zeros(3), False, 0.9,
    )
    for _ in range(8):
        buffer.add(record, demo=False)
        buffer.add(record, demo=True)

    sample = buffer.sample(32)
    assert int(sample[-1].sum().item()) == 8


def test_ma_d3qn_agent_train_step_is_finite():
    ma_d3qn.device = torch.device("cpu")
    agent = ma_d3qn.MAD3QNAgent(3, 4, hidden_dim=16, batch_size=8)
    buffer = ma_d3qn.DiscreteReplayBuffer(3, capacity=32, demo_fraction=0.25)
    for index in range(16):
        state = np.asarray([index / 10.0, 0.1, 0.2], dtype=float)
        record = (
            state, index % 4, 0.1, state + 0.01, False,
            0.2, state + 0.02, False, 0.9,
        )
        buffer.add(record, demo=index < 8)
    agent.bc_weight = 1.0
    result = agent.train(buffer)

    assert np.isfinite(result["loss"])
    assert result["margin_loss"] >= 0.0
