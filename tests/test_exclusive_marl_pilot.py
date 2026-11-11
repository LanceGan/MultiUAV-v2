"""Checkpoint-selection tests for exclusive-environment MADRL pilots."""

import numpy as np
import torch

from experiment.run_exclusive_marl_pilot import (
    _blend_teacher_actions,
    _completion_score,
    _score,
)
from trajectory import mappo


def _metrics(**overrides):
    metrics = {
        "feasible": False,
        "max_contiguous_outage_s": [10.0, 10.0],
        "outage_max_s": 20.0,
        "out_of_bounds": 0,
        "collisions": 0,
        "early_service_arrivals": 0,
        "early_terminal_arrivals": 0,
        "completion_step": [100, 100],
        "makespan_s": 100.0,
        "total_delay_s": 20.0,
        "uploaded_tasks": 20,
        "visited_tasks": 20,
        "remaining_distance_m": 0.0,
        "return": 1.0,
    }
    metrics.update(overrides)
    return metrics


def test_feasible_checkpoint_always_wins():
    assert _score(_metrics(feasible=True)) > _score(_metrics())


def test_feasible_checkpoint_with_larger_qos_margin_wins():
    safer = _metrics(
        feasible=True,
        max_contiguous_outage_s=[10.0, 9.0],
    )
    near_limit = _metrics(
        feasible=True,
        max_contiguous_outage_s=[20.0, 19.5],
    )

    assert _score(safer) > _score(near_limit)


def test_completed_mission_is_ranked_before_idle_clean_mission():
    clean_low_progress = _metrics(
        visited_tasks=10,
        uploaded_tasks=10,
        remaining_distance_m=100.0,
    )
    violating_complete = _metrics(out_of_bounds=1)

    assert _score(violating_complete) > _score(clean_low_progress)


def test_contiguous_outage_excess_is_ranked_before_other_constraints():
    small_outage = _metrics(max_contiguous_outage_s=[21.0, 10.0])
    larger_other_violation = _metrics(
        max_contiguous_outage_s=[10.0, 10.0],
        out_of_bounds=20,
    )

    # Outage count/excess is the first post-feasibility constraint, even when
    # another checkpoint has more boundary violations.
    assert _score(larger_other_violation) > _score(small_outage)


def test_task_progress_breaks_ties_after_constraints():
    more_tasks = _metrics(visited_tasks=20, uploaded_tasks=20)
    fewer_tasks = _metrics(visited_tasks=10, uploaded_tasks=10)

    assert _score(more_tasks) > _score(fewer_tasks)


def test_completion_checkpoint_preserves_high_task_progress():
    complete_with_outage = _metrics(
        visited_tasks=20,
        uploaded_tasks=20,
        max_contiguous_outage_s=[50.0, 10.0],
    )
    low_progress_cleaner = _metrics(
        visited_tasks=9,
        uploaded_tasks=9,
        remaining_distance_m=500.0,
    )

    assert _completion_score(complete_with_outage) > _completion_score(
        low_progress_cleaner
    )


def test_completion_checkpoint_prefers_better_qos_margin():
    safer = _metrics(
        feasible=True,
        max_contiguous_outage_s=[10.0, 9.0],
    )
    near_limit = _metrics(
        feasible=True,
        max_contiguous_outage_s=[20.0, 19.5],
    )

    assert _completion_score(safer) > _completion_score(near_limit)


def test_mappo_rollout_preserves_teacher_actions_and_anchors_actor():
    mappo.device = torch.device("cpu")
    buffer = mappo._RolloutBuffer(
        n_agents=2, local_state_dim=3, action_dim=2, capacity=4
    )
    max_action = np.asarray([1.0, np.pi], dtype=np.float32)
    min_action = np.asarray([0.0, -np.pi], dtype=np.float32)

    for index in range(4):
        obs = [
            np.asarray([index / 10.0, 0.0, 0.1], dtype=np.float32),
            np.asarray([index / 10.0, 0.2, 0.3], dtype=np.float32),
        ]
        actions = [
            np.asarray([0.4, 0.0], dtype=np.float32),
            np.asarray([0.8, -0.2], dtype=np.float32),
        ]
        teacher_actions = [
            np.asarray([0.5, 0.1], dtype=np.float32),
            np.asarray([1.0, -0.1], dtype=np.float32),
        ]
        log_probs, value = 0.0, 0.0
        buffer.store(
            obs, actions, [log_probs, log_probs], 0.1, value, False, obs,
            teacher_actions=teacher_actions,
        )

    np.testing.assert_allclose(
        buffer.teacher_actions[0],
        np.asarray([[0.5, 0.1], [1.0, -0.1]], dtype=np.float32),
    )

    model = mappo.MAPPO(
        n_agents=2, local_state_dim=3, action_dim=2,
        max_action=max_action, min_action=min_action,
        batch_size=2, epochs=1, teacher_weight=1.0,
    )
    stats = model.train(buffer)

    assert np.isfinite(stats["actor_loss"])
    assert np.isfinite(stats["teacher_loss"])
    assert stats["teacher_loss"] != 0.0


def test_teacher_action_blend_keeps_action_parameterisation():
    policy = [np.asarray([10.0, 3.0])]
    teacher = [np.asarray([20.0, -3.0])]

    blended = _blend_teacher_actions(policy, teacher, 0.5)

    np.testing.assert_allclose(blended[0][:1], [15.0])
    assert -np.pi <= blended[0][1] <= np.pi
