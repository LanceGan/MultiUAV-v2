"""Checkpoint-selection tests for exclusive-environment MADRL pilots."""

from experiment.run_exclusive_marl_pilot import _completion_score, _score


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


def test_constraints_are_ranked_before_task_count():
    clean_low_progress = _metrics(
        visited_tasks=10,
        uploaded_tasks=10,
        remaining_distance_m=100.0,
    )
    violating_complete = _metrics(out_of_bounds=1)

    assert _score(clean_low_progress) > _score(violating_complete)


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
