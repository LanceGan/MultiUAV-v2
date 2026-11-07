"""Exact-benchmark reporting state and aggregation tests."""

from plots.plot_stc_exact import _exact_state, summarise


def test_exact_state_distinguishes_timeout_from_complete_infeasibility():
    assert _exact_state({"timed_out": True}) == "timeout"
    assert _exact_state({
        "timed_out": False,
        "proven_optimal": True,
        "feasible": True,
    }) == "optimal"
    assert _exact_state({
        "timed_out": False,
        "proven_optimal": False,
        "discrete_search_complete": True,
        "feasible": False,
    }) == "no feasible"


def test_summarise_keeps_exact_and_proposed_records_paired_by_k():
    run = {
        "run_id": "exact-k4",
        "by_task": {
            4: {
                "exact_discrete_enumeration": {
                    "feasible": False,
                    "objective": None,
                    "runtime_s": 12.0,
                    "proven_optimal": False,
                    "discrete_search_complete": True,
                    "timed_out": False,
                    "explored": 120,
                    "gap": None,
                },
                "proposed_stc_alns": {
                    "feasible": False,
                    "objective": None,
                    "makespan_s": 150.0,
                    "total_delay_s": 80.0,
                    "runtime_s": 240.0,
                    "iterations": 1000,
                    "iterations_executed": 1000,
                },
            }
        },
    }

    stats = summarise([run])

    assert stats["task_counts"] == [4]
    assert stats["rows"][0]["exact_state"] == "no feasible"
    assert stats["rows"][0]["proposed_runtime_s"] == 240.0
