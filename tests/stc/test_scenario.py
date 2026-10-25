"""Task 2: SI scenario conversion and seed separation."""

from segment.stc.scenario import load_stc_scenario


def test_legacy_coordinates_convert_from_100m_units_to_metres():
    sc = load_stc_scenario(2, task_seed=42, map_seed=0)
    for (x, y) in sc.task_positions_m:
        assert 0.0 <= x <= 4000.0 and 0.0 <= y <= 4000.0  # metres, in 4 km region


def test_data_mbit_converts_to_bits():
    sc = load_stc_scenario(2, task_seed=42, map_seed=0)
    for d in sc.task_data_bits:
        assert 1e7 <= d <= 3e7  # 10-30 Mbit -> 1e7-3e7 bits


def test_scenario_deterministic_for_fixed_seeds():
    sc1 = load_stc_scenario(2, task_seed=42, map_seed=0)
    sc2 = load_stc_scenario(2, task_seed=42, map_seed=0)
    assert sc1 == sc2


def test_task_seed_changes_tasks_not_map_seed():
    sc1 = load_stc_scenario(2, task_seed=42, map_seed=0)
    sc2 = load_stc_scenario(2, task_seed=43, map_seed=0)
    assert sc1.task_data_bits != sc2.task_data_bits  # tasks changed
    assert sc1.map_seed == sc2.map_seed == 0          # map seed unchanged
