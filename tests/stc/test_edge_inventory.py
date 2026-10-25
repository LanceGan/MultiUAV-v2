"""Task 2 Step 1: all-pairs directed edge inventory (exactly the legal edges)."""

import numpy as np

from segment.stc.library import build_all_pairs_library
from segment.stc.types import RadioSample, STCScenario, ScenarioLimits, SegmentLibrary


def _scenario(n_tasks=3):
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    positions = tuple((800.0 + i * 1000.0, 1500.0) for i in range(n_tasks))
    data_bits = tuple(float(2e7 + i * 1e6) for i in range(n_tasks))
    return STCScenario(n_uav=2, n_tasks=n_tasks, task_positions_m=positions,
                       task_data_bits=data_bits, depot_m=(0.0, 500.0),
                       terminal_m=(4000.0, 3000.0), limits=limits, length_m=4000.0,
                       width_m=4000.0, n_bs=4, task_seed=42, map_seed=0)


class _Field:
    def __init__(self, n_sectors=4):
        self.n_sectors = n_sectors
        self.world_hash = "inventory-hash"

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.ones(self.n_sectors),
                           g2a_outage=np.zeros(self.n_sectors))


def _build(n_tasks=3):
    sc = _scenario(n_tasks)
    field = _Field()
    lib = build_all_pairs_library(sc, field, generator_config={"straight_hold": True})
    return sc, lib


def test_edge_count_is_exactly_k_times_k_plus_1():
    sc, lib = _build(3)
    k = sc.n_tasks
    assert len(lib.candidates) == k * (k + 1)


def test_no_self_edges():
    sc, lib = _build(3)
    for (s, t) in lib.candidates:
        assert s != t


def test_no_depot_to_terminal_edge():
    sc, lib = _build(3)
    assert (-1, sc.n_tasks) not in lib.candidates


def test_terminal_is_never_a_source():
    sc, lib = _build(3)
    for (s, t) in lib.candidates:
        assert s != sc.n_tasks


def test_edge_set_is_exactly_the_legal_directed_edges():
    sc, lib = _build(3)
    k = sc.n_tasks
    expected = set()
    for i in range(k):
        expected.add((-1, i))   # depot -> task
        expected.add((i, k))    # task -> terminal
    for i in range(k):
        for j in range(k):
            if i != j:
                expected.add((i, j))  # task -> task
    assert set(lib.candidates.keys()) == expected


def test_library_is_a_segment_library_with_world_and_seeds():
    sc, lib = _build(3)
    assert isinstance(lib, SegmentLibrary)
    assert lib.world_hash == "inventory-hash"
    assert lib.map_seed == sc.map_seed
    assert lib.task_seed == sc.task_seed
