"""Task 6 Step 1: EBTAS initial solution construction."""

import numpy as np

from segment.stc.assignment import library_aware_ebtas
from segment.stc.generators import generate_straight_hold
from segment.stc.operators import build_ebtas_initial_solution
from segment.stc.sequencing import library_aware_sequence
from segment.stc.types import RadioSample, STCScenario, ScenarioLimits, SegmentLibrary


class _MockField:
    def __init__(self, n_sectors):
        self.n_sectors = n_sectors

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.full(self.n_sectors, 1.0),
                           g2a_outage=np.full(self.n_sectors, 0.1))


def _setup(n_tasks=6):
    limits = ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                            arrive_radius_m=15.0, max_wait_s=60.0,
                            mission_deadline_s=7200.0, bandwidth_hz=1e6,
                            uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)
    sc = STCScenario(n_uav=2, n_tasks=n_tasks,
                     task_positions_m=tuple((800.0 + i * 600.0, 2000.0) for i in range(n_tasks)),
                     task_data_bits=tuple([1e6] * n_tasks), depot_m=(0.0, 2000.0),
                     terminal_m=(4000.0, 2000.0), limits=limits, length_m=4000.0,
                     width_m=4000.0, n_bs=4, task_seed=42, map_seed=0)
    field = _MockField(sc.n_sectors)
    candidates = {}
    for src in range(-1, sc.n_tasks + 1):
        for tgt in range(sc.n_tasks + 1):
            if src != tgt:
                candidates[(src, tgt)] = [generate_straight_hold(sc, field, src, tgt)]
    lib = SegmentLibrary(candidates=candidates, map_seed=0, task_seed=42, world_hash="h")
    return sc, lib


def test_ebtas_initial_covers_all_tasks_exactly_once():
    sc, lib = _setup(6)
    sol = build_ebtas_initial_solution(sc, lib)
    flat = sorted(t for r in sol.routes for t in r)
    assert flat == list(range(sc.n_tasks))


def test_ebtas_initial_matches_assignment_sequence_oracle():
    sc, lib = _setup(6)
    sol = build_ebtas_initial_solution(sc, lib)
    assign = library_aware_ebtas(sc, lib, seed=42)
    expected_routes = []
    expected_cids = []
    for cluster in assign.clusters:
        seq = library_aware_sequence(sc, lib, cluster, seed=42)
        expected_routes.append(seq.route)
        expected_cids.append(seq.selected_candidate_ids)
    assert tuple(sol.routes) == tuple(expected_routes)
    assert tuple(sol.candidate_ids) == tuple(expected_cids)


def test_ebtas_initial_is_deterministic():
    sc, lib = _setup(6)
    assert build_ebtas_initial_solution(sc, lib) == build_ebtas_initial_solution(sc, lib)


def test_ebtas_initial_has_valid_candidate_ids():
    sc, lib = _setup(6)
    sol = build_ebtas_initial_solution(sc, lib)
    for u, r in enumerate(sol.routes):
        edges = ([(-1, r[0])]
                 + [(r[i], r[i + 1]) for i in range(len(r) - 1)]
                 + [(r[-1], sc.n_tasks)]) if r else []
        assert len(sol.candidate_ids[u]) == len(edges)
        for (s, t), cid in zip(edges, sol.candidate_ids[u]):
            ids = {c.candidate_id for c in lib.candidates[(s, t)] if c.rankable}
            assert cid in ids
