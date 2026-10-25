"""Task 4 Step 1/3: route validity, directed-cost reversal, library-aware ordering.

The distance and library-aware solvers are fast (no legacy radio-map sampling);
the Jia / GA-EQTSP adapters (which sample the legacy maps) are exercised in
``test_jia_sequence_adapter.py`` with the map queries monkeypatched to constants.
"""

import numpy as np
import pytest

from segment.stc.library import build_all_pairs_library
from segment.stc.sequencing import (
    SequenceResult,
    distance_nn_2opt,
    edge_proxy,
    library_aware_sequence,
)
from segment.stc.slsqp_refine import candidate_cost
from segment.stc.types import RadioSample, STCScenario, ScenarioLimits


def _limits():
    return ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                          arrive_radius_m=15.0, max_wait_s=60.0,
                          mission_deadline_s=7200.0, bandwidth_hz=1e6,
                          uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)


def _scenario(n_uav, positions_m, data_bits):
    return STCScenario(
        n_uav=n_uav, n_tasks=len(positions_m),
        task_positions_m=tuple(tuple(float(v) for v in p) for p in positions_m),
        task_data_bits=tuple(float(d) for d in data_bits),
        depot_m=(0.0, 0.0), terminal_m=(4000.0, 4000.0),
        limits=_limits(), length_m=4000.0, width_m=4000.0, n_bs=4,
        task_seed=42, map_seed=0)


class _ConstField:
    def __init__(self, n_sectors):
        self.n_sectors = n_sectors
        self.world_hash = "const-field"

    def sample(self, pos):
        return RadioSample(a2g_efficiency=np.ones(self.n_sectors),
                           g2a_outage=np.zeros(self.n_sectors))


_POSITIONS = [(200.0, 200.0), (1200.0, 200.0), (200.0, 1200.0), (1200.0, 1200.0),
              (700.0, 700.0)]
_DATA = [1e7, 2e7, 3e7, 4e7, 5e7]


# ---- Step 1: route validity ----

def test_every_assigned_point_appears_exactly_once():
    sc = _scenario(2, _POSITIONS, _DATA)
    lib = build_all_pairs_library(sc, _ConstField(sc.n_sectors),
                                  generator_config={"straight_hold": True})
    task_ids = (0, 1, 2, 3, 4)

    results = {
        "distance_nn_2opt": distance_nn_2opt(sc, task_ids, seed=42),
        "library_aware": library_aware_sequence(sc, lib, task_ids, seed=42),
    }
    for name, res in results.items():
        assert isinstance(res, SequenceResult), name
        assert res.method == name, (name, res.method)
        assert sorted(res.route) == sorted(task_ids), name
        assert len(res.route) == len(set(res.route)), name
        assert res.proxy_cost > 0, name


def test_distance_nn_2opt_is_library_free_and_deterministic():
    sc = _scenario(2, _POSITIONS, _DATA)
    task_ids = (0, 1, 2, 3, 4)
    a = distance_nn_2opt(sc, task_ids, seed=0)
    b = distance_nn_2opt(sc, task_ids, seed=0)
    assert a.route == b.route
    assert a.selected_candidate_ids == ()
    assert a.proxy_cost == pytest.approx(b.proxy_cost)
    assert a.provenance["seed"] == 0


# ---- Step 1: directed-cost reversal ----

def test_directed_edge_proxy_reverses_source_payload():
    positions = [(200.0, 200.0), (1200.0, 200.0), (200.0, 1200.0)]
    data = [1e7, 3e7, 5e7]   # distinct payloads drive the asymmetry
    sc = _scenario(2, positions, data)
    lib = build_all_pairs_library(sc, _ConstField(sc.n_sectors),
                                  generator_config={"straight_hold": True})
    dt = sc.limits.delta_t_s

    # edge 0->1 carries I_0, edge 1->0 carries I_1: different upload delay
    fwd = edge_proxy(lib, 0, 1, dt)
    rev = edge_proxy(lib, 1, 0, dt)
    assert fwd != rev
    assert fwd < rev          # I_0 < I_1 => less upload delay on 0->1

    # depot edge carries zero payload; terminal edge carries the source payload
    depot_edge = edge_proxy(lib, -1, 0, dt)
    terminal_edge = edge_proxy(lib, 0, sc.n_tasks, dt)
    assert terminal_edge > depot_edge


def test_reversing_route_reverses_directed_costs_and_source_payloads():
    positions = [(200.0, 200.0), (1200.0, 200.0), (200.0, 1200.0)]
    data = [1e7, 3e7, 5e7]
    sc = _scenario(2, positions, data)
    lib = build_all_pairs_library(sc, _ConstField(sc.n_sectors),
                                  generator_config={"straight_hold": True})
    dt = sc.limits.delta_t_s

    def route_cost(route):
        edges = ([(-1, route[0])]
                 + [(route[i], route[i + 1]) for i in range(len(route) - 1)]
                 + [(route[-1], sc.n_tasks)])
        return sum(edge_proxy(lib, s, t, dt) for s, t in edges)

    fwd = route_cost((0, 1, 2))
    rev = route_cost(tuple(reversed((0, 1, 2))))
    # directed source-payload semantics => forward != reversed
    assert fwd != rev
    assert route_cost((0, 1, 2)) != route_cost((2, 1, 0))


# ---- Step 3: library-aware ordering ----

def test_library_aware_selects_best_rankable_candidate_per_edge():
    positions = [(200.0, 200.0), (1200.0, 200.0), (200.0, 1200.0), (1200.0, 1200.0)]
    data = [1e7, 2e7, 3e7, 4e7]
    sc = _scenario(2, positions, data)
    lib = build_all_pairs_library(
        sc, _ConstField(sc.n_sectors),
        generator_config={"straight_hold": True,
                          "comm_waypoints": {"grid_shape": (5, 5), "max_waypoints": 1}})
    task_ids = (0, 1, 2, 3)
    res = library_aware_sequence(sc, lib, task_ids, seed=42)

    assert len(res.selected_candidate_ids) == len(res.route) + 1

    edges = ([(-1, res.route[0])]
             + [(res.route[i], res.route[i + 1]) for i in range(len(res.route) - 1)]
             + [(res.route[-1], sc.n_tasks)])
    dt = sc.limits.delta_t_s
    expected_cost = 0.0
    for (s, t), cid in zip(edges, res.selected_candidate_ids):
        best = min([c for c in lib.candidates[(s, t)] if c.rankable],
                   key=lambda c: candidate_cost(c, dt))
        assert cid == best.candidate_id
        expected_cost += candidate_cost(best, dt)
    assert res.proxy_cost == pytest.approx(expected_cost)


def test_library_aware_sequence_deterministic():
    positions = [(200.0, 200.0), (1200.0, 200.0), (200.0, 1200.0), (1200.0, 1200.0)]
    data = [1e7, 2e7, 3e7, 4e7]
    sc = _scenario(2, positions, data)
    lib = build_all_pairs_library(sc, _ConstField(sc.n_sectors),
                                  generator_config={"straight_hold": True})
    task_ids = (0, 1, 2, 3)
    a = library_aware_sequence(sc, lib, task_ids, seed=3)
    b = library_aware_sequence(sc, lib, task_ids, seed=3)
    assert a.route == b.route
    assert a.selected_candidate_ids == b.selected_candidate_ids
    assert a.proxy_cost == pytest.approx(b.proxy_cost)


def test_library_aware_uses_only_rankable_candidates():
    # Straight-relaxation candidates are non-rankable and must be excluded from
    # the frozen single-UAV proxy (they are relaxation lower bounds, not
    # selectable trajectories).
    positions = [(200.0, 200.0), (1200.0, 200.0), (200.0, 1200.0)]
    data = [1e7, 2e7, 3e7]
    sc = _scenario(2, positions, data)
    lib = build_all_pairs_library(sc, _ConstField(sc.n_sectors),
                                  generator_config={"straight_hold": True,
                                                    "straight_relaxation": True})
    res = library_aware_sequence(sc, lib, (0, 1, 2), seed=42)
    for cid in res.selected_candidate_ids:
        found = None
        for edge, cands in lib.candidates.items():
            for c in cands:
                if c.candidate_id == cid:
                    found = c
        assert found is not None
        assert found.rankable
