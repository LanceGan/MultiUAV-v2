"""Task 3 Step 1/4: balanced-assignment invariants and balance metrics."""

import numpy as np
import pytest

from segment.stc.assignment import jia_ebtas, region_partition, spatial_kmeans
from segment.stc.types import STCScenario, ScenarioLimits


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
        depot_m=(0.0, 0.0), terminal_m=(1000.0, 1000.0),
        limits=_limits(), length_m=1000.0, width_m=1000.0, n_bs=4,
        task_seed=42, map_seed=0)


def _uniform_scenario(n_tasks=8, n_uav=2):
    rng = np.random.RandomState(0)
    positions = [tuple(rng.uniform(0.0, 1000.0, 2)) for _ in range(n_tasks)]
    data = [float(d) for d in rng.uniform(1e7, 3e7, n_tasks)]
    return _scenario(n_uav, positions, data)


def _methods():
    return {
        "region_partition": region_partition,
        "spatial_kmeans": spatial_kmeans,
        "jia_ebtas": jia_ebtas,
    }


# ---- Step 1: assignment invariants ----


def test_each_task_assigned_exactly_once():
    sc = _uniform_scenario()
    for name, fn in _methods().items():
        res = fn(sc)
        flat = sorted(t for c in res.clusters for t in c)
        assert flat == list(range(sc.n_tasks)), name
        assert set(res.labels.tolist()) == set(range(sc.n_uav)), name
        assert -1 not in res.labels, name


def test_cluster_count_equals_uav_count():
    sc = _uniform_scenario(n_tasks=9, n_uav=3)
    for name, fn in _methods().items():
        res = fn(sc)
        assert len(res.clusters) == sc.n_uav, name


def test_deterministic_seed_behavior():
    sc = _uniform_scenario()
    for name, fn in _methods().items():
        a = fn(sc, seed=42)
        b = fn(sc, seed=42)
        assert np.array_equal(a.labels, b.labels), name


def test_maximum_cluster_size_enforced():
    sc = _uniform_scenario(n_tasks=10, n_uav=3)
    for name, fn in _methods().items():
        res = fn(sc)
        sizes = [len(c) for c in res.clusters]
        assert max(sizes) - min(sizes) <= 1, name


def test_empty_clusters_only_when_k_lt_u():
    sc = _uniform_scenario(n_tasks=6, n_uav=2)   # K >= U -> no empties
    for name, fn in _methods().items():
        res = fn(sc)
        assert all(len(c) > 0 for c in res.clusters), name

    sc2 = _uniform_scenario(n_tasks=2, n_uav=5)  # K < U -> U - K empties
    for name, fn in _methods().items():
        res = fn(sc2)
        empties = [c for c in res.clusters if len(c) == 0]
        assert len(empties) == sc2.n_uav - sc2.n_tasks, name


# ---- Step 4: balance metrics ----


def test_balance_metrics_report_payload_count_cost_and_std():
    sc = _uniform_scenario(n_tasks=8, n_uav=2)
    res = jia_ebtas(sc)
    bm = res.balance_metrics
    for key in ("payload_sum", "task_count", "proxy_route_cost",
                "payload_std", "count_std", "route_cost_std"):
        assert key in bm, key
    assert len(bm["payload_sum"]) == sc.n_uav
    assert len(bm["task_count"]) == sc.n_uav
    assert sum(bm["task_count"]) == sc.n_tasks
    assert sum(bm["payload_sum"]) == pytest.approx(sum(sc.task_data_bits))


def test_high_payload_case_more_even_than_spatial_kmeans():
    # Synthetic high-payload fixture: four heavy tasks spatially mixed among
    # five light tasks so that a purely geometric split is lopsided, while the
    # payload-aware Jia-style assignment rebalances the workload.
    positions = [(355, 547), (823, 830), (786, 959), (135, 821), (788, 577),
                 (223, 114), (236, 514), (192, 659), (174, 296)]
    data = [29.7e6, 3.7e6, 3.6e6, 2.8e6, 4.0e6, 33.8e6, 36.8e6, 81.1e6, 1.7e6]
    sc = _scenario(2, positions, data)

    jia = jia_ebtas(sc, seed=0)
    spat = spatial_kmeans(sc, seed=0)
    assert jia.balance_metrics["payload_std"] < spat.balance_metrics["payload_std"]
