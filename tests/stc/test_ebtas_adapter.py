"""Task 3 Step 2/3: Jia-style adapter fidelity and library-aware features."""

import numpy as np

from assignment.baselines import jia_balanced
from segment.stc.assignment import (
    incident_edge_summary,
    jia_ebtas,
    library_aware_ebtas,
)
from segment.stc.library import build_all_pairs_library
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


# ---- Step 2: Jia-style adapter fidelity ----


def test_jia_ebtas_matches_legacy_jia_balanced_on_fixed_fixture():
    positions = [(100.0, 100.0), (900.0, 100.0), (100.0, 900.0), (900.0, 900.0),
                 (500.0, 500.0), (300.0, 700.0), (700.0, 300.0), (200.0, 200.0),
                 (800.0, 800.0), (500.0, 100.0)]
    data = [2.0e7, 1.5e7, 2.5e7, 1.0e7, 3.0e7, 1.2e7, 2.2e7, 1.8e7, 2.8e7, 1.6e7]
    sc = _scenario(2, positions, data)

    legacy_labels = jia_balanced(
        np.asarray(positions, dtype=float), 2,
        offload_volumes=np.asarray(data, dtype=float), random_state=42)[0]
    adapter = jia_ebtas(sc, field=None, seed=42)
    assert np.array_equal(adapter.labels, legacy_labels)


def test_jia_ebtas_records_weights_and_seed_in_feature_summary():
    sc = _scenario(2, [(100.0, 100.0), (900.0, 900.0), (500.0, 500.0)],
                   [1e7, 2e7, 3e7])
    res = jia_ebtas(sc, field=None, seed=7, weights={"payload": 2.0})
    fs = res.feature_summary
    assert fs["seed"] == 7
    assert fs["weights"]["payload"] == 2.0
    assert fs["weights"]["spatial"] == 1.0   # default resolved
    assert "rate" in fs["weights"]


def test_jia_ebtas_records_rate_feature_provenance():
    sc = _scenario(2, [(100.0, 100.0), (900.0, 900.0), (500.0, 500.0)],
                   [1e7, 2e7, 3e7])

    class _VaryingField:
        n_sectors = sc.n_sectors
        world_hash = "varying"

        def sample(self, pos):
            eff = np.full(self.n_sectors, 1.0)
            eff[0] = 1.0 + pos[0] / 1000.0   # sector-0 rate grows with x
            return RadioSample(a2g_efficiency=eff,
                               g2a_outage=np.zeros(self.n_sectors))

    no_field = jia_ebtas(sc, field=None, seed=0)
    with_field = jia_ebtas(sc, field=_VaryingField(), seed=0)
    assert no_field.feature_summary["rate_feature"] is None
    assert with_field.feature_summary["rate_feature"] == "max_a2g_efficiency"


# ---- Step 3: library-aware features ----


def test_library_aware_valid_partition_and_deterministic():
    positions = [(200.0, 200.0), (800.0, 200.0), (200.0, 800.0), (800.0, 800.0)]
    data = [1e7, 2e7, 3e7, 4e7]
    sc = _scenario(2, positions, data)
    lib = build_all_pairs_library(sc, _ConstField(sc.n_sectors),
                                  generator_config={"straight_hold": True})
    a = library_aware_ebtas(sc, lib, seed=3)
    b = library_aware_ebtas(sc, lib, seed=3)
    assert np.array_equal(a.labels, b.labels)
    flat = sorted(t for c in a.clusters for t in c)
    assert flat == list(range(sc.n_tasks))
    assert len(a.clusters) == sc.n_uav


def test_library_aware_incident_summary_reflects_payload():
    positions = [(200.0, 200.0), (800.0, 200.0), (200.0, 800.0), (800.0, 800.0)]
    data = [1e7, 1e7, 1e8, 1e7]   # task 2 carries 10x the payload
    sc = _scenario(2, positions, data)
    lib = build_all_pairs_library(sc, _ConstField(sc.n_sectors),
                                  generator_config={"straight_hold": True})
    summary = incident_edge_summary(sc, lib)
    assert summary.shape == (sc.n_tasks, 4)
    # upload-delay column (index 1) tracks payload: heavy task > light task
    assert summary[2, 1] > summary[0, 1]


def test_library_aware_enforces_equal_cluster_counts():
    positions = [(100.0, 100.0), (300.0, 100.0), (500.0, 100.0), (700.0, 100.0),
                 (900.0, 100.0), (300.0, 700.0), (700.0, 700.0)]
    data = [1e7, 2e7, 3e7, 4e7, 5e7, 6e7, 7e7]
    sc = _scenario(3, positions, data)   # K=7 >= U=3
    lib = build_all_pairs_library(sc, _ConstField(sc.n_sectors),
                                  generator_config={"straight_hold": True})
    res = library_aware_ebtas(sc, lib, seed=1)
    sizes = [len(c) for c in res.clusters]
    assert max(sizes) - min(sizes) <= 1


def test_library_aware_empty_clusters_only_when_k_lt_u():
    positions = [(100.0, 100.0), (500.0, 500.0)]
    data = [1e7, 2e7]
    sc = _scenario(4, positions, data)   # K=2 < U=4
    lib = build_all_pairs_library(sc, _ConstField(sc.n_sectors),
                                  generator_config={"straight_hold": True})
    res = library_aware_ebtas(sc, lib, seed=1)
    empties = [c for c in res.clusters if len(c) == 0]
    assert len(empties) == sc.n_uav - sc.n_tasks


def test_library_aware_feature_summary_lists_incident_features():
    positions = [(200.0, 200.0), (800.0, 200.0), (200.0, 800.0), (800.0, 800.0)]
    data = [1e7, 2e7, 3e7, 4e7]
    sc = _scenario(2, positions, data)
    lib = build_all_pairs_library(sc, _ConstField(sc.n_sectors),
                                  generator_config={"straight_hold": True})
    res = library_aware_ebtas(sc, lib, seed=3)
    fs = res.feature_summary
    assert set(fs["incident_features"]) == {
        "flight_s", "upload_delay_s", "outage", "sector_diversity"}
