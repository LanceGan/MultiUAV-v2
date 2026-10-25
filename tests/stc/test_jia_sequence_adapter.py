"""Task 4 Step 2: Jia Time-Energy / GA-EQTSP baseline adapter fidelity.

The adapters wrap the legacy ``sequence.radio_path`` / ``sequence.ga_eqtsp``
implementations without changing their internal criteria; these tests verify
the resolved edge weights and routes match the legacy code, and that the
resolved weights / seeds are recorded for provenance.

The legacy map queries are monkeypatched to constants (as in
``tests/test_radio_path.py``) so the tests are fast and deterministic; this
leaves the adapter's directed-weight assembly and unit conversion as the exact
things under test.
"""

import numpy as np
import pytest

from sequence import ga_eqtsp, radio_path
from segment.stc.sequencing import ga_eqtsp_sequence, jia_time_energy_tsp
from segment.stc.types import STCScenario, ScenarioLimits

DEFAULT_WEIGHTS = {"lambda_E": 0.0002, "lambda_out": 3.0, "lambda_tau": 2.0,
                   "tau_max": 12.0}


@pytest.fixture(autouse=True)
def _fast_radio(monkeypatch):
    monkeypatch.setattr(radio_path.radio_map_A2G, "getPointDateRate",
                        lambda loc: 10.0)
    monkeypatch.setattr(radio_path.radio_map_G2A, "getPointMiniOutage",
                        lambda loc: ([0.1], [10.0]))


def _limits():
    return ScenarioLimits(delta_t_s=0.5, v_max_mps=50.0, uav_height_m=100.0,
                          arrive_radius_m=15.0, max_wait_s=60.0,
                          mission_deadline_s=7200.0, bandwidth_hz=1e6,
                          uplink_power_w=0.2, eps_out=0.35, w_t=0.5, w_d=0.5)


def _scenario(positions_m, data_bits):
    return STCScenario(
        n_uav=2, n_tasks=len(positions_m),
        task_positions_m=tuple(tuple(float(v) for v in p) for p in positions_m),
        task_data_bits=tuple(float(d) for d in data_bits),
        depot_m=(0.0, 0.0), terminal_m=(4000.0, 4000.0),
        limits=_limits(), length_m=4000.0, width_m=4000.0, n_bs=4,
        task_seed=42, map_seed=0)


_POSITIONS = [(500.0, 500.0), (1500.0, 500.0), (500.0, 1500.0), (1500.0, 1500.0)]
_DATA = [10e6, 25e6, 15e6, 20e6]   # heterogeneous payloads so the delay term varies


def _legacy_cfg(scenario, seed):
    return {
        "v_max": scenario.limits.v_max_mps / 100.0,
        "tau_max": DEFAULT_WEIGHTS["tau_max"],
        "lambda_E": DEFAULT_WEIGHTS["lambda_E"],
        "lambda_out": DEFAULT_WEIGHTS["lambda_out"],
        "lambda_tau": DEFAULT_WEIGHTS["lambda_tau"],
        "uav_h": scenario.limits.uav_height_m / 100.0,
        "seed": int(seed),
    }


def _legacy_nodes(scenario, task_ids):
    """Return (node_xy_100, data_mbit) in [depot] + tasks + [terminal] order."""
    node_xy_100 = np.vstack([
        np.asarray(scenario.depot_m) / 100.0,
        np.asarray([scenario.task_positions_m[t] for t in task_ids]) / 100.0,
        np.asarray(scenario.terminal_m) / 100.0,
    ])
    data_mbit = np.asarray([scenario.task_data_bits[t] for t in task_ids]) / 1e6
    return node_xy_100, data_mbit


# ---- Step 1/2: route validity for the two legacy-wrapping adapters ----

def test_jia_and_ga_visit_every_point_exactly_once():
    sc = _scenario(_POSITIONS, _DATA)
    task_ids = (0, 1, 2, 3)
    for res in (jia_time_energy_tsp(sc, task_ids, seed=42),
                ga_eqtsp_sequence(sc, task_ids, seed=42, iteration=50)):
        assert sorted(res.route) == sorted(task_ids)
        assert len(res.route) == len(set(res.route))
        assert res.selected_candidate_ids == ()
        assert res.proxy_cost > 0


# ---- Step 2: Jia Time-Energy adapter fidelity ----

def test_jia_time_energy_edge_weights_match_legacy_edge_cost():
    sc = _scenario(_POSITIONS, _DATA)
    task_ids = (0, 1, 2, 3)

    res = jia_time_energy_tsp(sc, task_ids, seed=42)
    mat = res.provenance["edge_weights"]
    assert tuple(res.provenance["node_labels"]) == (-1, 0, 1, 2, 3, sc.n_tasks)

    node_xy, data_mbit = _legacy_nodes(sc, task_ids)
    cfg = _legacy_cfg(sc, 42)
    n = len(node_xy)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ds = data_mbit[i - 1] if 0 < i < n - 1 else 0.0
            expected = radio_path.edge_cost(node_xy[i], node_xy[j], ds, cfg)["cost"]
            assert mat[i, j] == pytest.approx(expected, rel=1e-6), (i, j)


def test_ga_eqtsp_adapter_matches_legacy_route():
    sc = _scenario(_POSITIONS, _DATA)
    task_ids = (0, 1, 2, 3)

    res = ga_eqtsp_sequence(sc, task_ids, seed=42, iteration=100)

    cluster_100 = np.asarray([sc.task_positions_m[t] for t in task_ids]) / 100.0
    data_mbit = np.asarray([sc.task_data_bits[t] for t in task_ids]) / 1e6
    ini_100 = np.asarray(sc.depot_m) / 100.0
    end_100 = np.asarray(sc.terminal_m) / 100.0
    cfg = _legacy_cfg(sc, 42)

    _, order_idx, cost = ga_eqtsp.ga_eqtsp(cluster_100, ini_100, end_100,
                                           data_mbit, cfg, iteration=100)
    expected_route = tuple(int(task_ids[i]) for i in order_idx)
    assert res.route == expected_route


# ---- Step 2: provenance recording ----

def test_adapters_record_resolved_weights_and_seeds():
    sc = _scenario(_POSITIONS, _DATA)
    task_ids = (0, 1, 2, 3)

    jia = jia_time_energy_tsp(sc, task_ids, seed=7, weights={"lambda_out": 1.5})
    p = jia.provenance
    assert p["seed"] == 7
    assert p["weights"]["lambda_out"] == 1.5
    assert p["weights"]["lambda_E"] == DEFAULT_WEIGHTS["lambda_E"]  # default resolved
    assert p["weights"]["lambda_tau"] == DEFAULT_WEIGHTS["lambda_tau"]
    assert "edge_weights" in p
    assert "world_hash" in p
    assert p["node_labels"] == (-1, 0, 1, 2, 3, sc.n_tasks)

    ga = ga_eqtsp_sequence(sc, task_ids, seed=7, iteration=50)
    g = ga.provenance
    assert g["seed"] == 7
    assert g["weights"]["lambda_out"] == DEFAULT_WEIGHTS["lambda_out"]
    assert "edge_weights" in g
    assert g["num_total"] == 25
    assert g["iteration"] == 50
