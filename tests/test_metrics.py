import numpy as np
from common import metrics
from common import config
from common.config import ENERGY_PARAMS

def test_power_energy_positive_and_increasing_in_speed():
    # 已知速度下的功耗与能耗
    v = 10.0
    p = metrics.rotary_power(v, ENERGY_PARAMS)
    assert p > 0
    # 能耗 = P * 时间 = P * d/v
    e = metrics.edge_energy(1.0, v, ENERGY_PARAMS)  # d=1(100m)
    assert abs(e - p * (1.0 / v)) < 1e-9

def test_metrics_makespan_is_max_of_uav_times():
    # 两架 UAV，完成时间分别 100, 150 步
    traj = {0: np.zeros((150, 2)), 1: np.zeros((100, 2))}
    times = {0: 150, 1: 100}
    m = metrics.compute_metrics(traj, times, data_sizes={0: [], 1: []})
    assert m["makespan"] == 150

def test_edge_energy_zero_length_is_zero():
    # d=0 的边（重复相邻点 / 悬停 / 结束）能耗应为 0，而非 NaN
    assert metrics.edge_energy(0.0, 10.0, ENERGY_PARAMS) == 0.0
    assert metrics.edge_energy(0.0, 0.0, ENERGY_PARAMS) == 0.0
    traj = {0: np.array([[1.0, 1.0], [1.0, 1.0]])}
    m = metrics.compute_metrics(traj, {0: 2}, data_sizes={0: []})
    assert np.isfinite(m["total_energy"])
    assert m["total_energy"] == 0.0

def test_get_data_sizes_shape_and_range():
    sizes = config.get_data_sizes(20, seed=42)
    assert len(sizes) == 20
    assert np.all(sizes >= 10.0)
    assert np.all(sizes <= 30.0)

def test_get_data_sizes_deterministic():
    a = config.get_data_sizes(10, seed=7)
    b = config.get_data_sizes(10, seed=7)
    assert np.array_equal(a, b)

def test_get_scenario_returns_expected_keys():
    sc = config.get_scenario(2)
    for k in ("user_num", "length", "width", "bs_loc", "ini_loc", "end_loc",
              "v_max", "delta_t", "tau_max", "T_out_max",
              "lambda_E", "lambda_out", "lambda_tau", "seed"):
        assert k in sc
    assert sc["user_num"] == 20
