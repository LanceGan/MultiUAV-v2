import numpy as np
from sequence import radio_path
from common.config import get_scenario


def test_radio_path_cost_is_deterministic_and_components_are_finite():
    # 确定性：无线电地图采用坐标派生的确定性快衰落，相同输入必须得到完全相同的
    # 边权与各分量（真实地图下 outage/时延逐边变化，不假设“更短几何必更小代价”）。
    cfg = get_scenario(3)
    a = np.array([5.0, 5.0]); b = np.array([6.0, 5.0])
    r1 = radio_path.edge_cost(a, b, 100.0, cfg)
    r2 = radio_path.edge_cost(a, b, 100.0, cfg)
    for k in ("cost", "C_ij", "t_ij", "E_ij", "d_ij"):
        assert r1[k] == r2[k]

    # 各分量有限且代价为正。
    assert r1["cost"] > 0
    for k in ("cost", "C_ij", "t_ij", "E_ij", "d_ij"):
        assert np.isfinite(r1[k])


def test_geometric_flight_time_monotone_under_constant_radio_field(monkeypatch):
    # 在“常数合成无线电场”下（各点 SINR/中断完全相同），通信项不再随位置波动：
    # 时延 τ = data/rate 与 d 无关，中断时长 outage·flight_time 与 d 成正比，
    # 故总代价随几何距离单调递增。真实地图下该单调性不成立（逐边 outage 变化）。
    cfg = get_scenario(3)

    monkeypatch.setattr(radio_path.radio_map_A2G, "getPointDateRate",
                        lambda loc: 10.0)                     # 恒定 SINR(dB)
    monkeypatch.setattr(radio_path.radio_map_G2A, "getPointMiniOutage",
                        lambda loc: ([0.1], [10.0]))          # 恒定 outage=0.1

    a = np.array([5.0, 5.0]); b = np.array([6.0, 5.0]); c = np.array([9.0, 5.0])
    assert radio_path.edge_cost(a, b, 100.0, cfg)["cost"] < \
           radio_path.edge_cost(a, c, 100.0, cfg)["cost"]


def test_edge_cost_penalizes_delay_violation():
    cfg = get_scenario(3)
    a = np.array([5.0, 5.0]); b = np.array([6.0, 5.0])
    big = radio_path.edge_cost(a, b, 1e6, cfg)["cost"]   # 数据量大→时延违约
    small = radio_path.edge_cost(a, b, 10.0, cfg)["cost"]
    assert big > small
