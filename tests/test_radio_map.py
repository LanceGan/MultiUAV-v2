import numpy as np
from radio_map import radio_map_A2G, radio_map_G2A

def test_radio_map_returns_finite_sinr_and_outage():
    # 区域中心附近一个点，坐标单位 km
    loc = np.array([[2.0, 2.0, 0.1]])  # 2km,2km,100m 高度
    sinr = radio_map_A2G.getPointDateRate(loc)   # 返回 SINR(dB) 标量
    outage = radio_map_G2A.getPointMiniOutage(loc)  # 返回 (outage_vec, sinr_vec)
    assert np.isfinite(float(sinr))
    assert 0.0 <= float(outage[0][0]) <= 1.0  # 单点中断概率 = outage_vec[0]

def test_users_files_have_expected_shape():
    for n in (20, 30, 40):
        pts = np.loadtxt(f"results/datas/Users_{n}.txt")
        assert pts.shape == (n, 3)  # x,y,h
        assert pts[:, :2].min() >= 0 and pts[:, :2].max() <= 40  # 4km×4km，单位100m
