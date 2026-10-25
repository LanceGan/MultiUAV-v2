import warnings

import numpy as np
import pytest

from common import config
from radio_map import radio_map_A2G, radio_map_G2A


def test_radio_map_returns_finite_sinr_and_outage():
    # 区域中心附近一个点，坐标单位 km
    loc = np.array([[2.0, 2.0, 0.1]])  # 2km,2km,100m 高度
    sinr = radio_map_A2G.getPointDateRate(loc)   # 返回 SINR(dB) 标量
    outage = radio_map_G2A.getPointMiniOutage(loc)  # 返回 (outage_vec, sinr_vec)
    assert np.isfinite(float(sinr))
    assert 0.0 <= float(outage[0][0]) <= 1.0  # 单点中断概率 = outage_vec[0]


def test_users_files_fit_their_scenario_bounds():
    # 区域随 N 缩放：N=2/3/4 → 4/5/6 km（length 40/50/60，单位 100m）。
    # 边界从场景 JSON 加载（config.get_scenario），而非在测试里复制常量，
    # 避免测试与场景配置两处口径漂移。
    for uav_num in (2, 3, 4):
        sc = config.get_scenario(uav_num)
        n = sc["user_num"]
        bound = sc["length"]
        assert sc["width"] == bound  # 区域为正方形
        pts = np.loadtxt(f"results/datas/Users_{n}.txt")
        assert pts.shape == (n, 3)  # x,y,h
        assert pts[:, :2].min() >= 0 and pts[:, :2].max() <= bound


def test_radio_map_covers_full_4km_area():
    # 右上角 (3.5, 3.5) km 必须在 4km×4km 覆盖内：有限 SINR + 中断概率在 [0,1]（无 IndexError）
    loc = np.array([[3.5, 3.5, 0.1]])
    sinr = radio_map_A2G.getPointDateRate(loc)
    outage = radio_map_G2A.getPointMiniOutage(loc)
    assert np.isfinite(float(sinr))
    assert 0.0 <= float(outage[0][0]) <= 1.0


def test_antenna_gain_is_python_scalar_without_warning():
    # 天线增益必须返回 Python 标量（而非 1 元素 ndarray）；若返回数组，把它赋给
    # (numBS, Sec_num) 矩阵的标量槽位会触发 "array ndim>0 -> scalar" 的
    # DeprecationWarning，这里将其升级为 error 以保证无法无声回归。
    for mod in (radio_map_A2G, radio_map_G2A):
        gain, _ = mod.getAntennaGain(90.0, 0.0)
        assert isinstance(gain, float)
        assert np.isfinite(gain)

    # 端到端：全图查询内部即进行该标量赋值，验证整条路径无 DeprecationWarning。
    loc = np.array([[2.0, 2.0, 0.1]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        radio_map_A2G.getPointDateRate(loc)
        radio_map_G2A.getPointMiniOutage(loc)


def test_antenna_gain_fixed_position_output():
    # 固定位置输出回归：两个模块共享同一套天线阵列模型，在固定 (仰角, 方位角)
    # 处的 dB 增益必须精确等于当前值。任何对天线模型的改动（哪怕仍返回有限值）
    # 都会让这里失败，防止无线电值被无声改动。
    expected = -291.75958991746415
    for mod in (radio_map_A2G, radio_map_G2A):
        gain, _ = mod.getAntennaGain(90.0, 0.0)
        assert gain == pytest.approx(expected, abs=1e-6)
