"""Layer-2 radio-path sampling: shared edge weight computation.

沿 i→j 直线以固定步长采样双无线电地图，累加得到：
- ``C_ij``：可卸载数据量 = Σ A2G 速率 · Δt_s
- ``t_ij``：中断时长 = Σ G2A 中断概率 · Δt_s
- ``E_ij``：遍历能耗（参考速度 v_max）
- ``d_ij``：欧氏距离

综合边权：
    cost = d_ij/v_max + λ_E·E_ij + λ_out·t_ij + λ_τ·max(0, τ_ij − τ_max)
其中 τ_ij = data_size / C_ij 为传输时延。

单位约定（与项目一致）：坐标/高度均为 100m 单位，查询无线电地图前须
``/10`` 转 km（高度同理，100m → 0.1 km）。
"""

import numpy as np

from common.config import ENERGY_PARAMS
from common.metrics import edge_energy
from radio_map import radio_map_A2G, radio_map_G2A

# 采样步长（100m 单位，沿用单机约定）
STEP = 0.0915
# A2G 带宽（Mbps）
BANDWIDTH = 1.0
# 可卸载数据量下限，避免 τ = data_size / C_ij 除零
C_FLOOR = 1e-3
# 默认 UAV 高度（100m 单位，即 100m → 0.1 km）
DEFAULT_UAV_H = 1.0


def edge_cost(i_xy, j_xy, data_size, cfg):
    """计算 i→j 边的综合代价。

    Args:
        i_xy, j_xy: np.array([x, y])，单位 100m。
        data_size: 该边需卸载的数据量（Mbit）。
        cfg: ``common.config.get_scenario`` 返回的场景配置字典。

    Returns:
        dict(cost=..., C_ij=..., t_ij=..., E_ij=..., d_ij=...)
    """
    i_xy = np.asarray(i_xy, dtype=float)
    j_xy = np.asarray(j_xy, dtype=float)

    v_max = float(cfg["v_max"])
    tau_max = float(cfg["tau_max"])
    lam_E = float(cfg["lambda_E"])
    lam_out = float(cfg["lambda_out"])
    lam_tau = float(cfg["lambda_tau"])

    # UAV 高度：与坐标同处 100m 单位，查询时 /10 转 km（100m → 0.1 km）
    h_km = float(cfg.get("uav_h", DEFAULT_UAV_H)) / 10.0

    # 欧氏距离（100m 单位）
    d_ij = float(np.linalg.norm(j_xy - i_xy))

    # 遍历能耗（参考速度 = v_max）。d_ij 与 v_max 均为 100m 单位；旋转翼功率
    # 模型常量（U_tip=120 m/s、v0=4.03 m/s）为 SI，故须将 d、v 同时 ×100 转 SI
    # （m、m/s）。v is in 100m-units/s; convert to m/s via ×100（v_max=0.5 → 50 m/s）。
    E_ij = edge_energy(d_ij * 100.0, v_max * 100.0, ENERGY_PARAMS)

    # 沿直线等步长采样（每段中点）
    if d_ij < 1e-12:
        samples_xy = i_xy.reshape(1, 2)
    else:
        n = max(1, int(np.ceil(d_ij / STEP)))
        fracs = (np.arange(n) + 0.5) / n
        samples_xy = i_xy + fracs[:, None] * (j_xy - i_xy)

    # 转 km（坐标 /10、高度 /10）
    samples_km = np.hstack([
        samples_xy / 10.0,
        np.full((samples_xy.shape[0], 1), h_km),
    ])

    # 每段遍历时间
    delta_t = STEP / v_max

    C_ij = 0.0
    t_ij = 0.0
    for k in range(samples_km.shape[0]):
        loc = samples_km[k:k + 1, :]
        # A2G：SINR(dB) → 速率（Mbps）
        sinr = float(radio_map_A2G.getPointDateRate(loc))
        rate = BANDWIDTH * np.log2(1.0 + 10.0 ** (sinr / 10.0))
        # G2A：中断概率（单点 = out[0][0]）
        outage = float(radio_map_G2A.getPointMiniOutage(loc)[0][0])
        C_ij += rate * delta_t
        t_ij += outage * delta_t

    # 除零保护：可卸载数据量下限
    C_ij = max(C_ij, C_FLOOR)

    # 传输时延（秒）= 数据量(Mbit) / 平均速率(Mbps)，平均速率 = C_ij / 飞行时间。
    # 注意：C_ij 是沿边的「累计可卸载数据量」(Mbit)，除以飞行时间才是速率；直接
    # data_size/C_ij 会得到无量纲比值（单位 bug），故须乘飞行时间。
    flight_time = d_ij / v_max
    tau_ij = data_size * flight_time / C_ij

    cost = (d_ij / v_max
            + lam_E * E_ij
            + lam_out * t_ij
            + lam_tau * max(0.0, tau_ij - tau_max))

    return dict(cost=float(cost), C_ij=float(C_ij), t_ij=float(t_ij),
                E_ij=float(E_ij), d_ij=d_ij)
