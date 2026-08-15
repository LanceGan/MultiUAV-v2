import numpy as np
from common.config import ENERGY_PARAMS

def rotary_power(v, p):
    P0, Pi, U_tip, v0, d0, rho, s, A = (p[k] for k in
        ("P0","Pi","U_tip","v0","d0","rho","s","A"))
    v = max(v, 1e-6)
    return (P0*(1+3*v**2/U_tip**2)
            + Pi*np.sqrt(np.sqrt(1+v**4/(4*v0**4)) - v**2/(2*v0**2))
            + 0.5*d0*rho*s*A*v**3)

def edge_energy(d, v, p=ENERGY_PARAMS):
    return rotary_power(v, p) * (d / v)

def compute_metrics(trajectories, times, data_sizes, energy_params=ENERGY_PARAMS):
    # trajectories: {uav_id: np.array([[x,y],...])}; times: {uav_id: 完成步数}
    makespan = max(times.values()) if times else 0
    # 能耗：逐边以实际速度估算（速度由相邻点位移 / delta_t 得到，简化用平均速度）
    total_energy = 0.0
    for u, tr in trajectories.items():
        d = np.linalg.norm(np.diff(tr, axis=0), axis=1)
        v = d / 0.5  # delta_t=0.5
        total_energy += float(np.sum([edge_energy(di, vi, energy_params) for di, vi in zip(d, v)]))
    return dict(makespan=makespan, total_energy=total_energy)
