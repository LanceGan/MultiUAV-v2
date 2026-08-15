import numpy as np

ENERGY_PARAMS = dict(P0=79.8563, Pi=88.6279, U_tip=120.0, v0=4.03,
                     d0=0.6, rho=1.225, s=0.05, A=0.503)

# N 架无人机 → 巡检点数 / 场景规模
SCENARIOS = {
    2: dict(user_num=20, length=40, width=40),
    3: dict(user_num=30, length=40, width=40),
    4: dict(user_num=40, length=40, width=40),
}

def get_scenario(uav_num, seed=42):
    sc = dict(SCENARIOS[uav_num])
    sc.update(
        uav_num=uav_num,
        v_max=0.5,            # 单位100m/步；对应 20m/s 的量级（沿用单机）
        delta_t=0.5,          # 秒/步
        tau_max=80.0,         # 时延 QoS 上限（秒）
        T_out_max=80.0,       # 中断时长上限（秒）
        lambda_E=0.001, lambda_out=1.0, lambda_tau=1.0,  # 权重，待灵敏度分析
        bs_loc=np.array([[9.54,9.12],[29.78,9.78],[9.26,27.56],[26.96,29.78]]),
        ini_loc=np.array([2.62, 2.65]),
        end_loc=np.array([18.95, 11.12]),
        seed=seed,
    )
    return sc

def get_data_sizes(user_num, seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(100.0, 300.0, user_num)  # MB，对齐单机论文测试的 100~300MB 量级
