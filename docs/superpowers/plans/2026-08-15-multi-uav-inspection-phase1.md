# 多无人机巡检系统 Phase 1 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `F:\Projects\Py\MultiUAV_New` 构建一套真实、可复现的通信感知三层多无人机巡检系统地基（4D 聚类 + 通信感知 GA_EQTSP + 参数共享独立 TD3），产出三层各对比 + 端到端的真实结果与图表。

**Architecture:** 忠实三层分解：层1 通信感知任务分配（点→无人机）→ 层2 通信感知序列（簇内排序，边权 `d/v_max + λ_E·E_ij + λ_out·t_ij + λ_τ·max(0,τ_ij−τ_max)`）→ 层3 通信感知轨迹（参数共享独立 TD3 + 真实数据卸载 + 能耗 + 避碰）。通信模型完全复用单机已发表代码（3GPP RMa + G2A/A2G 双地图）。

**Tech Stack:** Python（conda `pytorch` 环境 `F:/anaconda3/envs/pytorch/python.exe`）、numpy、torch、matplotlib、pytest。

---

## Global Constraints

- **禁止造假**：所有指标/曲线/图必须从原始日志（JSON/NPZ）自动生成；禁止任何手动调整数值。
- **固定随机种子**：环境、聚类、序列、训练全部 `np.random.seed()` / `torch.manual_seed()` 固定（种子值见各任务）。
- **单位**：距离单位 = 100m（区域 4km×4km → `length=width=40`）；无线电地图查询坐标须 `/10` 转 km、高度 `/10` 转 km。
- **能量模型**（旋翼功耗，Zeng 等）：`P(v) = P_0(1+3v²/U_tip²) + P_i·√(√(1+v⁴/(4v₀⁴)) − v²/(2v₀²)) + (1/2)·d₀·ρ·s·A·v³`，其中 `P_0=79.8563, P_i=88.6279, U_tip=120, v₀=4.03, d₀=0.6, ρ=1.225, s=0.05, A=0.503`。
- **目标/约束**：`min T + λ_E·E_total`，约束 `τ_k ≤ τ_max`（时延 QoS）、`T_out ≤ T_out^max`（中断）、运动学、边界、每点唯一访问、无碰撞。
- **轨迹算法**：参数共享的**独立 TD3**（一个共享 actor+critic，各机局部观测独立决策），**不用** MA-TD3 共享 Critic。
- **状态**：`s_i = [x, y, dx_target, dy_target, data_buffer, remaining_time, other_uavs_rel_pos...]`（含数据缓存，这是关键）。
- **动作**：`a_i = [v, θ]`，`v ∈ [0, v_max]`，`θ ∈ [0, 2π]`。
- **奖励**：`r_n = R_到达 + R_完成 − P_能耗 − P_传输 − P_中断 − P_碰撞`。
- **数据卸载**（真实仿真，逐点数据量 `I_i`）：每步 `data_size -= R_A2G · Δt`，`R_A2G = B·log2(1+10^(SINR/10))`，`B=1 MHz`，`Δt=0.5 s`。
- **数据文件来源**：从 `F:\Projects\Py\MultiUAV\results\datas\Users_20.txt / Users_30.txt / Users_40.txt` 复制（只读复制，不修改源项目）。
- **不要改动** `F:\Projects\Py\MultiUAV`（旧项目仅作只读参考）。

---

## File Structure

```
MultiUAV_New/
├── radio_map/                    # 通信模型（从 SingleUAV 复制，已验证）
│   ├── radio_map_A2G.py          # A2G: getPointDateRate(loc_km)->max SINR(dB)
│   ├── radio_map_G2A.py          # G2A: getPointMiniOutage(loc_km)->outage prob
│   ├── rural_world.py            # 建筑分布（3GPP RMa 地形）
│   └── entity.py                 # UAV / User 数据类
├── common/
│   ├── config.py                 # 场景配置（N→K、GBS、数据量、λ、τ_max 等）
│   └── metrics.py                # 指标计算（makespan/delay/energy/outage/rate）
├── assignment/
│   ├── clustering.py             # 4D 通信感知聚类（proposed）
│   └── baselines.py              # k-means / balanced-kmeans / jia
├── sequence/
│   ├── radio_path.py             # 沿路径采样双地图 → C_ij, t_ij, E_ij, cost_ij
│   ├── ga_eqtsp.py               # 通信感知 GA（proposed，修正边权）
│   └── baselines.py              # GA(欧氏) / PSO / ACO / NN+2-opt
├── trajectory/
│   ├── env.py                    # 通信感知多机环境（卸载/能耗/避碰/中断）
│   ├── replay.py                 # 经验回放
│   ├── td3.py                    # 参数共享独立 TD3
│   ├── train.py                  # 训练
│   └── test.py                   # 测试（rollout + 日志）
├── experiment/
│   ├── run_assign.py             # 层1 对比实验
│   ├── run_seq.py                # 层2 对比实验
│   ├── run_traj.py               # 层3 对比实验
│   └── run_end2end.py            # 端到端对比实验
├── plots/
│   └── plot_results.py           # 从日志生成全部论文图（纯自动）
├── tests/
│   ├── test_radio_map.py
│   ├── test_metrics.py
│   ├── test_clustering.py
│   ├── test_radio_path.py
│   └── test_env.py
├── results/
│   ├── datas/                    # Users_20/30/40.txt
│   ├── models/
│   ├── logs/
│   └── figs/
├── configs/                      # 场景/GBS/数据量 json 配置
├── requirements.txt
└── README.md
```

---

## Task 1: 脚手架 + 迁移通信模型（radio_map）

**Files:**
- Create: `radio_map/radio_map_A2G.py`, `radio_map/radio_map_G2A.py`, `radio_map/rural_world.py`, `radio_map/entity.py`
- Create: `requirements.txt`, `README.md`
- Create: `results/datas/` 目录，复制 `Users_20.txt / Users_30.txt / Users_40.txt`

**Interfaces:**
- Produces: `radio_map_A2G.getPointDateRate(loc_km)`、`radio_map_G2A.getPointMiniOutage(loc_km)`，`loc_km` 为 `np.array([[x_km, y_km, h_km]])`。
- Produces: `entity.UAV(x, y, h)`、`entity.User(x, y, h)` 数据类。

- [ ] **Step 1: 建目录并复制单机已验证代码**

Run:
```bash
cd /f/Projects/Py/MultiUAV_New
mkdir -p radio_map results/datas tests common assignment sequence trajectory experiment plots configs
cp /f/Projects/Py/SingleUAV/radio_map_A2G.py radio_map/
cp /f/Projects/Py/SingleUAV/radio_map_G2A.py radio_map/
cp /f/Projects/Py/SingleUAV/rural_world.py radio_map/
cp /f/Projects/Py/SingleUAV/entity.py radio_map/
cp /f/Projects/Py/MultiUAV/results/datas/Users_20.txt results/datas/
cp /f/Projects/Py/MultiUAV/results/datas/Users_30.txt results/datas/
cp /f/Projects/Py/MultiUAV/results/datas/Users_40.txt results/datas/
```

- [ ] **Step 2: 写失败测试 `tests/test_radio_map.py`**

```python
import numpy as np
from radio_map import radio_map_A2G, radio_map_G2A

def test_radio_map_returns_finite_sinr_and_outage():
    # 区域中心附近一个点，坐标单位 km
    loc = np.array([[2.0, 2.0, 0.1]])  # 2km,2km,100m 高度
    sinr = radio_map_A2G.getPointDateRate(loc)   # 返回 SINR(dB) 标量
    outage = radio_map_G2A.getPointMiniOutage(loc)  # 返回中断概率
    assert np.isfinite(float(sinr))
    assert 0.0 <= float(outage[0]) <= 1.0

def test_users_files_have_expected_shape():
    for n in (20, 30, 40):
        pts = np.loadtxt(f"results/datas/Users_{n}.txt")
        assert pts.shape == (n, 3)  # x,y,h
        assert pts[:, :2].min() >= 0 and pts[:, :2].max() <= 40  # 4km×4km，单位100m
```

- [ ] **Step 3: 运行测试确认失败/通过**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/test_radio_map.py -v`
Expected: 若 radio_map 导入路径或数据文件有问题则 FAIL，修到 PASS（`sys.path` 根目录即项目根，pytest 从项目根运行）。

- [ ] **Step 4: 写 requirements.txt 与 README 骨架**

`requirements.txt`:
```
numpy
torch
matplotlib
pytest
```

`README.md`：项目定位（真实可复现的多机巡检重构）、目录说明、运行环境（conda pytorch env）、"如何复现一个结果"占位（Task 11 填全）。

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: scaffold project and vendor verified radio map from SingleUAV"
```

---

## Task 2: 配置 + 指标计算

**Files:**
- Create: `common/config.py`, `common/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces `config.get_scenario(uav_num)` → `dict`，含 `user_num`, `length`, `width`, `bs_loc`, `ini_loc`, `end_loc`, `data_sizes`, `v_max`, `delta_t`, `tau_max`, `lambda_E/out/tau`, `energy_params`。
- Produces `metrics.compute_metrics(trajectories, data_sizes, ...)` → `dict`，含 `makespan`, `total_delay`, `total_energy`, `avg_uplink_rate`, `outage_duration`。

- [ ] **Step 1: 写失败测试 `tests/test_metrics.py`**

```python
import numpy as np
from common import metrics
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
```

- [ ] **Step 2: 实现 `common/config.py`**

```python
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
        bs_loc=np.array([[4.77,4.56],[14.89,4.89],[4.63,13.78],[13.48,14.89]]),
        ini_loc=np.array([2.62, 2.65]),
        end_loc=np.array([18.95, 11.12]),
        seed=seed,
    )
    return sc
```

- [ ] **Step 3: 实现 `common/metrics.py`**

```python
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
```

（`total_delay`/`avg_uplink_rate`/`outage_duration` 在 Task 8 由 env 日志提供，metrics 里预留接口，本任务先实现 makespan + energy。）

- [ ] **Step 4: 运行测试通过**

Run: `F:/anaconda3/envs/pytorch/python.exe -m pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add common tests/test_metrics.py
git commit -m "feat: add scenario config and metrics (energy, makespan)"
```

---

## Task 3: 层1 — 通信感知任务分配（4D 聚类 + 基线）

**Files:**
- Create: `assignment/clustering.py`, `assignment/baselines.py`
- Test: `tests/test_clustering.py`

**Interfaces:**
- Produces `assignment.clustering.kmeans_4d(points, k, offload_volumes, weights, uav_height, point_scale, random_state) -> (labels, centers, inertia)`
- Produces `assignment.baselines.naive_kmeans(...) / balanced_kmeans(...) / jia_balanced(...)` 同签名返回 `(labels, centers)`。

- [ ] **Step 1: 写失败测试 `tests/test_clustering.py`**

```python
import numpy as np
from assignment import clustering, baselines

def _pts(n=30):
    rng = np.random.RandomState(0)
    return rng.uniform(0, 40, (n, 2))

def test_kmeans_4d_balanced_and_deterministic():
    pts = _pts()
    vols = np.random.RandomState(1).uniform(50, 150, len(pts))
    l1, c1, _ = clustering.kmeans_4d(pts, 3, vols, (1.0,0.4,0.4,0.5),
                                     uav_height=0.1, point_scale=0.1, random_state=42)
    l2, c2, _ = clustering.kmeans_4d(pts, 3, vols, (1.0,0.4,0.4,0.5),
                                     uav_height=0.1, point_scale=0.1, random_state=42)
    assert np.array_equal(l1, l2)
    # 均衡：每簇大小差 <= 1
    sizes = np.bincount(l1)
    assert sizes.max() - sizes.min() <= 1
    assert set(l1) == {0,1,2}

def test_baselines_return_balanced_labels():
    pts = _pts()
    for fn in (baselines.naive_kmeans, baselines.balanced_kmeans, baselines.jia_balanced):
        lbl, cen = fn(pts, 3, offload_volumes=np.ones(len(pts))*100)
        assert set(lbl) == {0,1,2}
```

- [ ] **Step 2: 实现 `assignment/clustering.py`（4D k-means，从旧 MultiUAV `Clustering.py` 迁移并保留算法，去掉打印）**

关键：增强特征空间 `[spatial*w_s, g2a*w_g, a2g*w_a, vol*w_v]`，k-means++ 初始化，容量约束均衡后处理。G2A/A2G 特征通过 `radio_map` 逐点查询（`getPointMiniOutage` / `getPointDateRate`），`_min_max_norm` 归一化。

- [ ] **Step 3: 实现 `assignment/baselines.py`（普通 k-means / balanced k-means / jia 式空间+负载）**

- [ ] **Step 4: 运行测试通过** → `pytest tests/test_clustering.py -v`

- [ ] **Step 5: Commit**

```bash
git add assignment tests/test_clustering.py
git commit -m "feat: layer-1 communication-aware 4D clustering and baselines"
```

---

## Task 4: 层2 — 无线电路径采样（共享边权计算）

**Files:**
- Create: `sequence/radio_path.py`
- Test: `tests/test_radio_path.py`

**Interfaces:**
- Produces `sequence.radio_path.edge_cost(i_xy, j_xy, data_size, cfg) -> dict(cost=..., C_ij=..., t_ij=..., E_ij=..., d_ij=...)`，其中 `i_xy/j_xy` 为 `np.array([x,y])`（单位100m）。

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
from sequence import radio_path
from common.config import get_scenario

def test_edge_cost_decreases_with_shorter_distance():
    cfg = get_scenario(3)
    a = np.array([5.0, 5.0]); b = np.array([6.0, 5.0]); c = np.array([9.0, 5.0])
    cab = radio_path.edge_cost(a, b, 100.0, cfg)["cost"]
    cac = radio_path.edge_cost(a, c, 100.0, cfg)["cost"]
    assert cab < cac  # 短边代价更低

def test_edge_cost_penalizes_delay_violation():
    cfg = get_scenario(3)
    a = np.array([5.0, 5.0]); b = np.array([6.0, 5.0])
    big = radio_path.edge_cost(a, b, 1e6, cfg)["cost"]   # 数据量大→时延违约
    small = radio_path.edge_cost(a, b, 10.0, cfg)["cost"]
    assert big > small
```

- [ ] **Step 2: 实现 `sequence/radio_path.py`**

沿 i→j 直线以固定步长（如 0.0915 单位，沿用单机）采样，累加 A2G 速率得 `C_ij = Σ rate*Δt`、累加 G2A 中断得 `t_ij = Σ outage*Δt`、`E_ij = edge_energy(d_ij, v_ref)`；`cost = d/v_max + λ_E*E + λ_out*t + λ_τ*max(0, Data/C − τ_max)`。注意速率转换 `rate = B*log2(1+10^(sinr/10))`、`B=1`。

- [ ] **Step 3: 运行测试通过** → `pytest tests/test_radio_path.py -v`

- [ ] **Step 4: Commit**

```bash
git add sequence/radio_path.py tests/test_radio_path.py
git commit -m "feat: layer-2 radio-path edge cost sampler (dual-map + energy)"
```

---

## Task 5: 层2 — 通信感知序列（GA_EQTSP + 基线）

**Files:**
- Create: `sequence/ga_eqtsp.py`, `sequence/baselines.py`
- Test: `tests/test_radio_path.py` 追加序列有效性测试（或独立 `tests/test_sequence.py`）

**Interfaces:**
- Produces `sequence.ga_eqtsp.ga_eqtsp(cluster_points, ini_loc, end_loc, data_sizes, cfg) -> (ordered_points, order_indices, cost)`
- Produces `sequence.baselines.ga_euclidean(...) / pso(...) / aco(...) / nn_2opt(...)` 同返回 `(ordered_points, order_indices, cost)`。

- [ ] **Step 1: 写失败测试 `tests/test_sequence.py`**

```python
import numpy as np
from sequence import ga_eqtsp, baselines
from common.config import get_scenario

def test_ga_eqtsp_visits_all_points_once():
    cfg = get_scenario(2)
    pts = np.random.RandomState(0).uniform(0, 40, (8, 2))
    ds = np.ones(8) * 100.0
    ordered, order, cost = ga_eqtsp.ga_eqtsp(pts, np.array([2.,2.]), np.array([38.,38.]), ds, cfg)
    assert len(set(order)) == 8  # 每点访问一次
    assert np.isfinite(cost)

def test_ga_eqtsp_beats_euclidean_on_comm_cost():
    # 通信感知边权下，GA_EQTSP 的通信代价 <= 纯欧氏 GA 的通信代价
    cfg = get_scenario(2)
    pts = np.random.RandomState(1).uniform(0, 40, (6, 2))
    ds = np.ones(6) * 300.0
    from sequence import radio_path
    def comm_cost(order):
        return sum(radio_path.edge_cost(pts[order[i]], pts[order[i+1]], ds[order[i]], cfg)["cost"]
                   for i in range(len(order)-1))
    _, o_eq, _ = ga_eqtsp.ga_eqtsp(pts, np.array([2.,2.]), np.array([38.,38.]), ds, cfg)
    _, o_ga, _ = baselines.ga_euclidean(pts, np.array([2.,2.]), np.array([38.,38.]), ds, cfg)
    assert comm_cost(o_eq) <= comm_cost(o_ga)
```

- [ ] **Step 2: 实现 `sequence/ga_eqtsp.py`**：从单机 `GA.py` 迁移 GA 框架，`dis_mat[i][j] = edge_cost(...)["cost"]`，适应度 `1/path_cost`。

- [ ] **Step 3: 实现 `sequence/baselines.py`**：GA（纯欧氏距离）、PSO、ACO、最近邻+2-opt。

- [ ] **Step 4: 运行测试通过** → `pytest tests/test_sequence.py -v`

- [ ] **Step 5: Commit**

```bash
git add sequence/ga_eqtsp.py sequence/baselines.py tests/test_sequence.py
git commit -m "feat: layer-2 communication-aware GA_EQTSP and routing baselines"
```

---

## Task 6: 层3 — 通信感知多机环境

**Files:**
- Create: `trajectory/env.py`
- Test: `tests/test_env.py`

**Interfaces:**
- Produces `trajectory.env.MultiUAVInspectionEnv`（Gym-like）：`reset() -> obs_list`、`step(actions) -> (obs_list, rewards, dones, info)`、`get_obs(uav_id)`、`local_obs_dim`、`action_dim`。

- [ ] **Step 1: 写失败测试（物理正确性）`tests/test_env.py`**

```python
import numpy as np
from trajectory.env import MultiUAVInspectionEnv
from common.config import get_scenario

def _make_env(uav_num=2):
    cfg = get_scenario(uav_num)
    return MultiUAVInspectionEnv(cfg)

def test_data_offload_decreases_buffer():
    env = _make_env()
    env.reset()
    u = 0
    env._data_buffer[u] = 100.0          # 初始缓存
    env._offload(u, env.delta_t)         # 卸载一步
    assert env._data_buffer[u] < 100.0   # 卸载后缓存减少（SINR 非 -inf 时速率 > 0）

def test_collision_penalty_applied():
    env = _make_env()
    env.reset()
    env.UAVs[0].x, env.UAVs[0].y = 10.0, 10.0
    env.UAVs[1].x, env.UAVs[1].y = 10.0, 10.05  # 小于 safe_distance
    r = env._collision_penalty(0)
    assert r < 0

def test_obs_and_action_dim():
    env = _make_env()
    env.reset()
    # 自身位置(2) + 目标相对(2) + 数据缓存(1) + 剩余时间(1) + 其他机相对位置(2*(N-1))
    assert env.local_obs_dim == 6 + 2*(env.uav_num-1)
    assert env.action_dim == 2
```

- [ ] **Step 2: 实现 `trajectory/env.py`**

核心逻辑（严格按 spec 与单机 `World.py` 的物理）：
- 状态 `s_i = [x, y, dx, dy, data_buffer, remaining_time, (dx_other, dy_other)* ]`（归一化）。
- 动作 `[v, θ]`：`move_inside` 后 `x += v·cos θ·dt`，`y += v·sin θ·dt`，边界越界取消并惩罚。
- 数据卸载：`_offload(u, dt)`：`rate = B*log2(1+10^(SINR_A2G/10))`，`data_buffer -= rate*dt`，到达目标时若 `data_buffer>0` 且 `τ_k > τ_max` → `P_trans`。
- 能耗：`E += rotary_power(v)*dt`，`P_energy ∝ rotary_power(v)*dt`。
- G2A 中断：`_get_g2a_outage` → `P_out ∝ outage`（启用，修复单机注释掉的部分）。
- 碰撞：两机距离 `< safe_distance*2` → `P_collision`。
- 完成：每机访问完自己的序列点（由上层注入 `uav_traverse`），全部完成 → team reward + terminal。

- [ ] **Step 3: 运行测试通过** → `pytest tests/test_env.py -v`

- [ ] **Step 4: Commit**

```bash
git add trajectory/env.py tests/test_env.py
git commit -m "feat: layer-3 communication-aware multi-UAV environment"
```

---

## Task 7: 层3 — 参数共享独立 TD3 + 训练

**Files:**
- Create: `trajectory/replay.py`, `trajectory/td3.py`, `trajectory/train.py`

**Interfaces:**
- Produces `trajectory.td3.SharedTD3(state_dim, action_dim, max_action, ...)`，方法 `select_action(s)`、`train(replay)`、`save(path)`、`load(path)`。
- Produces `trajectory.train.main(uav_num, model_subdir, total_episode, ...)`。

- [ ] **Step 1: 实现 `replay.py`（经验回放，`add`/`sample`，容量可配）**

- [ ] **Step 2: 实现 `td3.py`**：从单机 `TD3.py` 迁移 Actor（tanh × max_action）/ Twin Critic，**参数共享**（全局单实例，不按 UAV 分 agent），状态维度 = `local_obs_dim`。TD3 三件套：target policy smoothing、twin critics 取 min、delayed actor update。

- [ ] **Step 3: 实现 `train.py`**：主循环（reset → 每机 `select_action` + 探索噪声 → `step` → 存 transition → 采样训练 → 记录 reward/energy/成功率 → 每 N 轮存 best/stable 模型 → 日志写 `results/logs/`）。固定 `np.random.seed(cfg["seed"])` + `torch.manual_seed(cfg["seed"])`。

- [ ] **Step 4: 冒烟训练（1 个小场景验证收敛不崩溃）**

Run:
```bash
F:/anaconda3/envs/pytorch/python.exe trajectory/train.py --uav_num 2 --total_episode 300 --model_subdir Ours
```
Expected: 训练日志正常输出 reward 曲线上升趋势，无 NaN，best 模型保存到 `results/models/UAV_2/Ours/`。

- [ ] **Step 5: Commit**

```bash
git add trajectory/replay.py trajectory/td3.py trajectory/train.py
git commit -m "feat: layer-3 parameter-shared independent TD3 and training loop"
```

---

## Task 8: 层3 — 测试（rollout）+ 统一评估

**Files:**
- Create: `trajectory/test.py`, `experiment/evaluate.py`

**Interfaces:**
- Produces `trajectory.test.rollout(env, model, uav_traverse) -> dict(trajectories, times, data_buffers, outage_steps, rates, energies)`
- Produces `experiment.evaluate.evaluate_log(log) -> metrics dict`（补全 makespan/delay/energy/rate/outage）。

- [ ] **Step 1: 实现 `test.py`**：加载 best/stable 模型，`reset` 后 rollout（`pure_policy`，无后处理），把每机轨迹/时间/每步卸载量/中断/速率/能耗写入 `results/logs/<run_id>.npz`。

- [ ] **Step 2: 实现 `evaluate.py`**：读日志 → `total_delay = Σ_k τ_k`、`avg_uplink_rate`、`outage_duration`、`total_energy`、`makespan`，输出 `results/logs/<run_id>.metrics.json`。

- [ ] **Step 3: 端到端跑通一个 N=2 场景并断言指标真实非空**

Run:
```bash
F:/anaconda3/envs/pytorch/python.exe trajectory/test.py --uav_num 2 --model_subdir Ours
F:/anaconda3/envs/pytorch/python.exe experiment/evaluate.py --run_id <run_id>
```
Expected: 生成 npz + metrics.json，`makespan > 0`、`total_energy > 0`、`total_delay >= 0`。

- [ ] **Step 4: Commit**

```bash
git add trajectory/test.py experiment/evaluate.py
git commit -m "feat: layer-3 rollout and unified evaluation"
```

---

## Task 9: 绘图（纯从日志自动生成）

**Files:**
- Create: `plots/plot_results.py`

**Interfaces:**
- Produces `plots.plot_results.plot_all(logs_dir, figs_dir)`，读 `results/logs/*.metrics.json` 生成论文图（训练曲线、轨迹对比、聚类可视化、三层对比柱状图、端到端对比）。

- [ ] **Step 1: 实现 `plot_results.py`**：所有数据从 metrics.json / npz 读取，**无任何硬编码数值**；输出到 `results/figs/`。含中文字体设置（沿用旧项目 `plot_test_statistics` 的做法）。

- [ ] **Step 2: 用 Task 8 的日志跑一次，确认图生成且无手工值**

Run: `F:/anaconda3/envs/pytorch/python.exe plots/plot_results.py`
Expected: `results/figs/` 下生成各图，脚本里 `grep` 不到任何手写数字常量（除坐标轴标签/颜色）。

- [ ] **Step 3: Commit**

```bash
git add plots/plot_results.py
git commit -m "feat: auto-generate all paper figures from logs"
```

---

## Task 10: 实验编排（三层各对比 + 端到端）

**Files:**
- Create: `experiment/run_assign.py`, `run_seq.py`, `run_traj.py`, `run_end2end.py`

**Interfaces:**
- 各脚本 `main()` 按固定种子跑「基线 × Proposed」全组合，输出 metrics.json 到 `results/logs/`。

- [ ] **Step 1: 实现 `run_assign.py`**：跑 4 种聚类（普通/balanced/jia/4D）× N∈{2,3,4}，输出各簇空间/通信/负载均衡指标。

- [ ] **Step 2: 实现 `run_seq.py`**：跑 5 种序列（GA/PSO/ACO/NN+2-opt/GA_EQTSP）× 固定聚类 × N∈{2,3,4}，输出通信代价/路径长度。

- [ ] **Step 3: 实现 `run_traj.py`**：跑「非通信感知基线轨迹 vs 通信感知 TD3」× N∈{2,3,4}，输出 makespan/时延/能耗/中断/速率。

- [ ] **Step 4: 实现 `run_end2end.py`**：三层基线组合 vs 三层 Proposed 组合 × N∈{2,3,4}。

- [ ] **Step 5: 真实跑 N=2/3/4 全量（有 GPU，可长时训练）**

Run: 依次 `run_assign.py`、`run_seq.py`、`run_traj.py`（含各自训练）、`run_end2end.py`，随后 `plot_results.py`。
Expected: 所有 metrics.json 由日志自动产生，图表可复现。

- [ ] **Step 6: Commit**

```bash
git add experiment/
git commit -m "feat: experiment orchestration for three-layer + end-to-end comparison"
```

---

## Task 11: 可复现收尾 + README

**Files:**
- Modify: `README.md`, `configs/`（场景 json）
- Create: `configs/scenario_uav2.json / uav3.json / uav4.json`

- [ ] **Step 1: 把 config 里硬编码的场景参数抽到 `configs/*.json`，`config.py` 读 json。**

- [ ] **Step 2: 写完整 README**：环境、依赖、目录、**一条命令复现全部结果**（`python experiment/run_all.py`，Task 10 补一个 `run_all.py` 串起四步）。

- [ ] **Step 3: 全量复现验证**：清空 `results/logs` + `results/figs`，跑 `run_all.py`，确认结果一致（种子固定）。

- [ ] **Step 4: Commit**

```bash
git add README.md configs experiment/run_all.py
git commit -m "docs: reproducibility docs and one-command reproduction"
```

---

## 完成定义（Phase 1 Done）

- [ ] 三层各对比 + 端到端均有真实 metrics.json + 图（N=2/3/4）
- [ ] `pytest` 全绿（test_radio_map / metrics / clustering / radio_path / sequence / env）
- [ ] 清空后一条命令可复现全部结果
- [ ] 仓库中无任何伪造/手动调整数值的脚本（对照旧项目 `regenerate_uav4_reward.py` 等一律不出现）
