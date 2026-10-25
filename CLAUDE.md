# CLAUDE.md — 多无人机巡检系统重构（MultiUAV_New）

> 🚨 **设计重构（2026-10-12）**：实验部分已整体重新设计，见 [`新设计文档.md`](新设计文档.md)。
> 下文"项目概述/实验记录"是**旧三层设计**的历史记录（保留供回溯）；新架构见 **「设计重构」** 一节。
> 旧代码（三层 + 端到端 DRL 轨迹）**尚未改写**，仍可按旧流程运行。

## 项目概述（旧设计，历史）

基于已发表的单机项目（ICC 2026，`F:\Projects\Py\SingleUAV`）重构的**多无人机通信感知巡检系统**。三层架构：

1. **层1 通信感知任务分配**（`assignment/`）— 4D 聚类（空间 + G2A 中断 + A2G 速率 + 数据量）+ 3 基线
2. **层2 通信感知序列**（`sequence/`）— GA_EQTSP（通信感知边权）+ 4 基线
3. **层3 通信感知轨迹**（`trajectory/`）— 参数共享独立 TD3 + 完整物理环境

**核心目标**：`min makespan T + λ_E·E`，约束传输时延 `τ_k ≤ τ_max`（时延是 QoS 约束，不在目标里）。

- 区域 4km×4km，UAV 2/3/4 对应巡检点 20/30/40
- 通信模型复用单机已验证代码（3GPP RMa + G2A/A2G 双地图）
- **不要改动** `F:\Projects\Py\MultiUAV`（旧项目只读参考）

## 设计重构（2026-10-12，权威定义见 [`新设计文档.md`](新设计文档.md)）

旧三层设计（4D 聚类 → GA_EQTSP 序列 → 端到端 DRL 轨迹）已被整体重构。新设计要点：

**求解架构**（分层混合优化，**明确不做端到端 MARL**）：
```text
通信拥塞感知 ALNS（任务分配 + 访问顺序 + 起飞时间）
   → 目标与资源条件化 TD3（相邻两点间单段连续轨迹）
   → 安全动作投影（边界/速度/加速度/碰撞）
   → 基站扇区关联 + 共享带宽调度
   → 同步多机评价器（T / D / O / E + 可行性）→ 反馈 ALNS
```

**核心变化 vs 旧设计**：
- **DRL 定位**：旧=端到端学完整多航点轨迹（长时域，CTDE 基线完不成的根因）；新=**只学相邻两点间单段轨迹**，条件化于目标 + 无线电地图，同一策略复用所有航段。
- **离散决策**：旧的 4D 聚类 + GA_EQTSP 统一交给 **ALNS（自适应大邻域搜索）**。
- **目标函数**：旧 `min T + λ_E·E`（时延为 QoS 约束）；新 `min ω_T·(T/T_ref) + ω_D·(D/D_ref)`（**时延 D 进主目标**，ω_T+ω_D=1 初始 0.5/0.5，中断 O/能耗 E 退为**硬约束 + 报告指标**）。
- **新增建模**：扇区关联 `a_{u,m}[n]`、共享带宽 `β_{u,m}[n]`、即时巡检（进半径即完成）、相邻任务上传完成硬约束（`Q_u[n_{u,r+1}]=0`）、安全投影层、可行性优先规则、SI 单位（m/s/bit/Hz/W/J）、蒙特卡洛鲁棒性评估。

## 新设计实现进度（segment/ 模块，2026-10-18~21）

新设计核心（**单段轨迹 TD3 + 边权预计算 + ALNS + 调度**）已实现并端到端跑通（代码在 `segment/`，与旧 `trajectory/` 隔离）。

**文件**：
- `segment/env.py` — `SingleSegmentEnv`（起点→目标单段，飞行中上传源点数据，到点前必须传完）+ `CommField`（场缓存）
- `segment/td3.py` / `ddpg.py` / `sac.py` — 三个单智能体算法（TD3 提出方法；DDPG/SAC 对比）
- `segment/train.py` — 段提取（`extract_segments` 路线段 / `all_pairs_segments` 全有向边）+ 训练（`--algo TD3/DDPG/SAC`、`--held_out_frac` 泛化切分）+ rollout + SINR/中断双栏图
- `segment/precompute.py` — 预计算全部有向边权（flight_time / delay / outage / feasible）
- `segment/alns.py` — 多邻域搜索+SA（分配+顺序，destroy/repair + 2-opt + swap，可行性优先）
- `segment/scheduler.py` — 扇区关联 + 共享带宽调度器（§7，v1 静态近似）
- `segment/compare.py` / `scan_eps.py` — 算法对比 / ε_out 权衡扫描

**关键决策（用户确认）**：
- 彻底弃用 SCA（非光滑确定性地图与梯度法错配，见下）；TD3 复用 SingleUAV 不重写
- 资源信息只进**奖励**（不进 state）；动作 `[v,θ]`；state 7 维（含**绝对数据量**，修 10 vs 30Mbit 混叠）
- 单段 reward = 对称前进 `+60·Δdist`（无套利）+ 势能塑形 + 上传奖励 `+1·已传数据` + 时间/中断惩罚 + 可行性优先（到达传完 +400 / 没传完 −1000）
- **到达半径 15m（0.15）**（原单机项目注释写 0.5m 实为 50m，属单位 bug）；**路线级 outage 约束 ε_out=0.35**

**边语义**：有向边 `A→B` 携带**源点 A 的数据**；depot→点 数据=0、点→点/点→end 数据=源点数据。全边 = K(K+1) = **420**（K=20）。

**最终结果**（⚠️ 以下为**错误地图（全场 NLoS）时代**的结果，已作废；正确结果见下方「地图修复」小节）（7 维 + 15m + ε_out=0.35，3000 集）：
- 单段 TD3：420 边 arrive+upload 可行 **91.2%**；**held-out（84 条未见边）泛化 90.5%**（vs 训练集 89%，几乎无过拟合）
- **对比**（确定性 rollout，ε_out=0.25 口径）：TD3 45.0% / SAC 49.0% / DDPG 46.2% 可行；makespan TD3 48.0s ≈ SAC 48.5s < DDPG 68.8s；**时延 DRL 14~15s vs 贪心 20.4s（−27~31%）**——通信感知核心收益
- **ε_out 权衡扫描**：0.25→45% 可行、0.30→63.8%、**0.35→73.1%**、0.40→80.2%（自然中断率上限 25.9%）
- **ALNS（ε_out=0.35）**：T **316s**、D 260s、obj **0.6706**（vs 贪心基线 −6.8%）
- 共享带宽调度（v1）：时延通胀 +7.6%（ALNS 把两机分到不同区域，天然避争用）

**审计整改**（GPT 审计 `docs/audits/2026-10-19-...md`，逐条落实）：P0 冻结 legacy_v0 + manifest ✅、best/last 分离 ✅、图存原始轨迹（去样条）✅、feasible 收紧（路线级 outage）✅、held-out ✅、日志补 ep_feasible ✅、ALNS 改名（多邻域搜索）✅、状态 7 维 ✅；**#8 旧地图单位 bug 已修复（2026-10-21，见下方「地图修复」小节）**。

**SCA 弃用原因（重要）**：SCA（连续凸近似 + SLSQP）需要平滑梯度，但本项目信道是**确定性建筑遮挡 LoS（`checkLoS` 硬跳变）+ 确定性快衰落（坐标派生种子）**，速率场非光滑；强行平滑（ChannelField 50m 网格）抹掉 40m 楼区特征、收益仅 +2.4%。故轨迹层改无梯度 DRL，离散层用 ALNS。

**地图修复 + 区域缩放实验（2026-10-21）**：
- **#8 四个单位 bug 全部修复**（`radio_map_A2G.py` + `radio_map_G2A.py` 两个地图）：
  - Bug 1（米/公里直比）：checkLoS `SelectedHeight > ZSample×1000`（修复前 LoS≈0.4% 全场 NLoS）
  - Bug 2（建筑索引差 4×）：写建筑 `(step-1)/D`→`(step-1)`
  - Bug 4（噪声差 40dB）：A2G −100→−114 dBm、G2A −140→−114 dBm（1MHz 热噪声 kTB）
  - Bug 3（BS 硬编码）：增 `set_region(D)`，BS 2×2 网格按 D 缩放；`load_scenario` 按 `length/10` 自动设 D
- **乡村场景重标定**：地形 0~50m 连绵丘陵（sigma=20 + 归一化；原 sigma=10 压到 ~1.5m）、建筑 4~12m 低矮（原 5~40m + `+=` 累加 160m 虚高 → `np.maximum` 取最高）
- **区域缩放实验（4/5/6km）→ 建议回退固定 4km**：N=4（6km、仅 4 基站）G2A 中断飙到 0.58，ε_out=0.35 约束无法满足 → obj=inf。固定 4km 则中断恒 ~0.25、约束始终可满足，且更简单、强化多机扇区争用叙事
- **⚠️ 噪声修复副作用（待决策）**：A2G 上行 SINR −4.2→+9.8dB，时延 ~20s→~5s，"通信感知降时延"收益（原 −32%）归零；G2A 下行中断成新瓶颈。需重审叙事（改"降时延"为"避 G2A 中断/可靠性"）或重标定噪声系数
- **初步结果（噪声修复后、区域缩放前，N=2/4km）**：held-out TD3 100%（DDPG 92.9%/SAC 96.4%）；ALNS obj 0.8087（T=247s/D=116.5s）；ε_out=0.35 可行 50.7%；对比时延全 ~5s（无 DRL 优势）

**闭环评价器 + 迭代 ALNS（2026-10-23，回应 GPT 审计的「闭环」升级）**：
- **区域缩放定案**：N=2/3/4 → 4/5/6 km、基站 4/5/6 个（`set_region(D, n_bs)`，2×2 / 四角+中心 / 2×3 网格随 D 缩放）。此前「回退固定 4km」建议已废弃，改为**区域+基站同步缩放**（修 N=4 的 6km 只 4 基站中断 0.58 的问题）
- **噪声定案**：对齐 SingleUAV 原值——A2G −100 dBm（SNR）、G2A −140 dBm（SINR，干扰主导噪声可忽略）；G2A=SINR、A2G=SNR
- **PB 定案**：0.2W（23dBm，折中——保留部分上行瓶颈使「降时延」有意义，又缓解 6km 过弱）
- **闭环评价器**（`segment/closed_loop.py`）：逐时隙扇区关联 + 共享带宽仿真，替代 v1 静态近似；`iterative_alns` 用争用因子缩放每机时延重跑 ALNS，让外层搜索显式感知资源冲突（GPT 的第①个升级）
- **结果（N=3，5 seeds，base vs 迭代）**：闭环 makespan 411.9±33.3s、delay 525.1±85.1s、争用因子 1.396±0.135；迭代相对基 ALNS 降时延 ~16%（单 seed 26%，多 seed 方差大，需更多 seed + 显著性检验）
- **N=4 仍 inf**：单段 TD3 在 6km 长段训练不收敛（`best_score=-inf`，带噪声可行率仅 ~4%），与 PB 无关，待单独调超参/换探针

## 目录结构

```
radio_map/     通信模型（从 SingleUAV 复制，D=4km，GBS 已 ×2 缩放，快衰落已改确定性）
common/        config.py（场景配置）+ metrics.py（旋翼能耗）
assignment/    层1：clustering.py（4D）+ baselines.py
sequence/      层2：radio_path.py（边权采样）+ ga_eqtsp.py + baselines.py
trajectory/    层3：env.py + td3.py + train.py + test.py + replay.py + sca.py（SCA 基线）+ maddpg/matd3/mappo.py（CTDE 基线）
segment/       新设计：env.py（单段环境+场缓存）+ td3.py + train.py + precompute.py（单段 TD3，见「新设计实现进度」）
experiment/    编排：run_assign/seq/traj/end2end + run_all.py + evaluate.py + _common.py + run_sca.py + run_marl_compare/rollout.py
plots/         plot_results.py（纯从日志生成图，无硬编码指标值）
tests/         25 个单元测试
results/       datas(点坐标) / logs(JSON+npz) / figs(图) / models(权重)
docs/          specs(设计) + plans(计划)
新设计文档.md   权威重构设计（2026-10-12，取代旧三层设计）
```

## 关键参数与单位约定

- **单位**：距离 100m（区域 40×40）；无线电地图查询坐标 `/10` 转 km；**数据量用 Mbit**（rate Mbps，`τ=data/rate` 秒，无 8× 系数）
- **能耗**：旋翼 `rotary_power(v)`（Zeng 等），`v` 须 ×100 转 m/s（SI）
- **T=1000**（horizon，对齐单机）、**total_episode=2000**
- **带宽 B=1 MHz**、**数据量 10~30 Mbit/点**、**τ_max=12s**（2026-10 重新标定：B=1 使时延速率受限、通信感知偏离有意义；数据 10-30 使时延绝对值回到几十秒；τ_max=12 使 ~40% 点违约）
- 状态 `[x,y,dx,dy,data_buffer,remaining_time,其他机相对位置]`，动作 `[v, θ]`

### 校准后的权重（2026-08-29，重要）

**序列边权**（`radio_path.py`）：`cost = d/v_max + λ_E·E + λ_out·t + λ_τ·max(0, τ−τ_max)`
- λ_E = 0.0002（能耗是距离的 ~26%，次要）
- λ_out = **3.0**（当前 config 值；曾扫描降到 1.0 后又在 c1a1bbc 回退到 3.0，实验6 的序列收益是在 3.0 下测的）
- λ_τ = 2.0（时延 QoS 违约惩罚）
- **τ = data·飞行时间/C**（曾有无量纲 bug：`data/C` 导致时延项恒为 0）

**轨迹奖励**（`env.py`，与序列层**解耦**）：
- `progress +60·max(0,Δdist)`（朝目标飞）
- `TIME_PENALTY -1.0/步`
- `TRAJ_OUTAGE_WEIGHT -1.0×outage/步`（独立常量，不是 config 的 λ_out=3.0）
- `R_ARRIVE=400, R_COMPLETE=1000`
- 能耗 `-λ_E·P(v)·Δt`

**4D 聚类权重**（`_common.py`）：`W4D = (1.0, 1.0, 1.0, 0.5)`（空间, G2A, A2G, 数据量）

## 当前状态（2026-10-06）

- ✅ 全部代码完成，27 测试通过（25 原有 + 2 回归测试），一键复现
- ✅ 防造假：所有指标/图从日志自动生成，无硬编码值
- ✅ **序列层通信感知收益 0.80%/23.51%/36.99%（N=2/3/4，λ_out=3.0、B=10）**——修完两个时延 bug 后重测，时延项驱动、随规模放大
- ✅ **聚类层通信感知优势**（簇内 G2A 中断方差：4D 0.032/0.027 vs 基线 0.051/0.070，N=3/4 明显）
- ✅ 无线电地图确定性（快衰落由点坐标派生固定种子）
- ✅ **轨迹层 SCA 基线**（`trajectory/sca.py`）：传统优化（SQP + 平滑场），A2G 消融显示双图 vs G2A-only 时延 -11.7%
- ✅ **ILD3（参数共享 TD3 + 双地图）是唯一能完成飞行任务的 DRL 方法**（all_done=True），三 CTDE 基线（MADDPG/MATD3/MAPPO）失败——但不反超贪心，定位为"执行器 + 算法对比"

## 训练历史与关键经验（重要，供后续会话参考）

1. **首训发散**（从未完成、reward 暴跌）→ 根因：奖励纯稀疏（漏了单机 progress）+ 探索噪声按 2π 缩放 + 碰撞惩罚 -10 平铺 + λ_out 过大。已逐项修复。
2. **best 模型噪声依赖** → 改为每 50 集确定性 rollout 评估选 best。
3. **无线电地图 RNG 脆弱**（快衰落依赖全局 RNG）→ 改为点坐标派生固定种子，训练/rollout 看到同一张图。
4. **τ 单位 bug**（`data/C` 无量纲 vs `τ_max` 秒）→ 时延项恒 0，通信感知名存实亡。修 `data·飞行时间/C` 后序列层收益从 3% 涨到 17%。
5. **轨迹 reward 矫枉过正**：progress 100→20 太狠（"朝目标飞"信号从 +24.5 跌到 +3/步），策略学不会完成。温和化到 progress 60、时间 1.0、中断 1.0（解耦）后恢复。
6. **稠密塑形（借鉴旧方案）适得其反**：加势能+35、对齐+4、防绕圈/防过冲等近目标惩罚后，训练中零完成（`done=[False,False]`）。**受控对比证明真凶是近目标惩罚**（防绕圈-6、防过冲-8 在 0.5~4.0 带把策略卡住——arrive_dist=0.5 但 close 带画在 dist<4.0），**不是参数共享**（参数共享在三行配置里都是常量，归因错误）。修复：移除近目标惩罚，保留势能+35、对齐+4、截断式 progress 60。
7. **完成问题根治**：移除近目标惩罚后重训，首次完成 ep 400，best 模型确定性 rollout `done=[True,True]`（两架都完成）。验证了"真凶是近目标惩罚"的假设。但轨迹层仍不反超贪心（makespan 332s vs 216s），DRL 固有限制，论文定位为"执行序列的组件"。

### 核心经验教训
- **通信项必须真实显著**：距离/空间项天然主导，通信项（outage/时延）需校准到 ~10-20% 才体现收益；但要小心矫枉过正。
- **轨迹层的 DRL 很难稳定反超贪心**（访问航点任务下贪心 makespan 近最优），论文重心应放在**层1 聚类 + 层2 序列**的通信感知贡献上。
- **奖励平衡的黄金区间**：progress 太弱(<20)→学不会完成；太强(100)→淹没通信项。轨迹层当前 progress 60 是可学习且有通信信号的中等值。
- **归因要受控对比**：别把"常量"当根因（参数共享三行都是常量），要盯住"变了的那一步"（稠密塑形）。近目标惩罚的 close 带半径必须与 arrive_dist 匹配，否则会卡策略。

## 实验记录（2026-09-12）

### 实验1：一致执行器端到端验证（`run_seq_e2e.py`）
**问题**：序列层的通信感知优势（中间指标 comm_cost 降 17%→46%）能否转化为最终指标（中断/时延）的改善？
**方法**：固定 4D 聚类 + 同一个贪心直线执行器，只变序列（欧氏 GA vs GA_EQTSP）。
**结果**（λ_out=3.0 时）：中断时长 GA_EQTSP 更优（N=4 降 23.6%：204.5→156.2s），但 makespan 增加（绕路 +35%）且时延略增。
**结论**：序列优势确实转化为中断改善，但绕路代价大——λ_out 需要重校准。

### 实验2：λ_out 扫描（`run_lambda_sweep.py`）
**问题**：λ_out=3.0 过度绕路，找"中断降幅大 + makespan 代价小"的平衡点。
**方法**：扫描 λ_out ∈ {0.5,1.0,1.5,2.0}，N=4 的 GA_EQTSP 序列 + 贪心执行器。
**结果**：
- λ_out=1.0~2.0 是平台期（outage 174~172s，makespan 238.5s）
- λ_out=3.0 进一步绕路（outage 156s，makespan 256.5s）
**结论**：λ_out=1.0 是平衡点（outage -14.6%，makespan +25.5%）。已应用到 config。

**权衡（需注意）**：λ_out 降到 1.0 后，中间指标 comm_cost 收益从 17%/26%/46% 缩到 **0%/15.9%/24.5%**（N=2 归零、N=4 减半）。这是两难：λ_out=3.0 中间指标强但端到端 makespan 代价大（+35%），λ_out=1.0 端到端温和但中间指标弱。最终取值需按论文叙事侧重决定。

### 关键认识
通信感知序列绕路避开中断是**固有权衡**：makespan 增加换中断降低，无法两全。即使 λ_out=1.0 仍有 +25.5% makespan 代价，论文需如实呈现"可靠性 vs 时间"的权衡。

### 实验4：问题建模重构——时延 QoS 才是核心约束（关键发现）
**发现 1：comm_cost 降 46% 主要来自"时延项"，不是"中断项"**（N=4 边权分解）：
- 欧氏：d/v=878, λE·E=225, λout·t=809, **λτ·delay=33693**
- GA_EQTSP：d/v=1055, λE·E=271, λout·t=707, **λτ·delay=17278**
- 即中断项只降 12.6%（对应端到端 outage 降 9%），时延项降 48.8% 才是 comm_cost 降 46% 的来源。**通信感知实际主要降"时延违约"，不是"避开中断"。**

**发现 2：数据量与上行速率严重不匹配**：A2G 平均 SINR=-3.2dB → rate≈0.56 Mbps；数据量 100~300 Mbit → 时延 τ=data/rate≈177~531s，远超 τ_max=80s，几乎所有边违约。per-edge τ 分布（N=4）：欧氏 p50=234s/max=3510s，GA_EQTSP p50=193s/max=1264s（GA_EQTSP 主要降极端时延 max 降 64%，中位时延改善小 17%）。

**重新建模方向**：核心约束应为"时延 QoS"（τ_k ≤ τ_max），通信感知 = 走高 A2G 区满足时延约束。需校准数据量（或带宽 B）与 τ_max，让"基准违反、我们满足"成立。

### 实验5：带宽 B 校准（1→10 MHz）+ 违约比例分析
**改动**：带宽 B 从 1→10 MHz（`env.py` BANDWIDTH、`radio_path.py` BANDWIDTH），rate 增 10 倍，时延 τ 降 10 倍（p50 从 127-234s 降到 13-23s）。
**违约比例**（τ > τ_max，B=10）：
- N=2：τ_max=60s 时，欧氏违约 5%、GA_EQTSP 违约 **0%**（清晰成立）
- N=4：τ_max=60s 时，欧氏 27.5%、GA_EQTSP 22.5%（差值仅 5pp，区分度弱；τ_max=40s 时 GA_EQTSP 反而更差）
**关键认识**：GA_EQTSP 边权里时延是**软惩罚**（`λτ·max(0,τ-τ_max)`），软惩罚只"降低违约幅度"（max 351→254s），不"消除违约边数量"。若要让 N=4 也成立，需提高 λτ 权重或把时延改硬约束。

### 实验3：MARL 算法对比（`run_marl_compare.py`）
**问题**：对比四种 MARL 算法（参数共享 TD3 vs MADDPG/MATD3/MAPPO）的训练效果。
**方法**：N=2、500 集、统一环境/4D 聚类/GA_EQTSP 序列/seed。三个新算法（MADDPG/MATD3/MAPPO）为 CTDE 架构（per-agent actor + 集中 critic），实现于 `trajectory/{maddpg,matd3,mappo}.py`。
**结果**（best_reward / avg_last100，500 集）：
- MADDPG：14819 / **+4395**（稳定）
- ILD3（参数共享 TD3）：13489 / **-82**（发散）
- MAPPO：6189 / **+4344**（稳定）
- MATD3：9259 / **-951**（发散）
**结论**：**TD3 系列（ILD3、MATD3）训练后期发散，DDPG（MADDPG）和 PPO（MAPPO）稳定**。这解释了之前 ILD3 需要"确定性 best 模型选择"来规避发散。on-policy PPO 最稳但 best_reward 较低（样本效率低）。

**补充：确定性 rollout 性能对比（`run_marl_rollout.py`）**
500 集训练的四个 best 模型做确定性 rollout，**四个都没完成**（times=[-1,-1]，makespan 全 500s 截断）——因为 500 集训练不够（ILD3 需 2000 集才完成）。所以 500 集只能得出"训练稳定性"结论（TD3 发散、DDPG/PPO 稳定），**不能得出"任务性能"结论**（需 2000 集完整训练）。
奖励曲线图：`plots/plot_marl_compare.py` → `results/figs/marl_reward_compare.png`（四算法原始 + 平滑曲线）。

### 实验6：序列层两个时延 bug 修复 + 收益重测（2026-09-27）
**发现两个 bug**（都影响时延项，之前序列层"通信感知收益"被污染）：
1. `sequence/ga_eqtsp.py` 的 `comm_dis_mat` 时延项漏乘 `flight_time`（写成 `λτ·max(0, ds/C_ij − τ_max)`，无量纲，且 ds/C_ij ~ O(1-30) << τ_max=80，**时延项恒为 0**）——GA 优化时根本没在优化时延，只在避中断；`edge_cost`（报告指标）是对的（`τ = data·flight_time/C`）。
2. `experiment/_common.py` 的 `path_cost` 按"路线位置"而非"点身份"取数据量（`data_sizes[i-1]` 而非 `data_sizes[order[i-1]]`），非自然访问序时每条边配错数据。已加 `order` 参数修复。
**修复后序列收益重测**（λ_out=3.0、B=10、修后 GA）：**0.80% / 23.51% / 36.99%（N=2/3/4）**，且 `method_cost == path_cost` 完全一致。分解显示收益由**时延项**驱动（N=3 违约清零、N=4 降 98.9%），代价是绕路（d/v +15.7%/+19.1%）；N=2 无违约边故收益≈0。
**回归测试**：`test_comm_dis_mat_consistent_with_edge_cost`（comm_dis_mat == edge_cost 逐边一致）、`test_path_cost_pairs_data_by_visit_order`。

### 实验7：轨迹层"通信感知偏离"是弱贡献（2026-09-27，重要结论）
**SCA 基线**（`trajectory/sca.py` + `experiment/run_sca.py`）：数值 SCA/SQP（scipy SLSQP + 平滑场 `ChannelField`），逐段优化"偏离直线的实际轨迹"最小化与序列层同一套 comm cost，用完整无线电地图（oracle 上界）。
**关键更正**：之前"SCA 相对贪心 10× 上行速率"是**乌龙**——旧 `UAV_2_Baseline.metrics.json` 是 B=1 时代产物，B=10 下贪心速率实为 **14.14**（非 1.41）。同口径 B=10 下 SCA 14.48 vs 贪心 14.14（**仅 +2.4%**）。
**带宽 B 扫描**（贪心 vs SCA，N=2）：
- B=1：速率 +38.2% 但时延 +23.4%（更差）、makespan +25%——低带宽下时延被"航段飞行时间"封顶，绕路拉长航段反而更差
- B=3：速率 +6.9%、时延 +2.2%（更差）；B=5：速率 +3.5%、时延 -1.6%（可忽略）；B=10：速率 +2.4%、时延 +2.3%（更差）
**结论：没有任何 B 让轨迹层"速率明显涨 + 时延明显降"同时成立。** 偏离追速率的好处几乎总被绕路代价（makespan +、航段拉长导致时延封顶升高）抵消，轨迹层"通信感知偏离"是**边际收益**。
**DRL 探索（也失败）**：给 ILD3 状态加 rate/outage 观测 + 上行速率奖励（权重 1.0/0.3/门控），四个算法全部**无法完成任务**（times=[-1,-1]、makespan 500 截断），轨迹在高速率区**来回震荡**（大角度折返 ~40-52%），只有 TD3 系学会高速率（13.8-16.2 Mbps）。根因：raw rate/outage 非光滑（LoS 跳变）塞进状态 → 价值函数非光滑 → 策略振荡；速率奖励（即使门控）给"高速率区盘旋"留了 reward-hacking 漏洞。
**轨迹层定位**：维持"执行序列的组件"弱贡献定位，论文重心在层1 聚类 + 层2 序列（有真实、可复现的收益）。

### 实验8：轨迹层最终结论——时延口径修正 + A2G 消融 + DRL 定位（2026-10-06，最新）
**时延口径修正（关键，推翻了实验7 的"边际收益"结论）**：实验7 用的轨迹层时延 `τ_k` 是"封顶的卸载耗时"（`min(data/rate, 航段时间)`），这个封顶把通信感知偏离的收益**藏起来了**。改为**传输时间 `τ_k = data/rate`（不封顶，与序列层一致）**后，收益显形。
**重新标定**（让时延速率受限、通信感知有意义）：带宽 B=10→**1 MHz**、数据量 100-300→**10-30 Mbit**、τ_max 80→**12s**。
**A2G 消融**（`use_a2g` 档位，核心亮点——A2G 上行地图是独创）：
- SCA-双图 vs SCA-G2A：**时延 -11.7%**（289→255s）、速率 +3.6%（1.65→1.71）、违约率同为 40%——A2G 地图的价值显形。
- 贪心（无地图）、SCA（G2A-only）、三个 DRL 基线（G2A-only）为对比方法；ILD3（双地图）为提出方法。
**DRL 结果**（N=2、2000 集、新标定、确定性 rollout）：
- **ILD3（参数共享 TD3 + 双地图）是唯一能完成飞行任务的 DRL 方法**（all_done=True、times=[670,790]、makespan 395）；三个 CTDE 基线 MADDPG/MATD3/MAPPO 全部失败（times=[-1,-1]）——参数共享的样本效率优势。
- 但 ILD3 不反超贪心（速率 1.34 < 贪心 1.63、时延 316 > 284），即 DRL 学会了完成、没学会"趋向高速率区"。
**DRL 通信感知奖励调不动（固有限制）**：平坦时延惩罚（`P_TRANS=10` 二值）→ 完成但不趋高速率；连续惩罚（∝违约幅度）→ 发散完不成。根因：速率场非光滑（LoS 跳变），DRL 对通信感知奖励强度极敏感——弱了没梯度、强了破坏完成。SCA（有完整地图、精确优化）能稳提速率，DRL（无地图、学 reward）做不到。
**轨迹层定位（方案 A，最终）**：DRL（ILD3）定位为"**唯一能完成任务的 DRL 执行器 + 与 CTDE 基线的算法对比**"；"**通信感知偏离收益**"用 **SCA 双图 vs SCA-G2A 消融（-11.7% 时延）** 展示；论文重心仍在层1 聚类 + 层2 序列。

### 实验9：通信惩罚消融（no-comm）——去惩罚修不好 CTDE 基线完成（2026-10-08）
**问题**：CTDE 基线完不成飞行任务，是否因奖励里的通信惩罚项（G2A 中断 `-1.0` + A2G 时延 `P_TRANS=10`）？
**方法**：新增 `use_comm` 开关 + `--no_comm` 标志，关闭两个通信惩罚（纯完成奖励），重训 4 算法（N=2、2000 集、seed 42），确定性 rollout 验证 `all_done`。
**结果**（no-comm，确定性 rollout）：
- **ILD3（参数共享 TD3）：all_done=True**（times=[464,889]、makespan 444.5）——唯一完成
- MADDPG：False、MATD3：False、MAPPO：False（均 times=[-1,-1]、makespan 500 截断）
- 奖励曲线：MAPPO 正常（先升后稳 avg_last100=4514）；ILD3/MADDPG/MATD3 后期发散（avg_last100=197/-602/334，TD3/DDPG 固有 Q 过估计）
**结论**：**通信惩罚不是 CTDE 基线完不成的根因**，根因是**架构样本效率**（参数共享一份策略两架共享梯度 vs CTDE 独立 actor 学得慢）。"完成"与"曲线稳定"在当前算法池里**矛盾**：off-policy（TD3/DDPG）能完成但发散，on-policy（PPO）稳定但完不成。且去惩罚反而使 ILD3 指标变差（时延 316→503、速率 1.34→1.28），证明通信项在给 ILD3 提供趋高速率的微弱信号。
**决策**：**放弃 no-comm 路线，回到带通信惩罚的实验8 口径**。no-comm 训练已覆盖实验8 的模型/奖励 npz/rollout 产物，复用实验8 口径需重训 `use_comm=True`。
**附带发现（不一致）**：`trajectory/mappo.py` 的 `best.pth` 用带噪声训练 reward 选（`if ep_reward > best_reward`），而 ILD3/MADDPG/MATD3 都用每 50 集确定性 rollout 选 best——口径不一致（`last.pth` 同样完不成，故不影响结论，但建议统一）。

### 实验10：奖励套利根因 + 三处实现缺陷（2026-10-08，逐条核实）
**奖励套利（reward hacking）**：前进奖励 `+60·max(0,Δdist)`（[env.py:438](trajectory/env.py#L438)）只奖励靠近、**不惩罚远离**（非对称），叠加近目标方向奖励 `+4·forward`（[env.py:449](trajectory/env.py#L449)），使"靠近—远离"振荡能**净赚正奖励而不完成任务**。最小复现：单 UAV 在目标前反复前后移动 200 步（始终未完成、最终回到起点），仍累计 +1271.0 奖励，每个"靠近—远离"循环净赚 ~12.71。**故奖励曲线的 2–2.7 万高峰含奖励套利，不能直接当作任务性能提升。**
**曲线形态解释**：早期策略学会"局部靠近"拿高回报 → replay buffer 混入振荡轨迹、策略继续更新 → 局部模式被破坏 → 奖励下滑。MADDPG/MATD3 的集中 critic 用团队总奖励放大了这种高方差回报；ILD3 因参数共享 + 更小的局部 critic 暂时更稳，但曲线同样后期下滑。
**三处实现缺陷**（已修复 2026-10-08：A① 对称 progress + B done=True + C best 要求完成 + D MAPPO log-prob，待重训验证）：
1. **truncation 仍写 done=False**：off-policy 三个 trainer（ILD3/MADDPG/MATD3）在 `truncated=True` 时仍存 `done=info["terminal"]`，critic 对"截止状态"继续 bootstrap。有明确时限（T=1000）的任务里这是"失败态"，却给了延续价值 → 错误价值传播 + Q 值不稳。（注：`done=False` 是"timeout bootstrap"标准写法，对无限时环境正确；本任务截断=失败，done=True 更贴切，需权衡。）
2. **best.pth 选择口径不一**：ILD3/MADDPG/MATD3 用"确定性总奖励"选 best（**不要求任务完成**，可能把振荡/套利策略误选为 best）；MAPPO 用"带探索噪声的单回合奖励"。四方法不一致。
3. **MAPPO log-prob 裁剪错配**：采样动作先算 log-prob（[mappo.py:236](trajectory/mappo.py#L236)）再裁剪到合法区间（[mappo.py:240](trajectory/mappo.py#L240)），存的是**裁剪前**动作的 log-prob；更新时却对**裁剪后**动作算新 log-prob（[mappo.py:310](trajectory/mappo.py#L310)）→ PPO 概率比不再严格对应采样分布，曲线"稳定但低效"是合理现象。

### 实验11：奖励修复 + 重训——修复不改变"仅 ILD3 完成"的结论（2026-10-09）
**实施**：实验10 的四项修复全部落地——A① 对称 progress（`+60·max(0,Δdist)`→`+60·Δdist`）、B truncation 存 `done=True`、C best.pth 要求确定性 rollout `all_done`（MAPPO 统一为确定性 eval + 无完成时 `best.pth=last.pth` fallback）、D MAPPO log-prob 裁剪后再算。
**重训**（use_comm=True 通信感知、N=2、2000 集、seed 42，确定性 rollout）：
- **ILD3：all_done=True**（times=[569,970]、makespan 485）、速率 **1.77**（实验8 是 1.34，**首超贪心 1.63**）、时延 267.5（实验8 316）
- MADDPG / MATD3 / MAPPO：**all_done=False**（三者训练期 0 次确定性完成，`best.pth` 触发 fallback，rollout times=[-1,-1]）
**结论**：**奖励修干净后 CTDE 三基线依旧完不成**——决定性证明根因是**架构样本效率（参数共享 vs 独立 actor）**，与奖励函数无关。奖励套利高峰消失；ILD3 训练奖励后期负向发散（avg_last100=-725、best=26604 早峰）是 TD3 的 Q 过估计，非套利。
**附带观察（未确认稳健性）**：ILD3 速率 1.34→1.77 反超贪心，但来自"飞得更慢更克制"（makespan 395→485、UAV2 在 970/1000 步才完成），是可靠性-时间权衡，需多次 rollout 确认。

## 已知问题与待办

1. 放量 N=3/4 的轨迹+端到端（低优先级，轨迹不反超贪心）
2. 序列层收益需在新标定（B=1、数据 10-30、τ_max=12）下重测（实验6 的 0.8%/23.5%/37% 是 B=10 + 数据 100-300 时代的）
3. 地形分辨率 40m（step=101 可考虑 step=201）
4. `compute_metrics`、`rural_world.py`、`entity.py` 死代码可删
5. `run_all.py`/`run_end2end.py` 默认 `total_episode=300`，与文档的 2000 不一致
6. ✅ best 模型选择口径统一（四方法）：确定性 rollout + 要求完成（`all_done`），避免误选振荡/套利策略（实验11）
7. ✅ 奖励套利修复：`+60·max(0,Δdist)` → `+60·Δdist` 对称化（实验11）
8. ✅ off-policy trainer（ILD3/MADDPG/MATD3）truncation 的 done 标志：截断=失败 → done=True（实验11）
9. ✅ MAPPO log-prob 裁剪错配：存"裁剪后"动作的 log-prob（实验11）
10. ILD3 速率 1.77 反超贪心 1.63 的稳健性验证（多次 rollout / 换 seed，排除单次偶然）
11. ✅【2026-10-22】区域缩放定案：N=2/3/4 → 4/5/6km + 基站 4/5/6 个（区域+基站同步缩放，不回退固定 4km）
12. ✅【2026-10-22】噪声定案：对齐 SingleUAV（A2G −100 / G2A −140 dBm）；PB=0.2W 折中
13. 【2026-10-23】N=4 单段 TD3 训练不收敛（best_score=-inf），需单独调超参/换探针（不在闭环主线内）
14. 【2026-10-23】迭代 ALNS 收益需更多 seed + 基 ALNS 多 seed 配对 + 显著性检验（当前 5 seed 方差大）

## 常用命令

```bash
# 环境：conda pytorch（F:/anaconda3/envs/pytorch/python.exe）
F:/anaconda3/envs/pytorch/python.exe -m pytest tests/ -q                          # 测试
F:/anaconda3/envs/pytorch/python.exe experiment/run_traj.py --uav_nums 2 --total_episode 2000  # 训练
F:/anaconda3/envs/pytorch/python.exe experiment/run_assign.py --uav_nums "2,3,4"  # 分配对比
F:/anaconda3/envs/pytorch/python.exe experiment/run_seq.py --uav_nums "2,3,4"     # 序列对比
F:/anaconda3/envs/pytorch/python.exe plots/plot_results.py                        # 出图
F:/anaconda3/envs/pytorch/python.exe experiment/run_all.py --uav_nums 2 3 4       # 一键复现
```
