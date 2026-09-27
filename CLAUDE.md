# CLAUDE.md — 多无人机巡检系统重构（MultiUAV_New）

## 项目概述

基于已发表的单机项目（ICC 2026，`F:\Projects\Py\SingleUAV`）重构的**多无人机通信感知巡检系统**。三层架构：

1. **层1 通信感知任务分配**（`assignment/`）— 4D 聚类（空间 + G2A 中断 + A2G 速率 + 数据量）+ 3 基线
2. **层2 通信感知序列**（`sequence/`）— GA_EQTSP（通信感知边权）+ 4 基线
3. **层3 通信感知轨迹**（`trajectory/`）— 参数共享独立 TD3 + 完整物理环境

**核心目标**：`min makespan T + λ_E·E`，约束传输时延 `τ_k ≤ τ_max`（时延是 QoS 约束，不在目标里）。

- 区域 4km×4km，UAV 2/3/4 对应巡检点 20/30/40
- 通信模型复用单机已验证代码（3GPP RMa + G2A/A2G 双地图）
- **不要改动** `F:\Projects\Py\MultiUAV`（旧项目只读参考）

## 目录结构

```
radio_map/     通信模型（从 SingleUAV 复制，D=4km，GBS 已 ×2 缩放，快衰落已改确定性）
common/        config.py（场景配置）+ metrics.py（旋翼能耗）
assignment/    层1：clustering.py（4D）+ baselines.py
sequence/      层2：radio_path.py（边权采样）+ ga_eqtsp.py + baselines.py
trajectory/    层3：env.py + td3.py + train.py + test.py + replay.py
experiment/    编排：run_assign/seq/traj/end2end + run_all.py + evaluate.py + _common.py
plots/         plot_results.py（纯从日志生成图，无硬编码指标值）
tests/         25 个单元测试
results/       datas(点坐标) / logs(JSON+npz) / figs(图) / models(权重)
docs/          specs(设计) + plans(计划)
```

## 关键参数与单位约定

- **单位**：距离 100m（区域 40×40）；无线电地图查询坐标 `/10` 转 km；**数据量用 Mbit**（rate Mbps，`τ=data/rate` 秒，无 8× 系数）
- **能耗**：旋翼 `rotary_power(v)`（Zeng 等），`v` 须 ×100 转 m/s（SI）
- **T=1000**（horizon，对齐单机）、**total_episode=2000**
- 状态 `[x,y,dx,dy,data_buffer,remaining_time,其他机相对位置]`，动作 `[v, θ]`

### 校准后的权重（2026-08-29，重要）

**序列边权**（`radio_path.py`）：`cost = d/v_max + λ_E·E + λ_out·t + λ_τ·max(0, τ−τ_max)`
- λ_E = 0.0002（能耗是距离的 ~26%，次要）
- λ_out = 1.0（中断，序列层通信感知——λ_out=3.0 过度绕路，扫描后降到 1.0，见实验记录）
- λ_τ = 2.0（时延 QoS 违约惩罚）
- **τ = data·飞行时间/C**（曾有无量纲 bug：`data/C` 导致时延项恒为 0）

**轨迹奖励**（`env.py`，与序列层**解耦**）：
- `progress +60·max(0,Δdist)`（朝目标飞）
- `TIME_PENALTY -1.0/步`
- `TRAJ_OUTAGE_WEIGHT -1.0×outage/步`（独立常量，不是 config 的 λ_out=3.0）
- `R_ARRIVE=400, R_COMPLETE=1000`
- 能耗 `-λ_E·P(v)·Δt`

**4D 聚类权重**（`_common.py`）：`W4D = (1.0, 1.0, 1.0, 0.5)`（空间, G2A, A2G, 数据量）

## 当前状态（2026-08-29）

- ✅ 全部代码完成，25 测试通过，一键复现
- ✅ 防造假：所有指标/图从日志自动生成，无硬编码值
- ✅ **序列层通信感知收益 17%/26%/46%（N=2/3/4）**——GA_EQTSP 相对纯欧氏 GA 的 comm_cost 降幅随规模放大，这是论文核心贡献
- ✅ **聚类层通信感知优势**（簇内 G2A 中断方差：4D 0.032/0.027 vs 基线 0.051/0.070，N=3/4 明显）——用正确指标"簇内方差"后体现，同样随规模放大
- ✅ 无线电地图确定性（快衰落由点坐标派生固定种子）
- ✅ getPointDateRate 多行输入守卫（会丢数据的隐患已封）
- ✅ 轨迹层能稳定完成（done=[True,True]，两架都完成），但不反超贪心（DRL 固有限制），定位为"执行序列的组件"

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

## 已知问题与待办

1. 放量 N=3/4 的轨迹+端到端（低优先级，轨迹不反超贪心）
2. tau_max/λ 灵敏度校准
3. 地形分辨率 40m（step=101 可考虑 step=201）
4. `compute_metrics` 死代码可删

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
