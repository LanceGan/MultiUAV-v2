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
radio_map/     通信模型（从 SingleUAV 复制，D 已改 4km，GBS 已 ×2 缩放）
common/        config.py（场景配置）+ metrics.py（旋翼能耗）
assignment/    层1：clustering.py（4D）+ baselines.py
sequence/      层2：radio_path.py（边权采样）+ ga_eqtsp.py + baselines.py
trajectory/    层3：env.py + td3.py + train.py + test.py + replay.py
experiment/    编排：run_assign/seq/traj/end2end + run_all.py + evaluate.py
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
- 奖励（当前）：`+100·max(0,Δdist) 前进 + R_ARRIVE=400 + R_COMPLETE=1000 − 能耗 − 中断(λ_out=0.1) − 碰撞(软，max -2) − 出界(soft clip) − 停留(-2)`

## 当前状态（2026-08-23）

- ✅ 全部代码完成，25 测试通过，一键复现（`python experiment/run_all.py`）
- ✅ 防造假：所有指标/图从日志自动生成，无硬编码值
- ✅ N=2 训练**收敛**（首次完成 ep 340，best 模型 727 步完成全部 20 点，无 NaN）
- ⚠️ **TD3 仍劣于贪心 baseline**（makespan 363.5s vs 216.5s，中断/时延也更高）

## 训练发散诊断史（重要经验）

首训 2000 集发散（从未完成、reward 暴跌），根因（已修复）：
1. **奖励纯稀疏** — spec 漏了单机的稠密 progress 奖励 → 已加 `+100·max(0,Δdist)`
2. **探索噪声按 2π 缩放** — 角度噪声 108°→28°，无法精确进点 → 已改 per-dim 缩放（角度用弧度）
3. **碰撞惩罚 -10 平铺** — 与缩小后噪声叠加，开局就碰撞 → 已改软惩罚（max -2，按距离渐变）
4. **λ_out=1.0 中断惩罚过大** — 死角区每步 -1，产生 -3578 尖刺 → 已降 0.1
5. rollout 用了**退化的最终模型**而非 best 模型 → 已改 `load_td3` 加载 best.pth

## 已知问题与待办（按优先级）

1. **TD3 劣于贪心**（当前最重要）— 奖励缺时间惩罚、通信项太弱，策略不学"快速完成 + 绕开死角"。待加：时间惩罚 + 加强 outage/时延项
2. 训练后期策略退化（ep~430 后 avg100 从 27000 掉到 6000）— 用 best checkpoint 规避，可考虑早停
3. 放量 N=3/4 全量实验（未跑）
4. tau_max/λ 灵敏度校准（未做）
5. 地形分辨率 40m（step=101 未随 D 增大，可考虑 step=201）

## 常用命令

```bash
# 环境：conda pytorch（F:/anaconda3/envs/pytorch/python.exe）
F:/anaconda3/envs/pytorch/python.exe -m pytest tests/ -q          # 测试
F:/anaconda3/envs/pytorch/python.exe experiment/run_traj.py --uav_nums 2 --total_episode 2000  # 训练
F:/anaconda3/envs/pytorch/python.exe plots/plot_results.py       # 出图
F:/anaconda3/envs/pytorch/python.exe experiment/run_all.py --uav_nums 2 3 4  # 一键复现
```
