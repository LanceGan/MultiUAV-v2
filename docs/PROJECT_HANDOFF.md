# MultiUAV-v2 项目交接文档

更新时间：2026-10-31  
当前分支：`master`  
远程仓库：`origin` → `https://github.com/LanceGan/MultiUAV-v2.git`

## 1. 项目目标

本项目实现面向蜂窝连接多无人机巡检的 STC-ALNS 仿真框架。研究任务由三个层次组成：

1. 为每条合法有向边生成多个通信感知连续航段；
2. 分配巡检任务并决定各 UAV 的访问顺序；
3. 在统一时隙和共享上行带宽下同步回放完整方案，并由 STC-ALNS 根据时空扇区冲突优化任务、顺序和航段选择。

正式目标为：

```text
J = w_T·T/T_ref + w_D·D/D_ref
```

其中 `T` 是系统 makespan，`D` 是总数据上传完成时延。

## 2. 不可改变的研究约束

- 使用完整已知 A2G 上行和 G2A 下行 CKM；不研究 Partial-CKM、Kriging 或在线地图补全。
- 不把能耗 `E` 放入目标、约束或正式结果表。
- 所有 UAV 在 `t=0` 同时起飞；禁止 departure shift 或起飞错峰。
- 点 `i` 到达后立即产生数据 `I_i`；`i→j` 航段携带 `I_i`，`depot→i` 携带 0，`i→terminal` 携带 `I_i`。
- 任务边库严格为 `K(K+1)` 条：depot→task、task→task（无自环）、task→terminal。
- 所有正式指标均由同一个槽同步共享带宽评价器计算。仅槽前缓存非空的 UAV 占用带宽。
- 学习型方法只可作为真实、可验证的候选或基线。缺少兼容 SI SAC checkpoint 时，GATSAC 必须显式 `unavailable`。

权威实验契约见 [final implementation plan](superpowers/plans/2026-10-25-final-multi-uav-experiment-implementation.md) 和 [STC experiment protocol](stc-experiment-protocol.md)。

## 3. 当前代码结构

| 路径 | 作用 |
|---|---|
| `segment/stc/` | STC 核心：场景、无线场、航段库、同步评价器、冲突归因、STC-ALNS、系统基线、统计与溯源。|
| `experiment/run_stc_segment_gate.py` | RQ1：相同有向边上的航段生成器比较。|
| `experiment/run_stc_main.py` | RQ2 与 RQ4：分配/排序比较和六种系统方法。|
| `experiment/run_stc_evaluator_audit.py` | RQ3：独立满带宽、静态拥塞和同步评价的差异。|
| `experiment/run_stc_ablation.py` | RQ5：七项机制消融。|
| `experiment/run_stc_scale.py` | RQ5：UAV 数量、任务规模/密度、数据量、带宽、上行功率和地图 seed。|
| `experiment/run_stc_gates.py` | Gate 0–4 的机器可读证据检查。|
| `plots/plot_stc_results.py` | 只从冻结 run 提取和绘制 STC 结果。|
| `tests/stc/` | STC 设计契约、评价器、运行器、Gate 和绘图回归测试。|

旧的 `assignment/`、`sequence/`、`trajectory/` 及 `experiment/run_all.py` 是早期三层实验代码。它们可用于回溯与历史对照，不应被当作当前论文的正式 STC 结果入口。

## 4. 已完成并接受的工作

- Task 1–8 已完成并通过此前的独立审查。
- Task 9（RQ1–RQ5 运行器）已接受。最后两次修复：
  - `9ed2d93`：直接航段回放、deadline 边界、内容绑定 reference/library hash、消融 seed、方法级 checkpoint。
  - `6fa4e3e`：RQ2 运行异常/中断时的记录保存与 failed manifest 语义。
- Task 9 最后一次回归：432 passed。
- 修复后的历史路线图位于：
  - `results/figs/segment_trajectory_uav2_corrected_radio.png`
  - `results/figs/alns_routes_uav2_corrected_radio.png`
  这些是旧路线在修复电磁地图上的重绘，不是正式 STC 论文结果。

## 5. 当前状态：Task 10 未验收

Task 10 覆盖 Gate、绘图、冻结与报告证据。其代码已有多轮加固，最新提交为：

- `b4a9530`：Gate 4 认证与 Gate 1 跨场景拼接修复。
- `f628825`：Gate 认证第二轮修复。

**截至本交接文档，`f628825` 尚未完成独立复审。** 因此以下动作仍被禁止：

- 不启动 bounded pilot。
- 不将现有 `results/runs/` 开发/冒烟产物作为论文实验结论。
- 不启动正式多 seed 矩阵。

下一位开发者应先独立审查 `d2e20bd..f628825`，至少验证：

1. Gate 1 对每个场景独立验证完整的 `K(K+1)` 合法边集，任何场景的非法、缺失或伪生成器证据都不能被其他场景掩盖。
2. Gate 4 绑定真实 `resolved_config.json`、输入哈希、实验特定产物契约和真实文件字节；报告 bundle 必须绑定图件、provenance、run ID 和输入身份。
3. Gate 2/3 的方法清单、同场景同 method-seed 配对、异常/失败策略、层次统计和共同可行集定义一致。
4. 绘图不混用不同代码、评价器、物理世界、地图或候选库版本的 run。

只有 Task 10 独立复审通过后，才能向项目负责人申请启动 bounded pilot。pilot 通过且获明确批准后，才可以启动正式多 seed 矩阵。

## 6. 运行环境与常用命令

建议解释器：

```powershell
F:\anaconda3\envs\pytorch\python.exe
```

从仓库根目录执行：

```powershell
F:\anaconda3\envs\pytorch\python.exe -m pytest tests/ -q
```

仅做配置检查：

```powershell
F:\anaconda3\envs\pytorch\python.exe experiment/run_stc_main.py --dry-run
F:\anaconda3\envs\pytorch\python.exe experiment/run_stc_ablation.py --dry-run
F:\anaconda3\envs\pytorch\python.exe experiment/run_stc_scale.py --dry-run
```

在 Task 10 审查通过并获得授权前，不要运行正式矩阵命令。Gate 和绘图命令在有效冻结 run 产生后使用：

```powershell
F:\anaconda3\envs\pytorch\python.exe experiment/run_stc_gates.py --runs-dir results/runs
F:\anaconda3\envs\pytorch\python.exe plots/plot_stc_results.py --runs results/runs/<run-id> --out-dir results/figs
```

## 7. 结果、模型与 Git

- `results/runs/`、`results/models/`、`results/logs/`、`results/figs/` 默认被 Git 忽略。正式 run 必须通过 manifest、输入哈希和 Gate 保存证据，不能只复制图片或汇总值。
- `results/frozen_v0_2026-10-19/` 与 `results/frozen_v1_2026-10-21/` 是历史冻结快照，仅作对照。
- 任务过程报告位于 `.superpowers/sdd/`，该目录通常被 Git 忽略；关键研究状态以本文件、协议和提交历史为准。
- 当前远程分支为 `origin/master`。推送后应使用 `git status -sb` 确认本地与远程一致。

## 8. 建议接手顺序

1. `git status -sb`、`git log --oneline -12`，确认当前提交与远程状态。
2. 审查 `f628825` 与 Task 10 的 Gate 反例测试；修复所有 Critical/Important 发现。
3. 运行全仓测试并进行独立复审。
4. 准备一个单场景、有限任务、限时的 bounded pilot 方案，提交给项目负责人批准。
5. 只有在 pilot Gate 通过后，执行 5 个 map/task 场景 × 5 个 method seed 的正式矩阵，并生成冻结 manifest、报告 bundle 和论文图表。
