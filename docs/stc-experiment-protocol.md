# STC 实验协议与闸门

STC-ALNS（Spatio-Temporal Conflict-Aware ALNS）最终实验的权威协议。所有正式
结果只从冻结 run 读取，任何表图不含硬编码指标值。

## 固定假设（Final Research Contract）

- **全知 CKM**：规划前已知完整 A2G 上行 + G2A 下行电磁地图。
- **无 Partial-CKM / Kriging / 在线补全**（本论文范围外）。
- **无能耗 E**：E 不在目标、约束与正式结果表中。
- **同时起飞**：所有方法（含基线）首航时隙 `t=0`，无起飞错峰、无 departure 维度。
- **有向边源点数据语义**：边 `i→j` 携带源点 `i` 的数据量 `I_i`；`depot→i` 数据为 0；
  `i→terminal` 携带 `I_i`。合法有向边恰为 `K(K+1)` 条。
- **巡检即时生成数据**：点 `i` 到达即产生 `I_i` bits；源点数据必须在下一个
  服务锚点事件前上传完毕。
- **统一时隙**：所有 UAV 用同一时隙 `dt`、同一全局时钟；带宽 `B` 共享，同扇区
  平分；仅槽前缓存非空的 UAV 占用上行带宽。

## 目标与约束

```text
J = w_T·T/T_ref + w_D·D/D_ref,   w_T + w_D = 1（默认 0.5 / 0.5）
```

- `T = max_u T_u` 系统 makespan；`D = sum_k D_k` 总上传完成时延。
- `T_ref` / `D_ref` 为同一实验族内所有方法共享的冻结参考值。
- 硬可行性：任务全覆盖、无重复任务、下一服务事件前上传完成、末点数据完成、
  路线级 G2A 中断阈值 `ε_out`、最大等待、任务 deadline。

## 六种系统方法（同一场景 / 航段库 / ObjectiveSpec / 同步评价器 / 失败规则）

| 方法 | 分配 | 排序 | 航段 |
|---|---|---|---|
| Region-Distance-Straight | 地理分区 | 距离 | Straight-Hold |
| Shortest-Balanced-Graph | 平衡空间 | 距离/3-opt | 图 / SLSQP |
| Jia-EBTAS-style | Jia 平衡 | Time-Energy / GA-EQTSP | 最近确定性通信感知 |
| GATSAC-style | 空间 K-means | 通信感知 TSP/3-opt | SAC 局部轨迹 |
| Library-Open-Loop | 库感知 EBTAS | 库感知 | 独立最优候选（无闭环） |
| **Proposed STC-ALNS** | 库感知 EBTAS | 库感知 | 同步冲突反馈 ALNS 精化 |

- GATSAC-style 只在存在**兼容 SI 版 SAC checkpoint**（源点负载、rankable、
  校验哈希一致）时可用；否则记录显式 `unavailable`，**禁止用其他轨迹冒充**。
- 基线不得静默使用 Proposed 的冲突反馈或多候选精化算子。

## 五个研究问题（RQ）

| RQ | Runner | 证据产物 |
|---|---|---|
| RQ1 航段质量 | `run_stc_segment_gate.py` | 各航段生成器可行性 / 共同可行 T/D/O / 扇区多样性 / 计算时间 |
| RQ2 分配与排序价值 | `run_stc_main.py`（`run_rq2`） | 分配方法 + 排序方法对比（固定航段/评价器） |
| RQ3 同步评价必要性 | `run_stc_evaluator_audit.py` | 独立满带宽 / 静态拥塞因子 / 同步评价 误差与反转 |
| RQ4 全系统对比 | `run_stc_main.py` | 六方法；主对比 Proposed vs Jia / GATSAC / Library-Open-Loop |
| RQ5 机制与适用性 | `run_stc_ablation.py` + `run_stc_scale.py` | 7 项单机制消融 + 规模/参数族（每族均评价 Library-Open-Loop 与 Proposed STC-ALNS；含 density 密度族：任务数+区域+基站同步缩放） |

## 四类 seed（互斥）

- `map_seed` 地形/建筑/无线图；`task_seed` 巡检点与数据量；
  `method_seed` ALNS/启发式搜索随机性；`train_seed` 仅学习型基线。

正式矩阵最低要求：5 个配对场景（不同 map/task seed）× 5 个 method seed；
学习型基线至少 5 个 train_seed checkpoint，否则显式标注为单 checkpoint 受限基线。

## 统计与失败策略

- 成功率：全部配对实例。代价对比：共同可行配对集（报告其规模）。
- 另报告预声明的 penalized 全实例目标；**永不静默丢弃失败**。
- 报告均值 / 中位数 / 标准差 / 95% CI / 配对差 CI / effect size / 配对置换或 Wilcoxon。
- 预声明对比应用 Holm 校正。主对比：Proposed vs Jia / GATSAC / Library-Open-Loop。

## 闸门（Gate 0–4，机器可读报告）

由 `experiment/run_stc_gates.py` 生成，证据缺失一律 `NOT_RUN`、绝不 `PASS`：

- **Gate 0** 评价器 / 全量测试通过。
- **Gate 1** 航段层证据（RQ1）。
- **Gate 2** 分配/排序（RQ2）+ 同步共享带宽（RQ3）证据。
- **Gate 3** Proposed 对比 + 至少两项消融机制贡献（RQ5）。
- **Gate 4** 溯源 / 复现（冻结 manifest、哈希一致、git 头、输入哈希）。

## 结果目录与溯源

- 所有正式结果写入 `results/runs/<run_id>/`，含 `manifest.json`
  （git 头/脏 diff/未跟踪源哈希、命令、版本、解析后配置、输入哈希、产物哈希）。
- 同一 run_id 不可覆盖；`finalize("success")` 要求全部 expected 产物在场且哈希一致。
- 冻结目录 `frozen_v0/v1/v2` 只读。

## 命令

```bash
# 冒烟（六任务、减少生成器）
python experiment/run_stc_main.py --smoke --run-id <id>
# 干跑（打印矩阵）
python experiment/run_stc_main.py --dry-run --map-seeds 42 --task-seeds 42 --method-seeds 0,1,2,3,4
# 正式矩阵（须经 bounded pilot 报告 + 用户明确批准后方可启动）
python experiment/run_stc_main.py --map-seeds ... --task-seeds ... --method-seeds 0,1,2,3,4
# 消融 / 规模
python experiment/run_stc_ablation.py --dry-run
python experiment/run_stc_scale.py --dry-run
# 出图（只读冻结 run）
python plots/plot_stc_results.py --runs results/runs/<run-id> [--runs-dir results/runs] --out-dir results/figs
# 闸门报告
python experiment/run_stc_gates.py --runs-dir results/runs [--tests] [--report gate-report.json]
```
