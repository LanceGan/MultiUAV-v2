# STC bounded pilot 方案（待项目负责人批准）

> 起草：2026-10-31（Task 10 第四轮复审通过并验收后）。
> 依据：[最终实验契约](superpowers/plans/2026-10-25-final-multi-uav-experiment-implementation.md) §6.1、
> [实验协议](stc-experiment-protocol.md)、[交接文档](PROJECT_HANDOFF.md) §8。
> **本方案获批前不执行任何命令；pilot 产物不构成论文结论。**

## 1. 目的与非目的

**目的**：在最小可信规模上，端到端验证「五个 RQ runner → 密封 manifest → 绘图/bundle →
Gate 0–4 机器可读报告」整条证据链在本参考机（macOS arm64，`.venv` 锁定栈）上真实可用，
为正式多 seed 矩阵的规模与时间预算提供实测依据。

**非目的**：pilot 不产生任何论文可用的数值结论；不验证方法优劣（K=6 冒烟场景的
对比/消融结果无统计意义）；不触碰 `frozen_v0/v1/v2`。

## 2. 前置条件（执行前逐项核对）

- [ ] Task 10 已验收（2026-10-31 已确认）。
- [ ] `results/runs/` 为**空**（Gate 4 扫描该目录下全部 run，任何残留目录都会污染报告）。
- [ ] `results/figs/` 内无 `reporting_bundle.json`（bundle 写一次性，残留会使绘图步骤拒绝写入）。
- [ ] `git status` 干净（manifest 记录 dirty diff；pilot 应在干净提交上跑）。
- [ ] 全量测试绿：`./.venv/bin/python -m pytest tests/ -q`（即 Gate 0 的输入）。

## 3. 范围与参数

单场景冒烟口径（协议 §6.1：`U=2, K=6`，单 map/task/method seed）：

| 参数 | 值 | 说明 |
|---|---|---|
| UAV 数 | 2 | `--uav-nums 2` |
| 任务数 | 6 | `--smoke`（smoke_tasks=6） |
| map/task seed | 42 / 42 | 单场景 |
| method seed | 0 | `--method-seeds 0`（单 seed；正式矩阵才用 0–4） |
| 场分辨率 | 1000 m | smoke 默认 `spacing_m` |
| ALNS 迭代 | 1000 | 默认值 |
| GATSAC | `unavailable` | 无兼容 SI SAC checkpoint（诚实记录，不替身） |
| 单 runner 限时 | 见 §4 | `--timeout-s`，超时产 failed manifest 并保留已写记录 |

## 4. 执行步骤（获批后按序执行，全部从仓库根目录）

```bash
# ① RQ1 航段层证据
./.venv/bin/python experiment/run_stc_segment_gate.py --smoke --method-seeds 0 --timeout-s 1800 --run-id pilot-rq1
# ② RQ3 评价器审计（独立/静态/同步误差）
./.venv/bin/python experiment/run_stc_evaluator_audit.py --smoke --method-seeds 0 --timeout-s 1800 --run-id pilot-rq3
# ③ RQ2+RQ4 六系统方法 + 分配/排序对比
./.venv/bin/python experiment/run_stc_main.py --smoke --method-seeds 0 --timeout-s 3600 --run-id pilot-rq2rq4
# ④ RQ5 七项机制消融
./.venv/bin/python experiment/run_stc_ablation.py --smoke --method-seeds 0 --timeout-s 3600 --run-id pilot-rq5-ablation
# ⑤ RQ5 规模/参数族
./.venv/bin/python experiment/run_stc_scale.py --smoke --method-seeds 0 --timeout-s 3600 --run-id pilot-rq5-scale
# ⑥ 绘图 + 溯源 bundle（只读冻结 run）
./.venv/bin/python plots/plot_stc_results.py --runs-dir results/runs --out-dir results/figs
# ⑦ Gate 0–4 报告（--tests 跑全量测试作为 Gate 0）
./.venv/bin/python experiment/run_stc_gates.py --runs-dir results/runs --tests --report results/figs/pilot-gate-report.json
```

每步记录墙钟时间。任一 runner 失败：failed manifest 与已写记录自动保留，**不删除、
不覆盖**，修复后换新 `--run-id` 重跑，旧目录留作证据。

## 5. 验收判据

**硬性（全部满足才算 pilot 通过）**：
1. 五个 run 全部 `finalize("success")`：密封 manifest、产物哈希齐全、符合各自产物契约；
2. 图件 + `figures_provenance.json` + `reporting_bundle.json` 生成且非空；
3. `pilot-gate-report.json` 生成，**无任何 Gate 处于 NOT_RUN**；
4. Gate 0（全量测试）与 Gate 4（溯源/认证）为 PASS。

**解释性（Gate 1–3，按下述规则处置）**：
- 预期 PASS。若 FAIL，先判定失败类别：
  - **管线缺陷**（证据缺失、记录无效、哈希不一致等）→ pilot 不通过，修复后重跑；
  - **尺度效应**（如 K=6 争用过低导致消融机制无贡献、共同可行交集为空）→ pilot 的
    管线目标仍达成，但须在报批材料中写明该 Gate 的失败证据与归因，并给出正式矩阵
    （或加压场景）下预期成立的理由，由负责人决定是否先做一次加压 pilot。
- 任何情况下**不得**为让 Gate 变绿而修改判据、伪造或挑选证据。

## 6. 交付物（报批材料）

- `results/runs/pilot-*/`（5 个密封 run 目录）
- `results/figs/`（图件 + provenance + bundle + `pilot-gate-report.json`）
- 一页报告：各步墙钟时间、Gate 状态表、异常与归因、据此估算的正式矩阵
  （5 场景 × 5 method seed × 3 UAV 档）时间预算

## 7. 批准记录

- [ ] 项目负责人批准执行 pilot：＿＿＿＿（日期）
- [ ] pilot 报告提交：＿＿＿＿
- [ ] 负责人批准启动正式矩阵：＿＿＿＿
