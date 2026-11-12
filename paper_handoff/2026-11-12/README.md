# 2026-11-12 实验交接包

本目录包含 Teacher-Anchored 学习型正式训练（Windows 侧）的密封结果。

## 内容

- `ta_marl_20261112.tar.gz`：两个正式 run
  - `results/formal_win/win-formal-u2-s42-sharedtd3bc-ta`
  - `results/formal_win/win-formal-u2-s42-mappo-ta`

每个 run 均含 `manifest.json`、`resolved_config.json`、`summary.json`、
`records.json`、`training_curve.npz`、`best_constraint.pth`、
`best_completion.pth`、`best.pth`、`last.pth`。

## 最终结果（best_constraint 确定性回放）

| 方法 | 可行 | T (s) | D (s) | 巡检 | 上传 | 最大连续中断 (s) | 碰撞/越界/扇区冲突 |
|---|---:|---:|---:|---:|---:|---|---|
| TA-SharedTD3-BC | 是 | 386.0 | 317.0 | 20/20 | 20/20 | 15.0 / 18.5 | 0 / 0 / 0 |
| TA-MAPPO | 是 | 392.5 | 319.0 | 20/20 | 20/20 | 15.0 / 18.0 | 0 / 0 / 0 |

## 验收门槛（全部满足）

- 20/20 巡检、20/20 上传、两架 UAV 均到达终点；
- 每架最大连续中断 ≤ 20 s；
- 零碰撞、零越界、零提前进入、零扇区冲突；
- `best_constraint.pth` 确定性回放可行。

## 论文使用边界

- 必须写作 Teacher-Anchored SharedTD3-BC（TA-SharedTD3-BC）与
  Teacher-Anchored MAPPO（TA-MAPPO），不得表述为“标准 SharedTD3-BC / 标准 MAPPO”。
- 旧标准版 SharedTD3-BC / MAPPO 未学会（发散），已作废，仅可作负结果。
- Windows 侧 MA-D3QN 已取消，MA-D3QN 结果以 Mac 侧为准。
- 本包结果仅来自本机（Windows），不得与 Mac 侧 run 混入同一统计族。

## 校验

```bash
shasum -a 256 -c paper_handoff/2026-11-12/SHA256SUMS
```
