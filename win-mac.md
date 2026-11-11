# Win–Mac 数据同步（win-mac.md）

> 维护者：Windows 机（feat/win）。本文件记录 Windows 侧实验执行状态，供 Mac 侧读取诊断。
> 更新：2026-11-11

## 1. 当前任务

按 `docs/teacher-anchored-marl.md` §5 与 `mac-win.md` §8，Windows 负责：

- TA-SharedTD3-BC：`win-formal-u2-s42-sharedtd3bc-ta`（3000 集）
- TA-MAPPO：`win-formal-u2-s42-mappo-ta`（3000 集）

MA-D3QN 已取消（Mac 侧 MA-D3QN 即将跑完，避免跨机重复，以 Mac 结果为准）。

## 2. 训练状态（2026-11-11）

| 方法 | 状态 | 预算 | 备注 |
|---|---|---|---|
| TA-SharedTD3-BC | 运行中 | 3000 集 | ep 150/3000，20/20，eval_feasible 连续 True |
| TA-MAPPO | 运行中 | 3000 集 | ep 200/3000，20/20，eval_feasible 间歇 True |
| MA-D3QN | 已取消 | 10000 集 | ep 5500 时停止，Mac 侧为准 |

## 3. 旧标准版结果（已作废）

旧 SharedTD3-BC（2100 集）与 MAPPO（4050 集）未学会、已停止，作为负结果留存；
不得与新 TA 版混为同一方法。

## 4. 执行合规

- 命令严格按 `docs/teacher-anchored-marl.md` §5，seed 42、horizon 2000、eval interval 50。
- 代码基于 `a56e8f7`；相关测试 34 项全通过。
- 环境 `.venv`：Python 3.13.14 + torch 2.15.0+cpu。
