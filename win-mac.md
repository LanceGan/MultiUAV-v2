# Win–Mac 数据同步（win-mac.md）

> 维护者：Windows 机（feat/win）。本文件记录 Windows 侧实验执行状态，供 Mac 侧读取诊断。
> 更新：2026-11-11

## 1. 当前任务

按 `docs/teacher-anchored-marl.md` §5 与 `mac-win.md` §8，Windows 负责：

- TA-SharedTD3-BC：`win-formal-u2-s42-sharedtd3bc-ta`（3000 集）
- TA-MAPPO：`win-formal-u2-s42-mappo-ta`（3000 集）
- MA-D3QN（原始 10000 集正式训练，继续运行）

## 2. 训练状态（2026-11-11）

| 方法 | 状态 | 预算 | 备注 |
|---|---|---|---|
| MA-D3QN | 运行中 | 10000 集 | ep 5300/10000（53%），仍 20/20；eval 间歇 feasible |
| TA-SharedTD3-BC | 运行中 | 3000 集 | ep 1 feasible=True，warm-start 加载成功 |
| TA-MAPPO | 运行中 | 3000 集 | ep 1 feasible=True，warm-start 加载成功 |

## 3. 旧标准版结果（已作废）

旧 SharedTD3-BC（2100 集）与 MAPPO（4050 集）未学会、已停止，作为负结果留存于
`results/formal_win/win-formal-u2-s42-{sharedtd3bc,mappo}`；
不得与新 TA 版混为同一方法。

## 4. 执行合规

- 命令严格按 `docs/teacher-anchored-marl.md` §5，seed 42、horizon 2000、eval interval 50。
- 代码基于 `a56e8f7`；相关测试 34 项全通过。
- 环境 `.venv`：Python 3.13.14 + torch 2.15.0+cpu。
