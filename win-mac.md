# Win–Mac 数据同步（win-mac.md）

> 维护者：Windows 机（feat/win）。本文件记录 Windows 侧实验执行状态，供 Mac 侧读取诊断。
> 更新：2026-11-11

## 0. 本机旧版 MA-D3QN 终止同步（2026-11-11，conda pytorch 3.10 / c21a5c7）

> 本机（conda pytorch 3.10 + torch 2.8，代码 c21a5c7）此前并行跑的旧版
> SharedTD3-BC / MAPPO / MA-D3QN 已全部停止：

- SharedTD3-BC 旧版：~5450 集停止，未学会（unguided visited 3~8/20）。
- MAPPO 旧版：10000 集跑完，不可行（best_visited 5/20，负结果）。
- **MA-D3QN 旧版：ep 8750/10000（87.5%）手动终止**。观察：
  - 任务完成稳定：unguided 回合 steps≈1143~1218、visited/uploaded 20/20；
  - 完整可行性 `eval_feasible` 后期衰落：ep 6800–7000 为 5/5 True（峰值）→
    之后稀疏 True → ep 8350–8750 连续 9 个全 False。
  - 疑似部分为旧 checkpoint 排序 bug（c21a5c7 会把「停摆策略」选为
    `best_constraint.pth`），该 bug 已在 `a56e8f7`（TA 公共环境修复）中修正。

**建议（待 Mac 决策）**：MA-D3QN 是最强学习型方法，但本机旧版结果出自旧代码、
不可采信；建议纳入「用新代码 `a56e8f7` 重跑」清单，与 TA-SharedTD3-BC /
TA-MAPPO 一起，并明确重跑环境（conda 3.10 还是 `.venv` 3.13）。

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
