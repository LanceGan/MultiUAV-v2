# Win–Mac 数据同步（win-mac.md）

> 维护者：Windows 机（feat/win）。本文件记录 Windows 侧实验执行状态，供 Mac 侧读取诊断。
> 更新：2026-11-12

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

- ✅ TA-SharedTD3-BC：`win-formal-u2-s42-sharedtd3bc-ta`（3000 集）——已完成并通过验收
- ✅ TA-MAPPO：`win-formal-u2-s42-mappo-ta`（3000 集）——已完成并通过验收
- 🔄 MA-D3QN U=3/U=4 规模探针：已转交另一台机器训练（见 §5），本机不重复跑。

## 2. 训练状态（2026-11-12）

| 方法 | 状态 | 预算 | 备注 |
|---|---|---|---|
| TA-SharedTD3-BC | ✅ 完成 | 3000 集 | 全程 20/20，eval_feasible 连续 True |
| TA-MAPPO | ✅ 完成 | 3000 集 | 全程 20/20，eval_feasible=True |
| MA-D3QN U=2 | 已取消 | 10000 集 | Mac 侧为准 |
| MA-D3QN U=3/U=4 | 转交另一台机器 | 3000 集 ×2 | 本机不跑，结果由该机器同步 |

## 3. 旧标准版结果（已作废）

旧 SharedTD3-BC（2100 集）与 MAPPO（4050 集）未学会、已停止，作为负结果留存；
不得与新 TA 版混为同一方法。

## 4. 最终结果（best_constraint 确定性回放，2026-11-12）

| 方法 | 可行 | T (s) | D (s) | 巡检 | 上传 | 最大连续中断 (s) | 碰撞/越界/扇区冲突 |
|---|---:|---:|---:|---:|---:|---|---|
| TA-SharedTD3-BC | 是 | 386.0 | 317.0 | 20/20 | 20/20 | 15.0 / 18.5 | 0 / 0 / 0 |
| TA-MAPPO | 是 | 392.5 | 319.0 | 20/20 | 20/20 | 15.0 / 18.0 | 0 / 0 / 0 |

验收门槛全部满足：20/20 巡检、20/20 上传、两架 UAV 到达终点；每架最大连续
中断 ≤ 20s；零碰撞/越界/提前进入/扇区冲突；`best_constraint.pth` 确定性回放可行。

## 5. 密封交接包

- 路径：`paper_handoff/2026-11-12/`
  - `ta_marl_20261112.tar.gz`（9.84 MB，含两个正式 run 的全部产物）
  - `SHA256SUMS`（tarball SHA256 = `d4b3abbd9e2541577ae50d8d8772af8fcf5d6c4293fd3b2e91e9e2eddc03b1a5`）
  - `README.md`（结果表、验收门槛、论文使用边界）
- 校验：`shasum -a 256 -c paper_handoff/2026-11-12/SHA256SUMS`

## 6. 执行合规

- 命令严格按 `docs/teacher-anchored-marl.md` §5，seed 42、horizon 2000、eval interval 50。
- 代码基于 `a56e8f7`；相关测试 34 项全通过。
- 环境 `.venv`：Python 3.13.14 + torch 2.15.0+cpu。