# Win–Mac 数据同步（win-mac.md）

> 维护者：Windows 机（feat/win）。本文件记录 Windows 侧实验执行状态，供 Mac 侧读取诊断。
> 更新：2026-11-13

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

Windows 负责的规模探针：

- ✅ TA-SharedTD3-BC U=2：`win-formal-u2-s42-sharedtd3bc-ta`（通过验收）
- ✅ TA-MAPPO U=2：`win-formal-u2-s42-mappo-ta`（通过验收）
- ✅ TA-SharedTD3-BC U=3：`win-formal-u3-s42-sharedtd3bc-ta`（完成，rejected，见 §7）
- ✅ TA-MAPPO U=3：`win-formal-u3-s42-mappo-ta`（完成，rejected，见 §7）
- 🔄 TA-SharedTD3-BC / TA-MAPPO U=4：转交其他机器训练，本机不跑
- 🔄 MA-D3QN U=3/U=4：转交其他机器训练，本机不跑

## 2. 训练状态（2026-11-13）

| 方法 | 状态 | 预算 | 备注 |
|---|---|---|---|
| TA-SharedTD3-BC U=2 | ✅ 完成 | 3000 集 | 通过验收 |
| TA-MAPPO U=2 | ✅ 完成 | 3000 集 | 通过验收 |
| TA-SharedTD3-BC U=3 | ✅ 完成 | 3000 集 | rejected：collision + 连续中断超限 |
| TA-MAPPO U=3 | ✅ 完成 | 3000 集 | rejected：collision + 连续中断超限 |
| TA 两方法 U=4 | 转交其他机器 | 3000 集 ×2 | 本机不跑 |
| MA-D3QN U=2 | 已取消 | 10000 集 | Mac 侧为准 |
| MA-D3QN U=3/U=4 | 转交其他机器 | 3000 集 ×2 | 本机不跑 |

## 3. 旧标准版结果（已作废）

旧 SharedTD3-BC（2100 集）与 MAPPO（4050 集）未学会、已停止，作为负结果留存；
不得与新 TA 版混为同一方法。

## 4. U=2 最终结果（best_constraint 确定性回放，2026-11-12）

| 方法 | 可行 | T (s) | D (s) | 巡检 | 上传 | 最大连续中断 (s) | 碰撞/越界/扇区冲突 |
|---|---:|---:|---:|---:|---:|---|---|
| TA-SharedTD3-BC | 是 | 386.0 | 317.0 | 20/20 | 20/20 | 15.0 / 18.5 | 0 / 0 / 0 |
| TA-MAPPO | 是 | 392.5 | 319.0 | 20/20 | 20/20 | 15.0 / 18.0 | 0 / 0 / 0 |

验收门槛全部满足：20/20 巡检、20/20 上传、两架 UAV 到达终点；每架最大连续
中断 ≤ 20s；零碰撞/越界/提前进入/扇区冲突；`best_constraint.pth` 确定性回放可行。

## 5. 密封交接包（U=2）

- 路径：`paper_handoff/2026-11-12/`
  - `ta_marl_20261112.tar.gz`（9.84 MB，含两个正式 run 的全部产物）
  - `SHA256SUMS`（tarball SHA256 = `d4b3abbd9e2541577ae50d8d8772af8fcf5d6c4293fd3b2e91e9e2eddc03b1a5`）
  - `README.md`（结果表、验收门槛、论文使用边界）
- 校验：`shasum -a 256 -c paper_handoff/2026-11-12/SHA256SUMS`

## 6. 执行合规

- 命令严格按 `docs/teacher-anchored-scale-handoff.md`，seed 42、horizon 2000、eval interval 50。
- 代码基于 `a56e8f7`；相关测试 34 项全通过。
- 环境 `.venv`：Python 3.13.14 + torch 2.15.0+cpu。

## 7. U=3 规模探针结果（rejected，2026-11-13）

两个 U=3 探针均完成 3000 集，但 `best_constraint.pth` 确定性回放不可行：

| 方法 | 可行 | T (s) | D (s) | 巡检/上传 | 碰撞 | 越界/扇区冲突 | 最大连续中断 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TA-SharedTD3-BC | 否 | 745.5 | 935.5 | 30/30 | 5 | 0 / 0 | 37.5 / 10.5 / 52.5 |
| TA-MAPPO | 否 | 748 | 936 | 30/30 | 5 | 0 / 0 | 37.5 / 10.5 / 52 |

失败原因：`collision`（要求 0，实际 5）+ `contiguous_outage_excess`
（要求 ≤20s/机，实际 37.5 / 52.5）。全程 61 次评估 `eval_feasible` 均为 False。
按契约以 rejected 封存，数据完整保留。

## 8. U=3 完整数据与轨迹导出（本次同步内容）

- 完整 run：`results/formal_win/win-formal-u3-s42-{sharedtd3bc,mappo}-ta/`
  （`summary/manifest/resolved_config/records.json`、`training_curve.npz`、
  `best/best_constraint/best_completion/last.pth` 共 8 项齐全）
- 日志：`results/formal_win_logs/win-formal-u3-s42-{sharedtd3bc,mappo}-ta.log`
- 逐步轨迹导出脚本：`experiment/export_exclusive_trajectory.py`
- 逐步轨迹产物：`results/formal_win_analysis/win-formal-u3-s42-*/trajectory_best_constraint.{json,npz}`
  - JSON 含 1491/1496 步逐步 frame + 端到端 metrics；
  - NPZ 含 12 个数组（positions/actions/buffers/energy/sectors/efficiency/
    outage/contiguous outage/uploaded/route_index/done/rewards）；
  - 导出口径与 `summary.json` 的 `best_metrics` 逐项一致（复用训练脚本同一套
    teacher-blend + 安全投影逻辑）。