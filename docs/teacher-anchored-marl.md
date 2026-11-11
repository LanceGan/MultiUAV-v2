# Teacher-Anchored SharedTD3-BC and MAPPO

更新：2026-11-11

## 1. 目的

原始 SharedTD3-BC 和 MAPPO 在 50/200 集短训中曾出现短暂可行检查点，
但继续训练后自主策略漂移。修复版保留算法主体，同时加入三个公共机制：

1. 持续教师锚定损失；
2. 固定比例的教师动作残差混合；
3. 上传保护投影，禁止 buffer 未清空时进入下一个 10 m 服务区。

这两个方法必须写作：

- Teacher-Anchored SharedTD3-BC（TA-SharedTD3-BC）；
- Teacher-Anchored MAPPO（TA-MAPPO）。

不能再表述为“标准 SharedTD3-BC”或“标准 MAPPO”。

## 2. 算法变化

### TA-SharedTD3-BC

- 修复旧 BC loss 的 `[speed, theta]` 顺序错误；
- demo fraction：0.5；
- BC weight：固定 1.0，不再衰减；
- persistent teacher-anchor weight：10.0；
- teacher action blend：0.5；
- MapGuide episode probability：0.5 恒定；
- 评估关闭 MapGuide，仅保留教师残差锚定、上传保护和无线电安全投影。

### TA-MAPPO

- 新增当前状态对应的固定 MapGuide 动作标签；
- PPO actor loss 增加均值动作教师损失；
- teacher-anchor weight：10.0；
- teacher action blend：0.5；
- MapGuide episode probability：0.5 恒定；
- 评估关闭 MapGuide，仅保留教师残差锚定、上传保护和无线电安全投影。

教师动作在进入损失或动作混合前先经过安全投影，但不重复计入安全投影统计。

## 3. 公共环境修复

`ExclusiveSectorFixedRouteEnv.project_safe_actions()` 现在同时执行：

- 越界保护；
- 无可信无线电扇区保护；
- buffer 未清空时禁止进入下一服务圆盘的上传保护。

最佳检查点排序改为：

```text
feasible
→ visited tasks
→ uploaded tasks
→ terminal completion
→ outage / collision / early-arrival violations
→ makespan / delay
```

因此不会再把“未移动、没有违规但只完成 2 个任务”的停摆策略选成
`best_constraint.pth`。

## 4. Mac 100 集稳定性短测

场景：U=2、s42、固定 4D + GA-EQTSP 路线、horizon 2000，
从已验收 warm start 启动。

| 方法 | 评估次数 | 可行次数 | best T | best D | 最大连续中断 | 碰撞 |
|---|---:|---:|---:|---:|---|---:|
| TA-SharedTD3-BC | 11 | 11/11 | 386.0 s | 317.0 s | 15.0 / 18.5 s | 0 |
| TA-MAPPO | 11 | 11/11 | 438.0 s | 318.0 s | 16.0 / 18.5 s | 0 |

短测 run：

```text
results/runs_teacher_anchor_v8/teacher-anchor-v8-sharedtd3bc-u2-s42-e100-s2000
results/runs_teacher_anchor_v8/teacher-anchor-v8-mappo-u2-s42-e100-s2000
```

两个 run 的 T 范围分别为：

- TA-SharedTD3-BC：386.0 s，D=317.0–318.0 s；
- TA-MAPPO：437.0–459.0 s，D=309.0–319.5 s。

## 5. Windows 正式训练

统一预算：3000 集、seed 42、horizon 2000、eval interval 50。

### TA-SharedTD3-BC

```powershell
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"

.\.venv\Scripts\python.exe experiment\run_exclusive_marl_pilot.py `
  --algorithm SharedTD3 --episodes 3000 --horizon-steps 2000 `
  --eval-interval 50 --guide-prob 0.5 --guide-min-prob 0.5 `
  --bc-weight 1.0 --bc-min-weight 1.0 --bc-decay-episodes 600 `
  --demo-fraction 0.5 --teacher-anchor-weight 10.0 `
  --teacher-action-blend 0.5 `
  --actor-lr 1e-5 --critic-lr 1e-3 `
  --safety-outage-threshold 0.2 `
  --init-checkpoint results\runs_short_v15\short-v15-u2-s42-sharedtd3bc-safe020-e200-s2000\best_constraint.pth `
  --routes-json configs\exclusive\u2_s42_4d_ga_routes.json `
  --run-root results\formal_win `
  --run-id win-formal-u2-s42-sharedtd3bc-ta
```

### TA-MAPPO

```powershell
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"

.\.venv\Scripts\python.exe experiment\run_exclusive_marl_pilot.py `
  --algorithm MAPPO --episodes 3000 --horizon-steps 2000 `
  --eval-interval 50 --guide-prob 0.5 --guide-min-prob 0.5 `
  --teacher-anchor-weight 10.0 --teacher-action-blend 0.5 `
  --actor-lr 1e-4 --critic-lr 1e-3 `
  --safety-outage-threshold 0.2 `
  --init-checkpoint results\runs_short_v18\short-v18-mappo-safe-e200-s2000\best_constraint.pth `
  --routes-json configs\exclusive\u2_s42_4d_ga_routes.json `
  --run-root results\formal_win `
  --run-id win-formal-u2-s42-mappo-ta
```

两台并行任务各限制 4 个线程；也可顺序运行以降低资源争用。

## 6. 验收门槛

正式结果只有在以下条件全部满足时才可接收：

- 20/20 巡检；
- 20/20 上传；
- 两架 UAV 均到达终点；
- 每架 UAV 最大连续中断不超过 20 s；
- 零碰撞、零越界、零提前进入；
- 使用 `best_constraint.pth` 做确定性回放；
- 完整 manifest、配置、训练曲线和指标均封存。

正式结果不得与旧标准 SharedTD3-BC/MAPPO 结果混为同一方法。
