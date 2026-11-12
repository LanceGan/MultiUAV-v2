# TA-SharedTD3-BC / TA-MAPPO U=3/U=4 Scale Handoff

更新：2026-11-12

## 1. 决策

U=2 正式结果已经通过验收：

| 方法 | 可行评估 | best T | best D | 最大连续中断 |
|---|---:|---:|---:|---|
| TA-SharedTD3-BC | 61/61 | 386.0 s | 317.0 s | 15.0 / 18.5 s |
| TA-MAPPO | 57/61 | 392.5 s | 319.0 s | 15.0 / 18.0 s |

论文将该组实验定位为 single-seed scale probes。允许启动 U=3/U=4，
但正式验收仍以 `best_constraint.pth` 的确定性回放为准，并同时报告
训练过程的可行性波动和失败原因。

## 2. 强制边界

- U=3/U=4 从零训练，不加载 U=2 checkpoint；
- U=2/U=3/U=4 的 local observation 维度分别为 17/19/21；
- 每个方法、每个规模只训练一次；
- seed 固定为 42；
- 不增加 method seed，不根据结果重新调参；
- 方法名称固定为 TA-SharedTD3-BC 和 TA-MAPPO。

## 3. 环境与并发

使用 Windows `.venv`：

```text
Python 3.13.14
torch 2.15.0+cpu
```

建议先并行运行两个 U=3 任务，每个进程 4 个线程。U=3 完成一个后，
再启动对应的 U=4 任务，避免四个训练进程同时争抢 CPU。

## 4. U=3 TA-SharedTD3-BC

```powershell
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"

.\.venv\Scripts\python.exe experiment\run_exclusive_marl_pilot.py `
  --algorithm SharedTD3 `
  --uav-num 3 `
  --episodes 3000 `
  --horizon-steps 2000 `
  --eval-interval 50 `
  --seed 42 `
  --guide-prob 0.5 `
  --guide-min-prob 0.5 `
  --bc-weight 1.0 `
  --bc-min-weight 1.0 `
  --bc-decay-episodes 600 `
  --demo-fraction 0.5 `
  --teacher-anchor-weight 10.0 `
  --teacher-action-blend 0.5 `
  --actor-lr 1e-5 `
  --critic-lr 1e-3 `
  --safety-outage-threshold 0.2 `
  --routes-json configs\exclusive\u3_s42_4d_ga_routes.json `
  --run-root results\formal_win `
  --run-id win-formal-u3-s42-sharedtd3bc-ta
```

## 5. U=3 TA-MAPPO

```powershell
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"

.\.venv\Scripts\python.exe experiment\run_exclusive_marl_pilot.py `
  --algorithm MAPPO `
  --uav-num 3 `
  --episodes 3000 `
  --horizon-steps 2000 `
  --eval-interval 50 `
  --seed 42 `
  --guide-prob 0.5 `
  --guide-min-prob 0.5 `
  --teacher-anchor-weight 10.0 `
  --teacher-action-blend 0.5 `
  --actor-lr 1e-4 `
  --critic-lr 1e-3 `
  --safety-outage-threshold 0.2 `
  --routes-json configs\exclusive\u3_s42_4d_ga_routes.json `
  --run-root results\formal_win `
  --run-id win-formal-u3-s42-mappo-ta
```

## 6. U=4 TA-SharedTD3-BC

```powershell
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"

.\.venv\Scripts\python.exe experiment\run_exclusive_marl_pilot.py `
  --algorithm SharedTD3 `
  --uav-num 4 `
  --episodes 3000 `
  --horizon-steps 2000 `
  --eval-interval 50 `
  --seed 42 `
  --guide-prob 0.5 `
  --guide-min-prob 0.5 `
  --bc-weight 1.0 `
  --bc-min-weight 1.0 `
  --bc-decay-episodes 600 `
  --demo-fraction 0.5 `
  --teacher-anchor-weight 10.0 `
  --teacher-action-blend 0.5 `
  --actor-lr 1e-5 `
  --critic-lr 1e-3 `
  --safety-outage-threshold 0.2 `
  --routes-json configs\exclusive\u4_s42_4d_ga_routes.json `
  --run-root results\formal_win `
  --run-id win-formal-u4-s42-sharedtd3bc-ta
```

## 7. U=4 TA-MAPPO

```powershell
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"

.\.venv\Scripts\python.exe experiment\run_exclusive_marl_pilot.py `
  --algorithm MAPPO `
  --uav-num 4 `
  --episodes 3000 `
  --horizon-steps 2000 `
  --eval-interval 50 `
  --seed 42 `
  --guide-prob 0.5 `
  --guide-min-prob 0.5 `
  --teacher-anchor-weight 10.0 `
  --teacher-action-blend 0.5 `
  --actor-lr 1e-4 `
  --critic-lr 1e-3 `
  --safety-outage-threshold 0.2 `
  --routes-json configs\exclusive\u4_s42_4d_ga_routes.json `
  --run-root results\formal_win `
  --run-id win-formal-u4-s42-mappo-ta
```

## 8. 验收口径

正式检查点必须满足：

- 全部巡检任务完成；
- 全部任务数据上传完成；
- 所有 UAV 到达终点；
- 零碰撞、零越界、零提前进入、零扇区冲突；
- 每架 UAV 最大连续中断不超过 20 s；
- 使用 `best_constraint.pth` 做最终确定性回放。

每个 run 必须保留并交接：

```text
summary.json
manifest.json
resolved_config.json
records.json
training_curve.npz
best_constraint.pth
best_completion.pth
last.pth
```

若最佳检查点不可行，该规模探针以 rejected 结果封存，不得改超参或增加
seed 后重跑。
