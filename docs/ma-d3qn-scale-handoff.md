# MA-D3QN U=3/U=4 Single-Seed Scale Probes

更新：2026-11-11

## 1. 决策

论文将 U=3/U=4 定位为 **single-seed scale probes**，不要求训练过程始终
`eval_feasible=True`。因此，U=2 已提供一个通过硬约束的 `best_constraint.pth`
后，可以启动 U=3/U=4 MA-D3QN 训练。

验收对象是封存的最佳检查点，不是最后策略；但必须同时报告完整评估历史中的
可行次数、失败原因和后期漂移。

## 2. 固定设置

| 项目 | U=3 | U=4 |
|---|---|---|
| 任务数 | 30 | 40 |
| 区域 | 5 km | 6 km |
| seed | 42 | 42 |
| episodes | 3000 | 3000 |
| horizon | 2000 | 2000 |
| eval interval | 50 | 50 |
| MapGuide probability | 0.5 constant | 0.5 constant |
| safety outage threshold | 0.2 | 0.2 |
| demo fraction | 0.25 | 0.25 |
| DQfD margin | 0.8 | 0.8 |
| n-step | 10 | 10 |
| action repeat | 2 | 2 |
| init checkpoint | none | none |

U=2/U=3/U=4 的 local observation 维度分别为 17/19/21，因此 U=3/U=4
必须从零训练，不能加载 U=2 checkpoint。

## 3. 环境与资源

使用 Windows 锁定环境，不使用旧 conda 环境：

```text
Python 3.13.14
torch 2.15.0+cpu
numpy 2.5.4
```

若 TA-SharedTD3-BC / TA-MAPPO 仍在运行，U=3/U=4 各限制 2 个计算线程；
若两个 TA 任务已经结束，可各使用 4 个线程。

## 4. U=3 命令

```powershell
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"

.\.venv\Scripts\python.exe experiment\run_exclusive_marl_pilot.py `
  --algorithm MA-D3QN `
  --uav-num 3 `
  --episodes 3000 `
  --horizon-steps 2000 `
  --eval-interval 50 `
  --seed 42 `
  --guide-prob 0.5 `
  --guide-min-prob 0.5 `
  --demo-fraction 0.25 `
  --demo-margin 0.8 `
  --actor-lr 1e-4 `
  --critic-lr 1e-3 `
  --safety-outage-threshold 0.2 `
  --n-step 10 `
  --action-repeat 2 `
  --pretrain-demo-episodes 0 `
  --pretrain-updates 0 `
  --routes-json configs\exclusive\u3_s42_4d_ga_routes.json `
  --run-root results\formal_win `
  --run-id win-formal-u3-s42-ma-d3qn
```

## 5. U=4 命令

```powershell
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"

.\.venv\Scripts\python.exe experiment\run_exclusive_marl_pilot.py `
  --algorithm MA-D3QN `
  --uav-num 4 `
  --episodes 3000 `
  --horizon-steps 2000 `
  --eval-interval 50 `
  --seed 42 `
  --guide-prob 0.5 `
  --guide-min-prob 0.5 `
  --demo-fraction 0.25 `
  --demo-margin 0.8 `
  --actor-lr 1e-4 `
  --critic-lr 1e-3 `
  --safety-outage-threshold 0.2 `
  --n-step 10 `
  --action-repeat 2 `
  --pretrain-demo-episodes 0 `
  --pretrain-updates 0 `
  --routes-json configs\exclusive\u4_s42_4d_ga_routes.json `
  --run-root results\formal_win `
  --run-id win-formal-u4-s42-ma-d3qn
```

两个训练可以在独立 PowerShell 窗口并行运行。若观察到明显 CPU 饥饿、episode
吞吐显著下降或内存压力，保持 U=3 继续运行，并暂停/重启 U=4，不得降低训练
步数或 horizon。

## 6. 验收口径

最佳检查点必须满足：

- 全部巡检任务完成；
- 全部任务数据上传完成；
- 所有 UAV 到达终点；
- 零碰撞、零越界、零提前进入；
- 每架 UAV 最大连续中断不超过 20 s；
- 使用 `best_constraint.pth` 做最终确定性回放。

训练过程中的其它评估点可以失败。报告时必须同时给出：

- 总评估次数与 feasible 次数；
- 各失败原因；
- `best_constraint.pth` 的 T/D/中断；
- `last.pth` 与最佳检查点是否存在性能差异。

如果最佳检查点仍不可行，该规模探针以 rejected 结果封存；不得增加 seed、
改超参或反复重跑直到可行。

## 7. 产物

每个 run 必须交接：

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

图与论文表只从 sealed run 生成，数值不得手工转录。
