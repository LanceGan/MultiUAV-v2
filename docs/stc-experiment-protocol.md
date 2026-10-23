# STC 实验协议与闸门

## 命令

- 冒烟：`python experiment/run_stc_main.py --smoke --run-id <id>`
- 干跑（打印矩阵）：`python experiment/run_stc_main.py --dry-run --map-seeds 42 --task-seeds 42 --method-seeds 0,1,2,3,4`
- 主实验：`python experiment/run_stc_main.py --map-seeds 42,43,44 --task-seeds 42,43,44 --method-seeds 0,1,2,3,4`
- 消融：`python experiment/run_stc_ablation.py --dry-run`
- 规模：`python experiment/run_stc_scale.py --dry-run`
- 出图：`python plots/plot_stc_results.py --runs results/runs/<run-id>`

## 四类 seed

- `map_seed`：地形/建筑/无线图
- `task_seed`：巡检点与数据量
- `method_seed`：ALNS/启发式搜索随机性
- `train_seed`：仅学习型基线

训练/验证/测试 seed 必须互斥。

## 闸门

- Gate 0 评价器守恒测试通过（带宽/缓存/扇区/时序）
- Gate 1 至少一种通信感知航段相对 Straight-Hold 稳定改善
- Gate 2 STC-ALNS 相对普通 LNS 稳定改善（配对 CI 不跨零）
- Gate 3 至少两项消融机制产生可复现贡献
- Gate 4 所有表图来自冻结 run、代码/配置/地图/命令哈希一致

## 结果目录

所有正式结果写入 `results/runs/<run_id>/`，含 manifest.json（git/版本/配置/hash）。
共享文件名禁止覆盖。
