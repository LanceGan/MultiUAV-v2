# MultiUAV 巡检 (Multi-UAV Inspection) — 阶段一重构

> ⚠️ **最终方法 STC-ALNS（2026-10-25）**：本论文的正式通信感知巡检方法在
> `segment/stc/`（时空扇区冲突感知 ALNS）。实验协议与闸门见
> [`docs/stc-experiment-protocol.md`](docs/stc-experiment-protocol.md)，整体思路见
> [`仿真实验.md`](仿真实验.md)。下方「三层流水线」是旧设计的**历史记录**，保留供回溯。

面向**真实可复现**的多机（Multi-UAV）巡检研究重构。本仓库在单机已验证的通信模型
（radio map，3GPP RMa 信道）基础上，加入多机任务分配、航迹规划、序列决策与
端到端实验编排，目标是让每一处实验脚本都能一键复现论文结果。

三层流水线（对应论文的三个层次）：

1. **任务分配（assignment）** — 将 K 个巡检点聚成 N 类（naive / balanced / jia / 4D 通信感知），
2. **序列决策（sequence）** — 每架 UAV 的访问顺序（ga_euclidean / pso / aco / nn_2opt / ga_eqtsp），
3. **航迹规划（trajectory）** — 通信感知的共享 TD3 强化学习（对比非通信感知贪心基线），

再加上统一评估（`experiment/evaluate.py`）与自动绘图（`plots/plot_results.py`）。

## 运行环境

- Python 解释器：`F:/anaconda3/envs/pytorch/python.exe`（conda `pytorch` 环境，
  含 numpy / scipy / torch / matplotlib / pytest）。
- 依赖见 `requirements.txt`：

  ```
  numpy
  scipy
  torch
  matplotlib
  pytest
  ```

- 从项目根目录运行（`radio_map` 等包需通过根目录 `sys.path` 解析）：

  ```bash
  cd f:/Projects/Py/MultiUAV_New
  ```

## 目录结构

```
radio_map/        通信模型（从 SingleUAV 验证代码迁移，只读）
  radio_map_A2G.py   地面基站 → 空中 UAV（A2G）信道，getPointDateRate()
  radio_map_G2A.py   空中 UAV → 地面用户（G2A）信道，getPointMiniOutage()
  rural_world.py     乡村宏蜂窝（RMa）地形/建筑生成
  entity.py          UAV / User 数据类
common/           公共工具
  config.py          场景配置（从 configs/*.json 读取）+ ENERGY_PARAMS + get_data_sizes
  metrics.py         能耗 / makespan 等基础度量
configs/          场景参数（JSON，可复现）
  scenario_uav2.json / scenario_uav3.json / scenario_uav4.json
assignment/       层 1：任务分配（聚类）
sequence/         层 2：序列决策（含 radio_path 边代价采样）
trajectory/       层 3：航迹规划（env / td3 / train / test / replay）
segment/          新设计（STC-ALNS，最终方法）
  segment/stc/     同步评价器 / 冲突归因 / STC-ALNS / 六系统基线 / 精确基准 / 统计 / 溯源
  segment/sac.py   SAC 局部轨迹（学习型基线执行器）
experiment/       实验编排（run_assign / run_seq / run_traj / run_end2end / run_all / evaluate
                  及 STC 的 run_stc_segment_gate / evaluator_audit / main / ablation / scale / gates）
plots/            自动绘图（plot_results.py + plot_stc_results.py 只读冻结 run）
docs/             设计/计划（specs、plans、stc-experiment-protocol.md）
results/datas/    用户位置数据（Users_20/30/40.txt，x,y,h 三列，单位 100m）
results/logs/     运行产出（*.metrics.json / *.npz / *.log，git 忽略）
results/figs/     论文图（*.png，git 忽略）
results/models/   训练模型（git 忽略）
results/runs/     STC 冻结 run（manifest.json + 密封产物，git 忽略）
tests/            pytest 测试
```

## 一条命令复现全部结果

```bash
cd f:/Projects/Py/MultiUAV_New
F:/anaconda3/envs/pytorch/python.exe experiment/run_all.py --uav_nums 2 3 4
```

这会按顺序执行五步，产出所有指标 JSON 与图（种子固定为 42，结果可复现）：

1. `run_assign`   → `results/logs/assign_<method>_uav<N>.json`
2. `run_seq`      → `results/logs/seq_<method>_uav<N>.json`
3. `run_traj`     → `results/logs/traj_<baseline|proposed>_uav<N>.json`（含 TD3 训练）
4. `run_end2end`  → `results/logs/end2end_<baseline|proposed>_uav<N>.json`
5. `plot_results` → `results/figs/*.png`

只跑 N=2 冒烟（更快）：

```bash
F:/anaconda3/envs/pytorch/python.exe experiment/run_all.py --uav_nums 2 --total_episode 200
```

其他 `run_all.py` 参数：

```
--uav_nums 2 3 4        空格分隔的 UAV 数量（默认 2 3 4）
--total_episode 300     TD3 训练回合数（默认 300）
--seed 42               随机种子（默认 42）
--log_dir / --figs_dir  覆盖默认的 logs / figs 输出目录
```

> 若想完全清空旧结果再复现：先删除 `results/logs`、`results/figs`（和
> `results/models`），再运行上述命令。三个目录均在 `.gitignore` 中，不影响仓库。

## 逐层运行

四层均可单独运行（`--uav_nums` 为逗号分隔）：

```bash
# 层 1：任务分配（四种聚类方法）
F:/anaconda3/envs/pytorch/python.exe experiment/run_assign.py --uav_nums 2,3,4

# 层 2：序列决策（五种方法，固定 4D 聚类）
F:/anaconda3/envs/pytorch/python.exe experiment/run_seq.py --uav_nums 2,3,4

# 层 3：航迹（Baseline 贪心 vs Proposed TD3）
F:/anaconda3/envs/pytorch/python.exe experiment/run_traj.py --uav_nums 2,3,4 --total_episode 300

# 端到端：全基线 vs 全提议
F:/anaconda3/envs/pytorch/python.exe experiment/run_end2end.py --uav_nums 2,3,4 --total_episode 300
```

单独训练 / 回放（Task 7/8 的独立入口）：

```bash
F:/anaconda3/envs/pytorch/python.exe trajectory/train.py --uav_num 2 --total_episode 300 --model_subdir Ours
F:/anaconda3/envs/pytorch/python.exe trajectory/test.py --uav_num 2 --model_subdir Ours
F:/anaconda3/envs/pytorch/python.exe experiment/evaluate.py --run_id UAV_2_Ours_best
```

单独绘图：

```bash
F:/anaconda3/envs/pytorch/python.exe plots/plot_results.py
```

## 测试

```bash
cd f:/Projects/Py/MultiUAV_New
F:/anaconda3/envs/pytorch/python.exe -m pytest tests/ -v
```

覆盖：`test_radio_map` / `test_metrics` / `test_clustering` / `test_radio_path` /
`test_sequence` / `test_env`。

## 场景配置

场景参数集中在 `configs/scenario_uav<N>.json`（`common/config.py` 的
`get_scenario(uav_num)` 负责读取），包括：

- `user_num` / `length` / `width`：巡检点数与场景规模（N=2/3/4 → 20/30/40 点，40×40）
- `v_max` / `delta_t` / `tau_max` / `T_out_max`：速度、步长与时延 QoS 上限
- `lambda_E` / `lambda_out` / `lambda_tau`：能耗 / 中断 / 时延权重
- `bs_loc` / `ini_loc` / `end_loc`：基站位置 / 起点 / 终点

`get_scenario(uav_num)` 的返回字典接口保持不变（`bs_loc` / `ini_loc` / `end_loc`
为 numpy 数组，`uav_num` / `seed` 由调用注入），因此既有代码与测试不受影响。

> **注意（权重未调优）**：阶段一的 4D 聚类权重 `W4D=(1.0,0.4,0.4,0.5)` 与
> `λ_E=0.001, λ_out=λ_τ=1.0` 为**占位/未调优**的硬编码初值，论文写作前须按
> §6.1/6.2 的端到端目标 + 敏感性分析进行调优。
