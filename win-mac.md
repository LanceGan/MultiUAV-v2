# Win–Mac 数据同步（win-mac.md）

> 维护者：Windows 机（feat/win）。本文件记录 Windows 侧实验执行状态，供 Mac 侧读取诊断。
> 更新：2026-11-11

## 1. 当前任务

按 `docs/exclusive-experiment-plan.md` §8/§9，Windows 负责全部学习型正式训练：
SharedTD3-BC、MAPPO、MA-D3QN（各 10000 集 × horizon 2000，seed 42，warm-start）。

## 2. 训练状态（2026-11-11）

| 方法 | 状态 | 进度 | 关键结论 |
|---|---|---|---|
| MA-D3QN | **运行中** | 3300/10000（33%） | 唯一真正在学习；确定性 eval 多次 feasible=True（ep 3050/3150/3250/3300）；best.pth / best_completion.pth 持续更新 |
| SharedTD3-BC | **已停** | 2100/10000（21%） | 学习策略发散：非引导集 0/20~1/20、2000 步截断、reward -55~-180；best.pth 自第 1 集后 18h 未更新 |
| MAPPO | **已停** | 4050/10000（40.5%） | 同样发散：非引导集 0/20 震荡；best.pth 停 8.5h |

## 3. 跨机一致性

Mac 独立实验同样发现 SharedTD3 / MAPPO 未学会、MA-D3QN 效果最佳，与 Windows 结论一致。
建议论文以 MA-D3QN 为学习型主方法，SharedTD3-BC / MAPPO 作为诚实负结果。

## 4. 执行合规

- 命令严格按 §9，单 seed 42，无超参改动；三份 warm-start `best_constraint.pth` 均正确加载。
- 环境 `.venv`：Python 3.13.14 + torch 2.15.0+cpu，与 requirements.lock.txt 一致。
- GPU 未使用：锁定 torch 2.15.0 无 CUDA 轮子；另以 2.14.1+cu130 实测 GPU 不加速。

## 5. 产物与交接

- 运行日志：`results/formal_win_logs/win-formal-u2-s42-{ma-d3qn,mappo,sharedtd3bc}.log`
- 终止记录：`results/formal_win_logs/STOPPED-sharedtd3-mappo.md`
- 正式结果待 MA-D3QN 跑完后按 §10 密封，经 `paper_handoff/` + SHA256 交接。
