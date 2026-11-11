# Mac–Windows 双机协同说明（mac-win.md）

> 读者：Windows 机器上的开发者/协作 agent。本文档是两台机器分工的**权威契约**，
> 由 Mac 侧维护（分支 `feat/mac`）；Windows 侧在 `feat/win` 上工作。
> 更新时间：2026-11-11。

## 1. 机器与分支

| 机器 | 平台 | 分支 | 角色 |
|---|---|---|---|
| Mac（参考机） | macOS 15.7 arm64, Apple Silicon | `feat/mac` | 执行全部正式实验（U=2/3/4 及其补充）；文档主干 |
| Windows 新机 | Windows 11, i5-12500H（4P+8E 12 核） | `feat/win` | 论文写作 + 指定的 U=2 teacher-anchored MADRL 正式训练 |

两分支**互不合并、互不 rebase**，只向 `origin` 各自推送；汇合时机与方式由项目负责人决定。

## 2. 硬规则：数值世界隔离（最重要）

**同一实验族的所有 run 必须出自同一台机器。** 跨平台浮点不逐位一致（实测：同一天线
零陷角 Windows x86 与 macOS arm64 差 0.55 dB，见 `docs/PROJECT_HANDOFF.md` §9），
跨机的轨迹、库内容、哈希都会微差，属于不同"数值世界"：

- 一张论文图/一组配对统计里的全部 run 必须同机（绘图层的 `check_compatible` +
  同源检查会自动拒绝混用——那是兜底，不要试探它）；
- 对已密封 run 的任何**补充 run**（如 GATSAC complement）必须回到原机器跑；
- 两机各自的 `results/` 永不互拷（本就 git-ignore）；冻结快照目录只读。

## 3. Windows 环境初始化（feat/win 的第一批工作）

1. `git clone` 后切 `feat/win`（基于 `feat/mac` 创建）；
2. Python 3.13 建 venv：`python -m venv .venv`，然后
   `.venv\Scripts\pip install -r requirements.lock.txt`
   （锁定版本以此文件为准：numpy 2.5.4 / scipy 1.18.1 / torch 2.15.0 /
   matplotlib 3.11.2 / pytest 9.2.0；torch 装 CPU 版即可，本项目 GPU 无用武之地——
   瓶颈是串行槽仿真，不是梯度计算）；
3. 跑全量测试：`.venv\Scripts\python -m pytest tests/ -q`；
4. **预期恰好 1 个失败**：`test_antenna_gain_fixed_position_output` 的零陷锚
   （90°,0°）——它是"环境身份金丝雀"，换机器**必然**失败。处理流程：
   读取本机实际值 → 在 `feat/win` 上把零陷锚改钉为 Windows 值（常规角度锚应当
   跨平台一致，若也失败则停下排查）→ 中文 commit 记录重钉依据。
   **不要**把这个改动合回 feat/mac；两分支各钉各的锚。
5. 文档里的 `./.venv/bin/python` 在 Windows 对应 `.venv\Scripts\python`。

## 4. 实验族分配（2026-11-04 时间盒化决定）

| 实验族 | 机器 | 当前范围 |
|---|---|---|
| U=2 主实验（s42/s43/s44，K=20） | **Mac** | 三场景 × method seeds 0–4，六种完整方法 |
| U=2 GATSAC 补充 | **Mac** | s42 已完成；s43 运行中；s44 待补 |
| U=3 规模探针 | **Mac** | 只跑 s42，`method_seed=0`，六种完整方法 |
| U=4 规模探针 | **Mac** | 只跑 s42，`method_seed=0`，六种完整方法 |
| RQ1/RQ3 | **Mac** | 各一个代表场景 |
| RQ5 消融 | **Mac** | s42 单场景 × method seeds 0–4 |
| exact benchmark | **Mac** | K=4/6/8 |
| 论文写作 | **Windows** | 只读 Mac 冻结结果，不运行实验 |

已取消：s45/s46、U=3/U=4 多场景、U=4 五场景矩阵、纯 UAV 数量五场景族、
完整参数敏感性和 DRL 多 train-seed 正式矩阵。论文必须将 U=2 解释为
three-scene evidence，将 U=3/U=4 解释为 single-seed scalability probes。

## 5. 运行纪律（与 Mac 侧一致）

- 正式 run 只写 `results/runs/<run_id>/`，不可覆盖；失败/超时 run 移入
  `results/runs_failed/` 隔离保存，绝不删除；
- commit 信息一律**中文**，结尾保留
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`；
- 不修改 `docs/` 下的协议/契约文档（它们归 feat/mac 维护；发现问题提给负责人）；
- 长任务防休眠：Windows 用 `powercfg` 关休眠或 PowerToys Awake（Mac 侧教训：
  电池+合盖休眠曾让 5 小时墙钟只算了 57 分钟）；插电运行；
- 笔记本跑夜间任务时确认电源计划为"高性能"，避免 E 核调度拖慢单线程求解。

## 6. 当前进度快照（截至 2026-11-04）

- Task 1–10 全部验收；bounded pilot 完成（Gate 0/1/2/4 PASS，Gate 3 为冒烟尺度
  效应 FAIL，见 `docs/stc-pilot-report.md`）；
- U=2 s42/s43/s44 三个主实验均已密封；15/15 配对点改善，平均目标值下降 18.7%；
- U=2 主实验 8 张正式图、provenance 和 reporting bundle 已重新生成于
  `results/figs_u2_main_20261104/`；
- GATSAC s42 补充已完成；s43 运行中，s44 待补；
- U=3/U=4 s42 单 seed 规模探针已启动，六种完整系统方法全部保留；
- DRL si-v1：SAC 95.2% / TD3 96.4% / DDPG 91.7%（held-out best），SAC checkpoint
  过正式加载器（SHA `d6d4b339…`）；
- 已完成：D1/D2 修复、TD3/DDPG 加载器、并行化与磁盘缓存、论文范围更新。

## 7. 同步节奏

- 两侧各自完成一个实验族节点后推送本分支，并在本文件 §4/§6 追加一行状态
  （Windows 侧改动本文件时只动状态行，结构性修改留给 feat/mac）；
- 有冲突风险的共享文件（本文件、seeds.json）：改动前先 `git fetch` 看对方分支
  最新版；
- Mac 侧的心跳监督不覆盖 Windows 机器——Windows 侧任务需自行监督并在推送时
  汇报状态。

## 8. Teacher-anchored MADRL 正式训练交接（2026-11-11）

Mac 已完成 TA-SharedTD3-BC 和 TA-MAPPO 的 100 集稳定性短测，两者均为
11/11 次确定性评估可行。Windows 按
[`docs/teacher-anchored-marl.md`](docs/teacher-anchored-marl.md) 的固定命令执行：

- `win-formal-u2-s42-sharedtd3bc-ta`
- `win-formal-u2-s42-mappo-ta`

正式预算均为 3000 集、seed 42、horizon 2000、eval interval 50。不得恢复
旧标准版本参数，也不得把一个正式 run 的结果与 Mac 短测混入同一统计族。
