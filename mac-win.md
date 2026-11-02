# Mac–Windows 双机协同说明（mac-win.md）

> 读者：Windows 机器上的开发者/协作 agent。本文档是两台机器分工的**权威契约**，
> 由 Mac 侧维护（分支 `feat/mac`）；Windows 侧在 `feat/win` 上工作。
> 更新时间：2026-11-02。

## 1. 机器与分支

| 机器 | 平台 | 分支 | 角色 |
|---|---|---|---|
| Mac（参考机） | macOS 15.7 arm64, Apple Silicon | `feat/mac` | U=2 实验族 + 其一切补充 run；DRL si-v1 已训产物；文档主干 |
| Windows 新机 | Windows 11, i5-12500H（4P+8E 12 核） | `feat/win` | 领取**整族**新实验（见 §4 分配） |

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

## 4. 实验族分配（当前）

| 实验族 | 机器 | 状态 |
|---|---|---|
| U=2 正式切片（s42/s43/s44，K=20） | **Mac** | s42 已密封；s43/s44 运行中（预计 11-03 下午收官） |
| U=2 的 GATSAC 补充 run、RQ1 DRL 生成器对比 | **Mac**（必须同机） | 待切片收官 |
| DRL si-v1 三算法训练（SAC/TD3/DDPG） | **Mac** | 已完成（附录 C） |
| **U=3 正式切片（K=30、5 km、5 基站、930 边）** | **Windows（建议首个任务）** | 待启动：`run_stc_main --uav-nums 3 --map-seeds X --task-seeds X --method-seeds 0,1,2,3,4`，场景 X∈{42,43,44} 逐个顺序跑，每场景独立 `--run-id formal-u3-sX-win` |
| U=4 切片（K=40、6 km） | 待定（看 U=3 实测成本） | — |
| RQ5 scale/消融正式矩阵 | 待定 | **先修 D1/D2**（超时/检查点粒度缺陷，见 pilot 报告 §3），未修前不要启动 scale |
| DRL 多 train_seed（协议要求 ≥5）× 多地图训练 | 可给 Windows | 入口 `experiment/train_stc_drl.py`，~1 小时/算法/seed |

**U=3 成本预警**：Mac 上 U=2 实测为库构建 ~2h + Proposed ~2.3h/seed（1000 迭代）。
K=30 的边数是 2.2 倍、时间线更长，单场景可能 20h+；i5-12500H 单核性能与 M 系接近
但持续负载注意散热降频。先跑一个场景当探针、记录各方法耗时，再决定排程。

## 5. 运行纪律（与 Mac 侧一致）

- 正式 run 只写 `results/runs/<run_id>/`，不可覆盖；失败/超时 run 移入
  `results/runs_failed/` 隔离保存，绝不删除；
- commit 信息一律**中文**，结尾保留
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`；
- 不修改 `docs/` 下的协议/契约文档（它们归 feat/mac 维护；发现问题提给负责人）；
- 长任务防休眠：Windows 用 `powercfg` 关休眠或 PowerToys Awake（Mac 侧教训：
  电池+合盖休眠曾让 5 小时墙钟只算了 57 分钟）；插电运行；
- 笔记本跑夜间任务时确认电源计划为"高性能"，避免 E 核调度拖慢单线程求解。

## 6. 当前进度快照（截至 2026-11-02 16:30，Mac 侧）

- Task 1–10 全部验收；bounded pilot 完成（Gate 0/1/2/4 PASS，Gate 3 为冒烟尺度
  效应 FAIL，见 `docs/stc-pilot-report.md`）；
- 正式探针 + s42 已密封：Proposed 0.7204±0.003 vs 开环 0.9102±0.076 vs Jia
  0.9908（5 seed，配对均值 −20.3%，5/5 为负）；s43 seed 0 复现 −21.9%；
- DRL si-v1：SAC 95.2% / TD3 96.4% / DDPG 91.7%（held-out best），SAC checkpoint
  过正式加载器（SHA `d6d4b339…`）；
- 待办：D1/D2 修复、TD3/DDPG 加载器接线、GATSAC 补充 run、附录 B、
  seeds.json 扩到 5 对场景（协议最低要求）。

## 7. 同步节奏

- 两侧各自完成一个实验族节点后推送本分支，并在本文件 §4/§6 追加一行状态
  （Windows 侧改动本文件时只动状态行，结构性修改留给 feat/mac）；
- 有冲突风险的共享文件（本文件、seeds.json）：改动前先 `git fetch` 看对方分支
  最新版；
- Mac 侧的心跳监督不覆盖 Windows 机器——Windows 侧任务需自行监督并在推送时
  汇报状态。
