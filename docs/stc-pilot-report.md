# STC bounded pilot 报告（2026-11-01）

> 执行依据：[stc-pilot-plan.md](stc-pilot-plan.md)（负责人 2026-10-31 批准）。
> 执行环境：macOS arm64 参考机，`.venv` 锁定栈（见 `requirements.lock.txt`），
> 分支 `feat/mac`。冒烟口径：U=2、K=6、单 map/task/method seed（42/42/0）。
> **本报告的全部数值仅用于管线验证与预算估算，不构成论文结论。**

## 1. 执行时间线（实测墙钟）

| 步骤 | run_id | 开始 | 用时 | 结果 |
|---|---|---|---|---|
| RQ1 航段证据 | pilot-rq1 | 10-31 20:24 | 6.5 min | ✅ success，7 记录 |
| RQ3 评价器审计 | pilot-rq3 | 20:31 | 5 s | ✅ success，1 记录 |
| RQ2+RQ4 六方法 | pilot-rq2rq4 | 20:31 | 11.8 min | ✅ success，6 记录（2 诚实 unavailable） |
| RQ5 消融（首轮） | pilot-rq5-ablation | 20:43 | 73 min | ❌ 超 3600s 预算 → failed manifest（设计行为），已隔离 |
| RQ5 消融（重跑） | pilot-rq5-ablation-r3 | 22:02 | 69 min | ✅ success，8/8 臂 |
| RQ5 规模族 | pilot-rq5-scale | 23:11 | 12.3 h 后人工中止 | ❌ failed manifest，已隔离（见 §3 缺陷 D1/D2） |
| 绘图 + Gate 报告 | — | 11-01 | 见 §2 | 首次运行暴露缺陷 D3，修复后重跑 |

隔离目录 `results/runs_failed/`（证据完整保留，Gate 扫描域之外）：
`pilot-rq5-ablation`（超时）、`pilot-rq5-ablation-r2-aborted-launch`（启动事故，
见 §4）、`pilot-rq5-scale-overbudget`。

## 2. Gate 0–4 状态（`pilot-gate-report.json`，随本报告归档于 `results/figs_pilot/`）

| Gate | 状态 | 说明 |
|---|---|---|
| 0 测试 | **PASS** | 全量套件通过（含 pilot 期间新增的 D3 修复测试） |
| 1 航段证据 | **PASS** | 1 个 run-scenario 组，4 个生成器（straight_hold/comm_waypoint/astar/slsqp_refined）真实共同可行交集 |
| 2 分配/排序 + 同步评价 | **PASS** | RQ3 审计 + RQ2 双层对比证据齐全 |
| 3 对比 + 消融 | **FAIL** | 无有限 objective 的 proposed 对比与消融贡献——即 §3 S1 尺度效应的直接后果（全方法 ε_out 超标 → objective=null），按方案 §5 判定为**尺度效应而非管线缺陷** |
| 4 溯源认证 | **PASS** | 4 个密封 run 全部通过认证（含图件 bundle 链校验） |
| 总体 | FAIL（仅因 Gate 3） | 方案 §5 硬性判据（无 NOT_RUN、Gate 0/4 PASS、bundle/报告生成）**全部满足** |

## 3. 发现（按处置分类）

### 尺度效应（冒烟口径固有，非管线缺陷）

**S1 — 冒烟场景全员不可行**：1000 m 粗网格无线场下，全部被评方法的路线级 G2A
中断率 0.40–0.61，超过 ε_out=0.35 → objective 全为 null，Gate 3 的对比/消融证据
无法在冒烟口径成立。方向性旁证：Proposed 已是离可行最近的方法（一机 0.00、
另一机 0.40，超标 0.05；基线超标 0.17–0.26），ALNS 的冲突/中断规避行为正确。
**处置**：正式口径（200 m 场）保真度探针验证可行性恢复（§5）。

### 管线缺陷（pilot 的核心产出，均需修复后才能跑正式 scale）

**D1 — 超时粒度**：`run_lifecycle` 的 `--timeout-s` 只在**场景边界**检查；scale 的
整个参数族矩阵是单场景，12.3 h 超预算运行从未触发 7200 s 限时。
**D2 — 检查点粒度**：同因，`records.json` 只在场景边界落盘；scale 跑 12.3 h 无任何
中间产物，中止后零记录可保全（消融 runner 已有记录级 checkpoint，scale 没有）。
**D3 — 兼容性检查误伤**（已修，`feat/mac` 提交）：第四轮复审的 I-4 修复把
`check_compatible` 键在各记录自身 `library_hash` 上，误拒诚实消融 run
（multi_candidate 臂的缩减库是**处理**不是**环境**）；scale 的族内重建库同理。
已改为优先 `reference_library_hash` + 条件键并入 (family, variant)，附 4 个
正反测试。

### 预算证伪

**B1 — scale 冒烟成本远超预估**：预估 ~1.5 h，实际 >12 h 未完成。原因：7 族 ×
多变体 × 2 方法 ≈ 40+ 次求解，且 payload/bandwidth/power 族按契约**重建航段库**。
正式 scale 预算必须按「每变体 = 库构建 + 2×求解」重算，并在修复 D1/D2 后执行。

## 4. 执行偏差（如实记录）

- 消融首轮 3600 s 预算低估（8 臂 × ~10 min）→ 按方案 §5 规则隔离重跑，判定正确。
- 一次启动事故：后半程链曾被误用 shell `&` 挂起脱管（职责在执行者），发现后
  SIGINT/SIGTERM 干净终止（损失 ~1 min 计算），事故 run 隔离为
  `pilot-rq5-ablation-r2-aborted-launch`，换 r3 正规后台重启。无数据覆盖/丢失。
- scale 12.3 h 超预算未被及时发现（预估 00:40 完成，实际 11:28 才人工中止）——
  暴露「长任务需中途心跳核对」的流程缺口，已计入后续执行纪律。

## 5. 正式实验预算（基于实测）

单位成本实测（K=6、200 m 场下另行探针校准 K=20）：
- Proposed STC-ALNS 单次求解（1000 迭代）：~600 s（增量评价 541 次 + 全量 353 次）
- 确定性基线单次：0.2–22 s；完整六方法一格：~12 min（K=6）
- 消融 8 臂一格：~70 min（K=6）

正式矩阵（协议最低要求）展开：
- RQ2+RQ4 主矩阵：5 场景对 × 5 method seed × U∈{2,3,4}... **K=20/30/40 的单格成本
  必须先由保真度探针实测**（K 增大使库构建、评价器时隙数、ALNS 邻域均超线性增长），
  冒烟数据不可外推。
- 保真度探针（§6 第一步）会给出 K=20 真实成本，报告将以附录补齐总预算。

## 6. 结论与请求批准事项

**管线判定**：五个 runner、密封/隔离、失败保全、绘图溯源、Gate 编排全部按设计工作；
pilot 抓出 3 个真实缺陷（D1/D2 待修，D3 已修）+ 1 个预算证伪（B1）——bounded pilot
的目的达成。

**请求负责人批准**：
1. 正式保真度探针（U=2、K=20、单场景、正式口径、单 method seed）——已获 2026-10-31
   晚「可开跑正式实验」的口头授权，先行执行，结果入本报告附录；
2. 探针可行 → 正式 RQ2+RQ4 首片（U=2 × 3 场景对 × 5 method seed）继续执行；
3. D1/D2 修复后再议正式 scale/消融矩阵与完整 5 场景对（需扩充 seeds.json）。

## 附录 A：正式保真度探针结果（formal-probe-u2k20，2026-11-01）

单元：U=2、K=20、200m 正式场、全生成器、单场景 (map 42, task 42)、单 method seed 0、
ALNS 1000 迭代。run 密封 success（6 产物）。**单 seed 单场景，方向性信号，非论文结论。**

| 方法 | 状态 | 可行 | obj J | T (s) | D (s) | 求解时长 |
|---|---|---|---|---|---|---|
| Region-Distance-Straight | unavailable* | — | — | — | — | — |
| Shortest-Balanced-Graph | unavailable* | — | — | — | — | — |
| Jia-EBTAS-style | evaluated | ✅ | 1.0876 | 481.0 | 269.0 | 161 s |
| GATSAC-style | unavailable（无 SI checkpoint，诚实声明） | — | — | — | — | — |
| Library-Open-Loop | evaluated | ✅ | 0.8871 | 330.5 | 263.5 | 1 s |
| **Proposed STC-ALNS** | evaluated | ✅ | **0.7256** | **265.5** | **219.0** | 8182 s |

\* 探针用修复前代码（git head `604c7ba`）构建库，两基线的 unavailable 是缺陷 **D4**
（跨族 Pareto 剪枝剪绝基线轨迹族）的表现；D4 已修复于 `21ffdaa`（族保底 + 3 测试），
正式首片用修复后代码，基线表将完整。探针 run 密封保留作 D4 发现证据；其 git head
与正式批不同，绘图层的同源检查会自动拒绝混用。

**探针回答的两个问题**：
1. **可行性在正式口径恢复** ✅——三个被评方法全部可行（中断率 0.20–0.23 < 0.35），
   冒烟全员超标确认为纯尺度效应（S1）。
2. **单元成本实测**：库构建 ~115 min；Jia 161 s；Open-Loop 1 s；**Proposed 8182 s
   （约 2.27 h）/seed**。正式首片（3 场景 × 5 seed）估算 ≈ 3×(1.9h 库 + 5×2.35h)
   ≈ **41 h 连续计算**。

**方向性信号（待正式矩阵检验）**：
- Proposed vs Library-Open-Loop：0.7256 vs 0.8871（**−18.2%**）——冲突反馈闭环的
  净效应（论文贡献 ③ 的核心对比），且 T（−19.7%）与 D（−16.9%）**同时**改善；
- Proposed vs Jia-style：−33.3%；Open-Loop vs Jia：−18.4%（库本身的价值，贡献 ②）；
- Proposed 把冲突损失压到 **0 bits**（Jia 方案冲突损失 4.2 Mbit 的同款场景）。

## 附录 C：SI 口径 DRL 单段重训（2026-11-02，负责人批准的重训线）

训练入口 `experiment/train_stc_drl.py`（commit `9f01956`，仅新建文件）；统一设置：
目标条件化单段环境（`StcSacEnv` + 文档定案奖励结构）、U=2/K=20 正式场景
（map/task seed 42、200m 场）、3000 集、train_seed 42、held-out 84 条边（20% 切分，
训练永不采样）、每 50 集 held-out 确定性评估选 best、4 线程（与首片同机并行）。
单算法训练用时约 40–55 分钟。

| 算法 | held-out 最佳可行率（@集数） | held-out 末次 | 训练末 100 集可行率 | best checkpoint SHA-256 前缀 |
|---|---|---|---|---|
| SAC | **95.2%**（@1300） | 90.5% | 98%（全程稳定） | `d6d4b33976065c15` |
| TD3 | **96.4%**（@1450，三者最高） | 8.3% | 8%（尾段发散） | `53236fd32aefa7ba` |
| DDPG | 91.7%（@1700） | 9.5% | 13%（尾段发散） | `64e3a875de53bfed` |

**与项目历史的一致性**：TD3/DDPG 的中期峰值 + 后期 Q 过估计发散、SAC 全程稳定，
与旧 100m 原型时代（实验 3/11 与 2026-10-21 对比）完全同构——"确定性 held-out
评估选 best"机制再次按设计捕获峰值模型。三算法的奖励曲线与可行率曲线原始数据在
`results/models/stc_drl/<algo>_u2_si-v1_rewards.npz`（`ep_rewards`/`ep_feasible`/
`eval_heldout_feasible`），provenance JSON 含 git head 与全部参数。

**用途与后续**：① SAC best 已通过 `_load_sac_policy` 正式校验 → GATSAC-style 基线
可解锁（首片三场景保持统一无 checkpoint 口径，收官后跑 GATSAC 补充 run）；
② 三算法作为 DRL 航段生成器进 RQ1 同边对比，按 spec §8.4 预注册门控判定升级与否
（负责人指示：若效果更好可不视作基线）；③ TD3/DDPG 的加载器接线需改现有文件，
待首片收官后实施。
