# MultiUAV_New 单航段 TD3、边权预计算与 ALNS 实验审计

**审计日期：** 2026-10-19  
**仓库：** `F:\Projects\Py\MultiUAV_New`  
**审计基线 HEAD：** `75d268a68b5433e72a8de4e16f34ac94850a8d03`  
**审计方式：** 只读代码检查、模型/NPZ/JSON 解析、确定性复算、现有测试执行、图件视觉与数值核验  
**结论等级：** 工程原型成果已成立；论文正式结果尚未成立

> 本报告描述的是审计时工作区的实际状态。仓库存在未提交修改和未跟踪的 `segment/`、模型及图件，因此 HEAD 不能单独代表本次实验源码。审计期间 `alns_routes_uav2.json` 和对应图片曾被外部进程覆盖一次；第 11 节记录的是审计结束时的最终哈希。

---

## 1. 执行摘要

当前新方案已经形成一条真实可运行的工程链条：

```text
固定 Users_20 场景
  → goal-conditioned TD3 单航段策略
  → 420 条有向边闭环 rollout
  → 航段边权表
  → 多邻域路线搜索
  → 简化共享带宽后处理
  → 奖励、航段轨迹和多机路线图
```

已独立核实的主要结果：

- `uav2_allpairs.pth` 在当前代码、seed 42、`spacing=0.25` 下能够逐项复现边权表；
- 420 条预期有向边中 384 条在当前单段环境内可行，覆盖率为 **91.4286%**；
- 当前两条 ALNS 路线覆盖 20 个巡检点且每点恰好一次，使用的 22 条边全部可行；
- 相对当前 cheapest-insertion 初始解，路线搜索使 makespan 改善 **10.45%**、上传时延改善 **4.26%**、标量目标改善 **8.41%**；
- 当前结果证明了“单航段策略 → 边表 → 外层路线搜索”的原型可行性。

但当前结果只能表述为：

> 在固定 Users_20、固定 seed-42 地图和训练支持集上，目标条件化 TD3 为 384/420 条有向边生成了确定性可行轨迹，路线搜索能够避开失败边并构造完整路线。

当前结果还不能支持：

- 任意连续点对的泛化成功率；
- 数学全局最优轨迹；
- 显著优于 Straight、A*、waypoint-SLSQP 或其他 DRL；
- 未见地图、未见数据量或多 seed 的鲁棒性；
- 完整共享带宽、关联、碰撞和安全约束下的多无人机系统结论。

---

## 2. 审计范围与工作区状态

### 2.1 重点审计文件

新单航段与外层流程：

- [`segment/env.py`](../../segment/env.py)
- [`segment/td3.py`](../../segment/td3.py)
- [`segment/train.py`](../../segment/train.py)
- [`segment/precompute.py`](../../segment/precompute.py)
- [`segment/scheduler.py`](../../segment/scheduler.py)
- [`segment/alns.py`](../../segment/alns.py)

相关旧模块：

- [`common/config.py`](../../common/config.py)
- [`radio_map/radio_map_A2G.py`](../../radio_map/radio_map_A2G.py)
- [`radio_map/radio_map_G2A.py`](../../radio_map/radio_map_G2A.py)
- [`sequence/radio_path.py`](../../sequence/radio_path.py)
- [`trajectory/env.py`](../../trajectory/env.py)
- [`trajectory/sca.py`](../../trajectory/sca.py)
- [`experiment/_common.py`](../../experiment/_common.py)

主要结果文件：

- [`results/models/segment/uav2.pth`](../../results/models/segment/uav2.pth)
- [`results/models/segment/uav2_allpairs.pth`](../../results/models/segment/uav2_allpairs.pth)
- [`results/models/segment/edge_weights_uav2.npz`](../../results/models/segment/edge_weights_uav2.npz)
- [`results/models/segment/alns_routes_uav2.json`](../../results/models/segment/alns_routes_uav2.json)
- [`results/logs/UAV_2_segment_rewards.npz`](../../results/logs/UAV_2_segment_rewards.npz)
- [`results/figs/segment_reward_curve.png`](../../results/figs/segment_reward_curve.png)
- [`results/figs/segment_trajectory_uav2.png`](../../results/figs/segment_trajectory_uav2.png)
- [`results/figs/alns_routes_uav2.png`](../../results/figs/alns_routes_uav2.png)

### 2.2 Git 状态

审计时仓库存在以下重要未提交内容：

- `segment/` 整个目录未跟踪；
- `results/models/`、`results/figs/` 中的新产物未跟踪；
- `trajectory/env.py`、多个 MARL 文件和实验脚本已修改；
- `CLAUDE.md`、`.gitignore` 已修改。

因此必须在继续实验前冻结源码和结果 manifest，否则后续无法证明某个模型、边表和图件由哪一版代码产生。

---

## 3. 已核实的单航段 TD3 成果

### 3.1 有向边总数

当前 Users_20 单航段集合包括：

```text
depot → inspection:       20
inspection → inspection: 20 × 19 = 380
inspection → terminal:   20
合计:                    420
```

`edge_weights_uav2.npz` 的矩阵 shape 为 `21×21`。文件中的 21 个 NaN 全部是结构空槽：20 个自环以及一个不使用的 depot→terminal 槽；没有异常 NaN/Inf。

### 3.2 可行率

| 边类型 | 可行 | 总数 | 覆盖率 |
|---|---:|---:|---:|
| depot → inspection | 18 | 20 | 90.0000% |
| inspection → inspection | 347 | 380 | 91.3158% |
| inspection → terminal | 19 | 20 | 95.0000% |
| **合计** | **384** | **420** | **91.4286%** |

审计使用当前代码、当前 checkpoint、seed 42 和 `spacing=0.25` 对 420 条边进行只读重算：

- flight time 最大差异：0；
- delay 最大差异：0；
- outage 最大差异：0；
- feasible mismatch：0。

因此边表与当前代码/模型的推理行为一致。

### 3.3 有向性

在 190 对互逆 inspection 点对中：

- 183 对 flight time 不同；
- 187 对 delay 不同；
- 190 对 outage 不同；
- 29 对可行性不同。

这说明当前边表保留了有向航段语义，不能通过翻转一条轨迹构造反向边。

### 3.4 91.4% 的正确口径

训练池就是相同 Users_20 的全部 420 条边，预计算也使用这 420 条边。5000 episodes 平均每条边约被采样 11.9 次；按 seed 42，单边出现次数约为 3–23 次。

因此 91.4% 是：

> 固定 Users_20、固定地图、固定 seed 的训练支持集边覆盖率。

它不是 held-out 点对泛化成功率。

---

## 4. 训练曲线和 checkpoint 审计

### 4.1 奖励日志

`UAV_2_segment_rewards.npz` 包含 5000 个 episode reward：

| 统计 | 数值 |
|---|---:|
| 全部 episode 平均 | 1106.09 |
| 前 100 episode 平均 | -1050.08 |
| 后 100 episode 平均 | 1411.23 |
| 100 窗滑动均值峰值 | 1714.59 |
| 峰值位置 | episode 2992 |

episode 4001–4500 存在明显退化。因此“训练结束模型”未必优于中期最佳模型。

奖励日志没有保存：

- episode 长度；
- 成功/失败标志；
- failure reason；
- transition 数；
- held-out 成功率；
- 地图、配置或代码身份。

所以奖励上升不能单独证明任务成功率。

### 4.2 best 被 final 覆盖

[`segment/train.py`](../../segment/train.py) 在训练过程中保存 probe best，但训练结束时又无条件把 final 模型写回同一个路径。因此 `uav2_allpairs.pth` 是 5000 episode 结束模型，而不是受保护的最佳模型。

### 4.3 checkpoint 不完整

两个 segment checkpoint 只包含 Actor/Critic `state_dict`，没有：

- target networks；
- optimizer；
- episode/global update；
- seed/RNG；
- 完整超参数；
- 地图、任务和代码 hash；
- best episode 与 held-out 结果。

因此当前模型可以推理并复算边表，但训练不能严格恢复，也不能仅凭 checkpoint 证明其训练参数。

---

## 5. ALNS 路线审计

### 5.1 当前路线

```text
UAV 0: [12, 3, 13, 7, 11, 16, 10, 8, 6, 0]
UAV 1: [2, 4, 18, 15, 9, 17, 5, 19, 1, 14]
```

20 个巡检点恰好各出现一次；22 条被使用的边全部在当前边表中标记为可行。

### 5.2 精确成本

| 指标 | UAV 0 | UAV 1 | 系统 |
|---|---:|---:|---:|
| flight time | 351.5 s | 349.0 s | makespan 351.5 s |
| upload delay | 154.5 s | 104.0 s | sum 258.5 s |
| outage exposure | 58.5185 s | 77.6685 s | sum 136.1870 s |

归一化参考：

- `T_ref = 373.3793 s`
- `D_ref = 525.4385 s`
- 当前标量目标：`0.71668594`

seed 42、2000 次搜索迭代可精确复现当前路线 JSON。

### 5.3 与当前初始解比较

代码实际使用的是 cheapest insertion 初始解，不是日志文字所称的 “greedy NN”。

| 指标 | 初始解 | 路线搜索结果 | 改善 |
|---|---:|---:|---:|
| makespan | 392.5 s | 351.5 s | 10.45% |
| delay | 270.0 s | 258.5 s | 4.26% |
| scalar objective | 0.782533 | 0.716686 | 8.41% |

### 5.4 当前实现不是严格自适应 ALNS

[`segment/alns.py`](../../segment/alns.py) 均匀随机选择算子，没有：

- 算子得分；
- 自适应权重更新；
- 分段更新周期；
- 起飞时间决策。

更准确的名称是：

> 带模拟退火接受准则的多邻域搜索。

它当前只优化 (T,D)，没有将 (O,E)、碰撞、共享带宽和起飞时间纳入搜索。

---

## 6. 约束与可行性问题

384 条当前“可行边”中：

- 223 条 `delay > tau_max=12 s`，占 58.1%；
- 102 条 `outage_time > T_out_max=20 s`，占 26.6%。

当前 `feasible` 实际只表示：

```text
到达目标 + 数据上传完成
```

没有执行现有配置中声明的 delay/outage 上限。因此当前 ALNS 路线“边均可行”并不意味着它满足完整论文约束。

---

## 7. 环境与策略语义审计

### 7.1 当前状态不是完整资源条件状态

状态仅包含：

```text
[x, y, dx_goal, dy_goal, data_fraction, time_fraction]
```

缺少：

- 初始/绝对数据量；
- 带宽比例；
- 扇区关联；
- 当前 A2G 速率与 G2A 中断；
- 当前速度；
- 其他 UAV 与资源状态。

10 Mbit 与 30 Mbit 在起点均表现为 `data_fraction=1`，但上传时间不同，存在状态混叠。因此当前策略是目标条件化，但不是严格的目标与资源条件化 Markov 策略。

### 7.2 当前环境更容易训练

当前 `SingleSegmentEnv` 的主要简化：

- 最大速度约 50 m/s；
- 每步可瞬时改变航向；
- 无加速度约束；
- 无安全投影；
- 无多机碰撞；
- 越界坐标直接 clamp；
- horizon 约为直达时间的 5 倍；
- 进度奖励强且稠密。

这些条件能够解释它为何比严格 v2 环境更容易训练成功。

### 7.3 当前没有 BC→TD3 交接问题

当前实现从随机 Actor/Critic 同步开始普通 TD3，没有 BC Actor、Critic-only 校准、专家 replay 或 BC anchor。因此它不会出现“已经训练好的 BC Actor 被随机 Critic 立即破坏”的特定故障。

这不表示当前算法更严格，只表示它解决的是另一个更宽松的问题。

---

## 8. 电磁地图与通信模型风险

当前旧 A2G/G2A 地图仍存在会影响论文结论的问题：

1. 地形和建筑高度使用米，而 LoS 线段高度使用公里，直接比较；
2. 建筑写入索引与 LoS 查询索引尺度相差约 4 倍；
3. BS 坐标在 radio 模块中硬编码，不能由 scenario JSON 真正控制；
4. A2G 和 G2A 使用不同的噪声定义，差约 40 dB；
5. A2G/G2A 各自选择“最佳扇区”，不保证关联一致；
6. 地图随机种子固定在模块内部，与场景实验 seed 分离；
7. 轨迹策略使用的是位置处已经聚合的最佳无线电值，而不是逐扇区资源条件。

因此当前无线电地图适合作为原型演示，正式论文实验应改用统一 SI、统一场景几何、逐扇区并带 provenance 的地图。

---

## 9. 共享带宽与多机协调审计

当前 scheduler 的逻辑是：

```text
计算整条路线平均同扇区 UAV 数
→ 基础总时延 × 平均拥塞因子
```

它没有逐时隙更新：

- 关联基站/扇区；
- 分配带宽；
- 实际速率；
- 剩余 buffer；
- 上传完成时刻；
- 活跃上传 UAV 集合。

因此当前共享带宽结果只能称为：

> v1 静态拥塞近似。

不能作为正式共享蜂窝资源调度实验。

---

## 10. 图件审计

### 10.1 单段轨迹图

[`segment_trajectory_uav2.png`](../../results/figs/segment_trajectory_uav2.png) 来自 `uav2.pth` 的真实、未平滑 rollout，但：

- 只画 UAV 0；
- 不使用 `uav2_allpairs.pth`；
- 11 条航段中只有 10 条可行；
- 失败段 13→19 在 144 步达到 horizon，buffer 已空但距目标仍约 952 m；
- 图中没有标出失败状态。

### 10.2 ALNS 路线图

[`alns_routes_uav2.png`](../../results/figs/alns_routes_uav2.png) 的底层来自 `uav2_allpairs.pth` 的真实 rollout，原始 22 条航段均可行，终点距目标不超过约 30 m。

但绘图前进行了 B-spline 平滑，且没有重新经过环境验证：

- 最大起点偏移约 144.5 m；
- 最大终点偏移约 140.6 m；
- 相对原始采样点最大偏离约 131.2 m；
- 有一段平滑曲线越过 4 km 边界约 58.4 m。

因此该图只能作为美化示意图，不能作为可执行轨迹证据。正式结果应保存并绘制原始 rollout positions，或在平滑后重新评价全部指标和约束。

---

## 11. 结果文件哈希

以下为审计结束时的最终快照：

| 文件 | SHA-256 |
|---|---|
| `uav2.pth` | `F44FBECBB2DD01398EC813FC6ADC0A8EA4F75E1AF210038FD1AED4980D203A41` |
| `uav2_allpairs.pth` | `FE70104364D0E28AB843491B03212DBDC8B72A2A689310D654B3075D4FB624DE` |
| `edge_weights_uav2.npz` | `C648D2804D08A0F0D972175A99C27692F29A003E6072B934A21A40F0861D5AC3` |
| `alns_routes_uav2.json` | `FDB37680B95F76AF83CA5CFBA85F107DFAF40229F72EE30CC97274D84273BAE3` |
| `UAV_2_segment_rewards.npz` | `1FF35C78CA809633A118B2FA2ED9EE03A8E00FEF6C3004D72FA571FB9CB403C7` |
| `segment_reward_curve.png` | `C1B594A67C354C81E90E83A1C3CEFF054ED4770332140D51EF4841477F434A07` |
| `segment_trajectory_uav2.png` | `7C3934EE1D75CEE8FB7F33CC8DAA4D86C97F0268CE06DC3BE1FB5D5F2CF97B48` |
| `alns_routes_uav2.png` | `6E11578AAB5A5DA1A8A4ECCE51FB01F6BF50A18DB0343D24FFDBE0C5FF7E20A2` |

审计期间 ALNS JSON 和图片曾被外部进程覆盖一次。正式引用前应把当前源码、模型、边表、路线和图件复制到只读快照目录，并生成包含文件 hash、命令、seed、依赖和 Git 状态的 manifest。

---

## 12. 测试状态

当前仓库完整测试执行结果：

```text
25 passed
2 failed
1,053,372 warnings
```

失败项包括：

- 数据量测试仍期待旧的 100–300 Mbit；
- `edge_cost` 距离单调性测试失败。

当前 `tests/` 尚未覆盖新的 `segment/` 模块。`CLAUDE.md` 中“27 tests passed”的描述与当前工作树实际状态不一致。

---

## 13. 模块成熟度与复用建议

| 模块 | 判断 | 建议 |
|---|---|---|
| 4 km 区域、Users_20/30/40、BS 坐标 | 可复用 | 转换为 SI canonical fixtures 并保存来源 hash |
| 配置、点位读取、绘图框架 | 可复用骨架 | 统一参数口径并禁止结果覆盖 |
| 旋翼能耗公式 | 适配复用 | 审计调用端单位 |
| ReplayBuffer、TD3 网络骨架 | 适配复用 | 增加完整 checkpoint、seed 和 provenance |
| 单航段环境与训练 | 有价值原型 | 补状态、约束、held-out 与测试 |
| 420 边表 | preliminary diagnostic | 保留为 legacy_v0，不称最优库 |
| 当前路线搜索结果 | 初始解/候选结果 | 修正名称和约束后继续 |
| assignment/sequence 旧算法 | 对比算法骨架 | 使用统一 v2 edge oracle 和局部 RNG |
| scheduler | 概念验证 | 重做逐时隙关联、带宽和 buffer |
| ALNS | 多邻域搜索原型 | 增加自适应算子权重及完整目标 |
| 旧无线电地图 | 不可直接用于正式结果 | 用统一 SI、逐扇区、带 provenance 的地图替换 |
| 当前 B-spline 图 | 仅示意 | 保存原始轨迹或重新验证平滑轨迹 |

---

## 14. 当前成果可支持的论文表述

可以支持：

> 在固定 Users_20、seed-42 场景中，目标条件化 TD3 为 384/420 条训练池有向边生成了确定性可行轨迹；基于该边表的多邻域搜索能够避开失败边，构造两条覆盖全部 20 个巡检点的可行路线，并相对当前 cheapest-insertion 初始解将 makespan 降低约 10.45%。

必须同时注明：

- 这是训练支持集覆盖率；
- 使用单地图、单 seed；
- 可行定义尚未包含全部 delay/outage/energy/safety 约束；
- 共享带宽为静态近似；
- 结果是 preliminary proof-of-concept。

不能支持：

- 任意两点泛化；
- 全局最优或近似最优保证；
- 对其他轨迹算法的显著优势；
- 完整多无人机通信协调；
- 正式论文主结论。

---

## 15. 最小补强路线

### P0：立即冻结当前成果

在继续训练或绘图前：

1. 冻结 dirty source snapshot；
2. 复制当前模型、边表、路线、奖励和图件；
3. 保存 SHA-256 manifest、Git HEAD/status、Python/NumPy/PyTorch/SciPy 版本、命令和 seed；
4. 禁止后续实验覆盖该目录。

### P1：修复证据链

1. 分离 `best.pth` 与 `last.pth`；
2. checkpoint 保存 targets、optimizers、counters、seed/RNG、config/map/task/code hashes；
3. 日志按 run ID 使用独立目录；
4. 保存 episode 成功率、长度和 failure reason；
5. ALNS 结果和图件写 manifest。

### P2：补最小测试

- 420 边枚举与反向边；
- source-point packet 语义；
- best 不被 final 覆盖；
- 边表索引和结构空槽；
- ALNS 每点唯一覆盖；
- 原始 rollout 指标重算；
- 图件轨迹来自原始 rollout；
- scheduler buffer/带宽守恒。

### P3：快速补齐轨迹库与公平基线

1. 保留当前 384 条 TD3 可行边；
2. 对 36 条失败边使用 Straight/A* fallback；
3. 所有轨迹通过同一个环境评价；
4. 加入 Straight、communication-aware A*、waypoint-SLSQP；
5. 统一输出 (T,D,O,E)、success 和 failure reason；
6. 保存每条原始轨迹，而不只保存标量边权。

### P4：修复物理与系统模型

1. 替换为统一 SI、逐扇区的双无线电地图；
2. 增加带宽/资源状态；
3. 增加速度和加速度状态；
4. 将 delay/outage/energy/safety 约束纳入可行性；
5. 重做逐时隙共享带宽 scheduler；
6. 为多机轨迹增加冲突检查与同步 evaluator。

### P5：正式实验

- 独立 held-out 点对；
- 未见地图；
- 至少 5 个预注册 seed；
- 同预算算法对比；
- 消融实验；
- 置信区间和完整失败分类。

---

## 16. 最终建议

不建议推倒当前成果，也不建议把当前仓库原样当作最终论文代码。推荐采用混合策略：

```text
冻结当前成果为 legacy_v0
  → 保留场景、点位、TD3骨架、384条可行边和当前路线
  → 修复 checkpoint、测试和实验 provenance
  → 用统一物理环境补齐失败边和基线
  → 修复无线电、安全和共享带宽
  → 在 held-out、多 seed 条件下重新形成正式结果
```

按此路线，当前成果可以作为可靠的 warm start 和 preliminary baseline，避免重新从零开发；同时不会把已经确认的地图、约束和证据链问题带入论文主结论。
