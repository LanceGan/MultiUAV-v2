# Windows–Mac 双机协同说明（win-mac.md）

> 读者：Windows 侧开发者 + Mac 侧负责人/协作 agent。
> 本文档由 **Windows 侧维护**（分支 `feat/win`），记录 Windows 机器的环境、任务分配与数据同步状态。
> 协作的权威契约见 Mac 侧维护的 [mac-win.md](mac-win.md)（分支 `feat/mac`），两文件并读。
> 更新时间：2026-11-07。

## 1. 机器与分支

| 机器 | 平台 | 分支 | 角色 |
|---|---|---|---|
| Mac（参考机） | macOS 15.7 arm64, Apple Silicon | `feat/mac` | U=2 实验族 + 补充 run；DRL si-v1 已训产物；文档主干 |
| Windows 新机 | Windows 11, i5-12500H（4P+8E 12 核） | `feat/win` | 论文写作（只读 Mac 冻结结果，不跑实验） |

两分支互不合并、互不 rebase，只向 `origin` 各自推送；汇合时机与方式由项目负责人决定。

## 2. 硬规则（与 mac-win.md 一致，Windows 侧重申）

- **数值世界隔离**：同一实验族的所有 run 必须出自同一台机器。跨平台浮点不逐位一致
  （实测天线零陷角 Windows x86 与 macOS arm64 差 0.55 dB，见 `docs/PROJECT_HANDOFF.md` §9）。
  一张论文图/一组配对统计里的全部 run 必须同机；绘图层的 `check_compatible` + 同源检查
  会自动拒绝混用（那是兜底，不要试探它）。
- **结果不互拷**：两机各自的 `results/` 永不互拷（本已 git-ignore）；冻结快照目录只读。
- **分支纪律**：`feat/win` 与 `feat/mac` 互不合并、不 rebase，只推各自分支。

## 3. Windows 环境状态（初始化进度）

> 按 mac-win.md §3 执行，完成一项勾一项。

- [x] 基于 `feat/mac`（commit `ec3e8c0`）创建 `feat/win` 分支（2026-11-02）
- [x] Python 3.13 建 venv：`uv python install 3.13`（3.13.14）→ `python -m venv .venv`
- [x] 安装锁定栈：`.venv\Scripts\pip install -r requirements.lock.txt`
      （numpy 2.5.4 / scipy 1.18.1 / torch 2.15.0 CPU / matplotlib 3.11.2 / pytest 9.2.0，
      全部 cp313 win_amd64 轮子）
- [x] 全量测试：首跑 **461 passed + 1 failed**（唯一失败 = 零陷锚金丝雀，符合预期）；
      重钉后全量复核 **462 passed**（17 分钟），无其他测试受影响
- [x] 零陷锚重钉：改钉为 Windows x86_64 值（中文 commit 记录依据；常规角度锚跨平台逐位稳定、未动）
- [x] 本机零陷锚值记录于 §6 机器身份

> 注：本机旧参考环境（conda `pytorch`，Python 3.10.18）是迁移前的历史栈，正式工作一律用上面的 `.venv`。

## 4. 实验族分配（Windows 视角）

| 实验族 | 状态 |
|---|---|
| ~~U=3 正式切片（K=30、5 km、5 基站、930 边）~~ | **已交还 Mac 跑**（负责人 2026-11-04 决定） |
| U=4 切片（K=40、6 km） | 待定 |
| RQ5 scale/消融正式矩阵 | 待定；先修 D1/D2（见 pilot 报告 §3）再启动 |
| DRL 多 train_seed（≥5）× 多地图训练 | 可给 Windows；入口 `experiment/train_stc_drl.py`，~1 小时/算法/seed |

**U=3 实测成本（Windows i5-12500H，供 Mac 侧排程参考）**：
- 航段库构建（16 核并行）：**~2.2h**（串行 12h+）
- proposed_stc_alns：**~20h+/seed**（K=30；K=20 实测 2.27h，超线性 ~9×）
- 5 seed 并行（各 1 核）：种子循环 ~20h+；单场景并行 ~22h

**关键提示**：并行化（`--workers`）代码在 `feat/win`，未合入 `feat/mac`；Mac 若直接用
feat/mac 串行跑 U=3，库构建 12h+ + proposed ~20h×5 ≈ **~112h/场景**。是否把并行化移植到
feat/mac 由负责人决定。

## 5. 运行纪律（Windows 侧，与 Mac 侧一致）

- 正式 run 只写 `results/runs/<run_id>/`，不可覆盖；失败/超时 run 移入 `results/runs_failed/` 隔离保存，绝不删除；
- commit 信息一律中文，结尾保留 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`；
- 不修改 `docs/` 下的协议/契约文档（它们归 feat/mac 维护；发现问题提给负责人）；
- 长任务防休眠：Windows 用 `powercfg` 关休眠或 PowerToys Awake；插电运行；
- 笔记本跑夜间任务时确认电源计划为"高性能"，避免 E 核调度拖慢单线程求解。

## 6. Windows 进度快照 + 机器身份

- **环境身份金丝雀（零陷锚 (90°,0°) 本机值）**：-291.75958991746415
  （逐位复现旧 Windows 参考值；macOS arm64 为 -292.30602416217334）
- **本机环境**：Windows 11 x86_64 (i5-12500H) · Python 3.13.14（uv 安装）· 锁定栈
  （numpy 2.5.4 / scipy 1.18.1 / torch 2.15.0 CPU / matplotlib 3.11.2 / pytest 9.2.0）
- run-id 命名约定：`formal-u3-sX-win`
- 已完成：`feat/win` 分支创建、win-mac.md 初稿、环境初始化（venv + 锁定栈 + 全量测试 + 零陷锚重钉）、并行化（库构建 + 种子循环，TDD 确定性回归锁定逐位一致，全量 464 passed）
- 进行中：—（U=3 已交还 Mac 跑）
- 备注①：首次串行 run（10:11 起，12h 库构建未完成）已中止，隔离于 `results/runs_failed/formal-u3-s42-win-aborted-20261103/`
- 备注②：第二次并行 run（23:42 起，库构建 2.2h 完成；种子循环 19.5h 时 proposed 仍 ~20h+/seed 未完成）已中止，隔离于 `results/runs_failed/formal-u3-s42-win-aborted-2nd-20261104/`
- 待办：D1/D2 修复（scale 前置）；并行化代码在 feat/win（未合入 feat/mac），是否移植由负责人决定
- **论文写作线（2026-11-07）**：已接收并核验 Mac 交接包 `paper_handoff/2026-11-05`（U=2 主实验 + 8 图 + s42 路线图）与 `paper_handoff/2026-11-06`（U=3 s42 单 seed 封存包，SHA256 PASS）。
  论文仓库 `F:\Codes\Latex\P2_journal` 已填入 RQ1/RQ2/RQ3/RQ4/RQ5 全部表值、GATSAC 补充基线、U=3 规模探针，修正数值口径（s44 Jia 标准差、vs-Jia 范围 19.6%→20.2%），接线 8 张主图 + s42 路线图，`pdflatex` 编译 15 页 0 警告 0 未定义引用，已中文 commit（论文仓库 `main`，commit `1b29ad0`）。

## 7. 同步节奏

- 完成一个实验族节点后推 `feat/win`，并在本文件 §4/§6 追加一行状态；
- 有冲突风险的共享文件（本文件、mac-win.md、seeds.json）：改动前先 `git fetch` 看对方分支最新版；
- Mac 侧的心跳监督不覆盖 Windows 机器——Windows 侧任务需自行监督并在推送时汇报状态。

## 8. 论文写作：待 Mac 决策/交接清单（2026-11-07）

以下问题由 Mac 侧负责人拍板，Windows 侧按其决定调整论文。

**需决策（阻塞）**
1. **RQ3 口径**：sealed run `formal-rq3-s42` 审计结果为
   `feasibility_reversal: none found`、`ranking_reversal: none found`
   （`n_plans=4`），近似误差 makespan 4.875 s / delay 18.625 s。
   论文已如实写「误差可测量、但 n=4 未发现排序反转」，并把贡献点 1 的
   「静态评估 mis-ranking 被量化」软化为「非同步评估的近似误差被量化」。
   请决定：(a) 接受该诚实表述；或 (b) 增大 `n_plans` 重跑 RQ3，尝试触发真实
   排序反转后再写贡献点 1。

**需交接（数据/图件）**
2. **RQ5 消融表** `tab:ablation`：7 个 single-mechanism ablation
   （s42 单场景 × method seeds 0–4）尚无封存 run，论文表仍为 skeleton。
3. **exact benchmark 表** `tab:exact`：K=4/6/8 实例尚无封存 run，仍为 skeleton。
4. **U=4 规模探针**：无封存结果，论文已标注 not yet frozen / excluded；
   如需纳入请补跑并交接。
5. **矢量图件**：论文当前接线的是 PNG 光栅（8 张主图 + s42 路线图），
   IEEE 投稿建议补 PDF/EPS 矢量版。

**需确认（写作口径，非阻塞）**
6. GATSAC 引用 `Yang2026CKM` 是否确为 GATSAC 论文（正文仍 TODO-verify）。
7. 数值口径修正：s44 Jia 标准差 `0.038→0.026`（统一 population std 口径）；
   vs-Jia 范围 `19.6%→20.2%`（per-pair 均值口径）。请确认无异议。
8. RQ2 已填表：固定 segment 层 = 每边 best independent candidate、
   单 method seed（= map seed）、三场景均值（只对 feasible 场景平均）。
   请核对语义无误。
9. 图注由 Windows 依据图名 + `plots/plot_stc_results.py` 推断书写，
   请核对图实际内容（尤其 `stc_balance` 箱线图与 s42 路线图）。
10. 作者列表 / 论文标题 / Acknowledgments 仍 TODO，需作者本人提供。
