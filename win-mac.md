# Windows–Mac 双机协同说明（win-mac.md）

> 读者：Windows 侧开发者 + Mac 侧负责人/协作 agent。
> 本文档由 **Windows 侧维护**（分支 `feat/win`），记录 Windows 机器的环境、任务分配与数据同步状态。
> 协作的权威契约见 Mac 侧维护的 [mac-win.md](mac-win.md)（分支 `feat/mac`），两文件并读。
> 更新时间：2026-11-02。

## 1. 机器与分支

| 机器 | 平台 | 分支 | 角色 |
|---|---|---|---|
| Mac（参考机） | macOS 15.7 arm64, Apple Silicon | `feat/mac` | U=2 实验族 + 补充 run；DRL si-v1 已训产物；文档主干 |
| Windows 新机 | Windows 11, i5-12500H（4P+8E 12 核） | `feat/win` | 领取整族新实验（U=3 切片等） |

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
      重钉后 `test_radio_map.py` 5 passed，全量复核后台进行中
- [x] 零陷锚重钉：改钉为 Windows x86_64 值（中文 commit 记录依据；常规角度锚跨平台逐位稳定、未动）
- [x] 本机零陷锚值记录于 §6 机器身份

> 注：本机旧参考环境（conda `pytorch`，Python 3.10.18）是迁移前的历史栈，正式工作一律用上面的 `.venv`。

## 4. 实验族分配（Windows 视角）

| 实验族 | 状态 |
|---|---|
| **U=3 正式切片（K=30、5 km、5 基站、930 边）** | **建议首个任务**，待启动 |
| U=4 切片（K=40、6 km） | 待定（看 U=3 实测成本） |
| RQ5 scale/消融正式矩阵 | 待定；先修 D1/D2（见 pilot 报告 §3）再启动 |
| DRL 多 train_seed（≥5）× 多地图训练 | 可给 Windows；入口 `experiment/train_stc_drl.py`，~1 小时/算法/seed |

U=3 启动命令（场景 X∈{42,43,44} 逐个顺序跑，每场景独立 run-id）：

```bash
.venv\Scripts\python experiment/run_stc_main.py --uav-nums 3 --map-seeds X --task-seeds X --method-seeds 0,1,2,3,4 --run-id formal-u3-sX-win
```

成本预警：Mac 上 U=2 实测为库构建 ~2h + Proposed ~2.3h/seed（1000 迭代）。K=30 的边数是
2.2 倍、时间线更长，单场景可能 20h+；i5-12500H 单核性能与 M 系接近但持续负载注意散热降频。
先跑一个场景当探针、记录各方法耗时，再决定排程。

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
- 已完成：`feat/win` 分支创建、win-mac.md 初稿、环境初始化（venv + 锁定栈 + 全量测试 + 零陷锚重钉）
- 进行中：全量测试复核（后台）
- 待办：U=3 首场景探针、D1/D2 修复（scale 前置）

## 7. 同步节奏

- 完成一个实验族节点后推 `feat/win`，并在本文件 §4/§6 追加一行状态；
- 有冲突风险的共享文件（本文件、mac-win.md、seeds.json）：改动前先 `git fetch` 看对方分支最新版；
- Mac 侧的心跳监督不覆盖 Windows 机器——Windows 侧任务需自行监督并在推送时汇报状态。
