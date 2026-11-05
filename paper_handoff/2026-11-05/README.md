# Mac 实验结果交接包（2026-11-05）

本目录供 Windows 论文写作线使用。压缩包只包含已 `finalize("success")` 的
sealed run，不包含 U=3/U=4 的未完成中间态。

## 解压

在仓库根目录执行：

```bash
tar -xzf paper_handoff/2026-11-05/u2_sealed_20261105.tar.gz -C .
```

压缩包内保留 `results/runs/...` 和 `results/figs...` 原路径，解压后结构与
Mac 工作区一致。

## 包含内容

### U=2 主实验

- `formal-u2-s42`
- `formal-u2-s43`
- `formal-u2-s44-r2`

每个主 run 包含六种方法、method seeds 0–4 的 raw records、solutions、
timelines、RQ2、summary、resolved config 和 sealed manifest。

### U=2 GATSAC 补充

- `gatsac-u2-s42-r2`
- `gatsac-u2-s43`
- `gatsac-u2-s44`

这三个 run 补齐 GATSAC-style 的 s42/s43/s44 行。

### 支撑实验

- `formal-rq1-s42`：正式航段生成器对比
- `formal-rq3-s42`：正式同步评价审计

### 图件

- `results/figs_u2_main_20261104/`
  - 8 张 U=2 主实验汇总图
  - `figures_provenance.json`
  - `reporting_bundle.json`
- `results/figs_u2_routes_s42/`
  - U=2 s42、method seed 0 的六方法最终路线对比图
  - PNG/PDF/SVG、输入 provenance、alignment 和 collision QA
- `results/figs/stc_drl_*.png`
  - 已有 SI 单段 DRL 训练曲线

### DRL 训练数据

- `results/models/stc_drl/*_provenance.json`
- `results/models/stc_drl/*_rewards.npz`

不包含 `.pth` 模型权重。

## 论文使用边界

- U=2 是 three-scene evidence，每场景 5 个 method seeds。
- 不得把 15 个场景×seed 配对点解释为 15 张独立地图。
- GATSAC 是 limited single-checkpoint style reproduction。
- 该压缩包不包含 U=3/U=4 最终结果；这两项仍在运行，完成后再单独交接。
- 所有正文数值必须从 sealed run 或自动图件读取，不手工改写。

## 校验

解压前可执行：

```bash
shasum -a 256 -c paper_handoff/2026-11-05/SHA256SUMS
```
