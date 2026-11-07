# Mac 补充分享包（2026-11-07）

本目录供 Windows 论文写作线使用，包含已经封存的 U=3 最终路线图和
RQ5 五-seed 机制消融。U=3 实验数据本身位于 2026-11-06 的 sealed run
交接包。

## 解压

在仓库根目录执行：

```bash
tar -xzf paper_handoff/2026-11-07/u3_routes_20261107.tar.gz -C .
tar -xzf paper_handoff/2026-11-07/rq5_ablation_20261107.tar.gz -C .
```

## 图件

- `stc_u3_s42_seed0_routes.png`：600 dpi 预览
- `stc_u3_s42_seed0_routes.pdf`：可编辑矢量投稿图
- `stc_u3_s42_seed0_routes.svg`：可编辑矢量图
- `stc_u3_s42_seed0_routes.provenance.json`：输入 run、文件 SHA-256 和图件参数
- `stc_u3_s42_seed0_routes.alignment-layout.json`：六面板最终几何
- `stc_u3_s42_seed0_routes.alignment.svg`：版式 QA 定位层
- `stc_u3_s42_seed0_routes.collision-audit.json`：PDF 文本/线条碰撞审计

## QA 结果

- Source strict validation：21 PASS
- Panel alignment：PASS，7 组比较，0 FAIL，0 WARN
- PDF collision：PASS，25 个文本框，0 FAIL，0 WARN
- PDF 最小字号：6.4 pt

## 使用边界

- 这是 U=3 单场景、单 method seed 的规模探针图。
- 不可把它解释成多 seed 稳健性或统计显著性证据。
- 图中红色叉号表示超过单航段最大等待时间的硬约束位置。
- 正文数值必须从 sealed run 或 provenance 所指向的数据读取，不手工改写。

## 校验

```bash
shasum -a 256 -c paper_handoff/2026-11-07/SHA256SUMS
```

## RQ5 机制消融

`rq5_ablation_20261107.tar.gz` 包含：

- `formal-ablation-s42-m0-r3` 至 `m4-r3` 五个 sealed run
- 每个 run 的 full + 7 个消融臂
- `stc_rq5_ablation_s42.png/pdf/svg`
- `stc_rq5_ablation_s42.statistics.json/csv`
- provenance、alignment、PDF collision audit 和 reporting bundle

五-seed 配对结果口径为 `ΔJ = ablation - full`，正值表示去掉该机制后变差：

- conflict-aware repair：`ΔJ=+0.013216`，5/5 seeds 变差；
- path switch：`ΔJ=+0.004327`，3/5 seeds 变差；
- adaptive weights：`ΔJ=+0.002647`，3/5 seeds 变差；
- EBTAS initialisation：`ΔJ=-0.000047`，2 worse / 3 better，基本中性；
- multi-candidate：`ΔJ=-0.004987`，1 worse / 4 better；
- hotspot destroy：`ΔJ=-0.000854`，2 worse / 3 better；
- incremental eval：目标值 5/5 完全相同，应只作为计算效率消融。

统计使用单侧 exact paired sign-flip test。conflict-aware repair 的 raw
`p=0.03125`，但 7 个消融臂 Holm 校正后 `p=0.21875`。因此正文应报告
方向、效应量和配对一致性，不能把 5 seeds 的结果描述成经过多重校正的
显著优势。

论文不能声称全部七项机制都改善目标值。可以重点报告
conflict-aware repair，并将 path switch、adaptive weights 作为方向性但
不稳健的补充证据。
