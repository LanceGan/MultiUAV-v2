# U=3 最终路线图交接包（2026-11-07）

本包供 Windows 论文写作线使用，对应 U=3、s42、method seed 0 的六方法
最终路线对比。实验数据本身位于 2026-11-06 的 U=3 sealed run 交接包。

## 解压

在仓库根目录执行：

```bash
tar -xzf paper_handoff/2026-11-07/u3_routes_20261107.tar.gz -C .
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
