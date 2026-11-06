# Mac 补充分享包：U=3 单场景规模探针（2026-11-06）

本目录供 Windows 论文写作线使用。压缩包只包含已经
`finalize("success")` 的 U=3 s42 sealed run，不包含 U=4 或 RQ5
的未完成中间态。

## 解压

在仓库根目录执行：

```bash
tar -xzf paper_handoff/2026-11-06/u3_sealed_20261106.tar.gz -C .
```

压缩包保留原始的 `results/runs/formal-u3-s42-single/` 路径。

## 内容

`formal-u3-s42-single` 是 U=3、map seed 42、task seed 42、
method seed 0 的单次规模探针，包含六种完整系统方法：

- Region-Distance-Straight
- Shortest-Balanced-Graph
- Jia-EBTAS-style
- GATSAC-style
- Library-Open-Loop
- Proposed STC-ALNS

run 已封存为 `success`，共有六条方法记录，并包含 raw records、
solutions、timelines、RQ2、summary、resolved config 和 sealed manifest。

## 已封存结果

- Jia-EBTAS-style 可行：`J=1.1601873389038178`
- Proposed STC-ALNS 可行：`J=0.8204105045385568`
- 其余四种方法在该 run 中不可行
- Proposed 相对唯一可行基线 Jia 的目标值下降约 29.28%

## 论文使用边界

- 这是单场景、单 method seed 的 U=3 scalability probe。
- 不能把它描述成多 seed 稳健性实验或统计显著性证据。
- 不能把 U=3 s42 与 U=2 的三场景结果混为一个重复实验矩阵。
- GATSAC 仍是 limited single-checkpoint style reproduction。
- 正文数值必须从 sealed run 或后续自动图件读取，不手工改写。
- 本包暂时不包含 U=3 最终轨迹图；图件完成后再单独交接。

## 来源

- 结果工作树分支：`codex/cache-parallel`
- 封存 run：`results/runs/formal-u3-s42-single`
- 封存时间：2026-11-05 15:47:54 CST

## 校验

```bash
shasum -a 256 -c paper_handoff/2026-11-06/SHA256SUMS
```
