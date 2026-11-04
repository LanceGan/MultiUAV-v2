# 两天版实验执行清单（2026-11-04）

> 本清单是当前执行范围的可操作汇总。权威假设和失败/溯源规则仍以
> `docs/stc-experiment-protocol.md` 为准；本清单覆盖其中旧的多场景正式矩阵要求。

## 执行边界

- 所有正式实验只在 Mac 运行；Windows 不再运行或补充实验，只负责论文写作。
- U=2 是主要证据；U=3/U=4 是 single-seed 规模探针。
- 所有图表只从 sealed run 自动生成，不手工填写数值。

## 实验矩阵

| 实验 | 场景 | method seeds | 方法数 | 状态 |
|---|---|---:|---:|---|
| U=2 主实验 | s42/s43/s44 | 0–4 | 6 | 已完成 |
| U=2 GATSAC s42 | s42 | 0–4 | 1 | 已完成 |
| U=2 GATSAC s43 | s43 | 0–4 | 1 | 运行中 |
| U=2 GATSAC s44 | s44 | 0–4 | 1 | 待运行 |
| U=3 规模探针 | s42 | 0 | 6 | 运行中 |
| U=4 规模探针 | s42 | 0 | 6 | 运行中 |
| RQ1 | 一个代表场景 | N/A | 多生成器 | 待运行 |
| RQ3 | 一个代表场景 | N/A | 3 评价模式 | 待运行 |
| RQ5 消融 | s42 | 0–4 | full + 7 ablations | 待运行 |
| exact | K=4/6/8 | N/A | 精确枚举 vs Proposed | 待运行 |

## 正式 run ID

已完成主实验：

- `formal-u2-s42`
- `formal-u2-s43`
- `formal-u2-s44-r2`

GATSAC 补充：

- `gatsac-u2-s42-r2`
- `gatsac-u2-s43`
- `gatsac-u2-s44`（待启动）

规模探针：

- `formal-u3-s42-single`
- `formal-u4-s42-single`

这两个 run 只使用：

```text
--map-seeds 42 --task-seeds 42 --method-seeds 0 --workers 5
```

## 当前正式图

U=2 三场景主实验的 8 张图、provenance 和 bundle：

```text
results/figs_u2_main_20261104/
```

数据来源仅：

```text
formal-u2-s42
formal-u2-s43
formal-u2-s44-r2
```

## 论文可声称内容

- U=2 three-scene paired study，每场景 5 个 method seeds；
- 15 个场景×seed 配对点全部改善；
- U=3/U=4 single-seed scalability probes；
- 不得声称五场景矩阵、跨地图规模泛化、多 train-seed DRL 或完整 Monte Carlo。

## 明确取消

- U=2 s45/s46；
- U=3/U=4 多场景；
- U=4 五场景矩阵；
- 纯 UAV 数量五场景族；
- 完整参数敏感性与 Monte Carlo 正式矩阵；
- 学习型基线多 train-seed 正式矩阵。
