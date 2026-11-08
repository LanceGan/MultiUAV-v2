# Exact K=4/6/8 交接包（2026-11-08）

本包包含 U=2、s42 场景下前 K 个任务的 tiny-exact 审计结果。它只说明
离散候选空间中的可行性边界，不是连续空间全局最优性证明。

## 解压

```bash
tar -xzf paper_handoff/2026-11-08/exact_k4_k8_20261108.tar.gz -C .
```

## 结果

| K | Exact 状态 | Exact 叶子数 | Exact 时间 | Proposed 可行 |
|---:|---|---:|---:|---|
| 4 | 完整枚举，无可行解 | 120 | 16.6 s | 否 |
| 6 | 完整枚举，无可行解 | 5,040 | 1,013.1 s | 否 |
| 8 | 6 小时超时，枚举 318,597/362,880 | 318,597 | 21,600.3 s | 否 |

K=8 的 manifest 为 `success`，但 exact record 明确记录：

- `timed_out=true`
- `discrete_search_complete=false`
- `proven_optimal=false`

不能把 K=8 描述成完整枚举或已证明最优。

## 图件与来源

`results/figs_exact_20261108/` 包含：

- PNG/PDF/SVG 汇总图
- statistics JSON/CSV
- provenance
- panel-alignment JSON/SVG
- PDF collision audit
- reporting bundle

图件 QA：源检查 21 PASS，alignment PASS，PDF collision PASS。

## 论文使用边界

- 只能写 “optimal within the declared discrete candidate space”。
- K=8 只能写 timeout，不能写完整枚举。
- 当前三个 K 前缀都没有可行解，因此不能据此计算正 optimality gap。
- 该实验是边界审计，不是主方法性能证据。

## 校验

```bash
shasum -a 256 -c paper_handoff/2026-11-08/SHA256SUMS
```
