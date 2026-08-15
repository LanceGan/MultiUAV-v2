# MultiUAV 巡检 (Multi-UAV Inspection) — 阶段一重构

面向**真实可复现**的多机（Multi-UAV）巡检研究重构。本仓库在单机已验证的通信模型
（radio map，3GPP RMa 信道）基础上，逐步加入多机任务分配、航迹规划、序列决策与
实验复现，目标是让每一处实验脚本都能一键复现论文结果。

## 目录说明

```
radio_map/       通信模型（已从 SingleUAV 验证代码迁移，只读）
  radio_map_A2G.py   地面基站 → 空中 UAV（A2G）信道，getPointDateRate()
  radio_map_G2A.py   空中 UAV → 地面用户（G2A）信道，getPointMiniOutage()
  rural_world.py     乡村宏蜂窝（RMa）地形/建筑生成
  entity.py          UAV / User 数据类
results/datas/    用户位置数据（Users_20/30/40.txt，x,y,h 三列，单位 100m）
tests/             测试（pytest）
common/            （待 Task 填充）公共工具
assignment/        （待 Task 填充）任务分配
sequence/          （待 Task 填充）序列决策
trajectory/        （待 Task 填充）航迹规划
experiment/        （待 Task 填充）实验编排
plots/             （待 Task 填充）绘图
configs/           （待 Task 填充）配置
```

## 运行环境

- Python 解释器：`F:/anaconda3/envs/pytorch/python.exe`（conda `pytorch` 环境，
  含 numpy / torch / matplotlib / scipy / pytest）。
- 依赖见 `requirements.txt`。

```bash
cd f:/Projects/Py/MultiUAV_New
F:/anaconda3/envs/pytorch/python.exe -m pytest tests/test_radio_map.py -v
```

> 从项目根目录运行 pytest，`radio_map` 才能通过根目录 `sys.path` 解析。

## 如何复现一个结果

（占位：由 Task 11 填全。届时在此给出从数据、到运行脚本、到产出图表的端到端复现步骤。）
