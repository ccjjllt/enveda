# 028 进展

- 实现：`notebooks/028/e1ext_cell.py`，完整 notebook 为 `notebooks/028/enveda-028a-e1ext.ipynb`（v465 加一个新 cell，再改 cell 19 的一行）。
- 本地单元检查（Py3.12 venv，可见测试的母离子窗口）：
  - 新增 14,237 个结构，池子从 710,701 增加到 724,938。
  - 质量保持单调排序。
  - 原有条目的指纹和碎片逐字节一致。
  - 新条目带有 FPNet 位和碎片质量。
  - 构建耗时约 45 s，碎片预算为 60 s。
- Kaggle 运行：推送 v9 被拒，提示“Maximum weekly GPU quota of 30.00 hours reached”。没有运行，也没有提交。
  - 下一步：等 GPU 配额重置后，推送 notebook `notebooks/028/enveda-028a-e1ext.ipynb`（推送目录 /workspace/k027b 里已是该版本），确认能在时限内跑完，再提交一次。
- v14（v465 fork，ref 57462581）的公榜分数仍待出。

## 2026-10-21 18:00 巡检

- GPU 周配额仍未重置（重置时间 2026-10-24 08:00 北京时间），028a 仍未运行。
- 公榜前列已到 0.495（我们 v14 为 0.456）。
- 新公开 notebook [lavinwins/casmi26-top-1-gold](https://www.kaggle.com/code/lavinwins/casmi26-top-1-gold)：v465 变体，PubChem 通道 `POP_LAM=0.25`、`POP_UNION=200`，池内流行度 `POOLPOP_MU=0.25`，Engine2 换用 `prvsiyan/chebi-lipidmaps-casmi26`，并加了 `hengck23/hengck23-dreams-enveda-casmi26` 数据集。分数未标注。
- 配额重置后的顺序：先跑 028a（池扩展），再对比上述 pop 参数（0.25/200 对比现在的 0.15）。

## 2026-10-22 12:05 巡检（北京时间）

- GPU 周配额仍满（30h/30h），重置仍为 **2026-10-24 08:00 北京时间**；028a **仍未** push/运行。
- 公榜：#1 **0.504**（Oleksandr Parfionov）；约第 20 名 **0.466**；我们 v14 仍 **0.456**（距金牌线约 0.010）。
- 本地复验 028a 池扩展产物（`/workspace/t028/work`）：724,938 结构、src=4 新增 14,237、质量单调 —— 与上次单元检查一致。
- **更正 18:00 条对 lavinwins 的参数描述**：对照 `/workspace/pub/lavinwins/casmi26-top-1-gold.ipynb` 源码后，其 `POP_LAM`/`POOLPOP_MU` 实际均为 **0.15**（与我们 v465 相同）；docstring 残留的 0.25 与版本表「0.25→0.15」一致。公开 changelog 中 **0.464** 来自 v16「N1 上 joint ICEBERG+proxy GLACIER 重排」，不是流行度权重。后续见 round 029。

