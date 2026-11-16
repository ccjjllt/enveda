# 数据目录

- `raw/`：原始比赛 Parquet 和示例提交，本地保留，不入 Git。
- `references/`：比赛元信息、人工核对和精简研究结果，入 Git；保留原始记录字节。
- `processed/`：后续本地预处理数据的约定位置，不入 Git。

原始数据与评估边界见[数据及远端工作流(../ops/remote-workflow.md)，研究历史见[已放弃研究总览(../archive/research-archive.md)。001–064 的旧路径不再维护；需要旧文件时从 Git 标签 `pre-cleanup-2026-11-16` 取回。
