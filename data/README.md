# 数据目录

- `raw/`：原始比赛 Parquet 与示例提交（2.8 GB），本地保留，不入 Git。
- `references/`：比赛元信息、人工核对与精简结果 JSON（684 KB），入 Git。
- `processed/`：后续预处理数据约定位置，不入 Git。

数据边界与评估口径见 [ops/remote-workflow.md](../ops/remote-workflow.md)；历史清理与恢复见 [archive/research-archive.md](../archive/research-archive.md)（标签 `pre-cleanup-2026-11-19`）。
