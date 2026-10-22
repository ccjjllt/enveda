# 数据目录

- `raw/`：原始比赛Parquet和示例提交，本地保留，不入Git。
- `references/`：按日期或轮次命名的来源、人工核对和精简研究结果，入Git；保持原始记录的字节与历史哈希。
- `processed/`：后续本地预处理数据的约定位置，不入Git。

原始数据与评估边界见[数据及远端工作流](../docs/remote-workflow.md)，逐轮结果见[文档索引](../docs/README.md)。历史JSON中的旧路径按[迁移表](../docs/workspace-paths.json)查找；研究脚本已在必要读取处适配，原始记录不批量重写。
