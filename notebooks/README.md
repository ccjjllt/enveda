# Notebook 入口

当前只保留 A 线四个候选；001–064 的 notebook 已清理，结论见[已放弃研究总览](../docs/research-archive.md)。

| 目录 | 用途与状态 |
| --- | --- |
| [065a](065a/) | N-1c 网络重排，public **0.488**（ref `58329677`），当前最佳 |
| [066a](066a/) | 训练库连续余弦 mid-rank，0.488 中性，不作底座 |
| [067a](067a/) | 自研 model3 mid-rank，0.488 中性，不作底座 |
| [068a](068a/) | 065a + N-1c `LAM=0.35`，对应公开 0.492；已备好，待 2026-11-21 GPU 刷新后推送 |

四个 notebook 都是自包含入口，只依赖 Kaggle competition/dataset/kernel 输入，不读取本地 `src/`、`artifacts/` 或 `datasets/`。metadata 已设为私有、断网、T4×2；`COMMIT_MODE='smoke'`，审计通过后再提交版本。

`kernels push` 只上传 `code_file` 与 metadata，不同步整个仓库。历史版本与公榜分数见[已放弃研究总览](../docs/research-archive.md)与[068 进展](../docs/progress/068-n1c-lam035.md)。
