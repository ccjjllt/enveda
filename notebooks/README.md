# Notebook 入口

当前保留 A 线 065–082 的关键候选；001–064 的 notebook 已清理，结论见[已放弃研究总览](../docs/research-archive.md)。

| 目录 | 用途与状态 |
| --- | --- |
| [065a](065a/) | N-1c 网络重排，public **0.488**（ref `58329677`），原最佳 |
| [066a](066a/) | 训练库连续余弦 mid-rank，0.488 中性，不作底座 |
| [067a](067a/) | 自研 model3 mid-rank，0.488 中性，不作底座 |
| [068a](068a/) | 065a + N-1c `LAM=0.35`，对应公开 0.492；自有 ref `58412029` 公榜 **0.492**，当前最佳 |
| [072a](072a/) | DreaMS Enveda180 kNN 中名次重排；ref `58413244` 公榜 **0.492**，中性 |
| [073a](073a/) | timsTOF 微调 ICEBERG 中名次重排；ref `58413231` 公榜 **0.491**，−0.001 |
| [076a](076a/) | DreaMS×Morgan 对比对齐中名次重排；ref `58415444` 公榜 **0.490**，−0.002；训练 kernel `jiyujia/enveda-align-train` |
| [082c](082c/) | Enveda-180 not-in-train 外部谱库 + 多谱一致性；**待 2026-11-17 提交验证 hidden-specific 假设** |

这些 notebook 都是自包含入口，只依赖 Kaggle competition/dataset/kernel 输入，不读取本地 `src/`、`artifacts/` 或 `datasets/`。metadata 为私有、断网、T4；`COMMIT_MODE='smoke'`，审计通过后再提交版本。

`kernels push` 只上传 `code_file` 与 metadata，不同步整个仓库。其余实验（069a/070a/071a/074a/075a/077a/078a/079a/080a/081a/082a/082b）保留在 jiyujia 账号的 kernel 记录中，分数与审计见[069 进展](../docs/progress/069-new-signals.md)与[068 进展](../docs/progress/068-n1c-lam035.md)。
