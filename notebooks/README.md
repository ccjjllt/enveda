# Notebook 入口

当前保留 A 线底座 065a–068a 与待验证的 082c；001–064 与 069 已终止变体（072a/073a/076a 等）的 notebook 已清理，结论见[已放弃研究总览](../docs/research-archive.md)第 7 节。

| 目录 | 用途与状态 |
| --- | --- |
| [065a](065a/) | N-1c 网络重排，public **0.488**（ref `58329677`），原最佳 |
| [066a](066a/) | 训练库连续余弦 mid-rank，0.488 中性，不作底座 |
| [067a](067a/) | 自研 model3 mid-rank，0.488 中性，不作底座 |
| [068a](068a/) | 065a + N-1c `LAM=0.35`，对应公开 0.492；自有 ref `58412029` 公榜 **0.492**，当前最佳 |
| [082c](082c/) | Enveda-180 not-in-train 外部谱库 + 多谱一致性；**待 2026-11-17 提交验证 hidden-specific 假设** |

已删除的 072a（DreaMS kNN，0.492 打平）、073a（timsTOF ICEBERG，0.491）、076a（对比对齐，0.490）本地副本可从 Git 标签 `pre-cleanup-069-2026-11-16` 恢复，Kaggle kernel 记录仍在。

这些 notebook 都是自包含入口，只依赖 Kaggle competition/dataset/kernel 输入，不读取本地 `src/`、`artifacts/` 或 `datasets/`。metadata 为私有、断网、T4；`COMMIT_MODE='smoke'`，审计通过后再提交版本。

`kernels push` 只上传 `code_file` 与 metadata，不同步整个仓库。其余实验（069a/070a/071a/074a/075a/077a/078a/079a/080a/081a/082a/082b）保留在 jiyujia 账号的 kernel 记录中，分数与审计见[069 进展](../docs/progress/069-new-signals.md)与[068 进展](../docs/progress/068-n1c-lam035.md)。
