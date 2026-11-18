# 研究轮次索引

一轮一目录：计划（`plan.md`）、实际进展（`progress.md`）、notebook 入口、专题调研都在同一目录下。接手先读对应轮次的 `plan.md` 与 `progress.md`，结束后更新 `progress.md`。

| 轮次 | 主题与状态 |
| --- | --- |
| [065-network-n1c](065-network-n1c/) | N-1c 网络重排，public 0.488；入口 `065a/`（ref `58329677`） |
| [066-libcos-midrank](066-libcos-midrank/) | 训练库连续余弦 mid-rank，0.488 中性；入口 `066a/` |
| [067-own-hn-midrank](067-own-hn-midrank/) | 自研 model3 mid-rank，0.488 中性；入口 `067a/` |
| [068-n1c-lam035](068-n1c-lam035/) | N-1c `LAM=0.35`，public **0.492**；入口 `068a/`（ref `58412029`） |
| [069-new-signals](069-new-signals/) | 新信号筛选（timsTOF ICEBERG / DreaMS / 对比对齐 / Enveda-180 谱库）；全部终止（072a 0.492 打平、073a 0.491、076a 0.490、082c 0.468），notebook 已清理；`research/` 下 5 份专题调研保留 |
| [070-generation](070-generation/) | 早期强打分/生成/class-2/e180变体已终止清理；082f3已出**0.492**；`082g/`与`083a/`待分保留。083a输入聚合缺陷及旧推断更正见071；`research/`保留6份历史调研 |
| [071-breakthrough-research](071-breakthrough-research/research-report.md) | 深度研究：历史证据和083a输入审计、最新MIST/MS-GPT资产、MSAlign式对齐与真实候选排序；研究报告与后续实验设计已形成，尚未执行新GPU实验 |
| [072-scale](072-scale/) | 提分空间排查：公开 59 臂表与自测旋钮最大 +0.006；072c mass prior 网格可提升 **+0.00000**；072d 留出仪器召回 **150/150**、48 个排序错误中 **47 个同分异构**；072e 首次启用 RR8+保证金升位 **0.487（−0.005）**。判决 **0.492 = 本配方族上限**，升位族 0/6 |
| [073-public-0495](073-public-0495/) | 复现公开 0.495（`heliosli` demo，神经 RRF `w_top=0.25 / w_mid=0.75` + 未训练碎片检查）。公榜复查：0.500→30 名、0.495→83–120 名、前 600 全 ≥0.492；公开 notebook 作者无人 >0.494 |

已终止方向的结论统一在[已放弃研究总览](../archive/research-archive.md)。notebook 入口都是自包含的，只依赖 Kaggle competition/dataset/kernel 输入；推送只上传入口目录：

```bash
kaggle kernels push -p rounds/068-n1c-lam035/068a
```

远端推送/看守脚本在 `../ops/063/`，比赛与远端操作在 `../ops/remote-workflow.md`。
