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
| [073-public-0495](073-public-0495/) | 复现公开 0.495 → **否证**：作者 7 个数据集均为 `PUBLIC-DEMO`，真实权重被替换为初始化 demo 权重、catalogue 为 3,683 个合成分子，且 Kaggle 拒绝提交（non-versioned datasets）。公榜复查：0.500→30 名、前 600 全 ≥0.492、0.500+ 无公开 notebook。**公开可复现前沿 = 0.494**（Dante fork），0.495 只是演示标签 |
| [074-public-0494](074-public-0494/) | 复现并提交最高分公开方案 `ccuuuuy/casmi26-0-492-test-set-network-rerank`（作者 Dante，公榜 **0.494**）：alexchilton 0.488 网络 + `LAM 0.35` 之上叠 `w_mid=0.75` 共识、物理碎片检查、K14 去重与文档先验。`074a` 已提交（ref 58505232），等公榜 |

已终止方向的结论统一在[已放弃研究总览](../archive/research-archive.md)。notebook 入口都是自包含的，只依赖 Kaggle competition/dataset/kernel 输入；推送只上传入口目录：

```bash
kaggle kernels push -p rounds/068-n1c-lam035/068a
```

远端推送/看守脚本在 `../ops/063/`，比赛与远端操作在 `../ops/remote-workflow.md`。
| [075-frontier](075-frontier/) | 前沿冲刺：只做 MS2 同分异构体判别。三条并行调研（SOTA/公开权重、未使用信号审计、公开资产与规则）+ 化学型整组留出仪器设计（075b）。本地已确认本机 RDKit Murcko 不可用、训练库类似物家族稀疏（5,000 抽样仅 77 个 2–4 成员簇） |
