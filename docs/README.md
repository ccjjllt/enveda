# 研究文档索引

整理日期：2026-11-16（北京时间）。001–064 的配对计划/进展与交接文档已删除，研究概况、结果与放弃原因统一保存在[已放弃研究总览](research-archive.md)。当前保留 A 线 065–069。

## 当前研究

| 轮次 | 主题 | 计划 | 实际进展 |
| --- | --- | --- | --- |
| 065 | 065a N-1c 网络重排，public 0.488 | [计划](plan/065-network-n1c.md) | [进展](progress/065-network-n1c.md) |
| 066 | 训练库连续余弦 mid-rank，0.488 中性 | [计划](plan/066-libcos-midrank.md) | [进展](progress/066-libcos-midrank.md) |
| 067 | 自研 model3 mid-rank，0.488 中性 | [计划](plan/067-own-hn-midrank.md) | [进展](progress/067-own-hn-midrank.md) |
| 068 | N-1c `LAM=0.35`，public 0.492 已核实（ref 58410619）；068a ref 58412029 PENDING | [计划](plan/068-n1c-lam035.md) | [进展](progress/068-n1c-lam035.md) |
| 069 | 新信号筛选：timsTOF ICEBERG / DreaMS / PubChem 尾召回 | [计划](plan/069-new-signals.md) | [进展](progress/069-new-signals.md) |

## 调研

- [069 前沿模型调研](research/069-frontier-models.md)：2023–2026 可接入的 MS/MS 结构解析模型、许可结论、三个特别评估问题与 P0/P1/P2 接入方案。不是实验计划，未运行内核。
- [069 数据资产审计](research/069-assets.md)：068a 已挂 10 个 dataset 的真实用途与死重、候选库/谱库/预训练权重的边际覆盖核算、最值得加的 5 个资产。只读审计，含 GNPS 与 train 的本地重叠核对。
- [069 公开前沿](research/069-public-frontier.md) 与 [069 检索设计](research/069-retrieval-design.md)：同日另两份调研，分别判公开配方与重排设计。
- [069 管线解剖](research/069-pipeline-map.md)：候选生成→证据→融合→排序→后处理的逐通道地图，含 400 题窗口实测、PubChem 通道入口上限（5000→25→5）与 Enveda-180 池外实测。

## 总览与运维

- [已放弃研究总览](research-archive.md)：001–064 的方向、分数、失败机制、删除清单与恢复方式。
- [比赛规则、数据与远端工作流](remote-workflow.md)：数据、指标、Kaggle CLI/提交与配额。
- [项目入口](../README.md)、[协作约定](../AGENTS.md)。

## 阅读边界

- 旧轮次已经结束，不再维护；需要细节时从 Git 标签 `pre-cleanup-2026-11-16` 取回。
- 065–069 中的分数、名次、GPU 余额与提交额度均按记录日期理解；当前状态以 069 进展最后一条为准。
- 公开 notebook 的标题分数不等于已核实分数；只有本账号实际提交并出分，或逐格核对且作者本人公榜一致，才记为已核实。
