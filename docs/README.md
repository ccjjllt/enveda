# 研究文档索引

整理日期：2026-10-22（北京时间）。**研究执行从030接手**；031记录本次目录整理，不代表新的模型实验或分数。

- [当前研究计划](plan/030-n1-challenger-and-score-strategy.md)与[实际进展](progress/030-n1-challenger-and-score-strategy.md)：v14实测0.456；030 kernel v9已完成并提交v15（ref57583178），待评分，等用户通知后查询。
- [研究历史与结论](research-history.md)：原README各轮摘要，包含当时状态与已限缩的旧判断。
- [比赛规则、数据与远端工作流](remote-workflow.md)：Kaggle/Colab说明和手动CLI参考。
- [根目录说明](../README.md)与[协作约定](../AGENTS.md)。

## 配对计划与实际进展

计划描述假设与方案，进展才是实际修改和结果的依据。旧文档里的“当前”“下一步”、配额、名次与分数均按其记录日期理解；若后续轮次已纠正，继承后续结论。

| 轮次 | 主题 | 计划 | 实际进展 |
| --- | --- | --- | --- |
| 001 | 数据理解与检索排序基线 | [计划](plan/001-baseline.md) | [进展](progress/001-baseline.md) |
| 002 | TPU 微调研究：Colab T4 核对兼容性，再适配 v5e-1 | [计划](plan/002-tpu-finetuning.md) | [进展](progress/002-tpu-finetuning.md) |
| 003 | 通用 Colab MCP 接入与训练平台切换 | [计划](plan/003-colab-mcp.md) | [进展](progress/003-colab-mcp.md) |
| 004 | Colab T4：真实候选排序、目标域训练与获奖路线交接 | [计划](plan/004-t4-candidate-ranking.md) | [进展](progress/004-t4-candidate-ranking.md) |
| 005 | 从 0.206 到多证据候选排序：研究与实现交接 | [计划](plan/005-multi-evidence-ranking.md) | [进展](progress/005-multi-evidence-ranking.md) |
| 006 | 同分子式辨别、能量条件与参考谱缺失：研究交接 | [计划](plan/006-isomer-and-domain-evidence.md) | [进展](progress/006-isomer-and-domain-evidence.md) |
| 007 | 可靠证据融合与覆盖缺口：下一轮研究计划 | [计划](plan/007-robust-evidence-and-coverage.md) | [进展](progress/007-robust-evidence-and-coverage.md) |
| 008 | 提交判断、真实候选召回与一致的残差评分 | [计划](plan/008-submission-readiness-and-recall.md) | [进展](progress/008-submission-readiness-and-recall.md) |
| 009 | 高分方案、外部候选可迁移性与新证据 | [计划](plan/009-frontier-and-transfer.md) | [进展](progress/009-frontier-and-transfer.md) |
| 010 | 新结果复核、提交决策与前沿 | [计划](plan/010-submission-review-and-frontier.md) | [进展](progress/010-submission-review-and-frontier.md) |
| 011 | 固定MassBank扩库的实际提交 | [计划](plan/011-massbank-expansion-submission.md) | [进展](progress/011-massbank-expansion-submission.md) |
| 012 | 同分复盘与互补证据研究 | [计划](plan/012-score-plateau-and-complementary-evidence.md) | [进展](progress/012-score-plateau-and-complementary-evidence.md) |
| 013 | 可追溯第二证据引擎：参考谱/analog 支撑审计与排序试点 | [计划](plan/013-analog-second-evidence.md) | [进展](progress/013-analog-second-evidence.md) |
| 014 | 新工作复核、支撑审计与提交判断 | [计划](plan/014-submission-readiness-and-support-audit.md) | [进展](progress/014-submission-readiness-and-support-audit.md) |
| 015 | 候选级analog排序实证与交接 | [计划](plan/015-candidate-analog-ranker-and-handoff.md) | [进展](progress/015-candidate-analog-ranker-and-handoff.md) |
| 016 | 新面板直接检索验证与提交准备 | [计划](plan/016-new-panel-direct-verification.md) | [进展](progress/016-new-panel-direct-verification.md) |
| 017 | 016复核、ICE接入修正与提交判断 | [计划](plan/017-review-and-corrected-panel.md) | [进展](progress/017-review-and-corrected-panel.md) |
| 018 | 同一批分子上分离参考库变化 | [计划](plan/018-same-molecule-reference.md) | [进展](progress/018-same-molecule-reference.md) |
| 019 | 同式候选的特异碎片信息 | [计划](plan/019-specific-fragment-evidence.md) | [进展](progress/019-specific-fragment-evidence.md) |
| 020 | 公开 E006＋PubChem Promo（rel>150）原样复现 | [计划](plan/020-e006-reproduction.md) | [进展](progress/020-e006-reproduction.md) |
| 021 | E006 基线的本地 NP250 验证 | [计划](plan/021-local-validation.md) | [进展](progress/021-local-validation.md) |
| 022 | v12：严格跨库直接参考提升 | [计划](plan/022-v12.md) | [进展](progress/022-v12.md) |
| 023 | v13 关闭 PubChem 提升；Engine1 生成路径研究 | [计划](plan/023-promo-off-and-engine1-generation.md) | [进展](progress/023-promo-off-and-engine1-generation.md) |
| 024 | 留出 NP 的 Engine1 指纹模型复核 | [计划](plan/024-leak-free-engine1.md) | [进展](progress/024-leak-free-engine1.md) |
| 025 | PubChem 在库、测试分布验证面板 | [计划](plan/025-pubchem-present-panel.md) | [进展](progress/025-pubchem-present-panel.md) |
| 026 | 按 PubChem 人气加权的面板 Q，以及公开 notebook 调研 | [计划](plan/026-popularity-panel.md) | [进展](progress/026-popularity-panel.md) |
| 027 | fork 公开 v465（0.456）作为新基线 | [计划](plan/027-fork-v465.md) | [进展](progress/027-fork-v465.md) |
| 028 | 把 PubChemLite 与 ChEBI/LIPID MAPS 候选并入 Engine1 候选池（基于 v465） | [计划](plan/028-e1-pool-extension.md) | [进展](progress/028-e1-pool-extension.md) |
| 029 | 对照 lavinwins/casmi26-top-1-gold，落地高价值差异 | [计划](plan/029-lavinwins-diff-and-joint.md) | [进展](progress/029-lavinwins-diff-and-joint.md) |
| 030 | N1 联合证据提交与冲奖研究路线 | [计划](plan/030-n1-challenger-and-score-strategy.md) | [进展](progress/030-n1-challenger-and-score-strategy.md) |
| 031 | 项目目录整理与冗余清理 | [计划](plan/031-workspace-organization.md) | [进展](progress/031-workspace-organization.md) |

## 研究交接

交接文档保留当时的研究问题和具体操作；今日状态以030进展为准。

- [015候选级analog排序研究](handoff/015-candidate-analog-ranker.md)
- [017修正面板与下一步](handoff/017-corrected-panel-and-next-step.md)
- [022 E006基线与后续](handoff/022-e006-baseline-and-next-step.md)

机器可读的精简研究结果与来源放在[`data/references/`](../data/references/)，大型数据、权重、运行缓存与日志放在本地忽略目录`artifacts/`。具体位置由各轮进展引用。
