# 075 当前研究

更新于 2026-11-26。比赛指标为 MRR@25。本方已确认最高 **0.506**（Pablo v29，`58820476`），同版本复投 `58920333` 尚待评分。公开榜首 **0.562**。Ishan 的公开 **0.509** 使用尚无生成代码的 VTLP 中间分。 [正式回执](../../data/references/075-public-0506-submission-2026-11-24.json) · [公榜](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/leaderboard)

## 仍有价值的方向

| 方向 | 实测证据与当前状态 |
| --- | --- |
| 融合后的同式前五重排 | 原 ICEBERG/GLACIER 分和实测谱，固定前五成员与第 6–25 位，零新增预测。NP60、E180 121、MassBank 101 的净 MRR 分别为 **+0.014444 / +0.058127 / +0.013531**。E180 的历史参比未按来源遮蔽。赛用包在 Kaggle CPU 归档上逐题复现 **181/181**，T4 实跑尚无确认。 [赛用入口](final_native_top5_production/build.py) · [逐题证据](../../data/references/075-c39-depth-routing-2026-11-25.json) |
| 等预算第 5 位补召回 | NP60 找回一题，净 **+0.003333**。MassBank 110 找回 **U062**，净 **+0.001818**，前五无流失。缓存复放中每题最多 4 个挑战者保住该命中，推算每模型 **633** 个追加谱–结构请求。FR 与 C34 的等预算盲选已完成，独立物理评分及整线收益待验证。 [FR 盲选](https://www.kaggle.com/code/jiyujia/enveda-075-mb110-fr-rank5-plan/versions/1) · [前向入口](virtual_np/build_mb110_fr_rank5_run.py) |
| FIDDLE v2.4.0 分子式 | CPU 预检完成，E180 121 题的 **299** 张明确 eV 的 timsTOF 谱盲预测已完成。独立计分以每题最多五个不同分子式对照 C34 并检查训练身份重叠。NP60 和 MassBank 110 缺匹配的仪器或明确 eV，未纳入这个检查点的比较。 [盲测 v2](https://www.kaggle.com/code/jiyujia/enveda-075-fiddle-v240-blind-formula/versions/2) · [计分入口](formula_fiddle/score_e180.py) |
| 可重建的结构编辑先验 | 公开单步规则和质量差审计保留。两次 8,192 母体供给运行超时，尚无封存供给或拟合分数；重启时先解决供给规模。 [供给入口](virtual_np/build_transform_prior_supply.py) · [拟合入口](virtual_np/build_transform_prior_fit.py) · [规则质量差](../../data/references/075-np-virtual-mass-delta-2026-11-25.json) |

## 排查优先级

- MassBank 110 的 45 道原各来源均漏中，29 道真结构未到 C34 第 26–100 位。其中 22 道仍在 ±10 ppm 窗内，17 道通过粗筛但排在第 100 位以后，5 道在粗筛被截。优先比较同一质量窗、同样预测预算下的结构排序。 [逐题诊断](https://www.kaggle.com/code/jiyujia/enveda-075-mb110-depth50-trace/versions/1)
- 本征带电的 011 用 `[M]+` 后落回窗内，但冻结主库和 PubChem 分区都没收录；033 已过全指纹筛选，排到第 85；034、035、178 在索引里，未通过现有跨库去重或主库截断。扩展名单用比赛规范键去重，并把入围、融合前后名次分开计数。 [五题审计](../../data/references/075-np-five-miss-index-audit-2026-11-24.json)
- 当前训练研究暂停。恢复时优先测已验证的最终前五重排。设计新模型时，固定真实候选与原排序，按来源、规范结构和近骨架整组留出，报告首位得失、前五得失和净 MRR；同质量错配谱用来确认模型确实读取谱。 [已终止实验和恢复标签](../../archive/research-archive.md)
