# 004 — Colab T4：真实候选排序、目标域训练与获奖路线交接

对应进展：[004-t4-candidate-ranking](../progress/004-t4-candidate-ranking.md)。日期：2026-10-01。

2026-10-04 接手提示：004已进行T4训练并提交v7，public 0.206。旧指标与验证边界的更正见对应进展开头；下一轮使用[005计划](005-multi-evidence-ranking.md)，不要按本文件初始状态重复R0。

## 本轮决定

主线是 **完整结构候选召回 → FPNet 真实打分 → 独立多谱融合 → 学习排序/领域训练**。随后用正向谱预测增加同分子式异构体的辨别能力，为库外分子保留一个小预算生成分支。当前使用 Colab T4；不先迁移 TPU，不从头训练大型生成模型，也不把时间继续耗在 MCP 或 GUI 操作上。

用户已授权 Colab T4 训练与手动研究。本轮已确认真实 T4 环境，完整训练实现尚待接手。用户随后明确指出 Computer Use 太慢，后续远端执行使用能暴露 notebook 执行工具的 MCP 客户端；不要默认继续 GUI 输入。这个执行限制不阻止完成代码和研究交接。

目标为最终奖金和金牌。2026-10-01 实时公开榜分页取得 2,001 队，第一条 0.471、第五条 0.423、第十四条 0.409。奖金前五；按当时队伍规模和官方公式，金牌约前十四。**这些是当前竞争参考，不能当作最终分数线或收益承诺。** 最终以私榜和资格为准；不以反复追逐公开榜微小变化代替研究。来源与时间口径见 `data/references/competition-update-2026-10-01.json`。

## 接手者首先继承什么

代码基点是 `5e22537`，工作树另有 003 配置与本轮文档修改，必须保留。已有：

- `data/raw/train.parquet`：3,033,286,496 B，2,539,608 行、18 列；无 `molecule_id/spectrum_id`。275,810 是 shipped `inchikey14` 去重数。
- `src/data.py`、`src/retrieval.py`、`src/evaluation.py`：可复用流式读取和 B0 基础，但评分、聚合与部分注释仍需按本计划修正。
- `artifacts/001-baseline/split_manifest.json`、`b0_metrics.json`、`b1b2_report.json`：历史实验记录，保留原样，不覆盖成新口径结果。
- `artifacts/001-baseline/weights/fp_single_s2.pt` 和 `fp_merged_m1.pt`：各 144,110,255 B，来源 `prvsiyan/casmi26-fp-models-v2` v1。
- 官方 de novo 教程 v9 是从头训练示范；Analog Propagation v32 是当前 FPNet 参考。公开权重实际训练结构列表未知。
- 原 `notebooks/baseline/` 仍不是完整有效的离线提交模型；尚无本项目排行榜分数。

## 已有结果的重新解释

1. **0.988 是非空输出率。** 不是 98.8% 真值召回。当前 unseen 从参考谱库剔除了真值结构，而输出结构也只来自该谱库，零 MRR 是这一候选构造的结果。
2. **seen 0.992 的干扰难度很低。** 参考库只有这 500 个评估结构的 3,712 条谱，平均约 1.24 个输出候选；行号不交叉尚不等于独立采集或无重复谱。
3. **B2 的 0.995–0.996 不构成融合增益证据。** merged 把参考库成员并入 query；scorefusion 也把原参考谱再次当查询参与融合。不能保留“偏乐观但可比较趋势”或“负模式有效”的结论。
4. **500/500 与 train 相交不证明 checkpoint 训练见过它们。** 留出本就取自 train。称为 `holdout_vs_competition_pool_overlap`，checkpoint 暴露仍为 unknown。
5. 当前 scorer 在排名前去重，偏离官方。例如 `CC;CC;CCO` 对真值 `CCO` 的 RR 应为 1/3，不能压缩为 1/2。

这些修正服务于选择有效方法，不搭建 CI、测试文件或形式化门控。用少量手工例子核对后直接开展研究。

## 目标域与数据划分

本轮标量扫描确认 `enveda-np-examples` 为 **1,184 张谱 / 250 个 shipped 结构，全部 timsTOF**。这 250 个结构与原 seen/unseen 各 500 结构的交集均为 0。原 unseen 只有 322/500 个结构带 timsTOF 谱；原 seen query 中 372/500 为 timsTOF。此前验证漏掉了最相关的公开天然产物示例域。

建立一个 NP/timsTOF 主观察集，并保留全库辅助观察集；NP250 不是隐藏集的替身，也可能含公开模型已见结构。

| 轨道 | Query Q | 训练/参考谱 L | 结构候选库 C |
| --- | --- | --- | --- |
| 已见结构、独立谱 | 目标域独立采集的谱组 | 可保留同结构其他来源谱，但剔除 Q 和重复谱组，加入完整背景干扰 | 合法结构库，允许含真值 |
| 未见结构 | 固定整组留出的规范化结构 | 跨全部来源排除这些结构；从头训练对照也排除 | 允许真值作为无谱结构存在，单独计算其覆盖 |
| 库外近似 | 独立结构留出 | 同上 | 从检索池剔除真值，测生成增量；不声称这些公开结构是真实未知天然产物 |

建议先把 NP250 固定为结构留出主观察集，训练侧跨所有来源移除其规范化结构；需要在 NP 域调参/微调时，再预先分成开发/保留部分或按结构做 5 折，不反复使用整个 NP250 选模型。既有随机 500 结构可作辅助，不能替代目标域分析。保存 canonical key、原键、来源、行号和重复谱组到新 manifest。

每分子先定义固定 Q/L，single、scorefusion、merged 使用相同 Q/L/C。single 以不依赖真值的规则选一张 Q；融合只能读 Q。先做同极性/同 adduct 融合，再比较候选级跨条件聚合，避免把跨 adduct 物理量硬合成一张伪谱。

## 评分和化学结构语义

官方当前 scorer 为 v13，RDKit 2026.03.3：SMILES 解析 → `TautomerEnumerator.Canonicalize` → InChIKey14。它**没有**显式 `RemoveStereochemistry`，不去候选重复；空分号项删除，非空无效/重复项保留排名；去空后零候选或超过 25 个候选会拒绝。

主办方新增答复确认 hidden solution 在上游已去立体。因此分开实现：

- `prepare_prediction_smiles`：候选导出预处理，可先去立体，再产生规范 SMILES；保留同位素/电荷语义，不随意中和或删盐。
- `official_match_key`：忠实复现 v13 helper，不私自把额外预处理塞进 scorer。
- `canonical_group_key`：明确记录用于 split 的去立体与互变异构处理顺序，使等价结构不跨训练/留出。
- 候选去重发生在**预测生成**阶段，评分直接读取最终输出顺序。无法规范化的结构留记录，不冒充成功匹配。

手动例子包括：重复错误候选占位、空项、无效 SMILES、互变异构、主办方讨论中的立体反例。具体原逻辑和链接已保存在本轮比赛证据中；不要用相似 Tanimoto 代替精确结构命中。

## 第一条模型路线：FPNet + 候选库

### 最重要的输入契约

公开 checkpoint 的 head 输出是 **6,930 位**，不是普通 4,096 位 Morgan。

```text
concat[
  Morgan radius=2, 4096 bits,
  Morgan radius=3, 4096 bits,
  RDKit maxPath=6, 2048 bits,
  MACCS, 167 bits
]  # 共 10,407 位
再按原 fp_bits.npy 取位 → 6,930 位
```

映射在 `prvsiyan/coconut-casmi26-candidates` **v2**，`fp_bits.npy` 为 55,568 B。下载 checkpoint 本身不够：不得截取前 6,930 位、重新选位后复用旧 head，或把 logits 直接当作 MIST 的 4,096 位条件。

沿用 v32 的 26 类 adduct 顺序、5 类 instrument 映射、`prep_peaks`、碰撞能和正负模式编码。原峰处理是 50 Da 窗内优先 8 峰再补足到 128，并带前体过滤、强度阈值和平方根变换；本地 `window_50da` 的平均配额算法不是同一个算法。

建立结构候选库与参考谱库两个独立对象。先用训练结构 + COCONUT，按证据再增补 ChEBI/LIPID MAPS/PubChem。质量过滤使用 float64 和 adduct 修正；逐步观察 5/10/20 ppm 的真值召回，不能仅按质量误差截掉大批同分子式异构体。当前数据没有明确提供 MS1 同位素包络或 CCS，不设计依赖这些隐藏输入的方案。

### 首轮实验顺序

| 实验 | 变化 | 要回答的问题 |
| --- | --- | --- |
| R0 | T4 + 固定公开 single checkpoint 的真实谱前向 | 权重/输入是否一致；实际吞吐、显存；不输出模型效果结论 |
| R1 | 固定 Q/L/C：mass-only → cosine → FPNet single | 真值在哪一步丢失？神经指纹是否提供排序增量？ |
| R2 | 同一候选池，single/logit 均值/候选级融合/原作者 merged 对照 | 多谱收益来自哪里？跨条件混合是否反而损失信息？ |
| R3 | 固定 encoder 训练头，随后末两层解冻；另设从头训练对照 | 目标域适配是否有效？公开 checkpoint 暴露未知与严格泛化分开解释 |
| R4 | FPNet + cosine/analog/质量/元数据的简单重排器 | 同候选集下能否修正互补错误？OOF 或独立 query 组生成 ranker 样本 |

开始真实训练时建议先用 200–500 步估计成本，batch 从 16/32 选能稳定工作的值，峰数 128；这只是预算建议，不是必须满足的门槛。依据测得的 step 时间再安排完整实验。以结构均匀抽样并限制每结构谱数，比较 timsTOF 混合比例，避免 115 万 enveda-180 谱单纯凭数量主导训练。

T4 首轮可尝试 FP16 AMP/GradScaler，m/z、正弦嵌入相关计算与质量过滤保持所需精度；不能照搬 TPU 的 BF16 方案。BCE 预测指纹，再加同质量窗口候选排序损失；先在训练结构中构造负例并排除真值等价结构，不从验证标签调负例。

作者训练代码默认将候选指纹全部展开后放 GPU。按公开池 711,705 × 6,930 的 uint8 数组计算，单此项约 **4.59 GiB**；Colab 实测主机内存只有约 12.67 GB，不能把全谱对象和全部解包指纹塞进内存。使用 packed bits/磁盘映射，按 batch 搬候选；谱按分片读。推理 checkpoint 没有已确认的完整优化器状态，把它当微调初始化，不称精确续训。

## 两条后续增量路线

**同分子式重排：ICEBERG / GLACIER。** 它们是结构→谱预测模型，本身不产生新结构。先用来源核实的 MassSpecGym 权重，对 FPNet top-10/25/50 候选做小批推理，衡量同候选池下的 RR 改善/恶化和每候选耗时；再决定是否扩大到 top-100。官方标准训练要求 2×24 GB，不把单 T4 完整重训作为默认。缓存键包含结构、adduct、CE、instrument 和模型版本。只使用规则允许的版本，排除 NIST 训练权重。

**库外生成：分子式候选 + MS-GPT / FRIGID，FoAM 作有限探索。** 优先读取现成权重，先做 10–20 个分子的可行性比较。MS-GPT 和 FRIGID 的公开结果依赖已知分子式和 MIST 4,096 位输入，本赛推理不能偷用训练真值分子式。分别运行 oracle-formula 诊断和预测 formula top-k 的真实路径，测公式召回、有效/独特/新增候选率、生成池真值召回及最后 MRR。它们在 RTX 6000 Ada 上的速度不能作为 T4 承诺。FoAM 需要反复调用正向模型，先限定单分子的调用数与时长。

DreaMS 可作为 Class1 检索或冻结编码器+指纹头的对照。先看与 FPNet 的错误互补，再决定微调；其默认输入域对负模式、高 m/z 需单列观察，不能因为是基础模型就预设更优。

研究资源先主要用于 FPNet/候选排序；forward 重排其次，生成保留少量探索预算。根据每单位 T4 时间带来的新增命中与 RR 改变调整，不按模型大小排优先级。

## 需要实现的代码：具体交接

以下为大部分尚未实现的工作，可按三次小交付推进，不需要一次搭完整框架。

| 代码/产物 | 必须完成的行为 | 复用与注意 |
| --- | --- | --- |
| `src/chemistry.py` | 上游去立体、官方 match key、split key、指纹 10,407→6,930 映射 | 明确三个用途，不混写一个 normalize |
| `src/evaluation.py` | 按最终候选位置计算 RR；非空率与真值召回分开；逐分子结果 | 替换当前评分前去重与静默截断；保留旧报告，不宣称分数可直接横比 |
| `src/data.py` + `src/splits.py` | 流式预处理、instrument family、重复谱组、NP/辅助 Q/L manifest | 已有 train/行号/来源可复用；注释同步真实 schema |
| `src/candidates.py` | 合法结构库、canonical 去重、float64 质量索引、packed 指纹 | 分开 C 与 L；每阶段记录真值是否存在 |
| `src/models/fingerprint.py` | 与 checkpoint 完全一致的 FPNet 和 feature contract | R0 可用 artifacts 原定义研究；正式入库先确认 notebook 代码许可/归属，不能把 Dataset CC0 自动套到代码 |
| `src/ranking.py` | single、多谱候选级聚合、简单学习排序 | 修正 `aggregate_by_molecule` 非真正轮转且可能超 25 的实现；选择一种明确规则 |
| `src/train_fingerprint.py` | 冻结编码器仅训练头/局部解冻/从头训练模式，实际谱与标签，保存 optimizer/config/split | 200–500 步小预算估计，再定完整轮数；不对随机假输入训练后宣称实验完成 |
| `notebooks/colab/004-t4-fpnet-feasibility.ipynb` | R0 环境、权重读取、真实谱前向与 JSON 结果 | 本轮提供；不等同上述训练和评分流水线 |
| `notebooks/colab/004-t4-candidate-ranking.ipynb` | R1–R4 手动运行入口 | 后续实现；不硬编码本地路径或当前 test IDs |
| `notebooks/inference/` | 自包含/显式挂载输入的离线 Kaggle 推理、动态 ID 输出 | 最终 <=9h；`push` 不同步整个 src，提前打包代码/依赖/权重 |

建议交付顺序：

1. **候选和评分可解释**：chemistry、固定 Q/L/C、逐分子候选召回与官方 RR；把已有 B0 与 FPNet 接到同一入口。
2. **真实 T4 模型研究**：R0 后做 R1/R2，接入训练脚本做 R3，再做 R4；把优胜方案整理成可提交的离线 notebook，取得本项目第一次隐藏重跑分数。
3. **补互补能力**：在失败分子分类结果上选择 forward 重排、DreaMS 或生成分支，不同时重写所有模型。

### 每次实验保存什么

`artifacts/004-research/<run-id>/` 保存 config、environment、metrics、checkpoint、日志和逐分子表。逐分子表至少包含 `query_group_id/canonical_key/split/source/instrument/mode/n_spectra/true_in_pool/true_after_mass/true_after_truncation/rank_before/rank_after/rr/seconds`；旧的 `coverage` 更名为非空预测率，不能改名后仍指同一含义。

只报告有真实观测的指标：候选池与各过滤阶段真值召回、MRR@25/Top1/Top25、候选存在条件下的排序、分层结果、训练 loss、稳定吞吐、峰值 CUDA 内存、推理总时长。小观察集同时给出命中数量和逐分子变化，必要时做分子级配对重采样；这属于研究分析，不是测试门控。

## 尚未解决的接口与执行条件

- 当前 Codex Colab MCP 只暴露连接工具，不能执行 cell；真实 T4 环境已通过 UI 确认，但用户已要求停止低效 Computer Use。下一执行者可在 OpenCode 等已能动态发现工具的客户端继续，必须明确目标 notebook 与 runtime，而不是复用“连接成功”的文字结论。
- 第一次真实 FPNet 前向仍待执行，映射/候选评分和训练更未完成。本轮没有新增模型分数或排行榜提交。
- 既有 Kaggle 排队作业未自动取消。Colab T4 的当前在线状态以后续实际查询为准；本轮不绕过用户对界面操作的限制。

## 来源

- `data/references/competition-update-2026-10-01.json`：实时榜单、奖项、官方 v13 和新讨论。
- `data/references/model-strategy-2026-10-01.json`：原始 FPNet 输入/输出契约、官方模型路线与预算证据等级。
- [立体信息说明](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744556#3530612)
- [官方评分代码](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank)
- [GLACIER/ICEBERG](https://github.com/coleygroup/ms-pred)、[FRIGID](https://github.com/coleygroup/FRIGID)、[MS-GPT](https://github.com/VIKI623/MS-GPT)、[FoAM](https://github.com/coleygroup/foam)、[DreaMS](https://github.com/pluskal-lab/DreaMS)
