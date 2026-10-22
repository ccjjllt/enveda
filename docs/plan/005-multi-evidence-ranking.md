# 005 — 从 0.206 到多证据候选排序：研究与实现交接

日期：2026-10-04（北京时间）。对应[实际进展](../progress/005-multi-evidence-ranking.md)。继承 [004 计划](004-t4-candidate-ranking.md)和[004 进展](../progress/004-t4-candidate-ranking.md)，但以下复核结果优先于旧结论。

2026-10-05接手提示：005已完成实际实验并提交v8（public0.307）。当前研究与旧结论修正见[006计划](006-isomer-and-domain-evidence.md)，不再按本文初始状态重复准备阶段。

## 研究结论与目标

**下一轮主线：在相同的完整候选库上，构建独立谱库 + analog + 神经指纹 + 正向碎裂的分子级排序。** 先取得可比的强基线，再研究多谱条件和计算预算分配；暂不继续单纯增加 FPNet 微调步数，不把训练迁移到 TPU 当作提分方法。

用户目标是最终金牌和奖金。已完成的 v7 submission `56799854` 为 **0.206**。本次官方 API 快照有 2,306 队，项目 metadata 排名 1,566；公开榜第 1/5/14/20 条分别为 **0.471/0.433/0.425/0.418**。这些仅用于认识当前竞争强度，既不是最终私榜线，也不是承诺能达到的目标。[实时榜单](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/leaderboard)

这轮要回答三件事：

1. 同一完整池、同一分子口径下，0.206 管线缺失的证据分别能改善哪些错误？
2. 250 个公开天然产物上的好结果，在新结构组和完整干扰背景下能否保留？
3. 在 T4 和最终 9 小时约束下，怎样把昂贵正向模型用于最有价值的候选，而不是平均花费？

## 必须继承的实际发现

| 发现 | 对研究方向的影响 |
| --- | --- |
| v7 只加载原始 `fp_single_s2.pt`，每分子选原始峰最多的一张谱，20 ppm、最多 2,000 候选、指纹点积；未用 rank-head、cosine、R4 或 merged | 0.206 尚未检验 004 大多数增量，不支持“微调已经没用”或“主要输在 Class3”的推断 |
| R1/R2/R3 只排 275,810 训练结构；v7 排训练 + COCONUT 的 712,199 结构 | 扩池增加干扰；必须重做同池对照，不能用离线 0.675 与 public 0.206 相减归因 |
| 旧 R4 把排序数组里的候选原下标当名次，1,184/1,184 个保存值与原窗口位置一致 | **撤回 R4 0.6257 及其增益解释**。已修代码，仍需重跑；旧 ranks 无法恢复正确排序 |
| cosine 0.7076、local CV 0.9796 的对应逐例产物未找到；当前同名 JSON 是旧结果 | 保留终端报告历史，不能拿它们做逐分子融合依据 |
| L 的 2,453 条谱只来自 Q 的 250 个结构，背景结构为 0 | 即使 Q/L 精确峰 hash 不重叠，仍存在参考库选择偏差；不能直接用“cosine 有命中就接管” |
| GEN500 有 215 个结构/630 条 query 出现在训练 200k；NP canonical 表只改 NP250，不扩展全训练别名 | GEN 不是严格结构未见；先建立实际整组排除后的新观察集，再选择训练方案 |
| 候选导出未接去立体/互变异构匹配键，仍按原 key 去重 | 修研究数据身份与预测导出，指纹输入身份单独保留，避免无意改变预训练模型输入 |

审计：[v7 与 004 证据](../../data/references/submission-v7-audit-2026-10-03.json)。其中日期按 UTC 文件命名，实际最终检查是北京时间 10-04。当前本地 `artifacts/submissions/submission-v7/submission.csv` 是**可见占位输入开发产物**，不是隐藏集逐行反馈。Kaggle 未返回隐藏真值、逐分子排名或 Class1/2/3 分数分解。

### 可复用的数值证据

本次从保存 ranks 重新截断到 25；仍是 shipped key 匹配，不冒称已完成官方 SMILES 规范化复评。所有数字来自旧训练结构池，公开权重预训练暴露未知。

| NP250 分子级方法 | MRR@25 | Top1 | Top25 | 相对 single |
| --- | ---: | ---: | ---: | ---: |
| 最多原始峰 single | 0.67520 | 0.528 | 0.992 | 基线 |
| 全谱 logit 平均 | 0.64828 | 0.488 | 0.976 | −0.02693 |
| 各 adduct 均值再平均 | 0.63631 | 0.476 | 0.976 | −0.03889 |
| 轮流取候选 `rrfuse` | 0.63000 | 0.472 | 0.980 | −0.04520 |
| 原 merged 路径 | 0.66457 | 0.504 | 0.988 | −0.01064 |
| **用真值选择最佳单谱，仅诊断** | **0.74716** | **0.620** | **0.992** | **+0.07196** |

10,000 次分子成对 bootstrap、seed=20261004：全谱均值相对 single 的区间约 **[−0.04971, −0.00500]**；merged 约 **[−0.04300, +0.02159]**。不同方法的质量窗也不同，因此不是纯融合算法的因果实验。merged 文件未保存 key，只能按已审阅代码中的相同分组顺序配对。

真值挑谱让 **51/250** 个分子 RR 改善，说明选择/加权仍有空间；它不是可部署算法，也不是多谱融合能保证达到的分数。旧训练池 single 的 Top25 已达 0.992，而 Top1 只有 0.528，**在这个观察集上**排序比扩大召回更值得优先。不能外推隐藏集也有 99.2% 召回。

已选谱中 `[M+H]+` 170 分子的 MRR 为 0.73783，负模式 48 分子为 0.58303，`[M+NH4]+` 25 分子为 0.51021。组成和结构难度不同，不能据此写死“永远选正谱”；应作为条件特征，并在同分子不同谱之间比较。

R3 的 BCE-only / BCE+rank 结果仍是旧每谱、不截断的汇总。rank-head 的 NP 0.6001→0.6217、GEN 0.7844→0.7611 是有价值的训练观察，但不同 seed/采样与留出交叉未隔离，不能把全部变化归因于 loss，也不能当最终分子 MRR。现有文件缺逐候选输出，需重跑同条件评估。

完整手动复算：[指标与逐分子表](../../data/references/004-metric-reanalysis-2026-10-04.json)。

## 当前竞争方案与前沿研究

已阅读当前按成绩排序靠前的四份公开源码及部分小型依赖源码，没有下载整套大权重、没有复现其分数。多个 notebook 共用上游资产，不算多个独立成功实验。

| 路线 | 已读证据 | 本项目如何使用 |
| --- | --- | --- |
| 公开完整管线 | 谱库、mass-shift analog、FPNet A/B、DreamsFP D、碎片特征、LightGBM、候选衍生及 GLACIER/ICEBERG | 作为架构参考；先补齐独立信息，再研究集成。不要将公开作者自报 0.415 当本项目成绩 |
| 条件化多谱 | v7 仅一谱；本地最佳谱诊断还有 +0.072 空间，简单平均下降 | 学习谱质量/离子条件/候选一致性权重，不混合正负谱的原始 m/z 成一张伪谱 |
| GLACIER / ICEBERG | 结构→谱，能区分同式异构体，不生成新结构；已有 MassSpecGym 来源路线 | 先 GLACIER，小批候选前向；当前 wrapper 对离子条件有限制，缺失分不能写成低相似度 |
| DreaMS / DreamsFP | 高分包的 D 是已训练的指纹模型；不是拿通用 embedding 余弦直接替换 FPNet | 研究与 FPNet 的错误互补。该包 7 块指纹布局与本项目 6,930 位不同，缓存与模型必须配套 |
| analog 结构衍生 | 高分源码包含受约束的局部衍生候选；宽泛生成可能引入更多错误 | 小预算生成新候选，按质量/价态/碎片证据重排，单列新增真值与挤掉真值的数量 |
| MS-GPT / FRIGID / DiffMS | 现有实现依赖分子式或特定指纹输入，不能无代价接到 6,930-bit FPNet | 先做真实 formula top-k 路径；oracle formula 只能作为上限诊断。T4 时延需实测 |
| MARLIN（2026-07） | 不依赖给定真值分子式，DreaMS→4,096 Morgan + 中性质量→block diffusion；论文说明代码/模型将在发表后提供 | 跟踪其思路，当前未核实可下载权重，不作为本轮可执行主线 |
| GenMol V2 | 是 SAFE 片段/模板条件分子生成，未提供本赛需要的谱编码器 | 不能作为现成谱→结构模型；重新训练适配成本目前不优先 |

来源：[v29 完整管线](https://www.kaggle.com/code/wangpenghua/casmi26-v29-fusion-pop-glmh)、[多引擎公开实现](https://www.kaggle.com/code/flexonafft/casmi26-multi-engine-fusion-spectral-reranking)、[ms-pred / GLACIER](https://github.com/coleygroup/ms-pred)、[DreaMS](https://github.com/pluskal-lab/DreaMS)、[DiffMS](https://github.com/coleygroup/DiffMS)、[MARLIN](https://arxiv.org/abs/2607.04774)。具体源码身份、资源和条件限制见本轮 `research-frontier-2026-10-04.json`。

公开 v29 说明其 NP250 与排行榜趋势多次不一致，并自报扩大生成/过强 PubChem 路由曾掉分；这只是作者观察，但与本项目小参考库偏差相符。**候选更多、模型更多、算力更多，都不等于排名更好。**

### 奖金路线的资产选择

- 继续使用已有 `prvsiyan/casmi26-fp-models-v2` **v1 / CC0**；记录公开 checkpoint 暴露未知。COCONUT 固定 v2，保留原始来源与 attribution。
- `ahmedberatozer/casmi26-v4b-models` **v4**、`casmi26-fpnet-full1` **v1**、`casmi26-v2-pool` **v1** 的 Dataset 为 Other，描述要求按源许可非商业使用；`fpnet-full1` 明确训练包含 NP 示例，因此不能以 NP250 证明其未见结构效果。
- 上述包和 DreaMS 的资格已有[745072](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745072)、[745604](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745604)、[745615](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745615)讨论，本次读取尚无主办方答复。**状态为待澄清，既不推断全部允许，也不把 NC 一概判为禁用**；比赛自身训练源也含 NC 组件。
- 已明确禁用的是 NIST 来源方案；不要用 NIST23 模型生成的 PubChem 预测谱 atlas 替代 MassSpecGym 权重。已明确准许的 GLACIER 具体权重路线与包装 Dataset 的附加文件许可分开记录。[NIST 主办方答复](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744470#3530300)
- `prvsiyan/chebi-lipidmaps-casmi26` v1 标注 CC BY-NC-SA 4.0；不能沿用“公开所以无条件可用”的假设。候选扩池可分别核原始数据库与具体发行版，再构建本项目所需子集。

资产快照：[candidate-assets](../../data/references/candidate-assets-2026-10-04.json)。研究与可独立实现的代码继续推进，不等待许可问题解决才做全部实验，也不擅自发布帖子或联系他人。

## 数据设计：Q、L、C 三者分开

先流式生成全训练结构的 `raw_key → canonical_group_key → output_smiles → official_match_key` 映射，记录 RDKit 实际版本，官方对齐使用 2026.03.3。NP250 中 16 个 key 规范化后改变；14 个目标键在全池还有其他原键记录，其中 8 个出现在旧训练 200k。需对两端 SMILES 真正规范化确认并扩展整组排除，不能只比较字符串。

指纹保持原模型规定的结构/bit 定义；输出去立体/规范化键用于匹配和去重。不要只改 SMILES 而保留错位的 key、mass、fp 行。原始身份与输出身份同时存储；互变异构等价候选可以先各自打分，再聚合为一个最终输出结构。

| 轨道 | Q | L / 本项目训练 | C | 能回答什么 |
| --- | --- | --- | --- | --- |
| NP 开发观察集 | 已有 250 结构/1,184 谱 | 所有背景结构都纳入；按任务排除 Q/重复谱或整结构组 | 固定完整 train+COCONUT | 与历史衔接；反复使用过，不当最终保留集 |
| 新 timsTOF 结构组 | seed=20261004，先固定约 500 个非 NP、非旧调参结构，按质量/模式分层 | 全来源排除规范化结构及谱重复组；新微调数据同样排除 | 可以保留无谱真值结构 | 本项目训练/参考库未见的结构排序，公开 checkpoint 暴露另标 unknown |
| 已见结构独立谱 | 从同一结构不同来源/采集组选择 Q | 保留其他来源独立谱 + 完整背景，排除精确/近重复 | 同一完整 C | 谱库信息的真实增量、错误高相似度匹配 |
| 库外模拟 | 使用上述有标签 Q | 与结构留出轨道相同 | 显式移除真值等价结构 | 受约束生成的新增召回；不等同隐藏 Class3 分布 |

无稳定采集 ID 时以可得来源、谱 hash、adduct/CE 和近重复关系构造组，并记录仍可能共享采集的局限。没有直接参考谱的候选也保留在 C，不能只评估“有 L 的那些结构”。训练 ranker 时按分子分组，不能随机拆候选行或谱行；每分子总权重相当，避免多谱分子被重复计权。

## 实验序列和研究假设

以下是人工启动的研究实验，不新增测试文件、测试框架或 CI。每步产出下一步所需的预测、分数与逐分子错误表；没有必要为每步建立审批或门控。

### E0：同池复现与身份修正

**假设：离线好分与线上弱分的一部分差异来自候选难度与未部署组件。**

1. 固定 v7 原模型，在旧 train-only C 和实际 712,199 并集 C 分别运行相同 Q，先记录未规范化结果以对齐历史。
2. 在并集 C 预计算合法导出 SMILES / 匹配键，测规范化后的候选数、去重损失和 MRR，单列这一变化。
3. 对相同候选与相同选谱分别评估原 head / `head_rank_5000.pt`。rank-head 仅是头部，必须先加载对应 encoder 再覆盖 head，不能直接 `strict=False` 忽略参数差异。
4. 为每分子预先定义统一质量候选 C(q)：单谱窗与多谱质量一致性窗分别作召回对照；排序方法之间使用相同 C(q)。20/10/5 ppm、cap=200/2000/无 cap 都重新测**完整池**召回与规模，不直接采用旧 10ppm+200 建议。
5. 不用绝对质量误差截断同分子式异构体作为默认；先保留同式组，必要时用廉价指纹分 shortlist，并报告截断造成的真值损失。

保留 `truth_in_full_pool / truth_in_mass_window / truth_in_shortlist / final_rank` 四层。这里的“真值”只来自有标签研究 Q。公开占位 test 只用于运行规模和格式检查。

### E1：完整背景谱库与 analog

**假设：单一指纹不能利用已知参考碎裂及相邻结构的共享片段，独立谱库与 shifted analog 提供互补证据。**

- 在全部合法训练背景建立按中性质量、极性、adduct 的检索索引；流式/分片处理 MS2，不把 3GB Parquet 全展开成 Python 对象。
- 参考谱先按结构聚合，再取结构 top-k。特征包括最大/次大相似度、匹配峰数、解释强度、谱熵相似度、同 adduct/CE 差、不同来源支持数和首二候选差距。
- 分开测普通匹配与质量偏移 analog；同一个峰不能在直接和偏移匹配里重复计证据。负模式、失水离子等单列。
- 谱库“有候选”“高 cosine”“真值命中”是不同概念。先保留 FPNet 与谱库全部候选证据，由 ranker 学习可信度，不用一个未校准阈值直接覆盖 FPNet。
- 已见独立谱与结构未见两轨都测；后者如果没有真值参考谱，analog 的目标是提供相近结构证据或衍生种子，不冒称直接找回不存在的真值参考。

### E2：分子级多谱与多证据 ranker

**假设：有效谱、离子条件和候选之间的共同支持比等权平均更重要。**

- 缓存每张 Q 的 encoder/head 输出及每候选各通道分数，GPU 前向只做一次；CPU 聚合/拟合独立进行。
- 先用可解释的线性/小 LightGBM ranker；正样本为真值等价组，困难负例来自相同质量窗、尤其同分子式结构。按分子组做 OOF，最终以 MRR@25 选方案，不把训练 BCE 或 NDCG 当比赛成绩。
- 分子特征：有效峰数、保留强度比例、谱熵、CE、前体/中性质量一致性、adduct、极性、同条件重复数、模型分歧、首二 margin。
- 候选聚合：各条件内 max/mean/top2，再跨条件保留 max、平均、方差、支持次数及互相矛盾的证据。避免高重复谱数的条件主导。
- 首轮比较 single、等权对照、只学选谱权重、分子候选 ranker；保持同一 C(q)。正负谱不直接平均原始峰坐标，merged 作为单独模型路径且保留其合法输入条件。
- 同源指纹相关性很高，不能将两个 checkpoint 当两份独立物理证据。是否接 DreamsFP/DreaMS 取决于条件适配、资产可用性以及**逐分子错误互补**，不取决于模型名新旧。
- 别把旧 160 特征 ranker 接到删减管线后补零。公开模型的特征名/顺序、指纹块、缺失处理都必须按其原定义；本项目新特征训练自己的 ranker。

### E3：GLACIER 同式重排与自适应预算

**假设：FPNet top-k 中的同式异构体错误能由碎片解释修正；收益集中在排名不确定且离子条件受支持的分子。**

先选 20–40 个有标签分子、预先包含容易/困难及不同谱数的样本，按不依赖真值正确性的规则形成 top-10/25/50 shortlist。记录这一小样本如何抽取，不只选已知会改善的例子。随后有价值时扩大到整个观察集。

- 从已核实 MassSpecGym 模型开始。当前公开 GL wrapper 仅支持正模式 `[M+H]+`/`[M+Na]+`，高分 v29 进一步限 `[M+H]+`；`timsTOF→QTOF` 和 CE 5eV bucket 是该 wrapper 的映射假设，需要保留对照。
- 先复用预训练推理，不在 T4 上重训完整 forward 模型。Colab 当前 Python3.13 与公开 cp312 wheels 不兼容；准备匹配 Python/CUDA 的依赖或原始实现，再做小批前向。
- cache 键包含结构、adduct、CE、instrument、模型/预处理版本。输出相似度同时保留匹配峰数和解释强度。
- unsupported / inference failed / no useful peaks 与有效但低分分开；ranker 接收缺失标记，不把未预测负谱写成 0 分惩罚候选。
- 比较无 forward、固定 top25、按 margin/同式歧义动态 top-k 三种方式，记录 RR 改善/恶化、shortlist recall、候选秒数、峰值显存。
- 只在支持子集有效时按证据加入最终排序；不因个别 forward 高分直接硬覆盖所有分子。ICEBERG 作为第二 forward 视角，在 GLACIER 有实际边际价值后再比较。

### E4：候选覆盖与小预算生成

**假设：缺失真值的分子需要新候选，但无条件扩池会降低已覆盖分子的排名。**

先根据 E0 的完整池缺失与公式/shortlist 统计安排资源。现有 NP250 在训练池真值普遍可用，不适合衡量外部库召回；新建“外库独有结构”观察子集，或用明确标记的库外模拟轨道。

1. 优先合法来源的结构子集与 analog 局部变换，测新增规范化结构、质量覆盖、真值召回和干扰增加，不只报告库规模。
2. 流行度/数据库出现频次作为弱先验单独消融；它可能偏向常见天然产物，损害新结构。不给 PubChem 来源默认最高优先级，不下载全库只为堆规模。
3. 公式候选从实际 precursor/adduct、多谱一致性和碎片/中性丢失推断；无 MS1 同位素包络时不假装具有此信息。可执行的首个正模式方案为 [MIST-CF](https://github.com/samgoldman97/mist-cf)：核实具体 NPLIB1 checkpoint 来源后，以 SIRIUS decomp 枚举质量对应公式，再用该模型排序。记录 SIRIUS/decomp 版本与依赖；其当前实现限正模式，负模式单列 unsupported 或另做独立方法，不能隐式套用。报告 formula top-k recall，不能偷用训练文件的真值 formula 做真实推理。
4. 再试现成有条件生成器的 10–20 分子小试，分别记录 oracle-formula 与 predicted-formula 路径。MIST 的4096指纹、本项目6930输出、公开DreamsFP D的selected-bit契约必须分开。通用DreaMS提供的是1024维embedding；MARLIN中的4096 Morgan来自任务专用预测头，不能从通用权重直接取得。需要的适配器在训练集学习，不用验证真值指纹替代预测。
5. 生成候选统一进入已有 ranker，记录“新增命中”和“挤掉正确候选”；不保证总是预留固定若干 slot 给生成器。

MARLIN 目前作为方向资料。只有可用权重/代码出现且满足来源条件后，才成为具体实现任务。GenMol 不作为本轮主线。

## 需要实现/修改的代码与产物

以下为交接范围，不代表已实现。本轮实际修改见配对进展。

| 路径建议 | 实现职责 | 必须留下的研究产物 |
| --- | --- | --- |
| `src/chemistry.py`, `src/splits.py` | 全池规范化映射/别名整组排除；区分预测导出、评分和 split 键；修读取合法 `<i8` npy 的现有问题；保留 isotope/charge 语义 | `structure_identity.parquet`、实际新 Q/L/train manifests、版本与 alias 数 |
| `src/candidates.py` | 固定并集候选池、统一分子质量窗、多谱一致性、阶段召回 | `candidate_manifest.json`、每分子各阶段候选/真值标记 |
| `src/retrieval.py` 或 `src/library_features.py` | 完整背景普通/shifted/entropy 检索，按结构聚合，多来源支持 | 稀疏候选级相似度表；完整 L 身份 |
| `src/models/fingerprint.py` | 与 v7 完全一致的输入路径、原 head/rank-head overlay；缓存逐谱输出 | 逐谱 logits/embedding、权重 SHA、各谱有效峰与条件 |
| `src/ranking.py` | 分子组 OOF；质量感知多谱聚合、缺失特征、轻量 ranker | 每折预测、特征顺序、逐分子最终 top25、RR 与 delta |
| `src/forward_rerank.py` | GLACIER 原始模型 adapter、条件映射、缓存与预算 | 实际预测谱/相似度、unsupported原因、时延/显存 |
| `notebooks/archive/colab/005-multi-evidence-ranking.ipynb` | 按 E0→E3 人工分格运行；依赖与设备记录；下载/训练/评估分开 | 唯一 run 目录及完整原始输出，不覆盖 004 同名文件 |
| `notebooks/submissions/submission/` 新版本 | 挂载相同模型/候选/特征定义；自包含、离线；按实际 NG 分块；输出1–25个有效候选 | 明确 notebook/version→assets→submission/ref→score 的对应关系 |

`src/train_fingerprint.py` 当前训练入口只算 BCE，不能复现旧 notebook 的 BCE+rank；要继续训练时才把真实 rank loss 移入该模块。`src/evaluation.py` 缺 RDKit 会回退字符串匹配：005正式指标路径需明确用官方 RDKit helper，不把回退结果标为官方效果。代码这些缺口没有进入 v7，不能称为已证实 0.206 原因。

每次实验保存：Q/L/C 版本、fold、seed、候选身份/特征schema、权重SHA、设备/软件版本、耗时/显存、最终预测、各阶段召回与逐分子 RR。证据文件中不要保存密钥、私有数据或完整训练谱；大产物继续放忽略目录/私有云存储。

## 计算安排与执行顺序

**Colab T4 继续作为已授权的主研究平台。** 本轮没有使用 GUI，也没有新的远端训练。当前 Codex 仍只暴露 Colab 连接工具，完整执行未可用；这不阻塞本地实现和交接。能实际操作 notebook 的已连接客户端可按本计划人工运行；不需要重新设计通用 MCP 服务。

建议首轮预算为分阶段上限，而非速度承诺：E0 神经前向/缓存约 0.5–1 T4 小时，E1 CPU 索引与缓存 1–3 小时，E2 CPU ranker 与聚合 0.5–1 小时，E3 小试 0.5–1 T4 小时。先用真实少量 batch 估计完整运行成本，CPU 化学处理尽量在本地完成再连接 GPU；实际时长覆盖这里的估计。最终推理仍为 Kaggle CPU/T4、关闭互联网、9小时内。

公开 v29 中 ICE 5,400 秒、GL 4,000 秒是其配置预算，不是实测上限保证；源码的开发 shortcut 只跑 12 分子。作者自报隐藏重跑4–5小时，不是本项目可复用的时延证据。

TPU v5e-1 适合后续已明确的批量 encoder/head 训练，不加速 RDKit、LightGBM、动态候选检索或现成 CUDA forward runner。这轮先提高信息和排序质量；等固定损失、固定形状训练确有收益，再迁移到 PyTorch/XLA，分别记录编译与稳定吞吐。

执行顺序为：**E0 同池重建 → E1 完整背景证据 → E2 条件化排序 → E3 同式重排 → 根据缺失证据选择 E4**。首个候选新版以原始公开 head + 完整谱库/analog + 自有 ranker 为主，rank-head/forward 作为独立可比较变化。提交记录与离线指标分开，不把公开榜反复试分当逐分子标签来源。

本计划未声称任何 005 方法已经提分。达到当前高分区域需要复现强管线，再在完整背景、负离子/多谱条件、稀有结构和预算分配上取得稳定增量；金牌和奖金最终由私榜与比赛规则决定。
