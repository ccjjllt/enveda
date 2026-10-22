# 010 — 新工作复核与下一次提交判断

日期：2026-10-14（北京时间）。对应[010计划](../plan/010-submission-review-and-frontier.md)。复核起点为`cee0641`，工作树干净；读取009后续全部四项主要结果，独立重算关键比较，并刷新官方、公开源码与研究论文。新结果保留在`artifacts/010-*`，精简数字见[研究结果](../../data/references/research-results-010.json)，原始009结果未覆盖。

## 决定：分开判断两项改动

**建议将“v9r2＋固定539个MassBank新结构”制作成一次单一变化的验证性submission；条件多谱规则暂不采用。** 这改变了009中把扩库零增益和R3融合退化合起来判NOT_SUBMIT的结论。一次验证性提交不等于最终获奖方案，也不保证public上涨。

当前仓库的009 Colab notebooks是研究计算入口，**尚不是已运行通过的Kaggle扩库提交成品**。本轮完成了应不应该推进的证据复核，没有push/submit，不能声称v10已经就绪或已提分。下一步是把539结构及同口径指纹作为离线输入接入原v9，跑通Kaggle版本后做一次外部验证；不必为此再开启一轮长期训练或等待所有研究方向结束。

## 539结构扩库：原判断使用了错误的问题

NP/G1的500个真值原本都在base质量窗内。该实验中新增结构只能提供干扰，因此完整ICE融合ΔMRR=0是**未观察到指标损失的证据**，不是“扩库无法召回新答案”的证据。独立重算：NP0.668996801→0.668996801、G10.907021645→0.907021645，Top1/Top25均不变；NP一例真值全序40→42，仍在25外，应保留真实排序退化记录。

R3旧表比较的是**已扩库FP→已扩库FP＋ICE**，故IFEJ首选1→2、B7的MRR下降衡量的是额外ICE证据，不能归因于添加539结构。新增[scripts/010/analyze_010_recall_audit.py](../../scripts/010/analyze_010_recall_audit.py)从相同候选分数和回传缓存为base/expanded分别重算z归一化，得到：

| 对照，R3质量可用46例 | 原库MRR | 扩库MRR | 新增Top25 / 损失Top25 | 原A25变化 |
| --- | ---: | ---: | ---: | --- |
| 原v9代表谱、原质量换算、仅用fused_merge | 0.408903 | 0.762629 | 20 / 0 | 无 |
| 固定R3多谱均值、两侧均采用未知CE的sum谱回退 | 0.448499 | 0.794979 | 21 / 0 | 无 |

两行不是同一种管线，不能把第二行更高归因于扩库。第一行用59谱已存logits重新按原notebook的最大原始峰数选代表谱、质量窗及canonical max评分；R3的523条`fused_merge`全缺失，原v9本来就退回代表谱FP。第二行额外使用有效的`fused`字段，是**未知CE回退规则的独立变化**。

23个有adduct支持的分子包含26条支持谱，但26条CE全部未知；523/523条`fused`有效、0/523条`fused_merge`有效，split/mean的fused相同。此为当前score_merge对sum桶的实现行为，不是“数据强制必须改用mean”。YTCZZ40→6是第二行的ICE回退成绩，第一行原v9仍rank40。首个提交候选建议只扩库，不同时加入该回退或条件多谱，以便解释public结果。

**限制仍然重要：** 46例排除了4个无明确adduct的NILU，A/B来自人工分层，分子也不是新盲集。MassBank索引与评价同源；统一移除query记录后，第二行仅新增8个Top25，移除query贡献者后新增0。原库外B只表示不在当前712K池，不代表官方Class3。以上大幅本地增量不能预测隐藏集增量。第一行是CPU float64/cache canonical重建，不是GPU float32/Kaggle新版本端到端复现。

## 条件多谱：确认结果成立，但不应否决其他改动

新增[scripts/010/analyze_010_heldout_audit.py](../../scripts/010/analyze_010_heldout_audit.py)，从原train.parquet核对1400谱，重新运行FPNet CPU前向并重建所有候选分数、代表SMILES、top100和最终Top25；谱字段、1400×6930 logits、候选分数及最终排序均零差异。原始临时构建代码7份已保存到`artifacts/runs/010-heldout-audit/recovered-build/`。

| 范围 | 原v9→冻结条件规则MRR | ΔMRR，95%分子bootstrap | Top1 | Top25 |
| --- | --- | --- | --- | --- |
| 新250全部 | 0.907838→0.905171 | −0.002667 [−0.010,+0.004] | 211→210 | 247→247 |
| ICE支持232 | 0.909308→0.909308 | 0 | 197→197 | 229→229 |
| 不支持18 | 0.888889→0.851852 | −0.037037 [−0.138889,+0.055556] | 14→13 | 18→18 |

这不是实现算错。232个恒等样本来自规则构造，真正改变的18例只有4个RR变化（1改善、3恶化）。本次未确认收益，也没有统计证据确认负效；默认继续使用原v9回退合理。不要扫阈值或追加样本直到转正。

纠正009的三个解释：

- 正确冻结描述是“**开发阶段选定规则后、使用新250前冻结**”，不是独立先验规则。内部配置/文件时间及SHA相符，但Git首次保存冻结与评分在同一后续commit，不能宣称有外部可信的预注册时间戳。
- v9在不支持条件下使用代表谱FP，正是可部署的回退行为，所以这个对照有效。原文“无ICE因而不是deployable比较”错误。
- 每个个体退化都要先解释清楚才允许提交，是过严条件。应看预先指定比较的净收益、区间、稳定性和实际风险；此处暂不采用的依据是确认未支持替换。

新250与研究250 canonical零交叠，但公开预训练权重暴露仍未知，不能称预训练未见结构泛化。该集已于10月13日使用，不再称保留集；本轮复核不是第二次独立确认。它只检验条件多谱，不是扩库的否决票。

## LMSD：数值成立，停止所有扩库的推断不成立

本轮[手动复算脚本](../../scripts/010/analyze_010_source_audit.py)独立从逐例rank重算：LMSD新增22765结构，在当前R3-50真值覆盖仍25→25，MRR0.356190→0.356190；新增15个窗口候选对，14个进入Top25，挤走7个旧尾部候选，正确答案排名均不变。4个质量未知者单列，B_known21新增命中0。

该B样本以非脂质结构为主，用脂质库出现零召回并不意外；不能推断ChEBI/PubChem也无价值。“暂不把LMSD混入本次539提交”合理，“停止其他来源”没有足够依据。下一来源要先用结构分层、与隐藏目标相关的观察集测**候选覆盖与增添的干扰**，无需先下载整个105M池，也不依据排行榜调候选预算。原文“PubChem许可混杂而禁下载”不是本项目用户要求或官方禁令，应撤回这种一刀切表述。

## 最新官方信息与奖金目标

10月14日Kaggle MCP重新读取：最佳有效提交仍为v9r2 ref57028399/public **0.323**，当前rank **1932/3277**；榜首0.491、第五0.467。与10月12日相比自身分数未变，其他队伍和参赛规模变化导致名次后移。奖金仅最终前五，公开榜快照不等于私人榜或金牌线。[官方榜单](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/leaderboard)

MRR@25、每分子最多25结构、RDKit2026.03.3身份匹配、离线≤9h仍有效，最终截止12月14日23:59 UTC/北京时间12月15日07:59。前三类结构在public/private都应预期出现，但官方不公开比例。不能把单一路线开发结果按假设比例加权成隐藏成绩。

本轮读取带`author_type=HOST`的原始回复，取代10月12日“未答复”状态：

- [综合答复748882#3538202](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/748882#3538202)，Marie Killian，10月13日：获准的MassSpecGym/GLACIER/ICEBERG/DreaMS checkpoint可交付发布权重、原作者训练代码与允许数据引用，加自己的微调/推理代码，不要求重跑预训练。参赛者共享模型同样要能凭训练代码和数据复现，或用允许数据自行训练；主办方保留审查所有提交的权利。
- [FRIGID明确答复746430#3538189](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/746430#3538189)：官方checkpoint可用于奖金方案，仍需提供原训练代码和允许训练数据。不能继续以“资格未明”搁置整个方向，但接线和效果尚未成功仍是事实。
- 综合答复还明确PubChem公开CID-PMID/Patent/SID计数可作特征，LOTUS/NPAtlas可用；公开许可仍需遵守。没有“只准白名单内数据库”的限制。人气特征被允许不等于它有跨域收益。
- 同一答复澄清电子质量修正**仅应用于隐藏test**，Enveda-180训练/Zenodo未修改；若构造同口径验证，可对这些来源的正模式m/z减0.000548 Da。**不要对已经修正的隐藏test再减一次**。本轮没有改动缓存或重算该校准下的分数，旧G1结果保留其原始质量口径。
- 官方确认添加有效且不同的尾部候选不会降低未改变首选的单行MRR；当前live scorer采用公开v13。旧“尾部投毒”提问不能作为删除尾部或改变MRR定义的依据。

原始快照在`artifacts/runs/010-official/`，精简[官方证据](../../data/references/submission-review-2026-10-14.json)。NIST一般禁止及GNPS-NIST14-MATCHES具体例外沿用已核实原始回复；不可从DreaMS训练语料包含某评估文件的帖子标题推断权重实际训练于NIST。

## 高分代码：新东西是覆盖缺口，不是把fork当独立验证

刷新ScoreDescending/DateRun并读取7份新/变化源码，保留版本、SHA与原始文件，详见[前沿源码摘录](../../artifacts/runs/010-frontier/findings.md)。搜索前五为Dual-Band MassBank+ICEBERG、Ahmed v7c FragGap、Closing the Evidence Gap、nursrijan MetFrag-lite及Apex v340；第五仅查看搜索元信息，未深读。**API不返回这些notebook的数值分数，0.441等数字是作者/下游自报，不是本轮独立测得。**

- [v7c FragGap，版本357736980](https://www.kaggle.com/code/ahmedberatozer/casmi26-v7c-e006-fraggap?scriptVersionId=357736980)：无GL/ICE分数的分子增加MetFrag-lite结构碎片解释，至多2次断键，固定λ=.5，按同式组内`z(base)+.5z(fragment)`重排后放回原槽位。它不依赖某个候选已经有参考谱，正好对应我们的支持条件缺口。实际触发条件却混合了“adduct不支持”和“运行失败/超预算”，实验需改为明确的metadata支持边界。只有两个非并列候选的组，这个λ不能逆转原排序；先确认≥3候选组覆盖，不把机械零变化误判为碎片无信息。作者后续引用0.440→0.441，不能作本地已验证增益。
- [Closing the Evidence Gap](https://www.kaggle.com/code/flexonafft/casmi26-closing-the-evidence-gap)：补融合后同式组的缺失forward分数。我们的ICE top100覆盖已经较完整，未见它足以解决负离子支持问题；不再重复花GPU做同一已完成的同式组合。
- 新v8b全训练数据hard-negative continuation当前未取得独立得分且未绑定完整训练记录，不能直接替换当前模型并用NP/G1声称未见结构提升。Dual-Band标题中的MassBank heuristic已在正文声明移除，不能仅看标题断定实现。

一个有实质价值的可复现性发现：**现用`fp_single_s2.pt`的原作者训练代码公开存在。** 在[prvsiyan原baseline v34](https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline)的raw cell18找到`TRAIN_SCRIPT_SRC`，cell47导出，静态提取11704字节到`artifacts/runs/010-frontier/original_train_fingerprint_model.py`，SHA `e2eaef630c5f09f157eea3652095b8b9780cbe7fdffa6f0652ae5845d7b18c8a`。含BCE＋同10ppm窗63 decoy的CE，不能拿仓库现有仅BCE的训练器当等价复现。

剩余缺口是**把公开源码绑定到该具体权重**：`spec.npz`的six/is_val与FP/候选构建来源、s2实际seed/merge_p/训练CLI及数据版本；公开脚本默认seed7/merge_p=.6不等于single的实际参数，checkpoint不保存这些字段，torch种子还在模型初始化后设置。本轮没有执行下载的训练代码，也未宣称已经复现权重。应补血缘记录，不因“有代码”或“Other许可”两个单一信号就判定整个最终方案合规/不合规。

## 前沿研究如何改变下一步

1. **优先做不依赖谱库的碎片通道。** 固定v7c参数，先20–30个明确不支持正H/Na的开发分子×top50，带负离子、钠/其他加合物和正模式对照，最多1 CPU小时。保留原v9其他分子排序；报告新增证据净收益、Top1得失及候选覆盖，不按已揭盲250错例挑阈值。这是后续独立实验，不混进本次539扩库提交。
2. **FRIGID从“资格未明”转为可研究，仍先打通最小真实谱链路。** [官方论文/代码](https://github.com/coleygroup/FRIGID)的MSG约18.29% Top1属于已知分子式设置，不能移植成本比赛未知式得分；6.58秒是RTX6000 Ada条件，非T4实测。先1个支持谱的真实4096向量、可推理公式、固定小预算生成，独立记录合法率/唯一数；不让真值SMILES roundtrip阻断，也不能用真值公式冒充推理。沿用008的单分子B=10起步，不因本次读到新论文便扩大采样预算；成功后再定10例固定观察集的有限实验。
3. **修正基准实现比照搬论文榜单更重要。** [MassSpecGym in the Wild](https://arxiv.org/html/2606.19624v1)检查了encoder暴露、公式泄漏、候选偏差与MIST batch padding错误，并发布MSG1.5。下一次MIST接线要人工核对单谱/批处理一致性、修正实现和训练来源。它支持保留严格结构分组及已见谱检索两条评价，不要求为本项目建立自动测试体系。
4. [GLACIER新版本](https://arxiv.org/abs/2606.29161v2)的MSG检索结果值得参考，但本地已有GL/ICE固定对照，不能据论文不同任务的Top1再宣布换模型必提分；MSG和NIST权重要分开。DreaMS及跨极性的MS2DeepScore2可作为以后谱库/analog证据，当前没有本地增量，不优先于上两项。

结构分组可行性已做无标签统计（只读研究集top100与候选公式）：NP不支持34例中32例有≥3候选同式组，共62组/1541候选；G1不支持21例全部有，共71组/1559候选。产物`artifacts/runs/010-frontier/fragment_group_coverage.json`。这说明有限碎片对照有可改变排序的空间，不代表碎片分数已有效；本轮未运行碎片打分。

## 下一次submission的具体范围

- 主变化只有**固定539结构**。沿用v9代表谱、质量窗、FP模型、ICE merged评分，不同时加LMSD、条件多谱、新权重或未知CE回退。
- 离线输入保留canonical、质量、SMILES与同6930位指纹；不按query真值动态添加结构，不在隐藏推理联网。修正原`712199`数量断言及候选取指纹分支，保持规范化去重和Top25。
- 先做一次Kaggle完整运行，确认原输入、输出契约和实际耗时。研究包回传成功并不等于最终notebook已跑通；本轮没有执行这一步。通过后一次验证性提交即可，不需要再等所有子方向显著正向。
- 使用public结果验证该单一改动并保留v9r2；同源开发结果不能预报分数，零涨分也不能据总分倒推隐藏类别。当前目标仍是补充足够多可泛化证据，539结构本身不可能证明已具备金牌/奖金实力。

## 人工复核与范围

执行三个新增手动脚本：`analyze_010_recall_audit.py`（约2.16s CPU缓存重算）、`analyze_010_heldout_audit.py`（14.63s CPU含1400谱重新前向）、`analyze_010_source_audit.py`（逐rank算指标）。没有训练、GPU作业、push、submission或测试文件/CI。

R3另人工重新加载59个峰文件做FP CPU前向：59×6930 logits与缓存逐元素相等，峰SHA/峰长全一致，50个代表谱选择全部一致；保存`artifacts/runs/010-recall-audit/verification.json`。该核验消除了缓存行顺序疑问，不消除CPU/GPU数值及RDKit版本差异，不冒充最终Kaggle运行。

范围覆盖：扩库两个评价域及R3比较、条件规则已用250、LMSD逐例指标、官方当前信息、7份新源码与训练原码、相关主论文。未重新扫描全部LMSD SDF、未执行第三方notebook/训练代码、未测新碎片/生成效果、未运行最终扩库notebook。主要结论已支持，最终提交集成仍待执行。语法、JSON和文档链接手动核对。
