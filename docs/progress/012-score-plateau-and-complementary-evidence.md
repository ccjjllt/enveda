# 012 — 同分复盘与互补证据：实际进展

日期：2026-10-17（北京时间）。对应[012计划](../plan/012-score-plateau-and-complementary-evidence.md)。本轮已完成同分复盘、官方/源码/论文复查，以及碎片、ChEBI、PubChemLite三项CPU研究。精简结果见[research-results-012.json](../../data/references/research-results-012.json)，以下只记录实际证据。

## v10已评分，同分解释的证据范围

Kaggle MCP重新读取submission57293821：COMPLETE/public **0.323**，508585字节；v9r2/ref57028399仍COMPLETE/public **0.323**，508590字节。两个提交均指向各自明确版本，文件大小不同，不能说误交了同一份CSV。public仅返回三位小数，同分不等于可证明未舍入MRR或逐行排名完全相同。

本轮尝试官方`download_competition_submission`读取自己的两份输出，均返回`Permission 'submissions.download' was denied`，没有绕过权限。隐藏逐行真值/排名不可见，不能从总分推断三类隐藏分子的比例或唯一失效原因。

已知改动规模是538/712199≈0.0755%；未修改任何排序证据。011已核查可见占位数据仅2行候选序列变化、首选变化0；此为改动触及范围证据，不作泛化指标。R3同源人工分层的本地大增益不能外推。最合理的待检验解释是扩库太窄、对公共计分样本有效排序影响有限；也可能存在显示舍入或相抵，不能把这些可能性写成已观察事实。

## 官方当前信息与实际差距

北京时间约12:51官方快照：3455队，当前排名2101；公开榜首0.495，第五0.468。奖金前五、共50000美元，最终以私人榜和规则为准；公开第五不是可保证的最终奖线。当前0.323距公开第五0.145，538结构本身不能代表已有奖金竞争力。

重新读取官方Evaluation：MRR@25按分子取第一个正确结构的倒数；错误候选仅在挤动正确答案位置时影响MRR。官方data-description仍明确可见test是训练占位数据，隐藏约400分子/1500谱、全部timsTOF、10种adduct、未知三类比例。正文范围157–1159Da；enveda-180虽然仪器匹配但为合成药物化学空间，NP250才是近目标域。因此不能只看G1高分或用假设类别比例加权成隐藏成绩。数据页adduct列仍残留“seven”表述，采用明确10项列表。

10月14日官方重评分公告确认早期版本已统一重评分，本轮重读两版本仍同分。10月16日主办方新答复745841#3540007：当前阶段不可完全复现的旧提交不具获奖资格，但不会使未来完全可复现且合规的新提交失去资格；不把历史第三方依赖不完整解读为今后无机会。现用s2完整训练血缘缺口仍须补，不能仅以“有训练源码”宣称奖金材料完整。

原始证据保存在`artifacts/012-official/snapshot.json`。来源：[官方数据](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data)、[评测](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview/evaluation)、[榜单](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/leaderboard)、[重评分公告](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/749430)、[新主办方答复](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745841#3540007)。

## 新实验一：固定结构碎片通道，有开发信号但不是确认收益

代码[scripts/analyze_012_fragment_pilot.py](../../scripts/analyze_012_fragment_pilot.py)。只载入已审阅的FRAG_SRC纯计算模块，固定SHA `eeacdcf0afb173a8b438054d93c30f061df7c9c69d1ae32db3530b2916f2e6a7`。来源为Ahmed v7c公开代码归功于Huseyin Emre Aksoy的结构碎片实现，未执行其notebook入口/网络/固定ID逻辑。

固定参数：最多2次断键，第二次权重0.6，线性强度，容差0.005Da，H位移−2…3，λ=.5；仅前50候选同式槽位重排，保留候选集合与第51—100位。路由只看metadata：没有正模式[M+H]+/[M+Na]+的分子才走新通道；只允许明确carrier，二聚体等不猜载荷。truth_canon在评分结束后用于rank统计，输入JSON虽整体包含标签，标签不参与评分/路由。

手动命令：

```bash
.venv-fp/bin/python scripts/analyze_012_fragment_pilot.py --probe
.venv-fp/bin/python scripts/analyze_012_fragment_pilot.py
.venv-fp/bin/python scripts/analyze_012_fragment_pilot.py --full-existing
```

初批按哈希/metadata固定24个无支持分子＋6个正H/Na对照；随后参数不变扩到现有全部55个无支持分子。后者包含初批且仍为已用开发集，不是独立确认。环境Python3.12.13、NumPy2.5.3、RDKit2026.03.6，seed20261017，本地CPU、无Numba/训练/GPU。初批墙钟15.59秒；全批31.14秒（CPU30.98秒）。

| 评价范围 | 原MRR@25 | 新MRR@25 | Δ及95%分子bootstrap | Top1 | Top25 |
| --- | ---: | ---: | --- | --- | --- |
| 初批24个不支持 | 0.480743 | 0.496775 | +0.016032，[+0.003975,+0.030383] | 8→8 | 23→24 |
| 全55个不支持 | 0.427799 | 0.460336 | +0.032537，[−0.000233,+0.076399] | 15→17 | 53→53 |
| NP完整250，相对原v9 | 0.668997 | 0.669629 | +0.000632，[−0.001800,+0.003138] | 138→138 | 237→236 |
| G1研究完整250，相对原v9 | 0.907022 | 0.913548 | +0.006526，[−0.000222,+0.016304] | 214→216 | 246→247 |

全55中17改善/7恶化，Top1得2失0，Top25得1失1；UAPNJSWNICTCCJ 29→10，JUVIOZPCNVVQFO 21→26。50个有明确carrier、1903候选成功评分；5个纯二聚体保持原排序。6个正H/Na对照仅FP诊断净−0.000690，未进入部署路由。不能只展示初批区间而隐瞒全55区间含0，也不按7个退化样本重新挑λ。

独立人工审查重新从原FP/ICE缓存构建v9全序，500/500一致；55条由原始碎片分数重建全部一致，445个支持分子全序不变，第51—100位不变，同式槽位和候选集合不变。候选公式与SMILES一致、无非有限分数。审查见`artifacts/012-fragment/reviewer-audit.json`，完整结果在`artifacts/012-fragment/full-existing/`。

限制：忽略CE、仪器、断键能量、电荷位置与重排机制；两个非并列候选的同式组在λ=.5下不能反转。28个含局部形式电荷候选不等于28个净带电离子，实查仅1个净带电。当前只支持保留为低成本新证据，不证明public增益，也不足单独弥补0.145差距。

## 新实验二：ChEBI三星完整来源覆盖

直接下载官方release256、2026-10-06的`chebi_lite_3_stars.sdf.gz`及LICENSE/README：15927307字节，SHA256 `74e1be399d581a65ffdc228f3cf088982aa2fc5f4d469d1364f6e37b30366363`，CC BY4.0。固定筛选为单连通、无通配、净电荷0、无显式同位素、100–1500Da；没有中和、删盐或根据query真值挑条目。先处理来源全体后才读取开发标签。

代码[scripts/analyze_012_chebi_coverage.py](../../scripts/analyze_012_chebi_coverage.py)，手动执行`/opt/homebrew/bin/python3.11 scripts/analyze_012_chebi_coverage.py`；Python3.11.16/RDKit2026.03.6，4个CPU规范化worker，无随机采样。首次完整运行137.86秒，随后缓存复核改为从原v10 cell5提取质量契约（SHA `1f3bff620986f54d7ba7701ed11f554b83fe16683630418864dc5db00a3277db`），全部查询统计相同。首次计时与摘要保留`initial-coverage-summary.json`，不把缓存重算时长当全量构建耗时。

- 原来源52963条；固定筛选剩33258条、身份失败57条；有效规范化结构26727个。
- 相对原库新增8474个，相对v10新增**8457个**。本项目canonical口径原库706242、v10为706780；不与shipped原始条目数712199/712737混用。
- NP/G1共500分子新增窗口候选对133＋78，分别影响86/60个query；真值已在v10，因此新增真值均0，仅是潜在干扰规模。
- R3-50中46个质量可用；新增13个候选对，原库及v10之外新增真值均0。未计算FP/ICE完整MRR，不声称干扰无损。

单独看来源自身可达率：NP为199/250（池内和质量窗均199），G1为0/250，R3为0/50。这揭示三星中性子集与各观察域的化学空间不同，不能只看总新增量。还要注意：NP/G1的500个真值原已在v10，R3的46个质量可用真值也已全部在v10窗内，因此这些集合上的“相对v10新增真值0”本身存在召回上限，不能据此否决任何后续来源。新的库外召回确认需要与扩库来源独立的观察集合，或明确标注的候选删除压力场景；不能反复使用已被MassBank补齐的R3来要求新增答案。

这只是三星、固定中性单连通子集和既有小样本结果，不能否定包含二星的完整ChEBI；公开98K ChEBI/LMSD包与本配置不是同一个集合。本轮不把它直接拼入submission。来源：[ChEBI官方说明](https://www.ebi.ac.uk/chebi/downloads)、[官方快照目录](https://ftp.ebi.ac.uk/pub/databases/chebi/SDF/)。产物`artifacts/012-chebi/`。

## 新实验三：PubChemLite全源质量窗与FP干扰

官方下载[PubChemLite 2026-09-25/v3.3.0](https://zenodo.org/records/22953038)，200142750字节、CC BY4.0，MD5 `3023e89b20e5b57bd3e2c70dc1366be1`、SHA256 `6a258459e1ccf29a5d08e82fcf5240a2fc8afbabe5ba9c678e39aa0801dd89cd`。下载完整后验证大小/MD5再计算，未使用部分文件。代码[scripts/analyze_012_pubchemlite_coverage.py](../../scripts/analyze_012_pubchemlite_coverage.py)。

先用原v9最大原始峰数代表谱和原质量函数固定550个已有开发query，其中546个有效10ppm窗；流式扫描全源、只对窗内结构规范化。依次保存`frozen_queries.json`、`frozen_window_candidates.json`、`frozen_fp_orders.json`，完成来源/排序后才访问truth字段评价。来源自身排序同步保存，不按标签决定走哪个来源。

手动命令：`/opt/homebrew/bin/python3.11 scripts/analyze_012_pubchemlite_coverage.py`。Python3.11.16、NumPy2.4.6、RDKit2026.03.6，CPU预算900秒，实际墙钟86.18秒/CPU84.71秒；无新神经前向、GPU、训练或submission。使用既有原代表谱logits、6930位同口径指纹、完整窗canonical-max排序；全程不加入真值。

- 完整扫描469602条，窗内15464条；1条结构无效；RDKit质量复核从16797个候选-query对剔除33个，保留16764对。
- 窗内有效canonical15399个，相对原库新增12098、相对v10新增**12092个**。这是546窗的并集统计，**不是PubChemLite全源canonical新增量**。
- 来源自身真值可达率：NP **242/250**、G1 **0/250**、R3 **25/46**。R3原库外B21为0/21。现有全部质量可用真值已在v10，不能把“相对v10新增0”当失败；相对base的R3新增0和来源自身覆盖则提供了这个小观察域内的独立来源信息。

| FP-only原代表谱口径 | v10 FP MRR@25 | v10＋PubChemLite FP | Top1 | Top25 |
| --- | ---: | ---: | --- | --- |
| NP250 | 0.514593 | 0.478352 | 93→85 | 230→224 |
| G1研究250 | 0.884250 | 0.879198 | 204→202 | 246→246 |
| R3质量可用46 | 0.762629 | 0.627743 | 29→22 | 45→43 |

新增候选对在NP/G1/R3分别6220/5886/978，挤掉原Top25候选973/697/146个。**该实验显示现有FP通道面对扩窗干扰有明显弱点；尚未运行新增候选的完整ICE，不能把这些值称为部署v10退化。** 不直接把PCL候选接入现有submission。

来源独占FP在NP得0.616989、Top1=121/250，然而G1覆盖为0且NP仍低于完整v9 ICE0.668997；这说明来源筛选能带来强化学先验，不说明可按真实未知分子的标签选择来源。不能仅凭NP涨分改成PCL-only或照搬人气权重。PCL预先排除了部分盐、断连和金属，零覆盖不代表完整PubChem无该结构。首轮按发布质量筛窗后再核对RDKit质量，未检查发布质量窗外但RDKit可能落窗的边界记录；这一限制保留。

原始产物、冻结阶段、逐例结果、来源元信息在`artifacts/012-pubchemlite/`。人工核对NP/G1共500例base全序与原bundle零差异，v10 Top100与009冻结表零差异；R3全部50例代表谱/质量/Top25与010 literal-v9零差异，三个冻结文件SHA未变。完整来源raw FirstBlock另查G1仍0/250，零覆盖不来自多谱或身份映射错误。详见`manual_verification.json`，没有把R3多谱均值混入该比较。

## 高分notebook复查：需要补的是真正独立的通道

实时按ScoreDescending/DateRun各搜前40项，读取8份当前源码并展开关键嵌入模块；还读取Aryan公开推理/训练notebook及实际6份小型实现。版本、SHA及逐cell差异在`artifacts/012-notebooks/`。未执行下载代码、未下载其中权重、未独立测到其宣称分数。

- Ahmed v11相对已存v7c真实新增只有PubChem提升阈值250→150及第二引擎首选插rank2。作者明确沿public sweep选择，不搬这个阈值到我们不同尺度模型。
- 搜索第一Closing Evidence Gap v23正文仍写`measured_score=None`；Apex“0.451”是作者自报；“0.452 SOTA”Consensus Cascade正文是Target Benchmark，不是实测。多个fork不是多次独立验证。
- 强recipe的实质差异是约105M PubChem质量窗检索、ECFP粗排→全指纹、人气候选并入；全train直接/adduct-shift参考谱；跨质量analog命中后传播结构Tanimoto；多组GBM/LightGBM学习排序、多FP视图，最后去重融合。当前v9/v10的FP＋ICE100并未包含这些通道。
- 当前可追溯的Aryan实现提供0.01Da fragment＋neutral-loss analog、FP/人气/碎片/analog特征与LambdaRank5折。作者S2的0.502→0.627→0.660为同740分子开发/CV，其最终public自报0.272反而低于我们；只借鉴机制，不能移植数值。原始InChIKey14排除尚不等价完整canonical别名排除。

来源：[Ahmed v11](https://www.kaggle.com/code/ahmedberatozer/casmi26-v11-e2rank2-rel150)、[Closing Evidence Gap](https://www.kaggle.com/code/flexonafft/casmi26-closing-the-evidence-gap)、[Apex](https://www.kaggle.com/code/saddamaditya/0-451-sota-forensics-lab-apex-synthesis)、[Consensus Cascade](https://www.kaggle.com/code/saddamaditya/0-452-sota-forensics-lab-consensus-cascade)、[Aryan实现](https://www.kaggle.com/code/aryanawasthiteykik/casmi26-gpu-fingerprint-fragmentation-ranker)。

## 前沿研究对执行顺序的影响

详细一手对照在`artifacts/012-literature/findings.md`。这里保留改变决策的部分：

1. [MS2DeepScore2](https://github.com/matchms/ms2deepscore)支持正、负及跨极性谱间化学相似度，可形成“query谱与近邻谱相似度”对“候选与近邻结构相似度”的一致性证据；不要求候选本身有参考谱。先用已有允许训练谱建小参考集，做同结构独立谱和整结构参考移除两域；不能把已有线性D2负结果当作此通道已经失败。权重具体来源与曝光须核对；当前未加载/运行该模型。
2. [MS2Query2原型](https://github.com/matchms/ms2query_2)参考包2.6GB、当前代码接口仍偏建库，不优先整体移植；先验证MS2DeepScore2小通道。
3. FRIGID公共输入链仍缺可推理公式、真实MIST前向、配checkpoint阈值二值化及独立长度先验。`scripts/mist_cf_formula.py`含`cand_raw_set.add(truth_raw)`，旧公式20例结果只能作oracle候选保障诊断，不能直接作为真实生成入口。当前FRIGID主分支已修padding mask，不能笼统声称仍有此bug；旧pin需实查。部分官方长度模型按benchmark test结构/公式建立，不能当本项目未见结构验证的独立长度先验。
4. [MS-GPT](https://arxiv.org/abs/2607.23607)与[FOAM](https://github.com/coleygroup/foam)均仍受分子式约束；前者也依赖MIST。先打通一条非oracle的公式→MIST→decoder链，B=10单谱起步，之后再比较decoder。未执行真实谱生成，不把论文已知公式Top1当本赛成绩。
5. [MassSpecGym in the Wild](https://arxiv.org/html/2606.19624v1)支持严格区分公式泄漏、encoder暴露和候选源捷径；官方仓库[#70](https://github.com/pluskal-lab/MassSpecGym/issues/70)、[#71](https://github.com/pluskal-lab/MassSpecGym/issues/71)仍有产物/CE处理讨论。我们未知CE必须用候选一致规则，不得用真值挑CE；不能从讨论推断现用ICE必然泄漏。

## 后续实施顺序与判断标准

1. **主投入改为可追溯的第二证据引擎。** 固定现有FP＋ICE为控制，建立直接谱/adduct-shift/mass-shift analog候选特征，再与碎片、质量误差共同学习小型非线性排序器。使用允许的train来源、canonical整组切分；无真值参考谱场景在构建特征前移除该结构全部参考，已见场景按谱hash/来源独立留出。输入标准化、特征、最终融合使用同一口径，避免重演D2训练α与推理α错配。
2. **广泛候选来源与排序分开推进。** 本轮PubChemLite完整快照已经证实直接FP扩库有干扰，下一步不能仅因新增数量大就加入；将来源/人气作为需验证的特征，并设置与来源独立的结构观察集。小型外部库没新增命中不否定完整PubChem；约7.2GB的完整PubChem资产可作为后续独立质量窗通道，无需展开全库指纹，先质量筛选，粗排后对窄列表精排。人气特征单独消融，不照抄public扫出的rel150，更不按query真值查询数据库。
3. **碎片作为候选特征保留，不把本批小增量包装成新高分方案。** 全部参数已固定；先用与现有开发样本不同的结构组检验增量，再决定独立提交。新样本独立于选型不等于公开encoder训练未见，须继续披露模型暴露；真正未见结构结论需从模型训练切分就排除。这里不要求每个切片都显著或每个个体都不退化。
4. **生成用有界最小链推进。** 先1个已选支持谱、可推理公式、真实MIST概率及配套二值化、独立长度来源、10个实际请求槽；成功后固定10例且总采样预算明确。代码接通、合法率、式满足率、结构召回及最终排序分别记录。生成不承担修补当前所有负离子条件的承诺。

当前最迫切的研究改变是从单一FP＋正模式forward，转到能够在缺参考谱、同式干扰、候选不在原池三种情况下提供不同信息的管线；已有D2/FIORA负结果仍保留，不用新名称覆盖旧失败。012尚无新排行榜提升，也没有再次push或submission。

## 本轮交付与人工核查

新增三份手动研究脚本、配对计划/进展与`data/references/research-results-012.json`；011和README已更新v10最终评分。原始谱、公开结构快照、权重和完整缓存仍留ignored artifacts，不加入Git。手动核对Python语法、JSON可读、文档本地链接与diff空白；三个CPU实验和独立指标复算已经完成。没有测试文件、测试框架、CI、自动任务或新的远端作业。
