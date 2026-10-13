# 009 — 高分方案与可迁移性：实际研究结果

日期：2026-10-12，北京时间。对应[009计划](../plan/009-frontier-and-transfer.md)。继承008新增结果，本轮读取官方当前信息、11份公开notebook源码，并执行CPU研究；不创建测试文件/框架/CI，不读G1保留250，无远端作业或submission。

## 官方现状与目标差距

Kaggle MCP本轮transport失败，官方Python SDK只读获取成功，原文快照在`artifacts/009-official/`，精简记录见[官方来源](../../data/references/research-frontier-2026-10-12.json)。最新有效提交仍为v9r2，ref57028399、public **0.323**；3099队，用户当前rank1787。公开榜第一0.491、第五0.461、第15个列表条目0.448（存在并列，不等于最终金牌线）。奖金只给最终前五，共$50,000；金牌和奖金是不同目标，当前公开榜分差既不是私人榜预测，也不是固定达标分数。

官方评测仍为MRR@25、每分子最多25个候选、去立体并规范互变异构后InChIKey14匹配，RDKit2026.03.3。API元信息仍显示旧名称Mean Average Precision，采用评测页而非旧字段。最终CPU/GPU≤9h、互联网关闭。最终截止2026-12-14 23:59 UTC，即北京时间12-15 07:59；入口/合队12-07 UTC。隐藏测试约400分子、1500谱、全部timsTOF；可见test为训练抽样占位，不能当验证集。

本轮核查原始作者回复，而非只复述参赛者汇总表：

- [David Healey，745072#3536169](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745072#3536169)：仅从train.parquet继承非商用限制的资产可使用，PubChem/COCONUT可使用；不能因Kaggle包级`Other`就断言失去奖金资格。
- [745841#3533343](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745841#3533343)：明确GLACIER/ICEBERG、DreaMS、ChEBI/LIPID MAPS，以及train.parquet训练与再分发资产可用。
- NIST一般禁用仍成立；但[745832#3536140](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745832#3536140)已有**GNPS-NIST14-MATCHES**这一具体旧公开集例外，允许库匹配及训练。不可外推至其他NIST或新衍生数据。本轮未使用该例外数据。
- 第三方训练代码交付问题748065及FRIGID非商用checkpoint问题746430，本轮未见官方结论。不把其他参赛者的许可推理当授权，也不为尚未澄清的生成权重追加大训练预算。
- 748068中“public约130分子”“0.43+均未CV”是参赛者主张；748029中尾部猜测导致降分是未获官方答复的报告。均不能据此修改官方MRR定义或推算隐藏类别比例。

配额API返回GPU `timeUsed=76915.244.0s`、`totalTimeAllowed=21600s`，前者格式异常且超总量，不能据此宣称尚余若干小时。TPU返回已用0/总72000s，刷新2026-10-17 UTC；本轮未启动GPU/TPU，也未假定Colab仍连接。

## 高分notebook实际提供了什么

按ScoreDescending与DateRun各读20项，拉取11份源文件、固定版本及SHA，另读7个资产说明；完整摘录见[源码审查](../../artifacts/009-public-notebooks/findings.md)。排序接口不返回数值分数，作者标题/源码自报不当作已核实版本成绩。

高分列表前五为flexonafft v86、canqiang E006 v1、nursrijan E006 v6、nursrijan E007 v1、Omniverse v13，许多共享同一上游，不是五次独立验证。flexonafft源码记历史v80为0.436，当前v86同式融合仍pending；E006的0.440是上游转述，Omniverse标题自报0.438，未拿到各当前版本绑定的独立分数。

与本项目v9的实质差距是：更广的训练/COCONUT/ChEBI/LipidMaps/PubChem结构来源，直接/shifted/analog库检索、结构碎片解释、多个FP/GB排序器、canonical去重后多引擎融合，以及同式内模拟谱重排。不能把整条高分管线概括为“换个ICE权重”。其中PubChem分支约105M结构是代码/资产说明，不是本轮下载或独立核实全量计数。

需要避开的照抄点：`fpnet-full1`说明明确训练于全部train（含np-examples），不能在NP250上宣称未见结构泛化；某些merged分支混合不同adduct再用首adduct和固定CE25；人气先验及多个阈值只见public驱动的解释。11份源码未见固定隐藏ID→真值覆盖表，但未读取所有依赖数据集，不能保证整个资产链无泄漏。

## 扩库：真实召回成立，独立发现率尚未成立

新增并手动运行[scripts/analyze_009_expansion_sources.py](../../scripts/analyze_009_expansion_sources.py)。重新扫描完整MassBank bulk建立canonical→所有允许记录/贡献者集合；008只保留前5 accession，不能直接用作来源排除。59谱重新做FPNet CPU前向，原008逐例rank复现零差异。

| B_known，21分子 | 正确结构进入质量窗/top100 | Top25 | MRR@25 |
| --- | ---: | ---: | ---: |
| 固定完整MassBank新增539结构 | 21 | 20 | 0.774830 |
| 统一排除query记录作为结构来源 | 8 | 7 | 0.333333 |
| 统一排除query贡献者的所有结构来源 | 0 | 0 | 0 |

13个Qingdao B在允许记录中仅有评价记录自身；其余8个mFam B有2–4条记录，但均同贡献者。4个质量未知的NILU另列，不放入上述21分母。该压力实验按来源规则移除所有候选，不利用真值决定删除；它检验来源可迁移性，**不否定合法包含完整MassBank的实际检索价值**。但20/21绝不能被外推为任意未知结构的95%发现率。

原R3 A25质量窗新增候选为0；B_known每窗恰好只新增1个且是真值。因此“A零损失”在这批样本中没有实际干扰压力。YTCZZ仍为rank40，其外部代表为含`.[Na+]`的断连钠盐，不能假设ICE必然支持并救回。

进一步将539结构真正加入现有NP/G1研究质量窗：NP11/250分子新增13个候选对，12进入top100、7进入top25；G1 5/250新增5对，3进top100、2进top25。两域FP MRR保持**0.514593 / 0.884250**，无Top1/Top25损失；NP一例全序28→29，仍在25外。G1有3个原top100尾部候选被挤出，所以**尚不能声称完整v9+ICE无损**。

完整对照的最小新增forward工作量已确定：新增进入top100的15对中12对有ICE支持谱（NP9+G1 3），另3对无支持条件。需要给这些候选补分并按新shortlist重算z标准化，不能仅把新结构尾接到旧融合排序。产物为`artifacts/009-expansion/`的来源集合、全候选分数、逐例排序、干扰统计及run_meta。

## 固定多谱实验：全局均值不采用

新增[scripts/analyze_009_multispectrum.py](../../scripts/analyze_009_multispectrum.py)，CPU复用NP1184+G11377谱logits。运行前固定代表谱、全部谱等权均值、先按mode/adduct组内均值再组间等权；固定质量窗、原代表SMILES、原FP top100与相同ICE逐CE合并缓存，不扫描权重。500/500分子代表谱选择与logits全局行号一致，全部候选基线分数最大误差0。

历史004/005已在训练结构池/旧匹配口径观察朴素均值劣于single；本轮新增价值是canonical并集池、固定完整ICE100及G1域的同口径比较，不能把均值聚合本身称为首次研究。

| FP证据方式＋固定ICE top100 | NP250 MRR | 相对v9，95%分子bootstrap | G1研究250 MRR | 相对v9，95%分子bootstrap |
| --- | ---: | --- | ---: | --- |
| 原代表谱 | 0.668997 | — | 0.907022 | — |
| 全部谱均值 | 0.643085 | −0.025912 [−0.047622,−0.005598] | 0.919651 | +0.012629 [−0.000626,+0.027599] |
| adduct平衡均值 | 0.641141 | −0.027856 [−0.049652,−0.007874] | 0.921095 | +0.014074 [+0.000044,+0.029806] |

NP全部谱均值改对4个Top1、改错16个；平衡均值改对3、改错15。全局均值不部署。G1平衡方式区间贴近0，且多个开发对照、来源狭窄与权重暴露未知，不称跨域泛化成功；不能用NP/G1标签选择生产路由。

没有ICE支持的切片却提供新线索：全部谱均值的固定100 MRR增量NP34为+0.033598、G1 21为+0.118601。此分层与“已有ICE分子平均多个质量不同的谱会冲淡有效证据”的假设相容，但没有证明因果。后续仅把条件支持情况作为可推理路由，单独冻结验证；不尝试把两个总体分数平均成隐藏效果。

本实验仅隔离固定候选集内的多谱评分，未评估改变shortlist后的全新ICE运行；全窗FP也固定原代表SMILES，不能称全量重算每个canonical的最优raw别名。输出`artifacts/009-multispectrum/{results,np_candidate_scores,g1_candidate_scores}.json`，3.26秒CPU，seed20261012，bootstrap20000。

独立复核另逐行核对原始NP/G1谱缓存的峰、前体、adduct、模式、仪器、CE，2561条零差异；两个logits与原输出SHA一致；500分子三种评分重算最大差0、Top25与RR零差异、代表谱分支和保存A完整100序完全一致。

事后追加一个仅依赖metadata的诊断：有正模式[M+H]+/[M+Na]+时保留v9，否则用全部谱平均FP的全质量窗Top25。NP **0.668997→0.673566**，Δ+0.004569，CI [−0.000641,+0.011761]，Top1 138→139（无改错），Top25 237→236；G1 **0.907022→0.916984**，Δ+0.009962，CI [+0.000366,+0.022263]，Top1 214→217（无改错），Top25 246→247。未覆盖的55例全窗与原100内Top25完全相同，没有额外候选进入Top25。

该路由的输入不含标签，但**选择路由时已看开发结果**；所以保存为`conditional_diagnostic.json`事后假设，不写成独立确认、不直接提交，不因两域都涨就忽略NP一次Top25损失。

## 借鉴高分机制的固定同式融合实验

新增[scripts/analyze_009_formula_fusion.py](../../scripts/analyze_009_formula_fusion.py)。从原NP250 F1 top100及GL100split/ICE100split的`fused_merge`出发，以v9全序为槽位基线；仅同式组≥2且两个模拟器对全组都有限时，组内`z(FP)+0.5z(GL)+0.5z(ICE)`（ddof1）重排并替换原公式占据的槽位。没有跨式比较新分数，不扫权重；这不是复现公开多引擎整套管线。

214分子/500同式组/12924候选符合条件，203分子全序改变，207个真值位于合格组。MRR **0.668996801→0.676017292**，Δ**+0.007020491**，分子bootstrap20000/seed20261012，CI **[−0.015437,+0.029246]**。31改善/28恶化/191相同；Top1 138→141，其中13改对、10改错；Top25 237→236。

候选canonical、原FP分、保存基线RR和公式槽位核查零差异。一个含铝候选在两个缓存中均缺行，作为None跳过该组，不造0分。**开发信号不够稳定，当前不部署、不为此单独追加GPU。** 不能把+0.007全归于“同式约束”或GL，因为本对照同时引入组内标准化、GL证据和混合权重；需进一步因子隔离才能归因。产物`artifacts/009-formula-fusion/{config,summary,per_molecule}.json`。

## D2结论的必要更正

008数值可复算：NP seen+0.053281，压力−0.012658并改错5个Top1，当前配置不部署仍成立。但训练`base_diff + D_gated @ w`隐含α=1；推理才用α=.25和clip。`verify_gated_consistency`只验证raw门控残差批量/逐行一致，不等于训练pair logit与部署分差一致。当前clip未触发，α错配仍存在。另008要求NP/G1共享模型，实际仍各自独立5折，未完成共享模型对照。因此只停止当前实现，不能断言一致残差学习已被否定。

手动新增[覆盖分层审查](../../scripts/analyze_009_d2_coverage.py)：NP无forward34例seen增量+0.131806，压力−0.003273（CI [−0.0190,+0.0114]），压力没有Top1损失；G1无forward21例seen+0.024026。5个NP压力Top1损失均在已有ICE覆盖域，错首选全为同式且cosine≥0.875，单加相似度阈值不是已证实解决办法。

500/500数据满足“无正模式[M+H]+/[M+Na]+记录 ⇔ 当前无ICE分数”；该域含纯负、纯正不支持及混合，不能等同负离子only。未来须区分模型条件不支持和运行失败/预算耗尽。产物`artifacts/009-d2-coverage/`、0.47秒CPU。以上覆盖分层为事后探索，不进提交。

## 下一步按价值排序

1. **最近的单一提交候选：固定539结构扩库。** 在现有NP/G1缓存上只补12对支持条件下的新候选ICE预测，3对无支持按v9缺失规则处理；重建扩展top100并重算全部z分数。R3另需完整v9基线（代表谱选择与008多谱均值不同），先排清断连盐支持契约，不把FP收益直接移植到ICE。输出实际新增forward任务数与成本，首轮仍沿用008建议的2h T4上限，实测小样本后再放大；该12对不是12次模型调用，也不是整个R3工作的上限。
2. **更有价值的新证据：补当前forward不覆盖的条件。** 冻结“支持条件→v9，不支持→多谱FP”这一单一规则，先做不看标签的完整管线构建，之后一次性用新数据确认；原G1保留250本轮仍未使用，不继续凭开发错例加阈值。此方向比继续所有候选统一加D2残差更有依据。中期研究按候选结构枚举的特异碎片/中性丢失，尤其负离子与钠加合物；先在现有55个未覆盖开发分子做有/无该通道的固定对照，再决定GPU训练。
3. **通往高分的覆盖差距：独立结构来源与多引擎。** 在一条来源上先做固定ChEBI/LipidMaps/PubChem候选召回与干扰实验；按每个来源新增canonical和命中成本决定是否扩展，不直接搬入105M后用public调阈值。对无参考谱结构及真正库外分子分别评价，不能用MassBank同源20/21或all-train模型的NP成绩替代。MIST生成继续保留，但完整谱接线、效果和checkpoint资格尚未成立，当前排在上述两项后。

**本轮不再提交同版，也不提交全局多谱均值、D2或尚未验证的同式组合。** 新submission的时机是一个冻结增量在完整v9口径下跑通并经新数据比较，已知退化得到解释，离线输入和实际运行成本可用；无需等全部路线完成。public可用于外部验证，但小幅涨跌不能代替结构分组验证，更不能保证金牌或奖金。

## 人工执行与产物

本地M5/CPU，Python3.11.16、NumPy2.4.6，扩库前向另用torch2.14.1、RDKit2026.3.6；官方SDK使用独立Kaggle环境。没有新训练或权重变更。权重、原始谱和完整缓存仍保留在ignored目录；本轮未发布数据。

```bash
/opt/homebrew/bin/python3.11 scripts/analyze_009_d2_coverage.py
/opt/homebrew/bin/python3.11 scripts/analyze_009_expansion_sources.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /opt/homebrew/bin/python3.11 scripts/analyze_009_multispectrum.py
/opt/homebrew/bin/python3.11 scripts/analyze_009_formula_fusion.py
```

扩库首跑29.6s，后续缓存复算约2s（run_meta记录最近一次），多谱3.26s，D2分层约0.47s；具体模型/数据SHA与逐例结果在各产物目录。[精简结果](../../data/references/research-results-009.json)随研究文档保存。多谱脚本拒绝覆盖已有results，复算请用`--output`指定新路径；首次使用的`/tmp/g1_build/g1_windows.npz`已逐字节复制到`artifacts/009-multispectrum/g1_windows.npz`（167252字节、SHA一致）并改为默认路径，跨机器可显式传`--g1-windows`。语法和格式仅手动核对，不创建自动化测试体系。

## 009-3 独立单来源固定召回与干扰：LMSD已执行（2026-10-12，CPU-only）

对应计划“下一步”第3条中“在一条来源上先做固定候选召回与干扰实验”。单来源先行，不搬大库、不拿public调阈值、不读G1保留250、不碰heldout。MIST生成线保持deferred（状态见本节末）。

手动执行（本地M5/CPU，零GPU，不push/submit，不下载大库全量）：

```bash
mkdir -p artifacts/009-source
/opt/homebrew/bin/python3.11 scripts/build_009_source_index.py
/opt/homebrew/bin/python3.11 scripts/evaluate_009_source.py
```

### 来源选择与依据

选定唯一单来源：**LIPID MAPS LMSD**本地快照`artifacts/pool-lmsd/LMSD.sdf.zip`（21921603字节，SHA256 `065644fb30b4c7c54acf62d3fdc856d24760a7ddd722687d63cb5ba93f02f908`；解包`structures.sdf`同步核对SHA；来源标注lipidmaps.org、CC BY 4.0、2026-10-04、20.91MB）。快照级许可CC BY 4.0∈{CC0/CC-BY/CC-BY-SA/MIT}，收录；SDF无逐记录LICENSE字段，快照级许可governing，本次快照收录NC/不明记录数为0。RDKit 2026.03.6重验22765条零差异（canon/mass/身份三零）。官方745841#3533343确认LIPID MAPS可用。

未选其他来源的理由（体量/许可/官方确认状态）：ChEBI官方确认可用但本地artifacts/data//tmp零快照/缓存命中，需下载，非最小先行，deferred；COCONUT已在base内（coco_extra 436389 shipped、coconut-casmi26-candidates v2、CC BY 4.0），不是新增来源；PubChem约105M结构（高分管线资产说明）体量禁下载、许可混杂，不搬入；MassBank已在008完成（8831 distinct/539新增），本任务为独立来源不重复。prvsiyan chebi-lipidmaps v1 CC BY-NC-SA重打包本次不用（005计划警告）。

### 固定单来源索引与base交叠/新增

008-B六条口径：canonical_group_key（RDKit2026.03.6）、ExactMolWt float64、无标签质量窗查询、未知adduct单列、真值仅oracle诊断分支、离线可用（.npy/.json/.txt）。复用`src/candidate_expansion.py`查询/oracle函数与许可约束；建索引模式参照`scripts/build_008_candidate_index.py`。

- LMSD全源：50586记录→50002 ok canon→43759去重；与base 712K（union canonical 706242）交叠**20994**，新增**22765**。
- 本次固定单来源索引（new-only）：**22765** canonical，质量float64，与base交叠0（构造即new-only，断言保证），新增22765；expanded全库734964（712199+22765）。
- 产物`artifacts/009-source/`：candidate_index_{masses,keys,smiles,lm_ids,records,meta}、sources_manifest.json、recall_eval_50.json、run_meta.json。seed 20261012；建索引566.6s、评估2.8s；输入SHA/seed/墙钟在meta/run_meta齐备。

### R3-50四阶段召回干扰（分层描述，不估隐藏比例）

R3-50共50分子/59谱：A25（库内、无参考谱，inC真、nref全0）/B25（当前库外、无参考谱，inC假、nref全0）；4张NILU无adduct单列质量不确定，1张Bruker [M-H2O-H]-扩展表支持后计准确窗。FP为fp_single_s2.pt orig-head CPU前向、prep_peaks v32 verbatim、多谱均值、collapse shipped→canon max、order(−score,canon)、RR@25。B_known=21（B中质量已知）。

| 分层（n） | 全库覆盖 base→exp | 质量窗覆盖 base→exp | FP top100覆盖 base→exp | top25：覆盖 base→exp；MRR@25 base→exp；Top1 | 候选量均值（窗shipped / FP）base→exp | A挤下损失 |
|---|---|---|---|---|---:|---|
| all无参考谱（50，含4不确定） | 25→25 | 25→25 | 25→25 | 25→25；0.356190→0.356190（known-only 0.387164→0.387164）；Top1 13→13 | 18.0→18.3 / 17.68→17.98 | lost_top25 0，rank_worsened 0 |
| A库内无参考谱（25） | 25→25 | 25→25 | 25→25 | 25→25；0.712381→0.712381；Top1 13→13 | 12.6→13.0 / 12.24→12.64 | lost 0，worsened 0 |
| B当前库外无参考谱（25，含4不确定） | 0→0 | 0→0 | 0→0 | 0→0；0→0；Top1 0→0 | 23.4→23.6 / 23.12→23.32 | —（无可挤真值） |
| B_known质量已知（21） | 0→0 | 0→0 | 0→0 | 0→0；0→0 | 27.86→28.10 / 27.52→27.76 | — |

干扰细节（真值rank全50零变化，top100 rank亦零变化）：窗新增15对（8分子：A6分子10对、B2分子5对，最大VNEAIGHVYDWTTQ 23→28）；新候选进FP top100计15对、进top25计14对（A 10/9、B 5/5）；旧top25尾部被挤出7个（A5、B2，均为非真值尾部，displaced_top100为0）。B全库/窗/top100/top25新增命中全部为0（LMSD不含本批B真值化学）。4不确定（NMVVBVMYPLMIOU/UTDAGHZGKXPRQI/JOECRAPIMXFQDL/ZTILAOCGFRDHBH）单列，不计准确窗/FP。以上不用MassBank同源20/21或all-train模型NP成绩替代结论。

### 下一来源判断（每来源新增/命中/干扰成本）

LMSD：新增22765 canonical / 全库·窗·top100·top25新增命中0 / 干扰成本+15窗对、top100内14对top25、7个旧top25尾部移位、MRR与Top1/Top25零变化、A零挤下损失。本批B为非脂质小分子化学，LMSD脂质库不覆盖其真值，符合预期 chemotype 失配；该零命中是本固定样本的观测，不得外推为隐藏（timsTOF、化学未知）无价值。判断：**不扩展下一来源**（ChEBI小子集/PubChem均不启动），不搬入大库，不用public调阈值；LMSD索引离线保留但不部署。下一来源仅当出现化学匹配的许可清晰小快照并配固定观察集时再议。

MIST状态一句话：008-C仅完成CPU接口修正与校验脚本跑通，完整谱条件前向与生成评价尚未执行，本轮继续deferred。

## 009-下一步第2项：条件管线冻结与新数据验证准备（2026-10-12后续执行）

对应009计划第18条（运行前固定三方式、不扫权重）与本进展“下一步”第2条。CPU-only，零GPU，不push/submit，不写测试文件/框架/CI，手动运行+人工核对。D2线保持封存（不重开残差训练；覆盖结论只读`artifacts/009-d2-coverage/`）。

冻结规则（单一label-blind管线，`artifacts/009-conditional/frozen_config.json`）：
- 有ICE支持条件（`metadata_has_supported_record=True`：存在正模式`[M+H]+/[M+Na]+`记录）→v9代表谱+ICE融合（`z(base)+z(sims)`，scored优先，冻结top100，Top25输出）。
- 无ICE支持条件（否则）→全部谱logits均值FP（`all_mean`）在全原质量窗、原代表SMILES上排序`(-score, canon)`，取Top25；无ICE、无新候选、不重建shortlist。
- `all_mean` vs `adduct平衡均值`二选一按先验简约固定为`all_mean`（等权每谱、无额外adduct分组超参；单谱时两者一致；与全窗诊断定义连续），未用开发MRR挑选。`observed-ICE`仅审计mismatch，不翻转路由；未来须区分不支持条件与解析/预测/预算失败。
- 原`conditional_diagnostic.json`（含`[M+H]+/[M+Na]+`路由）明确标为`POST_HOC_DIAGNOSTIC_NOT_DEPLOYABLE`，不得直接转正；本次冻结为独立先验冻结，开发数字仅诊断。

实现：新增`scripts/run_009_conditional.py`，输入只用谱metadata（模式/adduct/仪器/CE）与ICE支持判定，输出逐分子路由决策、两分支Top25及分数、最终排序。产物新目录`artifacts/009-conditional/`，记录seed 20261012、输入SHA、墙钟。

开发诊断（只作诊断，不加阈值、不改规则，不合成隐藏比例总分）：
- NP250：路由支持216/不支持34，`route==observed` mismatch 0，全窗vs固定100 Top25相等34/34。最终vs v9：MRR 0.668996801→0.673566149（Δ+0.004569348，CI [−0.000641,+0.011761]），Top1 138→139（改对1/改错0），Top25 237→236（得0/失1：`JUVIOZPCNVVQFO`），改善9/恶化5/持平236。与`conditional_diagnostic`重算差0、Top1/Top25全匹配。
- G1研究250：路由支持229/不支持21，mismatch 0，全窗vs固定100 Top25相等21/21。最终vs v9：MRR 0.907021645→0.916984127（Δ+0.009962482，CI [+0.000366,+0.022263]），Top1 214→217（改对3/改错0），Top25 246→247（得1 `UAPNJSWNICTCCJ`/失0），改善5/恶化2/持平243。与事后诊断差0。
- 不支持层（与009多谱`ice_uncovered/all_mean_ice_fixed100`一致）：NP34为0.320370→0.353968（Δ+0.033598），G1 21为0.601732→0.720333（Δ+0.118601）；支持层恒等（NP216/G1 229零变化）。一致性核对通过。

R3-50确认：不可行全量（CPU-only下）。路由已跑（metadata-only）：支持23（A16/B7）、不支持27（A9/B18），输出`r3_routing.json`。已有FP缓存（`r3_50_logits.npy` 59谱、`r3_50_full_candidate_scores.json`）可复用，但缺ICE前向缓存，支持分支（16 A+7 B需逐候选GL/ICE GPU前向）无法CPU-only评分；另4 B为未知adduct质量不确定无窗。本次只给路由，不评R3 MRR、不回流改规则。

保留250待命协议：G1保留250本轮仍未使用，未读其任何标签/谱，未执行确认。完整待命协议见`artifacts/009-conditional/protocol_g1_heldout_standby.md`（冻结配置SHA `a25d4503…`、输入清单、评分脚本、停损条件、不可逆声明），留待用户批准。

执行：`OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /opt/homebrew/bin/python3.11 scripts/run_009_conditional.py`（约0.48s CPU，`run_meta.json`记录墙钟与全部输入SHA）。人工核对：代表基线最大误差0、500分子三方式重算一致、路由与观测覆盖mismatch 0、Top25得失键已列、R3缓存存在性已查、heldout未触碰。

## 009-下一步第1项：扩库完整v9对照准备（2026-10-12后续执行，CPU-only）

对应本进展“下一步”第1条。CPU-only，零Kaggle GPU（无push/submit/远端），不读G1保留250，不扫阈值/权重，不写测试文件/CI，手动运行+人工核对。产物新目录`artifacts/009-v9recall/`（不覆盖009-expansion/008-expansion），输入SHA/seed/墙钟见`run_meta.json`。

12对需补ICE预测（NP9+G1 3，11分子），逐对核对`inputs.json`谱，ICE支持契约`code/ice/ice_runner.py:34 COV=([M+H]+,[M+Na]+)`+`is_covered`（正模式且adduct在COV，仪器任意，CE任意）：NP BHQCQFFYRZLCQQ×2（JSIQROACCRQDKK rank13、CWNPOQFCIIFQDM系WJXSWCUQABXPFS rank38等）、BMLIIPOXVWESJG、DFMAXQKDIGCMTL、HSMPDPBYAYSOBC、HYFHYPWGAURHIV、ILEDWLMCKZNDJK、LOIYMIARKYCTBW（rank4最高）、WJXSWCUQABXPFS；G1 GKBSLXWQUZXZPM（rank18）、PZQDWRSEJBKJJK（rank69，另4张[2M+H]+不支持）、SFHLFNWOCGHDOA（rank21）。12个新SMILES全RDKit可解析单组分、元素VALID。3对无支持按v9缺失规则scored续后、不补分不打0（BJRNKVDFDLYUGJ×2全`[M+CH2O2-H]-`负谱、RUDATBOHQWOJDD全`[M+NH4]+`不在COV）；另3新增在Top100外（rank109/126/131）不需forward。明细见`pairs_12_supported_detail.json`/`pairs_3_unsupported_detail.json`/`pairs_all_18.json`。

实际forward任务数（`unique(canon,adduct,instr_token,ce_bucket)`，timsTOF→QTOF）：增量12新候选mean 31、split 41（BHQC 4+4、BMLI/DFMA/HSMP/HYFH/ILED各4–5、LOIY/WJXS各2、GKB3、PZQD2、SFHL3之和已复算41）、union 51。12对≠12次调用。全11分子扩展shortlist（635候选）dry-run：split 2066、mean 1691。T4按26/s pure：增量split约1.6s、全量约79.5s，含load/overhead分别约3–6min与5–10min，均50倍以上低于008建议2h上限（G1全量56062约36min供量级参考）。首轮仍按2h上限，实测小样本后再放大。任务与成本见`forward_tasks.json`。

CPU扩展Top100复算（已有FP分数，全窗序截断100）：NP 0.5145934285521241、G1 0.8842502164502164，`repr`逐位一致；NP Top1 93/Top25 230、G1 Top1 204/Top25 246无损失；NP仅1例28→29（BHQC真值，仍25外），G1零恶化。G1三尾部被挤出逐例已列（GKB/PZQD/SFHL各原100→101）。新shortlist存`cpu_expanded_top100.json`。待ICE回传后全shortlist重算z（`z(base)+z(ice_merge)`等权，ddof1，missing不参与均值、scored优先、3无支持续后），不能尾接旧融合。

最小bundle（`bundle/`，tar 416KB，SHA见run_meta；大权重不进包，走Drive/私有dataset/私有存储三法+mismatch拒绝）：`inputs.json`（11分子worker兼容，扩展Top100）、`new_candidates_12.json`、`code/ice/`与`g1_worker.py` verbatim（SHA与G1验证包一致）、`install.sh`、`MANIFEST.iceberg.json`、`weights_reference.json`、`README.md`（Colab三步+truth/heldout声明+输出格式）。CPU dry-run已验（2分子436 jobs ok-dry-run，11分子2066 jobs规划0.61s）。待用户Colab运行回传，不假设会话在线。

YTCZZ契约（`ytczz_contract.json`）：query `[M+H]+`正QTOF 49峰为SUPPORTED；候选为断连钠盐（`.[Na+]`，C24H29NaO5，2记录），RDKit可解析、元素VALID，作业会被ATTEMPT而非跳过；但MAGMA/DAG对断连盐UNVALIDATED，无救回先验。判定CONDITIONAL/NOT GUARANTEED，不假设救回rank40，不把FP rank40移植到ICE，需实际forward分离失败类型。

R3完整v9基线缺口（`r3_baseline_gaps.json`，只排清单不跑新分）：代表谱、CE回退、仪器、覆盖（20/21不移植）、窗/shortlist/z重算、YTCZZ盐、权重runner复用、成本（2h上限小样本后放大）、真值边界。

可重跑脚本：`scripts/run_009_v9recall.py`（后补固化，`--check-only` 10目标全OK、关键数字与上文一致，`--update-run-meta`只加`reproducibility`小节不改原键；tar不重打，gzip时间戳不稳定，一致性以`bundle/inputs.json` SHA为准）。

## 009-下一步第1项后半段：ICE回传后离线融合评估（2026-10-12，CPU-only）

对应本进展“下一步”第1条后半段与`bundle/README.md §5`、`cpu_expanded_top100.json z_recalc_steps`。CPU-only，零GPU，不push/submit，不读G1保留250，不扫权重/阈值，不写测试文件/CI，手动运行+人工核对。脚本`scripts/evaluate_009_v9recall_fusion.py`，产物`artifacts/009-v9recall/fusion_eval.json`（seed 20261012，墙钟见文件内）。

入库：原包`/Users/zhaohuaxiang/Downloads/v9recall-scores.tar.gz`不动，复制到`artifacts/009-v9recall/scores/v9recall-scores.tar.gz`，SHA256 `f3c76409854595816b197d7685c24d57e2139fffac3c9b0d4dcae03d911799a5`两端一致；解包得`scores/`下split/mean两JSON+两meta齐备（split 1058444字节`feb6e0c9…`、mean 491058字节`5dd4df7b…`，meta见fusion_eval）。meta核对：split `n_mols 11/n_cands 635/unique_jobs 2066/n_pred 2066`、mean `11/635/1691/1691`，`prep_tag panel-A1:v1`、`ckpt gen deef0451…/inten c49e330e…`与`weights_reference.json`一致、`ms_pred_commit 708148c2…`、`status ok`、`n_mols_covered 11`、`n_spectra_unsupported 20`；`n_scored=0`为g1_worker已知`np.unicode_`计数器bug（007已记录），预测与相似度有效。12/12新候选`fused_merge`非空有限再断言通过（mismatch即停，未触发）；split 635/635 `fused_merge`有限、mean 635/635 `fused`有限。

融合（全扩展shortlist重算`z(base)+z(ice_merge)`等权，ddof1，<2/零方差→0，missing不参与均值贡献0；scored按`(-fused,canon)`、unscored按`(-base,canon)`续后；3无支持按base续后不打0；split用`fused_merge`主方案、mean用`fused`对照；truth仅排序后计RR）：

| 格 | 范围 | 基线MRR@25 | 扩展MRR@25 | Δ | Top1基→扩 | Top25基→扩 | 改善/恶化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| split主 | 11支持 | 0.784091 | 0.784091 | +0.000000 | 8→8 | 10→10 | 0/0 |
| split主 | NP支持8 | 0.703125 | 0.703125 | +0.000000 | 5→5 | 7→7 | 0/0 |
| split主 | G1支持3 | 1.000000 | 1.000000 | +0.000000 | 3→3 | 3→3 | 0/0 |
| split主 | NP无支持2 | 0.190476 | 0.190476 | +0.000000 | 0→0 | 2→2 | 0/0 |
| mean对照 | 11支持 | 0.696212 | 0.696212 | +0.000000 | 7→7 | 10→10 | 0/0 |

逐例（split，基→扩，RR同）：BHQC 40→42（RR 0→0，仍25外）、BMLI 8→8、DFMA 2→2、HSMP 1→1、HYFH 1→1、ILED 1→1、LOIY 1→1、WJXS 1→1、GKB/PZQD/SFHL均1→1；12新候选融合rank为23/26/26/41/47/14/20/3/46/24/77/19（仅LOIY新ZCIF rank3进Top25，真值仍rank1）。mean逐例同样零RR增损（BHQC 36→36等）。全250外推Δ均为0（NP分母250、G1分母250，其余分子无新增视作不变）。审计：FP重叠base差0；ICE新旧重叠623对split最大差5.25e-04（PZQD/SFHL各1对，其余<1e-6，系any-rdkit数值微差，已记录）。

Verdict：**NOT_SUBMIT，不构成单一提交候选**。四条：①单主要变化通过（固定539扩库）；②同口径可解释正向不通过（split+mean双零增益，无Top1/Top25得失）；③已知退化已处理通过（BHQC 40→42仍25外、G1尾部挤出未伤真值、BJRN 3/RUDA 21不变，R3缺口另列）；④离线可跑通过（T4全量约1.3min pure/5-10min含overhead，CPU融合0.92s）。缺口：真值已在库子集上新增全为干扰；R3 B_known 20/21的FP召回尚未做完整v9融合，YTCZZ盐等缺口未闭环；不得外推隐藏收益。下一步按009计划先冻R3完整v9基线或找库外真值融合正向证据。

手动命令：`mkdir -p artifacts/009-v9recall/scores && cp /Users/zhaohuaxiang/Downloads/v9recall-scores.tar.gz artifacts/009-v9recall/scores/v9recall-scores.tar.gz && tar xzf artifacts/009-v9recall/scores/v9recall-scores.tar.gz -C artifacts/009-v9recall/scores/`；`/opt/homebrew/bin/python3.11 scripts/evaluate_009_v9recall_fusion.py`（0.92s CPU）。

## 009-R3-50 完整v9基线融合评估（2026-10-12，CPU-only）

对应009计划下一步第1项（R3完整v9基线）与本进展“下一步”第1条后半段。CPU-only，零GPU，不push/submit，不读G1保留250，不扫权重/阈值，不写测试文件/CI，手动运行+人工核对。脚本`scripts/evaluate_009_r3baseline_fusion.py`（新），产物`artifacts/009-r3baseline/fusion_eval.json`（seed 20261012，墙钟2026-10-12T20:48:29+08:00，Python3.11.16、NumPy2.4.6，输入SHA见产物内）。

入库：原包`/Users/zhaohuaxiang/Downloads/v9r3baseline-scores.tar.gz`不动，复制到`artifacts/009-r3baseline/scores/v9r3baseline-scores.tar.gz`，SHA256 `d65e2cc18f1d8ed995a1a0dfb810a873446dace5b334aff9f9a2f5d2ffa97990`两端一致；解包得`scores/`下split/mean两JSON+两meta四件齐备（split 260651字节`f7b751a1…`、mean 240254字节`a851cea0…`，meta `89a33695…/cfab0998…`，tar内顶层`scores/`）。meta核对：split/mean均为`n_mols 23/n_cands 523/unique_jobs 523/status ok/n_mols_covered 23/n_spectra_unsupported 4`，`prep_tag panel-A1:v1`、`ckpt gen deef0451…/inten c49e330e…`与`weights_reference.json`一致、`ms_pred_commit 708148c2…`、`n_pred 1569/n_pred_fail 0/n_bad_smiles 0`（零失败）、`device cuda/rdkit 2026.09.1/torch 2.11.0+cu130`；`n_scored=0`是已知`np.unicode_`计数器bug（007已记录，23条同错），分数有效。断言通过（不一致即停，未触发）：split `fused 523/523`非空有限、`fused_merge 0/523`全空；mean `fused 523/523`真实有效；`sims`全为`(ce=null,ce_bucket=null,how=merged-sum)`，即R3谱CE全未知→null→sum桶，无逐CE预测可合并；split/mean `fused`最大差0、`base`与bundle最大差0。YTCZZ真值核对一致：`fused(mean)=0.12734447`，`base=-732.16483289`，FP rank40。

融合口径（数据强制，不是开发挑选）：“R3因CE未知无split-merge可用，改用mean路；与NP/G1的fused_merge主方案不同口径，不混为一谈”。如实改用ICE-mean路`fused`做`z(base)+z(ice_fused)`等权融合（ddof1，<2/零方差→0，missing不参与均值贡献0；scored按`(-fused,canon)`、unscored按`(-base,canon)`续后；支持23全scored，无续后分支触发；truth仅排序后计RR）。4 NILU与27不支持分子不参与ICE融合（维持原规则：不支持走CPU all-mean FP即`r3_50_full_candidate_scores.json expanded`已有分数；4不确定`NMVVBVMYPLMIOU/UTDAGHZGKXPRQI/JOECRAPIMXFQDL/ZTILAOCGFRDHBH`0候选单列排除，不入分母）。

23支持逐例基线（FP top100内rank/RR）vs融合后rank/RR（mean路，RR@25）：

| 分子 | 层 | n候选 | 基线rank/RR | 融合rank/RR | ΔRR |
| --- | --- | ---: | --- | --- | ---: |
| CLSDHBVHQKWEJR | A | 4 | 2/0.5000 | 1/1.0000 | +0.5000 |
| UVYPXLQJIYVPPA | A | 6 | 1/1.0000 | 1/1.0000 | +0.0000 |
| XTDLZZLFKOTTLJ | A | 6 | 1/1.0000 | 1/1.0000 | +0.0000 |
| ANDKUSLFYHFLNO | A | 1 | 1/1.0000 | 1/1.0000 | +0.0000 |
| BEGARHAEPVBWAE | A | 8 | 1/1.0000 | 1/1.0000 | +0.0000 |
| COYVFWSNQRUPIO | A | 6 | 3/0.3333 | 1/1.0000 | +0.6667 |
| JNMCFADRRBGAIM | A | 5 | 2/0.5000 | 1/1.0000 | +0.5000 |
| LWFZKXMCJDHBNJ | A | 28 | 7/0.1429 | 21/0.0476 | −0.0952 |
| MKNWXPPWASGIJB | A | 23 | 2/0.5000 | 1/1.0000 | +0.5000 |
| NYGVGDLVEMUBNK | A | 24 | 2/0.5000 | 1/1.0000 | +0.5000 |
| ROTFFYYCARWRHQ | A | 9 | 4/0.2500 | 1/1.0000 | +0.7500 |
| TYSLVBJOPFOBFU | A | 2 | 1/1.0000 | 1/1.0000 | +0.0000 |
| WKADCTRDFMBYPU | A | 33 | 1/1.0000 | 2/0.5000 | −0.5000 |
| YSNCVQPXJAIHHN | A | 6 | 1/1.0000 | 1/1.0000 | +0.0000 |
| ZSNZKFBMFHDMOM | A | 23 | 2/0.5000 | 2/0.5000 | +0.0000 |
| MLBROOJXUZKDHY | A | 19 | 1/1.0000 | 1/1.0000 | +0.0000 |
| BXKMAWZFXVEIMC | B | 27 | 1/1.0000 | 1/1.0000 | +0.0000 |
| FJEBZINYFPJJAW | B | 18 | 1/1.0000 | 1/1.0000 | +0.0000 |
| STLDIXSFXILIFT | B | 28 | 1/1.0000 | 1/1.0000 | +0.0000 |
| VMHJESQIJHKLBB | B | 55 | 1/1.0000 | 1/1.0000 | +0.0000 |
| GFJWACFSUSFUOG | B | 64 | 1/1.0000 | 1/1.0000 | +0.0000 |
| IFEJLMHZNQJGQU | B | 44 | 1/1.0000 | 2/0.5000 | −0.5000 |
| YTCZZXIRLARSET | B | 84 | 40/0.0000 | 6/0.1667 | +0.1667 |

分层（MRR@25/Top1/Top25/改善/恶化，RR口径）：23支持基线0.748965→融合0.857143（Δ+0.108178），Top1 14→18，Top25 22→23，改善7（CLSD/COYV/JNMC/MKNW/NYGV/ROTF/YTCZZ）/恶化3（LWFZ/WKAD/IFEJ）/持平13；A16基线0.701637→0.877976（Δ+0.176339），Top1 8→13，Top25 16→16，改善6/恶化2；B7基线0.857143→0.809524（Δ−0.047619），Top1 6→5，Top25 6→7，改善1/恶化1。独立numpy重算YTCZZ/LWFZ等rank一致，23基线rank与`r3_50_full_candidate_scores.json expanded`全匹配。

YTCZZ verdict：rank40→6，进入Top25（RR 0→0.1667），本次救回。query `[M+H]+`正QTOF 49峰为SUPPORTED；候选断连钠盐84/84 `fused`有限、peaks存在，作业被ATTEMPT而非跳过，即“ICE对断连钠盐可尝试打分”成立；但MAGMA/DAG对断连盐UNVALIDATED，本次单例救回不保证同类普遍可救，不把FP rank40移植为ICE先验。

冻结条件规则在R3的首次可验证数字（支持分支用本次ICE融合，不支持分支用已有CPU分数；4不确定排除并声明，分母46可评估=A25+B21）：不支持23（A9/B14）FP为MRR 0.732816、Top1 13、Top25 23/23；FP-only 46为MRR 0.740890、Top1 27、Top25 45；条件46（融合支持+FP不支持）为MRR 0.794979（Δ+0.054089）、Top1 31、Top25 46/46；A25为FP-only 0.712381→条件0.825238（Δ+0.112857，Top1 13→18，Top25 25→25）；B21为0.774830→0.758957（Δ−0.015873，Top1 14→13，Top25 20→21）。4 NILU单列排除，已声明。

Verdict：**NOT_SUBMIT**。四条：①单主要变化通过（支持分支FP→ICE-mean fused唯一变化，无扫描；但与NP/G1生产口径不同，仅为R3测量增量）；②同口径正向通过但不构成提交理由（23总体+0.108、A强正向，B层−0.048、小样本23不能外推隐藏timsTOF）；③已知退化未处理不通过（LWFZ 7→21、WKAD 1→2、IFEJ 1→2共3例RR恶化，B21条件MRR−0.015873/Top1 14→13，机制未查，需新数据一次比较）；④离线可跑通过（CPU融合0.01s，T4全量523 jobs pure约20s、含overhead 5–10min远低于2h）。缺口：口径不可混、3例恶化机制未解释、G1保留250待命未执行（本轮未读）。下一步按冻结协议先查因并经新数据比较，或找跨域同口径正向证据；不据public总分推断隐藏类别。

手动命令：`mkdir -p artifacts/009-r3baseline/scores && cp /Users/zhaohuaxiang/Downloads/v9r3baseline-scores.tar.gz artifacts/009-r3baseline/scores/v9r3baseline-scores.tar.gz && tar xzf artifacts/009-r3baseline/scores/v9r3baseline-scores.tar.gz -C artifacts/009-r3baseline/scores/`；`/opt/homebrew/bin/python3.11 scripts/evaluate_009_r3baseline_fusion.py`（0.01s CPU）；独立numpy重算YTCZZ/LWFZ等rank零差异、23基线与expanded全匹配、tar SHA两端一致已核。

## G1保留250一次性确认评分（2026-10-13，CPU-only；该集已转已用验证）

对应`artifacts/009-conditional/protocol_g1_heldout_standby.md`待命协议的唯一一次执行。本轮动用保留250标签做一次性确认，跑完该集即转已用验证，不再是保留集。零GPU，不push/submit，不扫参数，不写测试文件/CI，手动运行+人工核对JSON。脚本`scripts/score_009_heldout.py`（新，手动研究脚本非测试；冻结口径逐字复用`run_009_conditional`：zscore ddof1、scored优先+unscored按base续后、order(-score,canon)、RR@25、seed 20261012、bootstrap 20000；输出已存在即拒绝覆盖，不留第二轮确认接口）。产物`artifacts/009-heldout/heldout_score.json`（seed 20261012，墙钟2026-10-13T21:48:41+08:00，评分0.58s CPU，输入SHA见产物内）+ `run_meta.json` phase2增补。

冻结核对：`frozen_config.json` SHA256 `a25d45032c2bcc9ac822a74379f95836892625c1459cbd7fdb29d18274772ed3`逐字节一致（不一致即停，未触发）。

入库：原包`/Users/zhaohuaxiang/Downloads/v9heldout-scores.tar.gz`不动，复制到`artifacts/009-heldout/scores/v9heldout-scores.tar.gz`，SHA256 `5c2c66a94193be0c4547d4c623844288d5902ca50372a4f33757711d4198d738`两端一致；解包得`scores/`下split/mean两JSON+两meta四件齐备（split 34887795字节`4fc91332…`、mean 16319028字节`bfac112a…`，meta `df3381a6…/6438b54b…`，tar内顶层`scores/`）。meta核对：split/mean均为`n_mols 232/n_cands 18644/status ok/n_mols_covered 232`，`prep_tag panel-A1:v1`、`ckpt gen deef0451…/inten c49e330e…`与`weights_reference.json`一致、`ms_pred_commit 708148c2…`、`n_pred_fail 0/n_bad_smiles 0`（零失败）；`n_scored=0`是已知`np.unicode_`计数器bug（007已记录），分数有效。断言通过（不一致即停，未触发）：split `fused_merge 18644/18644`非空有限、mean只有`fused`（18644/18644有限，无`fused_merge`）、split覆盖==支持集、canon零缺失零多命中、FP representative/all_mean全覆盖full_keys。

路由复用：`routing.json`既有路由，不重路由，只核对计数：250 = 支持232 + 不支持18。支持分支= frozen_top100上FP representative + split `fused_merge`（按canon匹配，一canon一SMILES命中，无需取第一分支触发）做v9_order；不支持分支= all_mean全窗full_order，无ICE不补分。observed-ICE仅审计：mismatch 0（支持232全observed、18不支持全无ICE，符合预期）。

评分（最终final vs v9诊断基线，RR@25）：

| 范围 | 最终MRR | v9 MRR | Δ | Top1 终←v9 | Top25 终←v9 | 改善/恶化/持平 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 全体250 | 0.905171 | 0.907838 | −0.002667（ci95 [−0.01, 0.004]） | 210←211（改对1/改错2） | 247←247 | 1/3/246 |
| 支持232 | 0.909308 | 0.909308 | +0.000000（全持平） | 197←197 | 229←229 | 0/0/232 |
| 不支持18 | 0.851852 | 0.888889 | −0.037037（ci95 [−0.138889, 0.055556]） | 13←14 | 18←18 | 1/3/14 |

Top25得失键：得0、失0。不支持层4例RR变化：改对1（GLPAKWAJVAXLMJ rank2→1）、改错3（NJNOQHUTUOJWII rank1→2、NQCKXVOUOLCXPN rank1→2、YZSPJKXBERQUHI rank2→3）；Top1改对1（GLPA）、改错2（NJNO/NQCK）。支持层final与v9逐分子全同（符合构造：支持分支final即v9）。独立numpy重算AAIPSAYOWGJQIZ v9 Top25与fused_zsum零差异。

Verdict：**NOT_SUBMIT，不构成单一提交候选**。四条：①单主要变化通过（冻结单规则一次性执行，无扫描、无阈值、无重选，mismatch 0）；②同口径可解释正向不通过（全体Δ −0.002667为负、Top1净−1、Top25零得失；支持层恒等无增益、不支持层18例Δ −0.037小样本不能外推）；③已知退化未处理不通过（不支持层3例RR恶化+2例Top1改错机制未查；v9诊断基线在不支持层本就无ICE，不是对 deployable 分支的比较，不得据此改all_mean）；④离线可跑部分通过（CPU评分0.58s；但提交端新query仍需ICE forward，其GPU/Colab成本与CPU最终notebook可行性未在本轮验证）。缺口：G1保留集已耗用、无第二轮确认可用；不支持层样本小、恶化机制未解释；不得合成隐藏比例总分、不得据public总分推断隐藏类别。下一步：冻结规则保持部署候选状态，缺口需新数据一次比较或跨域同口径正向证据。

手动命令：`cp /Users/zhaohuaxiang/Downloads/v9heldout-scores.tar.gz artifacts/009-heldout/scores/v9heldout-scores.tar.gz && tar xzf artifacts/009-heldout/scores/v9heldout-scores.tar.gz -C artifacts/009-heldout/scores/`；`/opt/homebrew/bin/python3.11 scripts/score_009_heldout.py`（0.58s CPU）；独立numpy重算AAIPSAYOWGJQIZ Top25零差异、tar SHA两端一致已核。
