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
