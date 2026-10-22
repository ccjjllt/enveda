# 015 研究交接：从0.323平台期走向可验证的新证据

交接日期：2026-10-17，北京时间。目标是竞争奖金与奖牌；此目标不等于当前方法已具备获奖水平。本文件供下一次对话或接手研究者直接执行，配套[015计划](../plan/015-candidate-analog-ranker-and-handoff.md)、[015实际进展](../progress/015-candidate-analog-ranker-and-handoff.md)。项目根目录为`/Users/zhaohuaxiang/ccjjllt/enveda`，接手时Git基点`ca1abba`。

## 1. 先读这几条

- 已提交最佳仍为**public 0.323**，v9r2与v10显示同分；本轮没有新submission。015已跑完候选级OOF：NP允许同结构参考时MRR **0.668997→0.725342**，直接特征单独加入也达到**0.723109**。目标结构参考缺失时尚未确认增益，优先推进直接谱检索的独立来源确认。
- **不能用同分倒推隐藏数据类别或认定扩库毫无作用。** v10只是新增538个有效MassBank结构；本地R3覆盖改善真实，但与扩库同源、人工分层，不能预测隐藏收益。当前只观察到显示的0.323，无法从总分恢复逐分子变化。
- 后续核心问题是：候选已经在列表时，怎样用额外谱证据在同式异构体间排对；候选缺失时，怎样补库或生成且控制大量干扰。单纯再扩大库或换一种线性融合都没有已确认的获奖级收益。
- 所有NP/G1/R3及原保留250均已用于研究。015的canonical五折OOF只验证当前排序器的开发表现，**不是全新盲测，也不是预训练encoder未见结构证明**。
- 不创建测试文件、测试框架或CI。人工跑研究脚本、重放模型、核对指标与日志；每一轮写配对计划/进展。代码成功、运行成功、离线收益、排行榜收益必须分开。

## 2. 提交、数据与算力的真实状态

| 项目 | 已核实状态 |
|---|---|
| 比赛 | `enveda-CASMI26-molecule-id-mass-spectra`；MRR@25，代码提交，最终关闭互联网 |
| v9r2 | ref`57028399`，notebook版本9，scriptVersionId`356902571`，COMPLETE，public0.323 |
| v10 | `ccjjllt/enveda-submission-v10`版本1，scriptVersionId`359192425`，ref`57293821`，COMPLETE，public0.323 |
| v10实际修改 | 原候选712199→712737，canonical706242→706780；539来源条目中1个无有效身份剔除，实际新增538 |
| v10完整运行 | T4约41.45分钟；400×25输出；72336次ICE预测，零失败；占位test不能证明泛化 |
| 本轮远端复查 | 10月17日晚CLI确认上述两个提交同为COMPLETE/0.323；未启动新作业、未提交、未改变最终选择 |
| 配额快照 | 本轮CLI：GPU已用2.97h、剩27.03h/30h；TPU已用0、剩20h；返回刷新时间`2026-10-24T00:00:00`。这不是可分配设备保证 |
| 竞争距离 | 013当日榜单快照第一0.495、第五0.468；第五比我们高0.145。不是金牌分数线，也不是本轮重新查询的名次 |
| 当前工具边界 | 本会话没有可调用Kaggle MCP工具，已用Kaggle CLI完成只读查询；历史MCP成功不等于当前可调用。Colab当前执行通道未确认，不默认改用慢GUI |

可直接用的最终入口是`notebooks/submissions/submission-v10/enveda-submission-v10.ipynb`。metadata固定六项输入：`prvsiyan/casmi26-fp-models-v2/1`、`prvsiyan/coconut-casmi26-candidates/2`、`ccjjllt/casmi26-train-fp-pool/1`、`ahmedberatozer/casmi26-iceberg/1`、`ccjjllt/casmi26-cp313-wheelhouse/1`、`ccjjllt/casmi26-massbank539-expansion/1`。原镜像SHA见同目录`kernel-metadata.json`。

官方数据要点和规则来源保存在[README](../../README.md)与[比赛记录](../../data/references/competition-evidence-2026-09-30.json)。训练文件实际2539608谱、275810原始结构键，005重算canonical274186。官方隐藏集描述约400分子/1500谱，不可当作可见占位test统计。第三类de novo不在PubChem/COCONUT、类别比例未公布；只靠候选库检索不可能保证覆盖。

## 3. 015已冻结的研究方法与结果

完整特征500query、42580×10，179.99秒CPU、3.76GiB；五方案25折＋5个全量实验fit共15.67秒，零警告；保存模型重放误差0。随后追加固定参数direct-only归因消融（明确post-hoc、保持原结果），10折2.82秒、零警告。没有GPU运行或部署。

| OOF方案 | NP允许目标结构参考 | NP移除目标结构参考 | G1两场景 |
|---|---:|---:|---:|
| 原v9 | 0.668997 | 0.668997 | 0.907022 |
| LR原4特征 | 0.669852 | 0.669852 | 0.906673 |
| LR原特征＋同结构参考质量 | **0.723109** | 0.668829 | 0.906006 |
| LR原特征＋同结构及analog六特征 | **0.725342** | 0.670582 | 0.905994 |
| HGB原特征＋同结构及analog六特征 | 0.712199 | 0.667571 | 0.905696 |

NP B组合相对原v9 Δ+0.056345，95%CI[+0.038638,+0.075550]，Top1从138到151（得13失0），Top25从237到241。直接特征单独加入Δ+0.054113，CI[+0.036819,+0.072768]，Top1同为151。组合相对direct-only只多+0.002232，CI[−0.006480,+0.010874]：**本批主要增量可以由直接参考谱重现，尚未确认analog的额外增益**，不是证明analog绝对无用。

NP A组合相对v9仅+0.001586、CI跨0；G1−0.001028、CI跨0。B并非每个query都有同结构参考：NP203/250池内有真值参考，top50真值中185有正质量；G1为0/250有真值参考。因此G1两列不是两个有效的已见/缺失对照。不能外推NP B收益到隐藏全体。

固定五折种子20261017，同一canonical跨场景同折，所有CI为20000次配对分子bootstrap，属于反复使用开发面板上的描述性区间，不消除多轮选择偏差。独立审阅已重建500个v9序列、5000个输出排名和43项SHA，匹配；更多指标、得失、哈希见配套进展及产物。

方法固定如下，详细参数由计划与产物manifest承担：

1. NP250＋G1研究250，共500个canonical，沿用v9“原始峰最多”的代表谱，原v9 FP＋ICE完整top100；只在前50的同式槽位内重排。保留未进top50真值的负例，不改变候选集合。
2. train每50行采一条、先排除全部500查询的所有原始谱hash。B允许同结构其他参考；A再全局排除500目标canonical全部参考。A是已知标签驱动的评价干预，正式推理不可能知道真值，它不作为可部署分支选择。
3. 同极性、质量±200Da、最近至多2000条。floor=.002、top256、sqrt；去前体±1.5Da，已知多聚体另去完整单体±1.5Da。该删峰规则是冻结的新假设，仍需外部确认；记录剩余信号强度，不能只看删峰后重新归一化cosine。
4. direct与precursor差位移的峰对联合贪心一对一；位移仅两端单体、单电荷、同极性。quality压低单峰主导背景，按参考canonical去重后保留至多80个结构。
5. 四个原排序特征＋六个直接/类似物特征。`direct_quality`实际指**同候选canonical参考的联合谱匹配质量**，不是“只用未位移峰”的pure-direct cosine。类似物传播排除候选自身canonical。
6. 同一canonical全部场景同折，固定五折；NP/G1共享LR与HGB，各有baseline-only和新增证据两版，另有HGB打乱quality版。每query等权，两场景各一半；标准化只在训练折拟合。参数不按结果扫描。

负对照保留真实谱质量筛出的top80身份、候选Tanimoto、maxT和同结构direct_quality，只在query内打乱质量与参考的对应关系。因此它检验**已选参考内的谱权重对应贡献**，不是去掉全部谱信息。若与打乱版相近，不能直接说所有增益都是结构先验；若有差异，也要看分子级不确定性。

## 4. 必须继承的历史结论

| 方向 | 实际观察 | 后续处理 |
|---|---|---|
| 013/014 analog支撑 | 013只有query级支撑；固定36例中29例最佳匹配≥80%由一对峰贡献；前体/多聚体完整单体影响大 | 不拿支撑率当MRR，不沿用整谱direct/shift二选一；015改为联合配峰及候选证据 |
| 固定碎片 | 原开发55不支持adduct例MRR+.03254但CI跨0；完整NP仅+.000632、G1+.006526；R3纯负22例.720671→.716883 | 现配置不直接部署；不是证明所有碎片证据无效 |
| ChEBI子集 | 3-star、单连通中性100–1500Da子集新增8457 canonical；NP自身来源覆盖199/250，G1/R3为0 | 不当独立通用覆盖验证，也不是整个ChEBI |
| PubChemLite v3.3.0 | 546窗新增12092 canonical；FP-only扩库NP.514593→.478352、G1.884250→.879198、R3.762629→.627743 | 未做完整ICE；需要先评估真值召回和新增难负例，不能宣称扩库已提分或完全无用 |
| 条件多谱 | 已用原保留250：.907838→.905171，Δ−.002667，CI跨0 | 不部署；原保留集不能再叫盲集；全局均值已经多次无效，不重复扫 |
| D2 residual | 参考缺失场景退化；旧实现训练α1/推理α.25且NP/G1各自模型 | 继承失败配置，但不扩大为所有残差学习无效；任何新实现要训练/推理一致 |
| FIORA、F4a | 已运行配置未通过迁移 | 现配置不进入下一提交，不能靠再次命名成为新证据 |
| FRIGID | 完整MIST→匹配checkpoint阈值二值化→可推理formula/length→生成链未跑通 | 旧`mist_cf_formula.py`会加入truth formula，仅oracle诊断；不能以不完整链失败否定完整生成 |
| 奖金复现材料 | 原FP s2 BCE+hard-negative CE训练源码已找到 | exact input/split/seed/CLI/发布权重绑定仍缺；源码存在不等于血缘完整 |

阅读优先级：015→014→012；需要追踪旧残差/验证口径时读009/010，不只看旧摘要。R3全部50在完整train无canonical真值参考已核实；它们与MassBank扩库同源，46质量可用真值现都进入v10候选窗。因此“相对v10新增0”不能否认扩库此前的覆盖价值。

## 5. 高分实现与前沿研究如何转成动作

- 013已拉取并审阅[prvsiyan analog](https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline)、[Berat analog ranker](https://www.kaggle.com/code/beraterolelk/0-336-sota-enveda-casmi26-analog-ranker)、[Seyit PubChem](https://www.kaggle.com/code/seyitkaangunes/casmi26-e006-pubchem-promo-rel-150-lb-0-446)源码。能借鉴的是候选—参考结构一致性、直接谱检索、扩库难负例排序；标题分数不是我们独立测得的成绩。012多个0.451/0.452标题含自报/目标值，不能当多次独立验证。
- [matchms ModifiedCosineGreedy](https://matchms.readthedocs.io/en/latest/api/matchms.similarity.ModifiedCosineGreedy.html)定义direct与precursor差位移的联合匹配；贪心近似不是Hungarian全局最优。015据此纠正013的二选一实现，没有证明所有adduct适用。
- [MS2Query](https://www.nature.com/articles/s41467-023-37446-4)支持多参考化学一致性思路，但其谱库analog检索指标不等于本赛精确结构MRR。015先验证候选传播，不直接移植论文阈值。
- [MS2DeepScore 2.0正式论文](https://www.nature.com/articles/s41467-026-69083-y)及[官方代码](https://github.com/matchms/ms2deepscore)可作为下一种独立谱表示候选。其目标为Daylight-like4096，不是本轮Morgan2048；不能把模型分数与Morgan Tanimoto直接当同尺度差值。跨极性网络能力也不保证精确结构排序；Na/K、正/负/跨极性分开报告。
- 生成路线只在完整条件链和允许来源核实后开20例有界端到端研究。不得用truth formula、benchmark test训练的长度先验或不匹配checkpoint的阈值声称真实推理。它用于候选库外覆盖，当前不替代主线排序。

## 6. 接手后的执行顺序

**优先级已经收敛：先做直接谱检索的来源迁移确认，再决定下一次提交；参考缺失的analog仍是独立研究问题。** 不再只重做“有相似谱”的查询级审计，也不直接用本轮全数据模型替换比赛入口。

1. **先读取现成结果，不重复调015。** 确认输入/模型SHA与重放一致；保留原v9、LR baseline、LR direct-only及组合四个固定方案作为下一面板对照。三种LR都按015固定参数、权重和现有500开发分子全量拟合后冻结checkpoint，新面板只预测；已有两个全量模型，direct-only目前只有5个折模型，需补一次全量fit并记录。主线优先较简单的direct-only，理由是其已重现NP B大部分增量；这种结果驱动选择必须在下一面板确认。不得按NP/G1标签选择推理分支，也不使用真实结构有无参考作为部署开关。HGB direct-only在NP A下降约0.0093，本配置不作为候选成品。
2. **冻结新的确认面板与候选输入。** 首先清点所有旧query canonical/hash（NP250、G1研究250、已用保留250、R3等）并写排除清单；从未用于本项目决策的canonical选取，按来源/采集记录隔离参考、兼顾正负离子与多聚体。建议目标400个新canonical，分为有跨来源参考与自然缺库两组，数量不足如实记录，不能以重叠谱凑数。先完成来源可追溯性与canonical交叉检查，再生成真实FP/ICE候选；不得加入truth候选。面板产生前固定seed、名单和四个对照，禁止按结果重新采样。新面板只称“本项目此前未用”，不称预训练未见。已见结构独立来源评价和额外整组移除压力分别报告。
3. **如果现有约5万谱抽样有信号，再扩参考覆盖。** 先改为canonical/来源覆盖可追溯的索引，记录最近2000截断、空证据和来源密度；不要直接把253万谱展开成Python对象。先40例手动计时/内存，再决定完整库。索引预计算在CPU完成，T4只跑实际神经计算。
4. **优先补独立证据，而不是重复已有通道。** 有界尝试MS2DeepScore等合规谱表示，或有质量约束的直接谱库检索；若与现有通道无增量，停止该配置并记录。扩库须配同式难负例和完整ICE/排序；不重复FP-only全开。
5. **只有形成具体可部署增量才准备新提交。** 先集成一个有解释的变化到新版本，保存v9/v10，离线完整推理并检查实际读取test、候选合法/去重、全量25列、成本、padding/回退和变化行。达到可运行状态后才做一次验证性code submission；得分回来后再谈排行榜收益。

对以上步骤，不要求每个小切片都统计显著；但对主要部署域的明显退化必须解释和确认。public不能当高频调参集。奖金目标要求多个互补通道、真实验证和可复现训练材料，单次本地小增量不足以推出金牌。

## 7. 文件地图与重放命令

| 目的 | 仓库相对路径 |
|---|---|
| 015谱清洗/联合匹配 | `src/analog_evidence.py` |
| 候选特征生成 | `scripts/015/build_015_candidate_features.py` |
| 固定五方案训练/OOF/重放 | `scripts/015/run_015_candidate_ranker.py` |
| 事后直接特征归因 | `scripts/015/analyze_015_direct_ablation.py`；`artifacts/runs/015-analog/direct-only-ablation/`，原五方案未改 |
| 当前源码与原始train SHA | `artifacts/runs/015-analog/code-and-data-snapshot.json`；015结果JSON亦保存，明确为运行后的交接快照 |
| 输入与特征 | `artifacts/runs/015-analog/features/{config.json,manifest.json,features.npz,summary.json}` |
| 排除集合、查询与参考证据 | 同目录`frozen_reference_exclusions.json`、`frozen_queries.json`、`reference_evidence.jsonl` |
| 指标与逐分子完整排名 | `artifacts/runs/015-analog/ranker/{metrics.json,oof_full_rankings.json,oof_candidate_scores.npz,row_alignment.npz}` |
| 折、参数、环境、hash | 同目录`folds.json`、`run_manifest.json`、`training_details.json`、`reproduction.json` |
| 人工保存模型重放 | 同目录`manual_replay.json`；`models/`含25个折模型及5个experimental_full模型 |
| 原NP/G1输入 | `artifacts/legacy/f1-bundle/inputs.json`、`artifacts/legacy/g1-validation/inputs.json` |
| 原ICE缓存 | `artifacts/legacy/f1-scores/scores/f1e-4_ice_top100_split.json`、`artifacts/legacy/g1-scores/scores/g1e-4_ice_top100_split.json` |
| 完整参考与规范化身份 | `data/raw/train.parquet`、`artifacts/runs/005/structure_identity_train.parquet` |
| v10输出核查 | `artifacts/runs/011-v10-output/verification.json`及同目录submission/score文件 |
| 013/014、扩库、碎片 | `artifacts/runs/013-support-audit/`、`artifacts/runs/013-r3-fragment/`、`artifacts/012-{chebi,pubchemlite,fragment}/` |

在项目根目录执行，先读现有结果，**不用为接手而重跑全量训练**：

```bash
git status --short
.venv-015/bin/python scripts/015/run_015_candidate_ranker.py --replay
.venv-015/bin/python scripts/015/analyze_015_direct_ablation.py --replay
kaggle competitions submissions -c enveda-CASMI26-molecule-id-mass-spectra --csv
kaggle quota
```

确需重建015时，另选产物目录防覆盖原始记录：

```bash
/opt/homebrew/bin/python3.11 scripts/015/build_015_candidate_features.py --output-dir artifacts/runs/015-analog/rebuild-features
.venv-015/bin/python scripts/015/run_015_candidate_ranker.py --input-dir artifacts/runs/015-analog/rebuild-features --output-dir artifacts/runs/015-analog/rebuild-ranker
```

特征脚本自带900秒研究预算，不是自动化测试门控。随机种子20261017；ranker单线程。输入hash与原产物不同则先解释原因，不能直接覆盖原结果。环境：brew Python3.11.16、numpy2.4.6、RDKit2026.03.6、pyarrow25.0.1；项目`.venv-015`使用scikit-learn1.9.1、scipy1.17.1、joblib1.5.3、threadpoolctl3.6.0、narwhals2.24.0。来自已有离线缓存的重打包wheel和来源说明在`artifacts/runs/015-analog/ranker/environment-wheels/`与`environment-cache-provenance.json`，没有改全局环境。其他机器应重建相同版本环境，不能把macOS arm64 wheel直接带到Kaggle Linux。

`data/raw/`、`artifacts/`、权重和`.venv-015/`均被Git忽略，接手同一机器可直接访问；只克隆Git不具备这些资产。缺失时先恢复私有数据/版本和缓存；015特征manifest保存输入hash。自生成joblib只在兼容环境加载，不能把新的全数据模型的训练内分数写成OOF成绩。

源码与原始train当前SHA另存`code-and-data-snapshot.json`，属于运行后交接记录，不补写为预先冻结。train文件3033286496字节、SHA256`9423f90ec08b941a9646b535bf0606183e86ff0c5a297978f5a70693d41bcd98`。所有核心精简指标、环境及该快照还写入可入Git的`data/references/research-results-015.json`，避免只有ignored产物留下结果。

## 8. 下一次远端运行及奖金材料

本地M5/32GB继续承担索引、小样本与排序器。Colab T4有既往授权，v5e-1偏好保留；是否迁移取决于明确模型和吞吐需求。不要为排序器无理由重新搭TPU，也不推断历史Kaggle排队任务已取消。

`kaggle kernels push`会启动运行，并且只上传入口与metadata，不会同步整个仓库。最终入口须自包含或将代码、权重、依赖作为固定私有输入挂载；互联网关闭。用新版本进行完整运行后，必须按明确已完成notebook版本提交，不能直接传本地CSV。具体CLI参数见[README远程操作](../../README.md)与[011执行记录](../progress/011-massbank-expansion-submission.md)，勿把示例旧版本重新提交。

进入奖项争夺前仍需补齐：现用s2的输入`spec.npz`、six/is_val的生成与划分、训练seed/CLI、输入版本到发布checkpoint的绑定；所有新增模型/库的公开来源、许可和实际本项目训练/推理代码。官方允许特定公开权重不等于必须重跑预训练，也不等于可以省略自己的复现材料。现用原作者代码在`artifacts/runs/010-frontier/original_train_fingerprint_model.py`。

凭据、原始比赛谱和权重不进Git；云端数据保持私有。接手者可以直接推进已授权的例行研究，不需要重复询问。任何新结论都写入配对进展，留下实际命令、版本、设备、时长、指标、输出位置及局限。
