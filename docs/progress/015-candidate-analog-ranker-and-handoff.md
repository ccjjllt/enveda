# 015 — 候选级analog排序与交接：实际进展

日期：2026-10-17（北京时间）。对应[015计划](../plan/015-candidate-analog-ranker-and-handoff.md)。从`ca1abba`接手，初始工作树干净。用户授权继续研究并要求交接文档。

## 实际完成与结论

已完成500个开发分子的候选级特征、固定五方案canonical五折OOF、25个折模型＋5个全数据实验模型、保存模型人工重放及独立审阅。**直接检索＋analog组合在NP允许同结构参考场景有明确开发增量；目标结构参考移除场景与G1尚未确认改善。** 这比013“有相似谱”的支撑审计多了真正的候选排名结果，但不等于独立新面板或排行榜提高。本轮没有GPU作业、notebook push或新submission。

正式交接见[015交接文档](../handoff/015-candidate-analog-ranker.md)。原v9r2/ref57028399和v10/ref57293821仍COMPLETE/public0.323，本轮CLI重新核实。下一步优先确认直接检索的来源迁移，而不是立即把组合排序器部署或再扫描融合系数。

## 修改与运行

- 新增`src/analog_evidence.py`：精确峰hash、adduct质量、谱清洗、direct＋precursor-shift联合贪心配峰、质量权重、统一Morgan指纹。
- 新增`scripts/015/build_015_candidate_features.py`：冻结查询/排除集合，batch8192读取train，B/A两场景构造候选—参考证据，保存完整追溯记录。
- 新增`scripts/015/run_015_candidate_ranker.py`：固定LR/HGB、canonical同折、query等权、同式槽位重排、分子级bootstrap、模型和输入hash、人工重放入口。
- 本机brew Python3.11缺sklearn/scipy，从已有uv解包缓存离线重打包wheel，装到项目`.venv-015`，没有修改全局环境。缓存写受限后使用项目内缓存目录完成；`.gitignore`添加`.venv-015/`。
- 不创建测试文件、自动化测试、CI或新工程门控。下述检查均由本轮手动发起。

实际命令：

```bash
/opt/homebrew/bin/python3.11 scripts/015/build_015_candidate_features.py --max-queries 5 --output-dir artifacts/runs/015-analog/features-probe
/opt/homebrew/bin/python3.11 scripts/015/build_015_candidate_features.py
.venv-015/bin/python scripts/015/run_015_candidate_ranker.py
.venv-015/bin/python scripts/015/run_015_candidate_ranker.py --replay
kaggle competitions submissions -c enveda-CASMI26-molecule-id-mass-spectra --csv
kaggle quota
```

环境与成本：macOS26.6.2 arm64，本地CPU；Python3.11.16、numpy2.4.6、RDKit2026.03.6、pyarrow25.0.1；ranker sklearn1.9.1、scipy1.17.1、joblib1.5.3、threadpoolctl3.6.0，单线程；固定seed20261017。5query probe约11.35秒/RSS3.60GiB；全量特征179.987秒、RSS4037722112字节（3.760GiB），北京时间22:17:39完成。五方案训练及全部报告15.675秒、CPU15.621秒、RSS299024384字节；25折及5个全量fit均无收敛警告。重放0.80秒。未消耗新GPU时段。

## 数据、参考与方法核对

- NP250＋G1研究250，500个不同canonical；共42580候选行，每场景21290行、10列，不是每分子都恰有50候选。495/500真值在top50，NP缺4/G1缺1；所有负例保留。
- train2539608行，每50行采一条共50793原行；排54未知质量、55精确查询峰hash、520清洗后空谱，B剩50164谱/42276结构；A再排1098条500目标canonical参考，剩49066谱。A全局移除先于每query近邻选择。
- 与013约50681可用参考数不同：015完整v9质量表额外识别58条先前简表不支持的谱，含`[2M+K]+`1、`[2M+Na-2H]-`37、`[2M+NH4]+`7、`[M-H2O]+`13。未知质量未猜值。
- 全部500query的所有谱共有2561个精确hash；B排这些hash但不保证跨来源/采集独立，精确hash不能排除重处理版本。
- Query代表谱与完整v9基线独立重建500/500一致；ddof1、fused_merge、scored优先/缺失续后保持。只在原v9前50同式槽位重排，后50及候选集合保持。
- 每query最多2000参考，实际B221–2000/A213–2000。全部query清洗后非空；5例只剩<10%原始强度，可能存在重新归一化放大，已保留审计字段。
- B只“允许”同结构参考：NP203/250在实际参考池有真值参考，top50真值中185有正`direct_quality`；G1为0/250有真值参考。所以G1 B/A不是两套有效的已见/缺失目标参考对照。一个G1二聚体`OBVPXQQNIYPMEU`的2000条参考全quality0，特征按冻结规则给0。
- `direct_quality`是候选自身canonical参考的**联合谱匹配**质量，不是pure-direct cosine。analog传播显式去除候选自身，先按参考canonical去重。

## 五方案OOF实际指标

MRR@25。B允许同结构参考，A全局移除500目标canonical参考；不是“500个全新未见结构”。每格250分子，不把两个场景副本当500独立样本。

| 方案 | NP B | NP A | G1 B | G1 A |
|---|---:|---:|---:|---:|
| 原v9 | 0.668997 | 0.668997 | 0.907022 | 0.907022 |
| LR baseline-only | 0.669852 | 0.669852 | 0.906673 | 0.906673 |
| LR direct＋analog | **0.725342** | 0.670582 | 0.905994 | 0.905994 |
| HGB baseline-only | 0.662653 | 0.662653 | 0.905685 | 0.905685 |
| HGB direct＋analog | 0.712199 | 0.667571 | 0.905696 | 0.905696 |
| HGB质量打乱 | 0.710364 | 0.662263 | 0.906397 | 0.905658 |

关键配对比较（固定20000次canonical bootstrap，描述性95%CI；不覆盖反复开发或多方案选择的不确定性）：

| 对比 | ΔMRR | 95%CI | Top1 | Top25 |
|---|---:|---|---|---|
| NP B，LR组合 vs v9 | +0.056345 | [+0.038638,+0.075550] | 138→151，得13失0 | 237→241 |
| NP B，LR组合 vs LR基线 | +0.055490 | [+0.037745,+0.074805] | 138→151 | 239→241 |
| NP A，LR组合 vs v9 | +0.001586 | [−0.002541,+0.005746] | 138→138 | 237→241 |
| NP A，LR组合 vs LR基线 | +0.000730 | [−0.003237,+0.004782] | 138→138 | 239→241 |
| G1两场景，LR组合 vs v9 | −0.001028 | [−0.002989,+0.000358] | 214→214 | 246→247 |
| NP B，HGB真实质量 vs 打乱 | +0.001835 | [−0.010015,+0.014087] | 143→145 | 244→244 |
| NP A，HGB真实质量 vs 打乱 | +0.005308 | [−0.001739,+0.013631] | 134→137 | 243→242 |

NP B的LR组合相对v9有52例RR改善、9例变差；NP A为23/20；G1为2/4。不能只看Top25增加而忽略前排RR下降。HGB更复杂但本配置未胜过LR；不能据此否定其他树模型。

负对照没有去掉全部谱信息：真实谱选出的top80身份、maxT、同结构direct质量保持，仅打乱query内quality与身份对应。真实与打乱版相近不能推出“全部收益来自结构密度”；组合vsbaseline也不能单独归因analog。追加直接特征消融单独记录，不追溯改写预定五方案。

## 事后直接谱归因消融

主实验完成后，固定相同折、权重、LR/HGB参数、输入和排序槽位，仅用4基线＋`direct_quality`五列。新增`scripts/015/analyze_015_direct_ablation.py`，产物独立保存在`artifacts/runs/015-analog/direct-only-ablation/`；原五方案与所有原产物未覆盖。没有阈值/超参搜索，没有新增全数据模型。

```bash
.venv-015/bin/python scripts/015/analyze_015_direct_ablation.py
.venv-015/bin/python scripts/015/analyze_015_direct_ablation.py --replay
```

| 对比 | direct-only MRR | 相对v9 Δ/95%CI | 完整组合相对direct-only Δ/95%CI |
|---|---:|---|---|
| NP B，LR | 0.723109 | +0.054113，[+0.036819,+0.072768] | +0.002232，[−0.006480,+0.010874] |
| NP A，LR | 0.668829 | −0.000168，[−0.002323,+0.001907] | +0.001753，[−0.001923,+0.005553] |
| G1 B/A，LR | 0.906006 | −0.001015，[−0.002682,0] | −0.000013，[−0.000987,+0.000923] |
| NP B，HGB | 0.703632 | +0.034635，[+0.021595,+0.048689] | +0.008567，[−0.001156,+0.019419] |
| NP A，HGB | 0.659691 | −0.009305，[−0.018908,−0.000590] | +0.007880，[−0.000378,+0.016770] |

LR direct-only在NP B Top1同为151，主要已观察增量可由同结构参考谱重现；追加analog尚未确认独立增量。HGB direct-only在参考缺失NP A反而下降，不能全局部署此配置。这里的“直接”仍指结构身份相同的联合谱匹配，不表示pure-direct峰相似。

10折2.82秒、零warnings；saved-model重放0.482秒，逐候选误差0，1000条完整记录及所有指标一致。独立按概率与公式槽位重建2000个排名，错误0/MRR差0；追加15项输入/模型SHA核对一致。结合原实验，独立复查共58项SHA、7000个方案排名。没有将事后消融包装成预注册确认。

## 人工验证与独立审阅

- `features/manual_verification.json`检查1000个query×scenario、78805条参考记录：重复canonical、跨极性、queryhash、A目标结构引用、candidate-self analog赢家均为0；质量公式复算误差0；X/XS有限且4基线＋direct不变、maxT不变、打乱只影响4个权重相关特征。
- `ranker/manual_replay.json`：5方案全部OOF逐候选最大误差0；1000条完整排名与所有指标/CI逐项相同；输入和模型hash不变。
- 不导入015计算代码的独立核对：原bundle＋ICE重建500个完整v9序列与代表谱全部相同；从train原峰独立枚举NP/G1各1query的A/B首参考，共4条，cosine/quality误差0、配峰数一致。
- 另一组固定SHA选8个query/场景/参考，重新batch读取train原峰、独立广播配对矩阵＋贪心重算，cosine/quality差0、匹配数和来源一致。
- 独立重算5000个输出排序、rank/RR及20组MRR全部相同；43项输入/输出/模型SHA一致；25个折模型canonical无训练/验证交叠。抽样模型前向误差0，LR训练折加权均值误差<4.8e-9。
- 独立审阅素材在`artifacts/runs/015-handoff/independent-audit.json`与`independent-review.md`。这些是本轮手动研究核对，不是新建自动化测试体系。
- 交接收尾补记当前train.parquet及4份源码SHA到`artifacts/runs/015-analog/code-and-data-snapshot.json`，并进入可入Git的精简记录。train为3033286496字节，SHA256`9423f90ec08b941a9646b535bf0606183e86ff0c5a297978f5a70693d41bcd98`。这是22:25的事后快照，不冒充事前冻结证明。
- 4份源码用brew Python3.11内存AST解析通过，JSON与文档本地链接核对通过，`git diff --check`无报错。系统`python3`指向3.9且py_compile写系统缓存受限，改用上述不写缓存的检查；实际研究一直使用指定3.11环境，不受影响。

## 远端状态与遗留限制

10月17日晚只读CLI：v10/ref57293821和v9r2/ref57028399均COMPLETE/public0.323；GPU2.97h已用/27.03h剩余，TPU20h未用，API返回刷新时间2026-10-24T00:00:00。没有新push、submit、最终选择或公开发布。当前会话未暴露Kaggle MCP，使用CLI不代表旧MCP配置已损坏。

现有NP/G1反复用于开发；精确谱hash排除不等于来源独立。A干预使用真值集合只为参考移除评价，不是可部署oracle开关，不证明encoder预训练未见。±1.5Da去峰、最近2000/stride50/top80都有覆盖与质量限制。全数据模型仅实验资产，尚无Kaggle离线集成、端到端运行或新分数。

下一轮首先冻结未用于本项目决策的来源/采集确认面板，验证直接谱检索，再决定是否扩参考索引及制作下一提交。参考缺失能力仍需独立通道；不把本轮NP seen增量外推到所有隐藏分子。详细路线、来源及准确重放入口写在交接文档。
