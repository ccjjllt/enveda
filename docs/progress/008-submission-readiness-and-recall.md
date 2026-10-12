# 008 — 新研究与提交时机审查：实际结果

日期：2026-10-12（北京时间）。对应[下一步计划](../plan/008-submission-readiness-and-recall.md)。本轮只读查询Kaggle、审代码/日志，并手动复算已有产物；未训练、未GPU推理、未push或submission。

## 决定

**目前建议继续一轮有明确目标的研究，不提交当前产物。** 最新有效提交仍为v9r2，ref57028399、版本356902571、public **0.323**；10月12日Kaggle MCP返回的提交列表没有新有效版本。[只读查询快照](../../data/references/submission-decision-2026-10-12.json)。

实质进展是：D落实了先前指定的5项修正并保存OOF；外部观察集从模板变成50分子/59谱；FRIGID/MIST权重加载阶段已实跑。这三项分别属于实现、数据和加载成果，尚未形成适合提交的新方法。

## D：结果可复现，但训练与推理还有错配

新增手动入口 [analyze_008_r1d_evidence.py](../../scripts/analyze_008_r1d_evidence.py)，从候选特征、已保存模型与OOF结果重建。16组指标、改善/恶化及20,000次分子bootstrap与汇总一致；α=0和全无库时完整排序均为250/250一致。10个模型的训练折四类增强均值/标准差可精确复算，未发现标准化读入验证折的证据。

| 集合/场景 | A MRR | D α=0.25 | D α=0.50 |
| --- | ---: | ---: | ---: |
| NP 有参考 | 0.668997 | 0.729408 | 0.755970 |
| NP 真值参考移除 | 0.668997 | 0.641915 | 0.611737 |
| NP 移除riken来源 | 0.668997 | 0.726075 | 0.753653 |
| G1 原场景 | 0.907022 | 0.912117 | 0.915784 |
| G1 移除enveda-180来源 | 0.907022 | 0.907022 | 0.907022 |

NP压力损失分别−0.027082/−0.057260，11/23个原Top1被改错；当前D不部署的决定成立。来源表这次确实重新构造且不含目标来源，不过此移除情景已参与训练增强，不能称未见来源泛化。G1真值仍全部无参考，其pressure与seen相同。仍为两域分别训练模型。

**剩余评分函数错配：** `run_D.py:193–205`训练所有候选的未门控标准化残差，`:97–100`推理才在`has_lib=0`时强制清零。NP10305、G14252条原场景无库候选的原始残差都非零；G1无库真值为+0.0977～+0.2154，却在部署式评分时被抹掉。缺失增强训练的评分函数与实际评价不相同，不能据本次失败否定一致门控的残差方法。

此外`:117`始终按v9为无forward候选排序，忽略库残差；NP34/G1 21个无forward分子因定义而永远不变，其中NP有33个真值实际上有参考。α=0复现基线，不要求α>0仍锁死这部分排序。下一步只给对齐评分函数和可学习候选对的D2一次固定对照，不继续扫描α。

## R3扩池：0→0.64是oracle插入，不是候选发现成功

新增手动入口 [analyze_008_pool_pilot.py](../../scripts/analyze_008_pool_pilot.py)，只读已存JSON，不运行FP。

- `r3_pool_pilot.py:143–169`读取5个B评价分子的已知SMILES/公式，核对其canonical等于truth，然后仅加这5个结构。没有无标签的外部候选搜索/枚举/生成。
- 5个B排名5/2/2/1/1、MRR **0.64**可复算，含义是“正确结构已进入池时的条件排序”。不代表推理时能发现这些结构。
- 插入后的canonical窗为22/3/14/29/12，4个窗≤25。因此4个Top25在模型打分前已保证命中，原“找回至少2个就CONTINUE”没有有效区分能力。
- 每个新增结构只进入自己的B窗，没有进入其余9个query。A质量约760–1021，B约142–238；“A不退化、零新增干扰”不能证明实际扩库安全。
- 原产物写着下一步“仍只加约束内真值”，该方向需要撤回。可以保留oracle诊断，但不能扩大它并作为提交依据。下一步应固定外部结构库并召回真实干扰候选。

50分子数据确已落地，按定义分成25A/25B；C覆盖A全有/B全无来自选样规则，不是新召回方法的成绩。`coverage_50.json`按质量误差/pool index截100，本批最大窗口84，等同整个质量窗；不是FPNet top100，更不是ICE融合后的最终Top25。

首批10例MRR0.900也只有FP原head＋检索诊断，没有完整v9融合；两谱分子采用logits平均而非v9代表谱，6/11谱CE回退25eV。这些限制必须带入方法对比。50集中5张谱的adduct未知/未识别，把precursor当QM的窗口不能当准确中性质量窗；应先修正或单列。272条discards是历史筛选记录，其中35条后来已选入，不应称当前最终丢弃272条。

已记录逐条来源、许可字符串、峰SHA与unknown暴露信息是实际产物；本轮未重新作许可法律判断，也未从发布时间推断保证未见结构。本地B仅表示当前712K候选库外。

## MIST：SAFE调用错误，完整谱条件仍未运行

读取仓库脚本和用户下载的实跑notebook日志，确认DLM Sampler已初始化；MIST两套权重均记录missing=0/unexpected=0；脚本约38.2秒停于SAFE检查。没有MIST encoder forward、de novo或公式生成结果。

`run_mist_single.py:444`使用`SAFEConverter(slicer=None).encoder(truth_smiles)`。[SAFE 0.2.1官方源码](https://raw.githubusercontent.com/datamol-io/safe/0.2.1/safe/converter.py)明确`slicer=None`产生空切分，而`allow_empty`默认False，因此报错不支持“B04太小”的解释。保持同一调用而换大分子也不能解决。官方FRIGID的无切分编码使用`allow_empty=True`，条件生成不要求先编码真值SMILES。[同pin sampler](https://raw.githubusercontent.com/coleygroup/FRIGID/d242a2c06d671e0fb6fbc1a5b31f0e19e24b456d/src/dlm/sampler.py)

`run_arm:255`仍硬编码`fingerprint=None`，峰数组没有进入encoder。因此即使越过SAFE并显示GEN_OK，也仅验证公式条件生成，不能称完整MIST/FRIGID谱条件成功。首批5个B全为负谱，新增50例中已有支持正模式的候选；选样应按模型输入契约，不能为躲SAFE错误按分子大小换样。

其他待修口径：日志缺`data/len.pk`而回退随机长度；按过滤后返回数循环凑B不能反映实际请求槽位；脚本以“检索miss”为由令union MRR等于生成MRR，没有真实联合排序依据。均尚未产生成绩，不能补造效果。

## 下一次提交的具体时机

最快路径是用现有缓存完成一次训练/推理一致的D2；并行把R3变成无真值输入的候选召回实验。新方案与完整v9在同口径下有可解释收益、重要已知退化得到处理、并完成离线notebook后，就可做一次验证性submission，不必等待全部研究完成。public提升无法保证；保留v9r2。

目前不提交D、不提交oracle插入池、不以SAFE_BLOCKED改生成管线。具体有限实验和文件职责见[008计划](../plan/008-submission-readiness-and-recall.md)。

## 产物与人工核对

本地M5、Python3.11/NumPy，手动执行两个新增脚本；D审计seed=20261004、bootstrap20,000次，沿用已存模型而不重新拟合。输出`artifacts/008/r1d-audit.json`、`pool-pilot-audit.json`、`mist-single-audit.json`及日志，原始实验产物不覆盖。MIST审查含官方源码URL、输入SHA和已执行阶段。

```bash
/opt/homebrew/bin/python3.11 scripts/analyze_008_r1d_evidence.py
/opt/homebrew/bin/python3.11 scripts/analyze_008_pool_pilot.py
```

精简记录见[审查证据汇总](../../data/references/research-review-2026-10-12.json)。脚本语法、输出JSON、配对文档链接和`git diff --check`均已手动核对。实跑notebook另确认SAFE版本为0.2.1，与本次核对源码一致。

本轮未查询配额、未启动远端作业、未使用或评价保留250谱，也未引入测试文件/测试框架/CI。README与007的最新解释已同步更正。

## 008-B 真实候选召回：实际结果（2026-10-12，CPU-only 手动执行）

替代 oracle 插入的无标签召回已按 008-B 六条落地。新增 `src/candidate_expansion.py`、
`scripts/build_008_candidate_index.py`、`scripts/evaluate_008_expansion.py`；
输出新增目录 `artifacts/008-expansion/`（不覆盖 007-r3）。未写测试文件，未用 GPU，
未 push/submit，未读保留集；索引离线可用，不联网查询。

索引来源清单：MassBank-data 2026.03 bulk 快照
`/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/massbank-2026.03.zip`
（139240 txt，bytes 239602156，SHA256 `496fd822e22bc4104726cd1199dad41b580b695d91658899e1f01afb5d6a00f9`，
top 前缀 `MassBank-MassBank-data-85e5599`，release CC BY 4.0、per-record LICENSE 为准，
DOI https://doi.org/10.5281/zenodo.19073053）。
许可筛选仅收 CC0/CC BY/CC BY-SA/MIT；bulk 许可分布 CC BY-SA 21086、CC BY 48404、
CC0 7031，排除 dl-de/by-2-0 20658、CC BY-NC-SA 33478、CC BY-NC 8434、CC BY-NC-ND 149
（62719 未收，许可不明/禁再分发不收）。
canonical 为 `canonical_group_key`（RDKit 2026.03.6，去立体→互变异构→InChIKey14）；
质量为 ExactMolWt float64；指纹口径 v32 10407→`fp_bits.npy` 选位 6930
（SHA `a61e372fc41091bf5c7467e08134081266262665077cb9e2c56d73e4887f3013`，
全量指纹本次按窗按需算，同口径）。
去重后外部 8831 canonical（12885 distinct SMILES→8831，1 不可解析），
与 base 712K union 706242 交叠 8292，新增 539；全库 base 712199→expanded 712738。
评价 50 谱同源边界：评价谱亦来自 MassBank，覆盖偏乐观，已记录不修正。

查询规则：`query_candidates_unlabeled` 只收谱峰/前体/adduct/仪器/CE，
`predicted_formula` 仅允许谱预测输入（本次 None，不过滤）；
真值公式仅 `oracle_diagnose_truth_in_list` 诊断分支。
未知 adduct 返回 `(None,False,True)`，不把 precursor 当准确中性质量；
扩展表在 15 类主表上加 `[M-H2O-H]-`（delta -19.01784115344，仅因记录自带
PRECURSOR_TYPE 支持，不猜无标注谱）。
质量窗 10ppm float64 全窗枚举（按误差升序+下标升序确定性排序，不预截断）；
FP top100 为预先固定 WMAX=100 按 FP 分截断（窗<100 全留），最终 top25 按 FP 分；
R3 mass-ranked top100 不当 FP shortlist，本次 FP top100 均为 FPNet orig-head
（`fp_single_s2.pt`，SHA `c19aa29016ed4d7a67bac38105b8fa5076c696c29eb66a504558c9b2e52b614a`）
CPU 前向 + `prep_peaks` v32 + shipped→canon max + order(-score,canon) + RR@25，
多谱 logits 均值、CE 缺失回退 25eV（10 集 6/11，50 集 46/59），与 v9/检索评分同口径。

10 例（先行核对流程）：全库 5/10→10/10；质量窗 5→10；FP top100（FP 计分）5→10；
最终 top25 MRR_all 0.45→0.77。分层 A MRR 0.90→0.90（5/5 top25，rank [2,1,1,1,1] 不变，
与 `retrieval_scoring_10.json` base 一致，确认为同口径）；
B MRR 0→0.64（5/5 top25，rank [5,2,2,1,1]，top1 2/5），与 oracle 试点同值但本次为
真实召回（含真实干扰，B 窗均值 15.0→16.0，+1/窗）；挤下损失 0，rank 恶化 0。

50 例（开发结果）：全库 25/50→50/50；质量窗 25→46（4 不确定单列除外）；
FP top100 25→46；最终 top25 MRR_all 0.356→0.682，MRR_known-only 0.387→0.741。
分层 A 0.712→0.712（25/25 top25 全留，rank 全同，lost 0/worsened 0，
窗均值 12.6→12.6）；B_all 0→0.651（20/25 top25，14/25 top1），
B_known-only 0→0.775（20/21 top25，14/21 top1）。
B 新增 top100 21、top25 20、top1 14；窗均值 23.4→24.24（+0.84 真实干扰/窗）。
YTCZZXIRLARSET 窗 84→85 但 FP rank 40（>25，top100 命中、top25 miss），
GVQNVXICGHLXAI rank 14，为 FP 排序局限实例；其余 19 B_known 均 top25。

5 张问题谱处置：4 NILU（NL0227/NL0322/NL0370/NL0230，adduct None）单列质量不确定，
QM None、窗 0、FP 跳过，全库 oracle 仍 False→True（结构在库但质量不可用）；
UTDAG（positive prec 433.1217≈M+1.007）与 JOECR（prec- mass 18.04）等无标注不猜 adduct。
1 Bruker（MC21_000277 `[M-H2O-H]-` prec 289.0718）经扩展表纠正 QM 308.08964，
窗 65→66、FP rank 1（top25 命中）。原 `coverage_50.json` 对 5 张的 precursor 回退窗
（24/23/17/33/22）不作准确中性质量窗，已废止口径。

与 v9 同口径对比结论：base 重算 A MRR 0.90（10）/0.712（50A）与 R3 FP 原 head 一致，
确认同口径；expanded 在 A 零挤下（lost 0、MRR 不变）前提下，B_known 新增 5/5（10）与
20/21（50）top25，属真实干扰下的无标签召回，非 oracle 插入。
同源乐观、YTCZZ rank40 FP 局限、4 不确定未覆盖须带入后续；本次不跑 ICE、不花 GPU，
首轮额外 T4 预算未动用。产物 `candidate_index_*`、`sources_manifest.json`、
`expansion_eval_10/50.json` 均在 `artifacts/008-expansion/`；`git diff --check` 已核对。

## 008-A D2固定对照：实际结果（2026-10-12，主审复核，判STOP）

单配置 α=0.25、τ=1.0，不扫描。新增 `src/evidence_residual.py`（门控残差+标准化+分组排序同一函数）
与 `scripts/run_008_d2.py`（只读冻结 `artifacts/007-r1d/` 特征/A_full/folds，复用 forward 缓存，
不重建库、不覆盖D、不重训FPNet）。NP/G1分域独立5折，4副本同折，μ/sd只用训练折。
本地CPU全量墙钟143.0s（NP69.5s/G173.4s），seed全冻结20261004，bootstrap 20,000/分子。
零GPU（只用标准库+numpy，不导入torch）、未push/submit、未读保留集。产物 `artifacts/008-d2/`
（models/oof/metrics/run_meta，不动007-r1d）。

| 域/场景 | A MRR | D2 MRR | delta[95%CI] | 改善/恶化 | Top1 w2r/r2w | D冻结α0.25参考 |
| --- | ---: | ---: | --- | --- | --- | ---: |
| NP seen | 0.668997 | 0.722278 | +0.053281[+0.0342,+0.0740] | 55/3 | 17/1 | 0.729408 |
| NP 全无库 | 0.668997 | 0.668997 | 0.0 | 0/0 | 0/0 | 0.669 |
| NP 真值缺库 | 0.668997 | 0.656339 | −0.012658[−0.0232,−0.0039] | 1/14 | 0/5 | 0.641915 |
| NP 来源移除riken | 0.668997 | 0.716605 | +0.047608[+0.0293,+0.0673] | 53/3 | 14/1 | 0.726075 |
| G1 原场景 | 0.907022 | 0.916468 | +0.009447[+0.0027,+0.0181] | 7/0 | 4/0 | 0.912117 |
| G1 全无库 | 0.907022 | 0.907022 | 0.0 | 0/0 | 0/0 | 0.907 |
| G1 真值缺库* | 0.907022 | 0.916468 | +0.009447同seen | 7/0 | 4/0 | 0.912117 |
| G1 来源移除enveda180 | 0.907022 | 0.909022 | +0.002[0,+0.006] | 1/0 | 1/0 | 0.907022 |

*G1真值seen已全无库（0/250有库），pressure==seen，非独立鲁棒证据，与D结论一致。

主审独立复算：两域8场景MRR与oof逐分子rr均值分毫不差；α=0全序NP/G1各250/250一致；
全无库250/250等于A；`has_lib==0`残差全零（抽查0违反）；10折训练/推理一致性max_abs_diff=0.0；
clip在α=0.25下两域0触发（NP 0/14874×4，G1 0/20809×4），只记录不称保护。
训练对过滤（同组且至少一端有库）每折剔除跨组NP 0–4/G1 0、两端无库NP约12–13K/G1约30–31K；
零可学习副本落空NP每折14–20/G1 3–5，零可学习分子0/折，据实记录。

判定STOP：门控对齐后NP压力损失由D的−0.027/23恶化/11r2w收窄至−0.0127/14恶化/5r2w
（5个r2w为HYBRYAPKQCZIAE/QGXBDMJGAMFCBF/RODXRVNMMDRFIK/XFZJEEAOWLFHDH/IVOMOUWHDPKRLL），
但同类损失仍明显且CI全负；且D2新增seen恶化3（含1个r2w IVOMOUWHDPKRLL 1.0→0.5，
另两例CGTADGCBEXYWNE 0.5→0.333、CRSFLLTWRCYNNX 0.143→0.125），D seen恶化0。
按008-A停损规则停止当前线性门控配置，不扫α、不动保留250，资源转向新证据。
G1来源唯一改善PYBOHFRFXPUXAC 2→1系uncovered组内干扰负残差让位（真值门控0），
非真值库增益，不作跨域泛化证据。

## 008-C 生成接线：实际结果（2026-10-12，CPU可查部分，主审复核）

零GPU，未动 `artifacts/mist-single/` 原始包。修正副本
`artifacts/008-mist-fix/run_mist_single_fixed.py`（563行原包的修正版）+
`INTERFACE_DIFF.md` + `positive_candidates.json` + `check_fingerprint_interface.py`。
`py_compile`（原包/修正版/5个新py）与JSON校验均通过；校验脚本四节手动跑通。
未下载大权重、未push/submit、未读保留集；未写测试文件/CI。

SAFE：原 `:444` 为 `SAFEConverter(slicer=None).encoder(...)`，对照SAFE 0.2.1官方
`converter.py`（slicer=None→空切分，allow_empty默认False→必抛错）与FRIGID pin
`sampler.py`（无切分一律allow_empty=True），修正为 `encoder(..., allow_empty=True)`，
失败只记诊断不设SAFE_BLOCKED、不raise；真值roundtrip与生成评价分离。
保持checkpoint `use_bracket_safe=False`，加representation_note禁改表示绕过。
换大分子不能解决同一调用错误；B04保留作诊断不替换。

指纹：原 `run_arm:255` 硬编码 `fingerprint=None`，新增 `validate_fingerprint()`
（None=显式formula-only；非None须1-D+4096+finite+SHA否则抛错）与
`run_arm(fingerprint/fingerprint_bits/...)` 透传、`fp_check` 记录。
CPU四态验证通过（None/4096-good shape[4096] range[0.0001,0.9997] SHA12示例/
2048错维/2-D/非有限均抛错）；5条预选峰SHA/keys/峰数通过。
MIST `SpectraEncoderGrowing` 前向与Spec-featurizer适配明确未接通（待T4），
当前fp_status=formula-only，不补造向量。`data/len.pk` 本地无命中，
`record_length_config` 把fallback-range(20,100)显式落盘。
B计数改为请求槽位/返回分别计数（假sampler请求10实回8验证分离）；
oracle与预测公式分开；单分子包无检索C，union记None/NOT_COMPUTED，
检索miss不得令union==generation；skipped行union亦为None非0分。

预选正谱（按ion contract非尺寸）：primary BEGARHAEPVBWAE/ED128204
（A类inC、[M+H]+、Orbitrap、35峰、C29H45N5O7/575Da、单组分）；
backup1 MLBROOJXUZKDHY/EC140207（39峰，S3待GPU确认）；
backup2 NYGVGDLVEMUBNK/ED124507（25峰，同分子双CE对照）；
minimal-alt MKNWXPPWASGIJB/EC150407（449Da最小但仅10峰，边际对照）；
larger-only LWFZKXMCJDHBNJ/ED103106（694Da最大，仅对照不首选）。
7条mFam B类正谱因Na盐双组分、5条unknown因质量不确定暂不作接通首选。
A类正谱均为inC，只验wiring不度量库外；B=10真实调用（oracle/预测各B=10、
subformula→MIST4096前向、生成RR与联合排序）仍待T4，见INTERFACE_DIFF §4，
本轮不点提交。

## 本轮提交判断（主审复核 2026-10-12，不提交）

D2判STOP（压力CI全负+新增seen r2w），不部署。008-B为FP orig-head口径的真实
无标签召回（A零挤下、B_known 10例5/5与50例20/21 top25），但仍是FP-only：
无ICE融合、无完整v9融合、无离线notebook；评价与索引同源（MassBank）偏乐观；
4不确定未覆盖；YTCZZ rank40为FP排序局限实例。尚不满足008计划v10的
“单主要变化+同口径可解释正向+已知退化已处理+离线可跑”四条，不做验证性submission。
008-C仅完成CPU接线，不进管线。保留v9r2（public 0.323）。
下一步：B补ICE/完整v9融合与离线运行包后再议v10；D2线封存；C按INTERFACE_DIFF §4
待T4单分子B=10。本轮零GPU配额消耗（D2 143s CPU、B约102s CPU、C纯校验），
首轮2h T4预算未动用；无测试文件/CI；检查均手动发起。
