# 006 — 同分子式辨别、能量条件与参考谱缺失：实际进展

更新：2026-10-10（北京时间）。对应[006研究计划](../plan/006-isomer-and-domain-evidence.md)。

**当前最佳有效提交为v9r2，ref `57028399`，public 0.323，已重新查询Kaggle确认COMPLETE。F1已完成并用于提交；F2/F3尚未证明适合部署，F4a两套新head均不采用。非NP timsTOF500已固定manifest，尚无新方案在该集合上的评估结果。**

以下按日期保留研究历史。10月5日仅完成复盘和缓存分析，此后Colab实验、Kaggle提交与F4a训练结果见后续各节，不能将早期“未执行”当作当前状态。

## 实际比赛核实（2026-10-05历史快照）

本轮Kaggle MCP只读恢复可用。核实时间北京时间13:34左右：

- `56821309` COMPLETE，public **0.307**，scriptVersionId `355132734` 对应 `ccjjllt/enveda-submission-v8`。历史列表只有v1/v7/v8三条，没有新的submission；仓库也没有v9 notebook。
- 当前2,479队，公开榜第1/2/5/14条分别为0.471/0.466/0.435/0.428。这些是当前公开成绩，不是最终获奖线。
- 数据和规则读取未发现推翻既有研究约束的新规定。Other/NC权重包、DreaMS等讨论仍未得到主办方新的明确回复；不把参与者追问当官方结论。
- 官方data明确多CE列表表示**不同能量采集后合并的谱**。合并权重/归一化顺序没有公开；不能把`[20,50]`说成连续区间，也不能声称平均35eV等同实际合并。

记录：[competition-update-2026-10-05.json](../../data/references/competition-update-2026-10-05.json)。本轮未使用Computer Use、未读取凭据内容、未下载大模型。

## 手动复算：ICEBERG替换GLACIER尚无充分证据

新增[analyze_006_forward_evidence.py](../../scripts/analyze_006_forward_evidence.py)，只读取005已有候选/forward分数：

```bash
/opt/homebrew/bin/python3.11 scripts/analyze_006_forward_evidence.py
```

本机M5/32GB、Python3.11、NumPy，约0.2秒；seed=20261005，按分子配对bootstrap 10,000次。首次序列化遇到NumPy整数类型，改为Python `int` 后完成。未调用神经模型。

产物：

- [汇总、来源SHA与条件统计](../../data/references/forward-reanalysis-2026-10-05.json)。
- `artifacts/006/forward_per_molecule.json`：250个分子的分层、RR、oracle与候选信息，继续受Git忽略。

四个已有方法（base/GL25/GL50/ICE25）的1,000个逐分子RR与保存报告**零不一致**。

| 方法 | MRR@25 | Top1 | Top25 |
| --- | ---: | ---: | ---: |
| base | 0.514593 | 0.372 | 0.920 |
| GL25 | 0.619828 | 0.500 | 0.920 |
| GL50 | 0.639061 | 0.516 | 0.940 |
| ICE25 | 0.638172 | 0.528 | 0.920 |
| GL/ICE各0.5，top25，缓存探索 | 0.633352 | 0.512 | 0.920 |

- ICE25相对GL25：+0.018344，38改善/31恶化，bootstrap区间[−0.00782,+0.04425]。
- ICE25相对GL50：−0.000888，33改善/43恶化，区间[−0.03042,+0.02791]。**尚不能据此把v9定为更优的ICE替换版。**
- GL50相对GL25：+0.019232，26改善/14恶化，区间[+0.00635,+0.03447]。
- oracle选择GL50/ICE25可得0.690026，仅诊断；固定平均没有兑现该上界。
- 将缺失forward分数作为中性项、对全部候选统一排序，与原scored优先策略在这250个分子上结果相同；没有证明隐藏数据上也一样。

仍是NP开发集、canonical键口径与旧缓存，不是官方2026.03.3逐SMILES复评。NP被反复用于选择方法，区间不修正这种偏差；未执行ICE top50，不能凭已有文件补造该结果。

## 新发现的错误和计算重点

GL50首位错121个，其中115个错首选与真值同分子式、6个不同。111个真值已在top50，另外10个在质量窗中但被FPNet截断。真值公式过滤完整窗后的FPNet MRR仅0.532182（base0.514593），说明在此观察集上，主要问题是同式结构辨别。

base候选真值数：top10=205、top25=230、top50=240、top100=248、top200=250；全部250真值都在质量窗。扩大shortlist值得测，不能拿“窗口有真值”当最终命中。

| 分层 | 分子数 | base MRR | GL50 MRR |
| --- | ---: | ---: | ---: |
| 有forward分数 | 216 | 0.545166 | 0.689225 |
| 无forward分数 | 34 | 0.320370 | 0.320370 |
| 只有负谱 | 18 | 0.490467 | 0.490467 |
| 同时有正负谱 | 130 | 0.531868 | 0.614137 |

649张受当前正模式forward支持的谱中，220张有多CE。按分子内(adduct,5eV bucket)去重计算候选条件任务数：

| shortlist | mean CE | 每个记录CE |
| --- | ---: | ---: |
| 25 | 13,152 | 16,246 |
| 50 | 23,021 | 28,274 |
| 100 | 36,768 | 44,724 |

top50逐CE增加约22.8%的计划任务，而不是按多CE条目数直接数倍放大，因为许多条件会重复。此计数未实际调用模型，不包含跨分子缓存、候选失败等因素，也不等同GPU时长。它为006先做多CE与top100提供了可负担性依据。

## 本轮代码/实现审阅的实质结论

[v8审计](../../data/references/submission-v8-audit-2026-10-05.json)保存了源码位置、哈希和历史临时E2脚本的关键代码片段，没有保存私有谱。

- **v8与NP配置不同**：实际v8为20ppm、raw-key、global top50；NP对照为10ppm、canonical合并。005身份表已经算好，但未完整部署到v8。
- **E2后补模块不等同历史训练**：原临时脚本有分子等权，后补模块没有；若干特征定义也不同。保存OOF数值仍是实际记录，不能因logit复算一致就声称重训已复现。
- **线性上下文无条件路由能力**：8个分子上下文在同query内相同，线性打分的候选差中会抵消。006考虑交互/非线性残差。
- **真值去库干预并非简单实现错位**：找到的原脚本确实把全部canonical阳性别名的原始库特征置零后再标准化；其数学顺序合理。0.1000是固定seen模型的压力观察，不等于重新训练了适合缺库的模型；缺完整逐例干预文件。
- **FRIGID未完成预期条件路径**：formula确实编码，fingerprint=None；长度配置/采样返回过滤也不同。原“4000次生成”不是可核实的实际请求槽数量；从生成输出删掉真值后得到0不构成库外能力评价。暂停该不完整路径合理，否定完整谱条件生成不成立。

本轮没有擅自修改v8或后补ranker的模型行为。新代码仅为上述手动分析入口；生产与大规模实验改动已分解到006计划。

人工检查：250个输入对应250个不同canonical真值；分析脚本语法可解析，4份本轮证据JSON可读取；配对文档本地链接已核对；`git diff --check`无空白错误，逐分子产物仍被Git忽略。没有新增测试文件、测试框架或CI。

## 前沿与下一轮选择

新查的FIORA-OS有明确MSnLib来源参数及负离子支持，提供与当前正forward不同的证据入口。固定OS权重约53.8MB；本轮只读源码、参数、资产元信息和论文，没有下载/加载该checkpoint。34个无现有forward证据的分子中，19个有`[M-H]-`可作小试；HCD训练输入与timsTOF之间的域差要实测，另15个不因此被覆盖。

FIORA论文已有多能量分别预测再合并的实例，支持F1作为有根据的模型实验；不能把它当本赛已证增益。MS-GPT/FRIGID完整条件路径仍可研究，但NP当前主要是同式排序问题，大规模生成不是第一投入方向。最新高分公开源码中的固定ID预测覆盖不作为可泛化方法复刻，也未下载或保存其映射。

具体一手来源、FIORA checkpoint身份、条件限制及前沿判断见[research-frontier-2026-10-05.json](../../data/references/research-frontier-2026-10-05.json)。

006执行顺序：**F0对齐基线与新观察集 → F1多CE/top100/同条件forward比较 → F2缺库场景残差ranker → F3负离子FIORA-OS → F4困难负例训练或正确条件生成小试**。这里没有新的模型效果结论，当前研究成果是发现可验证的改进方向并给出完整交接。

## F0–F3 执行与主审验证（2026-10-06，四子任务并行，主审逐项验货）

**F0（双基线 + 三 manifest）**：`artifacts/006/` 六文件。主审从逐分子 top25 独立重算：规范化 10ppm 0.514593 / v8 原配置 0.504453，与报告分毫不差。新非 NP timsTOF500（研究 250/保留 250 未评估）、参考谱双条件（canonical 组层移除 57,273 行）、pool-removed 遮蔽轨道（移除 275 shipped/250 canon，真值覆盖 0/250）齐备。v8 原 vs 规范化差 +0.010，系窗口 + 去重口径，不归模型名。

**F1（逐 CE + top100）**：因子配额停下（主审亲查 MCP：GPU 已用 98,245/108,000，剩 2.7h < 4h 门槛），push 0 次。停得对——F1a–e 全量约 1–2 T4 小时，硬上会烧穿。待 10-10 刷新后按 F1a→F1e 跑。输入（shortlists/logits/旧 scores）就绪确认。

**F2（残差 ranker）**：`src/molecule_ranker.py` 增补（import 通过）。主审从逐分子 ranks 独立重算三情景：seen 0.9073 / 无库 0.5146（恰回 base，旧 0.1000 崩塌已修）/ 压力 0.2572，与报告分毫不差；缺失曲线单调。verdict 接受：**不进管线**——压力情景（真值无参考 + 干扰有高相似参考，正是 Class2 形态）下把真值压到 base 之下，缺失路径修了崩塌没修误导。下一步是真值独占式不对称 dropout + 校准，不是部署。

**F3（FIORA-OS 负离子）**：权重主审实测验证（53,845,076B + sha `27380712…`，与报告一致）；卡点有日志原文证据（CLI 索要不存在的 `_state.pt`），22 分子零有效预测。verdict 接受：**无法判定，不进管线**——否定的是本次 loader 卡点，不是负离子物理路线。`src/fiora_adapter.py` + notebook 已落盘，解开加载契约后重跑。

配额现状：GPU 剩 2.7h（10-10 刷新），TPU 剩 19.7h。F1/F4 训练类工作全部等刷新；F2 分析类已闭环。

## F3 续：加载契约解开，22 分子实测信号弱，不进管线（2026-10-06，主审验货）

- 卡点根因（源码为准）：`FioraModel.load_from_state_dict(.pt)` 只拿 `.pt` 做前缀派生 `_params.json` + `_state.pt`，本体字节不读；v2 只下了两件，补官方同 commit `_state.pt`（53,814,639B）即解。`ccjjllt/fiora-os-pilot-neg` v4 COMPLETE（156.5s，配额消耗约 5 分钟）。
- 实测（`artifacts/006-fiora/fiora_scores_22.json`，主审独立重算）：746 候选 591 scored（0.792），均值多 <0.06；仅 CZMR 0.211/RNKM 0.250/QIVC 0.128/FBPF 0.114 中等；JDUW/QFJC/UEDU 全零；零峰 155 条集中在 SEEG/QQTZ/RTIX/APAJ。与报告分毫不差。
- verdict 接受：**不进管线**。仅可作带缺失路径的残差特征，须本地先验 `base+residual` RR 增量（未做）。离线捆绑（53MB×2 + 源码）暂缓。

## v9 已构建，未能推送：GPU 周配额见顶（2026-10-06，主审验货）

- `notebooks/submission-v9/` 新建（15 格，13 code 全 AST 通过；无 glacier/20ppm 残留；融合 `_z` 与 v8 逐字一致；复算 AQCH 行与 filed 一致）。kernel id `ccjjllt/enveda-submission-v9`，GPU，离线，挂载比赛 + fp-models-v2 + coconut + train-fp-pool + casmi26-iceberg + cp313-wheelhouse。口径为已验证配置（union/10ppm/canonical/top25 + ICEBERG 融合），未混 v8 的 20ppm/raw-key。
- push 被拒：`Maximum weekly GPU quota of 30.00 hours reached`（远端无版本）。GPU 周 30h 用完，10-10 刷新后再推。CPU 干跑只验链路不作提交，未执行。

## F1 可移植包（2026-10-06，用户自备 Colab 跑，主审验货）

- Kaggle GPU 周配额见顶，F1（纯前向）改走用户 Colab。包 `artifacts/f1-bundle/` 131MB/80 文件 + `f1-bundle.tar.gz`：250 分子 top100 shortlist（与 filed top50 同序已逐分子断言）+ E0 logits + 三权重（大小/sha 全验，零下载运行）+ gl/ice runner 与 shim（cp312 轮子已剔）+ install.sh/run_f1a_e.py + README（Colab 逐条命令、truth 声明）。
- 主审验：目录结构/权重齐备、inputs 250 分子、README 起跑三步清晰；任务量本地实算 top50 split +22.2%/top100 1.92×，与计划一致。包内无凭据（grep 确认）。
- 待用户 Colab 回传 scores/ 后做融合评估（F1a–e 对照 + verdict）。

## F1 矩阵评估（2026-10-09，用户 Colab 跑回，主审验货）

- 8 格全 ok（GL 423–799s，ICE 851–1664s；权重 sha 全验；`n_scored=0` 系 `np.unicode_` 缓存落盘 bug，不影响文件内 fused/sims 有效——F1a 重算 0.63906059 与 filed 差 4.7e-7）。主审从逐分子 rr 独立重算 12 格，与汇总 6 位一致。
- 对照：F1a 0.6391 / F1b(top100) 0.6541（+0.0151，CI 不含 0，Top1 净 +6）/ F1c split +0.005（GL，CI 含 0，噪声地板）/ F1d 最佳 0.6583 / **F1e ICE top100+merge 0.6690** 全局最佳 / ICE-merge 显著优于 sims-mean（+0.011~0.015），sims-mean 对 ICE 是错误融合 / ICE−GL 的 CI 均含 0（方向 5/6 向 ICE）。
- verdict：**top100 进管线；CE 策略选 fused_merge；管线选 ICE top100+merge（0.6690），GL 留回退**。v9 笔记本（ICE top25 口径）需更新到 top100+merge 后再推——待配额刷新 + 用户批准。

## v9 更新到 top100 + fused_merge（2026-10-09，未推，主审验货）

- 改动（仅 v9 notebook，+366/−52）：shortlist top25→top100（validated 口径不动）；forward 改逐记录 CE 展开（单 CE 原路径，多 CE 逐 CE 预测 + 等权合并，缓存键与 f1 同格式，BUDGET 3600）；融合以 fused_merge 为主、sims-mean 为辅；顺带修掉一个必炸的 `gl_out` NameError。
- 主审验：15 格 AST 全过；glacier/20ppm 零残留；引用的 9 个 ice_runner 原语（merge/prep_query/entropy_sim/is_covered/spectrum_ce/ce_bucket/instr_token/prep_spectrum/Predictor）逐个核对真实存在；sims-mean 复算一行精确成立。
- 诚实边界：merge 路只审未算（本地无预测峰），第一次执行就是配额刷新后的 v9 提交运行本身，fail-loud 断言兜底。预计 forward 40–50 分钟（隐藏 400 分子），总量 <9h。

## v9 提交（2026-10-10，PENDING）

- 配额刷新后 push，`ccjjllt/enveda-submission-v9` v1 COMPLETE。主审独立验证：400×25、400 唯一、顺序对 sample、首候选裸 CCO=0（35 行尾部 CCO 补位有计数）；5 blocks 全零异常；merge 路实跑（371 分子、29014 候选 scored）；rdkit 2026.03.6。
- 已提交排行榜，ref `57027339`，PENDING。今日还剩 4 次。

## v9r2 修好重交（2026-10-10，PENDING）

- v9r1（ref 57027339）在隐藏重跑报 unhandled error：5 个 block 格只覆盖 400 分子 + `assert len(out_rows)==len(mids)`，隐藏集分子数不同即炸。修法：25 格覆盖到 2000 分子（空转零开销，日志证实 20 个空 block 均为 rows 0 瞬时过）+ forward 预算 3600→7200。
- v2 COMPLETE，主审验证：400×25、顺序对 sample、首候选裸 CCO=0；merge 实跑 371 分子/29014 候选。已重交，ref `57028399`，PENDING。今日还剩 3 次。

## v9r2 public 0.323（2026-10-10）

- ref `57028399` COMPLETE，public **0.323**（v8 0.307 → +0.016）。对应NP250开发配置为GL50 mean-CE 0.639061 → ICE100 split-merge 0.668997，增量约 **+0.029936**；原记录中的+0.124属于上一轮FP基线→GL50，不能混用。v8→v9同时改变窗口、身份合并、候选数、CE和模型，不能只归因于ICEBERG，也不能据差值诊断隐藏类别或退化原因。r1（57027339）仍无分数（隐藏重跑error，已修）。
- 此前公开榜第五约0.435是历史快照，本次未刷新榜单，不能作当前奖线。F1逐CE/top100已进v9，下一步优先补非NP验证，再研究残差校准与负离子证据。

## F4a 可移植训练包（2026-10-10，用户 Colab 跑，主审验货）

- 包 `artifacts/f4a-bundle/` + tar.gz（8.3MB）：train_f4a.py/f4a_prep.py/f4a_model.py + fp_bits + np250 评测包 + 双臂口径（Arm A 原 head 纯 BCE / Arm B BCE+CE-K63 同式优先，seed 2026，分子等权）；大文件（分片/池/encoder ~1.1GB）走 Colab 内用户自认证下。
- 主审验：tar 内容 12 项齐、目录合规（ccjjllt 下无新建、git 干净）、fp_bits sha `a61e372f…` 一致、README 起跑链完整、凭据 grep 干净。11/11 本地口袋验证通过（含 BCE/CE 手算对照）。
- 待用户 Colab 回传 head.pt + metrics.json 后做对照 verdict（原 head vs BCE+rank，同 seed/同样本）。

## F4a 对照结果：双head不部署，当前配置封存（2026-10-10，用户Colab实跑；口径经本次复核更正）

- 250分子、seed=2026、同训练样本顺序、T4、500步：固定top100内重排，Arm A纯BCE MRR **0.255103**，Arm B BCE+rank **0.478995**；逐rank手算与报告一致，B比A高0.223892。产物 `artifacts/f4a-results/`（双head、curves、metrics）。
- 两者数值均低于历史filed基线0.514593，但 `train_f4a.py` 的 `filed_base` 直接读取历史全窗结果，未在同一个 `eval_pack_metrics` 中重算训练前原head；本次只验证固定 `order_top100` 内重排，不能据此报告新head全窗召回。`recall100=0.992`来自固定shortlist的覆盖上限。
- A的BCE首/末步0.3123→0.1902；B的rank项前/后50步均值0.9455→1.0606，未显示下降。目标或分布不匹配仍是待查假设，尚未确证过拟合根因。原“grad_norm约1e5说明梯度尺度失控”撤回：代码默认AMP，先 `scaler.scale(loss).backward()`，随即记录norm，之后才 `scaler.step()`；日志范数未经反缩放，不能作为真实梯度爆炸证据。
- verdict保持：**两个头都不部署，当前数据/超参配置封存，不继续消耗训练资源**。现有证据不等于否定所有head训练；如果未来重开，先补同评测器训练前基线和正确的梯度诊断，再决定是否训练。

## 研究现状复核与交接（2026-10-10）

- 本次按用户要求梳理现状，读取005/006配对文档、F1/F2/F3/F4a结果文件与v9入口。Kaggle MCP `search_competition_submissions`（比赛ID、按Date、pageSize=10）返回5笔提交：v1 0.000、v7 0.206、v8 0.307、v9r1无分错误、v9r2 0.323；最新ref为57028399，版本356902571。没有发起训练、推理、push或submission。
- 手动核对F1汇总、F2三情景、F4a逐rank与训练/评测代码；纠正本轮开发增量和F4a因果解释，更新README与计划入口，保留按日期展开的历史。
- 当前研究空缺：非NP timsTOF500已有研究250/保留250 manifest，尚未见新方案评测；F2在干扰有库、真值缺库时仍退化；FIORA尚无融合后的MRR增量；完整MIST谱条件生成仍未复现。不能从public总分反推隐藏类别比例。
- 后续优先固定v9配置在非NP研究250上做独立于NP调参过程的验证，保留250供最终比较；再研究参考谱不对称缺失与残差校准。预训练权重暴露仍为unknown，不将新集合自动称为严格未见结构验证。
- 检查均为本地人工读取、标量复算与 `git diff --check`，未新增测试文件、测试框架或CI。未查询当前GPU配额、Colab会话或公开榜位置，旧快照不作为当前状态。
