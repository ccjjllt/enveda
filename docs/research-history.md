# 研究历史与结论索引

整理日期：2026-10-22（北京时间）。本页保留原 README 的研究记录；各节分数、名次、额度、执行顺序均属于标注日期的历史快照。**当前研究入口为 [030计划](plan/030-n1-challenger-and-score-strategy.md)与[030进展](progress/030-n1-challenger-and-score-strategy.md)**，实际最佳 public 为0.456，030尚未运行或提交。

原 README 最初提供的首个链接显示文字为 ARC Prize，但实际链接属于 Enveda；本项目只研究 Enveda。逐轮完整记录见[计划与进展配对索引](README.md)。

阅读旧结论时继承030的边界：省略ICEBERG/GLACIER的CPU面板不能替代完整融合验证；不能从单个public总分推断隐藏类别比例；full1与fp0表现差异不能单独证明训练成员身份。029a保留为历史准备版本，完整N1路径以030为准。

## 023历史记录（2026-10-21）

**v13 = v11＋关闭 PubChem 提升：public 0.432（v11 0.446，Δ −0.014），不保留。** 本地遮蔽一致看好关闭，隐藏集相反，本地 NP250 面板不能评估 PubChem 相关改动。 本地新增 R 遮蔽（真值无参考但在池，最接近隐藏集，R 基线 0.4547 与 public 0.446 接近）：关闭 PubChem 提升 0.4547→0.5068（+0.052，44/1）。Engine1 研究：is_gen 确认 C 命中全部来自生成；扩大生成（12/120/400）在 C 上 Engine1-only +0.067（29/3）但在 B 上 −0.022（4/23），不上线；保护 Engine1 生成第一名在 B/R 基本不生效、C +0.03～+0.16。Engine1 使用的 `fpnet_full1` 可能训练时见过 NP250，Engine1-only 与 C 收益偏乐观，v14 暂不提交，下一步换留出 NP 的 `fpnet_0` 复核。见[023计划](plan/023-promo-off-and-engine1-generation.md)与[023进展](progress/023-promo-off-and-engine1-generation.md)。

## 022研究记录（2026-10-20）

**v12 public 0.434，低于 v11 的 0.446（Δ −0.012），严格跨库直接参考规则不保留；当前最佳仍为 v11 0.446。** v12 = v11 原样＋末尾一个 cell：候选有 q≥0.8 的跨来源参考谱匹配、且当前第一名没有强参考（差≥0.2）时提到第一名。本地 B 遮蔽 +0.0140（7/0）、C 遮蔽 −0.0046（0/5），public 与 C 同向，说明隐藏集以无跨来源参考的分子为主。kernel v5，提交 ref 57436327。

下一步：加强 Engine1 生成与库外召回（以 C 遮蔽 top25 召回为主指标）；用已修复的排序损失做困难负例训练；关闭 PubChem 提升作为 v13 候选，须先确认库外不伤害；领奖前核实上游数据集许可与来源。接手先读[022交接](handoff/022-e006-baseline-and-next-step.md)，再看[022计划](plan/022-v12.md)与[022进展](progress/022-v12.md)。

## 021研究记录（2026-10-20）

**本地 NP250 验证器已跑通（E006 去模拟器，CPU 约 80 分钟/次）：B 遮蔽 0.9569，C 遮蔽 0.1168（Δ −0.840，2/244）。** C 的 89 个 top25 命中全部来自 Engine1 的生成路径（排除法：PubChem 通道与 Engine2 均 0/89，真值已从各池移除）。MassBank538 扩池 Δ +0.0001（1/0），不并入；门参数扫描中 REL_TH、POP_TAU、窗口零效应，PubChem 提升阈值单调（关闭 +0.0045，3/0，仅 B）。ICEBERG/GLACIER 未计入。见[021计划](plan/021-local-validation.md)与[021进展](progress/021-local-validation.md)。

## 020研究记录（2026-10-19）

**原样 fork 公开 E006＋PubChem Promo（rel>150）为私有 kernel，v11 public 0.446（ref 57405363），与作者页面一致，比 v9r2/v10 的 0.323 高 0.123。** 代码未改；失败均来自环境（源 metadata 锁定 CPU 镜像、默认新镜像为 Python 3.13），最终锁定 Kaggle GPU 镜像 v170（Python 3.12）的 kernel v4 完成。见[020计划](plan/020-e006-reproduction.md)与[020进展](progress/020-e006-reproduction.md)。

## 019研究记录（2026-10-18）

**38 个同式错例的峰×候选矩阵已算完：少数候选支持的峰组合没有可写入排序器的方向。** 37 对可计算，23 对保留谱里没有任何一边独占的峰。少数支持偏向真值 5 对、偏向错候选 6 对、26 对相同，中位差为 0。去掉同位素和氢系列后，没有组合峰指向真值，因此没有新增排序特征。15 对骨架相同且无法区分，逐 CE 预测仍未跑：28 个 query 各一张谱，21 张是合并的 20/40/60。

接手先读[019研究计划](plan/019-specific-fragment-evidence.md)与[019实际观察](progress/019-specific-fragment-evidence.md)，精简记录见[019结果](../data/references/research-results-019.json)。018的关键解释已补充：重训的原始quality斜率实际下降61.6%；margin没有提供新的碎片信息。优先特异峰是研究选择，多CE仍保留为有具体需要时的小规模后续方向。

## 018研究记录（2026-10-18）

**同一批500分子上，只把参考库从约5万换成251万，就复现了缺参考退化；线性重训保不住跨来源增益。暂不提交。** 索引上的 stride-50 子集把015已存 `direct_quality` 复现到最大误差 0.00025。旧 direct 模型在全库上：NP有参考 MRR 0.724→0.786，NP去掉目标参考 0.670→0.646，G1 0.907→0.866。最近2000条里约98%被换成新谱，但真值谱几乎没被挤出；变的是错误候选的谱证据。用全库特征重训后，400面板外部跨来源增益从 +0.015 缩到 +0.001，自然缺库损失从 −0.031 缩到 −0.005，区间仍不含0。同式 quality 差只在真值本身领先时加分。下一步看特异碎片，不先加多CE，也不扫 quality 阈值。v9r2/v10 仍 public 0.323。本轮无 GPU、push 或提交。

接手先读[018进展](progress/018-same-molecule-reference.md)和[018计划](plan/018-same-molecule-reference.md)，精简记录见[018结果](../data/references/research-results-018.json)。

## 017复核记录（2026-10-18）

**016已复核并完成纠错重算，当前暂不新增submission。** 原016将SMILES键的ICE回包按canonical读取，全部ICE特征误丢失，并把FP-only排序标成v9。现已保留旧产物、复用21869个有效ICE分数，修正真实v9前50并CPU重算两套400分子特征。跨来源外部B，direct MRR0.936458→0.951625，Top1 180→184；自然缺库A却0.765440→0.734150，不能全域部署。主要问题是全库下同式错误候选也获得高谱质量，下一步优先对齐训练/实际参考分布并校准证据可靠性，而非直接扫阈值。v9r2/v10仍public0.323；本轮无新GPU、训练、push或提交。

接手先读[017交接与下一步](handoff/017-corrected-panel-and-next-step.md)，实际结果见[017进展](progress/017-review-and-corrected-panel.md)、[017计划](plan/017-review-and-corrected-panel.md)、[017精简记录](../data/references/research-results-017.json)。016原始工作见[016进展](progress/016-new-panel-direct-verification.md)，其中旧排序结论已显著标注失效范围。

## 015研究记录（2026-10-17）

**已实际跑完候选级排序研究，下一步优先确认直接谱检索的来源迁移；尚未提交新版本。** 500个开发分子、42580行特征、固定五折OOF及事后direct-only消融已完成。NP允许同结构参考时，原v9 MRR0.668997→LR组合0.725342，Top1从138到151；只加直接参考质量也达到0.723109，追加analog增量尚未确认。移除目标结构参考后与G1均没有确认收益，不能外推隐藏集提分。保存模型人工重放误差0，主实验与消融结果分别保留。v9r2/v10仍public0.323。

**接手先读[015完整交接](handoff/015-candidate-analog-ranker.md)**，含下一轮具体步骤、输入/模型/命令/算力与奖金复现缺口。实际指标与边界见[015进展](progress/015-candidate-analog-ranker-and-handoff.md)，冻结契约见[015计划](plan/015-candidate-analog-ranker-and-handoff.md)，精简机器记录见[015结果](../data/references/research-results-015.json)。

## 014复核记录（2026-10-17）

**013新工作已复核，本轮不新增submission。** 现有结果仍是query级支撑率，没有候选级analog排序器、OOF MRR或新离线入口。014固定36例诊断发现29例最佳匹配≥80%由单峰对贡献，原direct还包含12个查询首谱的精确重复；但不能据此否定全部analog信号。另按012原参数补做R3纯负22例：MRR0.720671→0.716883、Top1/Top25不变，未确认碎片跨域增益。下一项核心工作是候选—参考结构一致性特征及canonical分组排序对照。见[014计划](plan/014-submission-readiness-and-support-audit.md)、[014进展](progress/014-submission-readiness-and-support-audit.md)、[014精简记录](../data/references/research-review-014.json)。v9r2/v10仍为0.323。

## 013前置工作（2026-10-17历史记录）

**012 已关闭：v10 public 0.323 与 v9r2 同分已复核；碎片/ChEBI/PubChemLite 三项 CPU 研究完成，未再提交。** 013 主线为可追溯参考谱/analog 第二证据：已用 ccjjllt 账号核实提交与榜单（榜首 0.495/第五 0.468，GPU 剩 27.39h），拉取 prvsiyan analog、Berat 0.336、Seyit 0.446 三份源码（只读未执行），完成首个 CPU 支撑审计（550 query、train 抽样约 5 万、34.5 秒）：NP direct≥0.3 156/250、shift≥0.3 242/250；G1 direct 19/250、shift 246/250；R3 direct 0/46、shift 32/46。支撑率非排序增益，排序试点（HGB/GroupKFold）下一步。最新[013计划](plan/013-analog-second-evidence.md)与[013进展](progress/013-analog-second-evidence.md)，精简[013记录](../data/references/research-results-013.json)。012 结论见[012进展](progress/012-score-plateau-and-complementary-evidence.md)。

## 已完成提交：011（2026-10-17）

用户已授权实际运行和提交。**v10版本1已COMPLETE并通过输出核查，比赛提交ref57293821也已COMPLETE/public0.323。** T4完整日志约41.45分钟，400×25输出、72336次ICE预测全部成功。私有离线资产v1已上传，原539来源条目剔除1个无有效身份的Markush组，实际新增538、候选总数712737，原研究指标不变。550例实际候选/规范化流程核查通过。

两次Kaggle CPU小样本完整运行成功，但估算全量超9小时，因此没有提交CPU版本。10月17日配额恢复后复用GPU主版本，已在北京时间11:43提交明确完成的版本1（scriptVersionId359192425）；12:51复查确认public0.323。见[011进展](progress/011-massbank-expansion-submission.md)、[011计划](plan/011-massbank-expansion-submission.md)和[10月17日执行记录](../data/references/submission-execution-2026-10-17.json)。成品入口[submission-v10](../notebooks/submissions/submission-v10/enveda-submission-v10.ipynb)。v9r2及其0.323分保留。

## 研究决策入口：010（2026-10-14）

目标为最终奖金与金牌。最新有效提交 **v9r2，public0.323**（ref57028399，10月14日Kaggle MCP核实）；当前rank1932/3277，榜首0.491、第五0.467，均为公开快照。最新[010复核与决策](progress/010-submission-review-and-frontier.md)、[010计划](plan/010-submission-review-and-frontier.md)。

**010决策为只验证固定539结构扩库；条件多谱暂不采用。** 正确的R3原v9逻辑base→expanded对照在46个质量可用分子上MRR0.408903→0.762629，新增20个Top25、无损失；NP/G1已有500真值的完整ICE指标不变，属于干扰回归通过。R3与MassBank同源且经人工分层，不可预测隐藏涨幅。010结束时最终离线扩库notebook尚未集成/运行；011已完成集成，剔除1个无有效身份条目后于10月17日启动538结构扩库版，当前执行状态见上节。009研究Colab入口不作为比赛成品。

- 条件多谱在原保留250上的确认：0.907838→0.905171，Δ−0.002667，95%CI[−0.010,+0.004]，Top1净−1。CPU独立重做1400谱前向与完整评分均复现；尚未确认收益，不按错例扫阈值。该250已用，不再称保留集。
- 009-R3的0.740890→0.794979是已扩库FP→已扩库ICE未知CE回退；原v9只用fused_merge，在这些CE未知谱上回退FP。不能把ICE退化当扩库退化，也不把未知CE回退混入首次扩库提交。
- LMSD新增22765结构，在当前非脂质R3样本零新增命中；仅支持暂不部署本配置，不能否定其他结构来源。D2、FIORA、F4a既有退化配置继续不部署。
- 新高分v7c FragGap在缺模拟谱条件补结构碎片，0.441为作者自报非本轮独立测分；下一研发优先明确不支持adduct的碎片证据，避免重复全局logits均值和线性库残差扫描。
- 官方10月13日已允许FRIGID checkpoint，用已发布官方预训练权重不要求重跑预训练，但要提供原训练代码/允许数据引用及自己的代码。完整MIST谱指纹仍未跑通，生成只是后续有界研究路线。
- 现用fp_single_s2原作者BCE＋hard-negative CE训练源码已从公开baseline v34找到；具体s2参数、spec.npz切分与输入资产仍需绑定，不能以本仓库仅BCE训练器冒充精确复现。


## 024–029 后续研究记录

以下为原 README 末尾的增补记录，时间晚于上面的023及更早历史；030已经替换其中的下一步执行顺序。

## 024 验证缺口诊断
- 可见测试为 enveda-180 训练谱副本（标签已恢复）；此前可见测试分数 0.03 为 smoke 伪影（只处理 12/400 分子）。
- E180/H 面板：v11 0.0154，promo_off 0.0158；真值不在 PubChem/Engine2 列表（0/400），无法复现 v11>v13。
- 隐藏测试分布与可见测试/NP250 均不同；尚无可复现公榜排序的面板，未提交。


## 025 PubChem 在库面板（P）
- 250 分子/700 谱（gnps/riken/pluskal，[M+H]+，280–380，PubChem 在库，R 式遮蔽）。
- v11 0.5006，v12 规则 0.4983（复现 v11>v12），promo_off 0.5380（未复现 v11>v13）。未提交。


## 026 人气加权面板 Q 与公开 notebook
- Q：v11 0.4915，v12 0.4885，promo_off 0.5168；人气偏差无法解释 promo 反转（疑与 CPU 跳过 ICEBERG/GLACIER 有关）。未提交。
- 公开 notebook guhongbin/casmi26-v465-v15-pop015 公榜 0.456（Apache-2.0），建议作为新基线 fork。


## 027 fork v465（0.456 公开 notebook，Apache-2.0）
- 代码未改，作为 ccjjllt/enveda-020-e006-repro v8 运行（固定 Py3.12 GPU 镜像）；提交 v14 ref 57462581，**公榜 0.456（新最佳）**。排行榜 10-21 17:50：#1 0.495，#20 0.464（金牌线约 0.465+）。
- 以后的比较一律用 GPU 完整流程 A/B；分数阶梯与后续方向见 docs/progress/027。


## 028 Engine1 候选池扩展（PCL + ChEBI/LMSD）
- notebook 已备好（`notebooks/028/`，推送目录 `/workspace/k027b`）；本地池扩展单元检查通过（710,701→724,938）。
- **GPU 周配额已满**，重置 **2026-10-24 08:00 北京时间** 前不 push。详见[028计划](plan/028-e1-pool-extension.md)与[028进展](progress/028-e1-pool-extension.md)。


## 029 lavinwins 对照与 joint ICE/GL
- 纠正：lavinwins 实际 POP/POOLPOP 为 0.15（非 0.25）；其 changelog 0.464 来自 N1 上 joint ICE+GL。
- 029a 为历史准备版本（+DreaMS、GEN_N_ANALOG=20）。**030复核发现漏改N1 complement开关，不直接运行；当前先单独运行 `notebooks/030` 的完整N1 challenger，再考虑028扩池。** 见[030进展](progress/030-n1-challenger-and-score-strategy.md)。旧记录里的 `/workspace/k029` 不在当前本机环境中。


## 初始阶段记录（001–003）

以下内容属于早期建设记录，保留当时判断与入口；其中“当前”“尚无”等说法不描述今天的项目状态。已完成前向、训练与提交的实际历史见004之后各轮。

截至提交 `5e22537`，已完成训练数据实测、初步 split、B0/B2 本地运行和指纹权重下载检查；这些指标尚未接入官方 RDKit 规范化，B2 的 merged 路径也是占位比较，不能当作正式泛化结果。尚无真实模型前向、微调结果或排行榜分数，详见 [001 进展](progress/001-baseline.md)。

## 开始研究

当前执行 [003 Colab 接入计划](plan/003-colab-mcp.md)，先核实 Kaggle 已关联账号能否复用现有工具，再决定通用 Colab MCP 的配置；实际结果见 [003 进展](progress/003-colab-mcp.md)。本轮范围是连接能力核实与配置，不是训练。

后续继承 [001 基线计划](plan/001-baseline.md) 与 [001 进展](progress/001-baseline.md) 中的数据、划分和 B0/B2 记录，补齐官方评分规范化与真实权重前向；按 [002 TPU 微调计划](plan/002-tpu-finetuning.md) 先做 Colab T4 兼容性核对，再推进 v5e-1 适配。平台变更、已 push 的 Kaggle 作业和未验证项见 [002 进展](progress/002-tpu-finetuning.md)。

可复用参考：[官方 de novo 教程](https://www.kaggle.com/code/inversion/casmi-denovo-tutorial-notebook)、[公开 Analog Propagation 基线](https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline)。公开运行、代码注释或作者分数不算本项目实测结果。使用前记录版本，并分别确认代码、模型权重及外部候选库的许可和来源。
