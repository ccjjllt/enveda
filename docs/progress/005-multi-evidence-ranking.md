# 005 — 从 0.206 到多证据候选排序：实际进展

更新：2026-10-04（北京时间）。对应[研究与实现交接计划](../plan/005-multi-evidence-ranking.md)。

**本轮完成了比赛与前沿资料研究、004/v7 实现审计、旧指标手动复算、R4 名次错误修正和下一阶段交接。没有运行新的 GPU 训练、推送 notebook 或提交；排行榜成绩仍为 v7 的 0.206。005 的完整证据融合管线尚未实现。**

## 本轮确定的研究方向

优先做相同完整候选库下的 **谱库 + shifted analog + FPNet + 分子级多谱 ranker**，再用 MassSpecGym 来源的 GLACIER 对同式 shortlist 重排。下一次计算首先复现原 head/rank-head 在实际并集池上的结果，而不是继续增加训练步数。

理由是 v7 尚未部署 004 多数增量；旧验证还混入候选池、统计粒度和参考背景差异。当前公开强源码也主要靠多种互补证据，并非单一模型长训练。库外生成保留为独立小试，等有标签研究能够区分候选缺失与排序失败后安排；不能从 0.206 倒推出隐藏类别占比。

## 已核实的比赛状态

- `56799854`：COMPLETE，public **0.206**；metadata 未返回私榜和逐分子真值。本人 notebook 当前 v7、private、T4、关闭互联网。提交与 v7 的关联来自用户及仓库记录，API 分别核实了 submission 和 notebook 状态。
- 最新读取 2,306 队，项目 metadata rank 1,566；公开榜第 1/5/14/20 条为 0.471/0.433/0.425/0.418。只是当前公开快照，不是最终金牌/奖金线。
- 发布文件仍为原来的三份文件，名称/大小/创建时间未见变化；本次没有重新下载并计算全部数据哈希。
- 近期部分公开模型包的 Other/NC 许可适用性与 DreaMS 来源问题还在讨论，没有读到主办方新答复。与已明确禁止的 NIST 路线分开处理。

证据：[比赛更新](../../data/references/competition-update-2026-10-03.json)、[资产版本与说明](../../data/references/candidate-assets-2026-10-04.json)。两个 10-03 命名文件按 UTC 研究日期保留，内容记录了最终核实时间。

Kaggle MCP 在根会话仍列有工具，但读取 submission/asset 时发生 HTTP transport failure；子会话工具目录刷新后不再暴露相应工具。随后通过已配置的官方 Kaggle CLI 2.2.4/SDK 只读成功核实。没有为此查看或复制凭据，也没有使用 Computer Use。

## 004/v7 复核的关键结果

1. **v7 与离线模型不同。** v7 仅原始 `fp_single_s2.pt`，single 选谱、20ppm、点积；没有 rank-head、cosine、R4。在线并集 712,199 与旧离线训练池 275,810 不同。
2. **旧 R4 0.6257 无效。** part2 cell 11 原代码 `int(i)+1` 将排序后列出的原索引当名次。CPU 重建窗口发现 1,184/1,184 个保存 rank 全等于原始窗口位置，完全不体现排序，故撤回其 +0.004 增益解释。正确结果需重跑，不能从该 ranks 文件恢复。
3. **强 cosine 证据不完整。** 当前 `r1_cosine_np1184.json` 仍是旧实验的 1,184 个 None；当前 `r4_np1184.json` 还是旧五维 ranker/ranks。文档记载的 .7076/.9796 没有对应可复核的完整逐例文件。
4. **参考库存在选择偏差。** L2453 仅含 Q 的 NP250 结构，背景结构为0。Q/L 精确峰数组 hash 重叠为0，不等于没有来源/结构选择偏差；大 C 中只有真值集合有谱特征，也可抬高结果。
5. **“未见”边界不足。** train200k 含113,955个 shipped结构；GEN500中215个结构/630条query出现在该训练分片中。NP canonical表有16个变更键，其中14个目标键存在于全池其他非holdout原键、8个在train200k；这是待两端SMILES规范化确认的别名风险，不单凭字符串认定泄漏。
6. **开发输出与隐藏结果分开。** 本地 v7 CSV 是占位输入开发结果。400×25、首候选裸CCO=0 仅说明输出情况；仍有515个槽位带 `@`、18行含填充CCO、17行有精确重复。这不能定位隐藏得分损失，但表明计划中的导出规范化未接入，且无需强行填满25个候选。

完整源码定位、输入/源码 hash、人工分析与待复现实验见 [v7 审计](../../data/references/submission-v7-audit-2026-10-03.json)。审计里的 notebook hash 是修正前身份；下述 R4 代码修正发生在审计后。

## 本轮实际手动计算

新建 [analyze_005_existing_results.py](../../scripts/analyze_005_existing_results.py)，读取已有 JSON，不运行模型、不使用隐藏标签。命令：

```bash
/opt/homebrew/bin/python3.11 scripts/analyze_005_existing_results.py
```

本机 M5/32GB；Python3.11、NumPy；约5.2秒。固定 seed=20261004，10,000次按分子成对 bootstrap。结果写入 [004-metric-reanalysis-2026-10-04.json](../../data/references/004-metric-reanalysis-2026-10-04.json)，包括来源文件 SHA、250分子明细、分层结果与局限。

| 已保存方法 | n/单位 | MRR@25 | Top1 | Top25 |
| --- | --- | ---: | ---: | ---: |
| R1 single，每谱诊断 | 1,184谱 | 0.599114 | 0.439189 | 0.955236 |
| R2 single，最大原始峰数 | 250分子 | 0.675203 | 0.528 | 0.992 |
| R2 mean_all | 250分子 | 0.648276 | 0.488 | 0.976 |
| R2 mean_ad | 250分子 | 0.636313 | 0.476 | 0.976 |
| R2 rr 轮流取候选 | 250分子 | 0.630003 | 0.472 | 0.980 |
| merged | 250分子 | 0.664566 | 0.504 | 0.988 |
| 真值选择最佳单谱，仅诊断上界 | 250分子 | 0.747158 | 0.620 | 0.992 |

人工检查：从 Q/R1 重新构造“最多原始峰谱”的 ranks，与保存的 R2 single **250/250一致**。merged 只有顺序 ranks，没有独立 key；按已读 notebook 的同一分组顺序对齐，不能证明原 runtime 从未改过顺序。

- mean_all 比 single 低0.026927，95%分子bootstrap区间约[−0.049710,−0.004995]；28个分子改善、38个变差。
- merged 差−0.010638，区间[−0.043000,+0.021593]；47改善、52变差，不能据此认定有提升。
- 真值选谱让51个分子改善，MRR增量0.071955。这个上界支持研究条件化选谱，但真值不可用于推理。
- single 正模式202分子0.697106，负模式48分子0.583031；`[M+NH4]+`25分子0.510206。不同结构构成，不作离子条件因果结论。

边界：这里只修正 rank>25 截断，没有重跑 RDKit官方SMILES匹配。不同融合方法仍有质量窗差异；NP250反复用于研究、预训练暴露未知。bootstrap只反映此样本内部差异，不能校正这些系统偏差。R3没有逐例数据，不能补算分子MRR或区间。

## 实际代码修改与人工核对

- 修正 `notebooks/colab/004-part2-consolidated.ipynb` cell11 的 R4：使用 `enumerate(order, start=1)` 返回排序位置。
- 手动例子：`order=[2,0,1]`、真值原索引2；原表达式输出3，修正后输出1。手动编译该 cell 的 Python源码通过。**没有远端重跑，不宣称修正后的R4分数。**
- 新增上述旧结果分析脚本及结构化报告；未修改004原始大产物，未写测试文件或引入自动化测试。
- 更新004状态/错误解释、README当前研究入口和AGENTS接手位置，避免下一位再次从未执行R0的旧描述开始。
- 手动读取本轮5份JSON成功；005配对文档本地链接均可解析；分析脚本和修正cell的Python语法可编译；notebook与原版逐cell比较只有cell11改变；`git diff --check`无空白错误。这些是本次人工检查，没有新增测试文件或门控。

## 前沿研究实际完成情况

读取并记录四份公开源码、小型引擎源码/manifest、官方模型资料与许可说明：[前沿研究证据](../../data/references/research-frontier-2026-10-04.json)。仅获取研究必需小文件，没有下载大模型/全PubChem。

确认了公开强管线的多个独立通道及160特征ranker；不能将旧6930-bit缓存塞到其7块指纹模型，也不能删除通道后补零复用ranker。已确认 `fpnet-full1` 训练包含NP示例，NP250不能作为其未见效果证据。

GLACIER有可用MassSpecGym路线，但当前公开wrapper与Colab Python3.13的依赖版本、支持adduct和条件映射需实际适配。前沿MARLIN无需真值formula但未找到已发布官方权重；DiffMS/FRIGID等仍受formula/指纹条件约束；GenMol不是现成谱条件模型。这些研究支持先做证据融合和小型正向重排，而不是马上重训一个大型生成器。

## 未完成与接手顺序

**005 E0–E4 均未完成模型实验。** 未新增GPU日志、训练权重、完整背景ranker、GLACIER预测或submission，不存在新的提分结果。

按配对计划实现：

1. 全池身份映射与实际整组留出，固定相同并集候选池；原head/rank-head重新计算分子预测，保存候选级分数。
2. 完整背景谱库及shifted analog，按结构聚合；人工分析互补错误和高相似度误命中。
3. 条件化多谱与自有轻量ranker；再运行GLACIER支持条件下的小预算重排。
4. 按候选真值缺失/shortlist损失决定扩池或生成；把方法、代码完成、离线效果与排行榜分数分别写回本文件。

本轮没有等待Colab MCP执行能力，也未用GUI绕行。大规模实现与实验的函数职责、文件、依赖、条件和计算预算已写到[005计划](../plan/005-multi-evidence-ranking.md)，可直接作为下一位执行者的交接。

## E0–E4 执行与主审验证（2026-10-04，四子任务并行 + E2 跟进，主审逐项验货）

**E0（同池复现与身份修正）**：`artifacts/005/` 下身份映射（train/coco parquet + summary）、36 组召回（`e0_cq_recall.json`）、双模型 logits（1184×6930 finite 且互异）、canon 重算全明细。主审验证：summary 数字与文件一致（train unique canon 274,186/变更 4,370；coco 432,610/45,540；并集 canonical 706,242，554 组横跨双池；NP16 单列）；canon 重算 train orig 0.688468 / union orig 0.504854 与报告一致且带逐分子明细；train shipped orig 0.675203 逐位复现历史 R2。结论：扩池代价 orig head 下 train→union −0.183（23 分子掉出 top25）；rank-head union −0.045（53 改善/84 恶化），train 上基本持平。`src/train_fingerprint.py` 只有 BCE、与 notebook 的 BCE+CE 不是同一公式——决定不动，需训练时再统一口径。遗留：`e0_head_compare.json` 的 summary/detail 为空（数字以 canon 明细 + logits 为准）。

**E1（完整背景谱库 + analog）**：`src/retrieval.py` +697 行（主审 import + 直接/偏移 disjoint smoke 通过；早前一次报错是主审误传参数格式，代码无问题）。完整背景 seen 谱 MRR 0.9263/分子 0.9228（相对旧 L2453 −0.049/−0.056，Top25 同为 1179/1184——召回同、排序变差）；unseen 真值证据 MRR 0.0（tie-break 0.136 已明确剔除，不冒充）。互补：谱库修 94 个 FPNet Top1 错误，反向 8 个；高相似误命中 121 例（含同 adduct 多源仍错），证阈值不可硬接管。局限如实声明：偏移全背景穷举本地未跑。

**E2（分子级多证据 ranker）**：同一 C(q)（union|single|10ppm nocap）四种对照——single 0.5149、等权 0.4956、选谱权重 0.5112、ranker **0.9211**；lib_only 0.8933；unseen 真值去库模拟 0.1000。主审独立重算：1000 个 rank 声明（250 分子×4 法）与 top25 实际位置 0 错位，ranker MRR 重算 0.921143 分毫不差。代码初轮未落盘，已补派落盘 `src/molecule_ranker.py` 并重跑确认 ranker MRR 差 0（single 对照差 −0.000333 系并列决胜口径不同，真值 rank 仅 1 分子差一位，不影响 verdict）。**verdict：不进提交管线**——+0.406 几乎全来自谱库（ranker 仅再 +0.028），无库真值下系统性降权（215 loss），不满足缺失处理要求。

**E3（GLACIER 试点准备）**：`artifacts/005/glacier.pt` sha256 `9192e4c2…b397a12` 主审实测一致（60,531,887B；任务写的 61,108,325B 在任何来源都对不上，已弃用）；`src/forward_rerank.py` 新建（import 通过，条件映射/五元组缓存键/四类计数/三种预算入口齐备）；28 分子试点集规则落盘；CPU prep 136 谱 0.006s。数字 pending（无 shortlist + 无 GPU 环境），未编造。卡点明确：缺 ms-pred 推理子集 + RDKit 版本须子进程隔离 + cp312/3.13 绕行方案已备。

**E4（公式路径 + 缺失分析）**：MIST-CF 两 ckpt URL 主审对官方 `download_model.sh` 逐字验证通过（zenodo 8151490）；SIRIUS/decomp 版本约束、正模式限制（`still only positive mode`，6 加合；`[M+K]+` 与全部负谱 unsupported）已核实；缺失表：NP 池 100%/窗 99.7% vs holdout 窗 82.5%——NP 不适合衡量外部召回。生成小试只设计未执行（待权重/环境）。

本轮仍无 GPU 训练、无 submission、无提分；成绩保持 0.206。E2 verdict 指向：下一步不是把当前 ranker 部署，而是解决无库真值的排序（E4 生成召回 / E3 forward 重排补同式证据），或把 ranker 的缺失处理做对后再谈上线。

## E3/E4 执行与主审验证（2026-10-04）

**E3 GLACIER 试点（28 分子，Kaggle GPU 实跑，主审复算）**：
- 配额：GPU 剩约 17.1h（此前“47 分钟”为过期快照），本次只 push 两次开发运行（v1 挂载失败 ERROR 约 24s、v2 修好挂载后 pip 失败 ERROR、v3 成功），配额消耗可忽略。
- 两次修 bug 都是主审亲手修的：① 数据集挂载在 `/kaggle/input/datasets/{owner}/{slug}` 而非 `/kaggle/input/{slug}`——v1 全 0 挂载的原因，改全树按文件名搜索后解决；② cp312 轮子在 python3.13 worker 上装不上（意料之中），改 PyPI 直装（rdkit/einops/lightning/torchmetrics）+ `site='none'` + `--any-rdkit`。rdkit 实际为 2026.03.6（非 panel-exact 2025.03，已记偏差）。
- 实测（v3 COMPLETE，T4，41.5s，68.46 pred/s，峰值 0.42GB，632 候选 506 scored）：base MRR 0.5162 → GLACIER 重排 **0.5791（+0.0629）**，改善 8 / 恶化 6 / 持平 14。逐分子由主审从 `glacier_scores.json` 独立重算。大胜：WJXS 24→1、CKEX 13→1、IKMD 3→1、FXNF 3→1；大亏：VNON 1→5、XRWQ 1→3。结论：同式重排有净增益但硬覆盖会亏 rank-1，必须做融合而非替换；28 例太少，扩到全观察集前不报成绩。产物：`notebooks/glacier-pilot/`（开发运行，非提交）、`artifacts/005/e3_pilot_rerank_eval.json`（逐分子 base/glacier rank，主审生成）。

**E4 公式路径（20 分子，本地 CPU 实跑，主审验权重 + 落盘复跑）**：
- 双 ckpt md5 主审实测与 zenodo API 一致（mist bff0db38…/fast 505eced4…）；`download_model.sh` 主审对官方逐字验证通过。
- MIST-CF 真实前向：质量基线 Top1 9/20 → MIST 重排 **Top1 15/20**（+6，恶化 3）。SIRIUS 缺席（本机无 Java），用 E0 窗内公式枚举代替 decomp——候选偏乐观，已在 `scripts/mist_cf_formula.py` docstring 与结果 meta 双记；timsTOF→Unknown override 已记。
- 落盘补派：`scripts/mist_cf_formula.py` + `artifacts/005/mist_cf_formula_topk_20.json`，重跑 Top1 15/20 零差异，主审验 JSON 得同数。
- FRIGID 2.8GB @~90KB/s 下不动（8–9h ETA）；MS-GPT 无直接权重 URL。只有 MIST-CF 这条公式路跑通，生成器本身仍 pending。

本轮仍无 submission、无提分；成绩保持 0.206。下一步：E3 扩全观察集 + 只做融合（不硬覆盖）；E4 等 FRIGID 可下时再开生成小试。
