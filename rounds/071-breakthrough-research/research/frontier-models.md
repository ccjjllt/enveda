# 071 前沿模型与可落地框架核查

核查日期：2026-11-17，北京时间。范围：原论文最新版本、作者代码、原始权重发布页；只做资料与源码阅读，没有运行训练、推理或提交。本文的“优先”是下一轮实验的优先级，不表示已经在本赛提分。

## 结论

最值得补测的模型资产是 **2026-10 发布的 official MIST MSG / 4M-aug**，其次是 **2026-10-29 修订版 MS-GPT 的 official MSG 编码器与配对 adapter**。前者应先作为候选检索分数测，后者作为有条件的池外候选提案器测。它们解决的问题不同，不能用生成命中率否决检索排序。

MSAlign 的候选对比框架同样值得推进，但真正的 MolDeBERTa 版本不是本项目 076a 的 Morgan MLP；其许可、替代编码器与训练设计见本轮主报告。长线方向是用合法前向模型生成结构覆盖更广的合成谱，改进逆向编码器，而不是继续从零训练 FP→分箱谱小 MLP。

**两个看起来很强的新方法不能作为高分依据：MSFlow 和 SpecBridge 均已撤稿。** 搜索摘要和旧 HTML 仍返回它们的高分，必须打开最新摘要页才能看到撤稿状态。

## 1. 首先修正旧研究对 MIST 的判断

旧记录将“MIST+MolForge 的 de novo top-1 约 0.107”与 ICE 的候选集 R@1 放在一起比较，并据此否定 MIST 排序。这是任务口径不一致：生成要自己产生正确结构，检索假设正确结构在给定池里。

MassSpecGym 官方审计论文 v2 **Table 3 的公式已知检索榜**中，MIST 4M-aug 的 R@1 / R@20 为 **40.31% / 73.02%**；同表 ICEBERG 为 **36.18% / 78.83%**，提示两者可能各有强项。该论文 **Table 2 的质量候选/公式泄漏实验**另报：MIST-CF top-1 预测公式接 MIST 后 R@1 / R@20 为 **32.14% / 60.92%**。Table 2 的真实公式对照 52.84% 属另一实验口径，不作为通用检索分数或与 Table 3 跨表排名。以上均不是本赛 MRR；它们只支持“应补测新版 MIST 检索”，不支持承诺增益。[官方评测论文 Tables 2–3](https://arxiv.org/html/2606.19624v2#S5.SS1)

### 可拿到什么

- [coleygroup/MIST](https://github.com/coleygroup/MIST) 是新的完整训练/推理实现，保留公式感知注意力，输出 4096 位、radius-2、无手性 Morgan 指纹；支持紧凑缓存、分布式训练和合成谱混合训练。
- [Zenodo 23303289](https://zenodo.org/records/23303289) 于 2026-10 发布 `mist_msg_noaug.ckpt`、`mist_msg_aug.ckpt`，各 **289.2 MB**；MSG 与 CANOPUS 的 checkpoint、GLACIER generator 和数据分开列出。
- [MSG 增强训练配置](https://raw.githubusercontent.com/coleygroup/MIST/main/configs/aug/msg.yaml) 明写从零训练，数据源为 MSG 真谱和 refined-4M 的 MSG-GLACIER 合成谱。不存在“必须沿用旧 CANOPUS/NIST 编码器”的技术依赖。
- 代码 MIT。代码许可证不等于所有权重、数据都继承 MIT；[NOTICE](https://raw.githubusercontent.com/coleygroup/MIST/main/NOTICE.md) 要求分别查看下载资产 provenance/license。此次 Zenodo 页面抽取没有展示具体权重授权值，落地打包时应保留原发布元数据与许可，不用泛化的“MIT 全包”表述。

### 本赛接入的关键限制

[当前化学词表源码](https://raw.githubusercontent.com/coleygroup/MIST/main/src/mist/utils/chem_utils.py) 的 `ION_LST` 只有 7 种正离子模式；没有 `[M-H]-`、formate、Cl 等负离子。`timsTOF` 也不在显式 instrument map 中，未经映射会归为 unknown。`[M+NH4]+` 与源码 `[M+H3N+H]+` 等应做化学等价名称归一；不能把负离子伪装成 `[M+H]+`。

**建议实验（本轮提出，尚未执行）：**

1. 第一阶段只覆盖受支持的正离子谱，保留底座处理其他谱；分别记录“分子有可支持谱”的覆盖率与各加合物效果。
2. 对同一真实候选池同时比较现有 FPNet、新 MIST noaug、新 MIST aug，不能额外把真值塞入正式召回评估。用额外的真值注入面板单独诊断条件排序能力。
3. 本赛没有真实公式输入。先用 precursor/adduct 得质量窗口中的公式假设，再用合法、未见结构验证过的 MIST-CF 排序；对 top-K 公式分别算指纹，公式后验参与候选分数。不要在验证时拿标签公式冒充可用输入。
4. 除了独立 R@1，测与底座的错误互补、召回增量和最终 MRR@25；保留多谱分数而非过早平均。新 MIST 如果只重复 FPNet 的错误，停止该集成。
5. 训练数据的公式特征、固定候选表示与权重训练放在提交前；最终隐藏谱的公式推断和子公式特征仍须在断网运行时计算，再做模型推理与候选评分。289 MB 权重并不证明 9h 可行，仍需远端手动测峰值显存、每谱预处理与每公式耗时。

## 2. MS-GPT：现在确实可用，但旧高分已被修订

[作者当前 README](https://github.com/VIKI623/MS-GPT) 明确更新了 MIST 编码器、注意力 mask 和评测。默认生成预算是每谱 100 个 decoder samples（20 个后验阈值 × 5 次采样），按公式匹配、生成频率和后验相似度整理候选。

| 指标口径 | MSG top-1 / top-10 | 状态 |
| --- | --- | --- |
| 搜索缓存/早期 README | 23.91% / 28.65% | 已被当前版本取代，不引用为现状 |
| 2026-10-29 v2，official MIST，真实公式 | **10.22% / 13.21%** | 当前主结果 |
| 同版，MIST-CF top-5 预测公式 | **6.83% / 8.58%** | 更接近未知公式场景 |
| 同版，50M 从零分子 decoder，真实公式 | 8.49% / 11.44% | 更严格结构距离控制的变体 |

论文另报 MSG 的 100 个 samples 约 **2.80 秒/谱**（RTX 6000 Ada + 16 核 CPU，生成与 top-10 后处理，不等于包含全部 MIST/公式预处理的本赛端到端时长）；扩大到 1000 次约 21.16 秒/谱、top-10 15.07%。因此先做有限预算的提案覆盖测量，比先堆采样次数更合理。[MS-GPT v2 Tables 4、6、7](https://arxiv.org/html/2607.23607v2)

### 干净的资产组合

应逐文件选择：`shared MLM + tokenizer/SAFE base + mist_official/mist_massspecgym.pt + 对应 MSG rank-16 adapter`。不要默认运行整个压缩包里的 NPLIB1 profile，也不要将 `frigid_legacy` 与 official profile 的阈值、adapter 混配。

作者[issue #1 的许可说明](https://github.com/VIKI623/MS-GPT/issues/1) 允许其 MLM 和 adapter 使用 Apache-2.0，重分发的 MIST 保留上游许可。shared decoder 的训练是分子结构而非谱：上游 [SAFE-GPT 数据卡](https://huggingface.co/datasets/datamol-io/safe-gpt) 标示 ZINC20/UniChem 来源和 Apache-2.0。MSG-only 谱编码器和 MSG adapter 的谱数据链应独立核对；“共享 decoder 同时用于两数据集”不等于它训练过两边的谱。

50M 版本的目的主要是对 MassSpecGym 做更严格的结构相似性隔离，不是仅为规避 NIST。它不自动比 100M 主 decoder 更适合本赛。当前 Kaggle 新挂载包及源码 pin 见本轮 [公开资产核查](public-frontier.md)。

### 如何避免又做一次低收益尾槽插入

**建议实验（尚未执行）：**先在真实候选池未命中的验证分子上运行 100-sample MS-GPT，记录新正确结构是否进入生成池，以及它与库/PubChem候选的重复比例。再与底座合并后由统一候选排序器打分，比较 MRR，而不是把生成第 1 名直接升到全局第 1 名，也不是永远塞第 23–25 名。

生成频率是模型内部一致性，不是跨化学分布校准的正确概率。应使用公式后验、多个独立谱的一致性、与前向谱匹配、候选来源和基线置信度来决定有限名额。若增量 oracle recall 几乎为零，则停止；若存在真实增量但最终 MRR 不涨，问题转向融合/排序而不是增加采样。

MIST 的正离子限制也适用于该分支。独立谱生成再做分子级合并，比随意平均后验更容易审计；第一轮应固定总采样预算，防止多谱分子不公平获得数倍算力。

## 3. 更值得投入的训练框架：真实谱 + 合成谱的逆向预训练/适配

官方 MIST 新资产的重要性还在于，它发布了生成器、训练特征和可复用训练框架，而不仅是一个旧 benchmark 权重。MSG augmented 路径实际使用数百万结构的合成谱扩展，再由真谱验证选择 checkpoint；它与本项目几分钟训练 FP→谱 MLP 并不是同一个研究问题。[官方训练说明](https://github.com/coleygroup/MIST#train)

**研究假设：**本赛未见天然产物泛化差可能来自结构覆盖和谱侧域差异；比增加小 MLP 宽度更有价值的是：先保留合法 MSG 预训练的结构先验，加入本赛 timsTOF 真谱，再用合法 ICE/GLACIER 在公开天然产物结构上合成多能量谱，训练谱→指纹或谱→分子表示的模型。

这条路线必须分成可否证的阶段：

- 先测官方 checkpoint；没有增益时定位是公式、加合物、谱预处理还是结构域差异。
- 接着只在训练结构内做 timsTOF 适配，加入旧域样本防止遗忘；用整结构留出的真实谱报告结果。
- 只有前两步显示改善才上合成谱。模拟器的训练与结构选择也必须与验证拆分隔离，不能按验证真值的近邻挑合成结构后仍称无泄漏。
- 不用模拟谱上的高分当成功指标，最终只看真实谱候选检索及 MRR；同时报告正负离子、天然产物、质量段与近邻距离分层。

训练预算不能从参数量推算保证。官方审计附录的单模型训练约 12h 是 RTX PRO 6000 Blackwell 上的参考，不是 T4 预算；完整增强训练应作为中长期方案，先缓存/小样本吞吐后再估时。[官方审计训练细节](https://arxiv.org/html/2606.19624v2#A1.SS2)

## 4. 需要降级或排除的“前沿高分”

| 模型/路线 | 本次核查 | 对本项目的处理 |
| --- | --- | --- |
| **MSFlow** | 最新 arXiv v2 已于 **2026-03-12 撤稿**，作者称潜在数据泄漏；v1 的 MSG 32.00/42.53% 与“45%”摘要仍能被搜索到。作者 repo 当前 MIT 且有权重链接，但不能恢复已撤回的结果可信度。 | 可研究连续 CDDD 条件表示的思想，不能按旧高分投入主线。[撤稿声明](https://arxiv.org/abs/2602.19912)、[作者代码](https://github.com/ghaith-mq/MSFlow) |
| **SpecBridge** | 最新 arXiv v3 已于 **2026-03-03 撤稿**，原因是预处理/评估流程问题；作者给的 GitHub 链接此次返回 404。Zenodo 仍有 3 个约 1.7GB 权重，页面未展示明确许可值。 | 不将 84.7% R@1 或“只训两层就大幅提升”当证据。[撤稿声明](https://arxiv.org/abs/2601.17204)、[原资产](https://zenodo.org/records/18357418) |
| **MS-BART** | 代码和 Figshare 权重链接存在；原 MSG 资产目录却注明 MIST 来自“clean CANOPUS”，另支持旧 FRIGID 参数字典。clean 指 benchmark 清理，不等于已证明不含禁止数据。README 还明确列出待修正的路径与 beam 参数。 | 不优先于新版 MS-GPT。需重新核整条权重谱源及一致的 ik14 指标，不能凭“5/12 SOTA”选择。[作者 README](https://github.com/OpenDFM/MS-BART) |
| **FRIGID / 旧 MIST-MolForge** | FRIGID 作者承认旧 MIST 合成训练结构选择使用了测试信息；已换 official MIST 并重报结果。旧 checkpoint、旧 batch bug 修正后仍不是同一基线。 | 本项目旧失败不等于否决所有新 checkpoint；但算力优先给更轻的检索/MS-GPT提案。复现时必须固定编码器版本。[作者更新](https://github.com/coleygroup/FRIGID) |
| **CSI:FingerID / MSNovelist** | 依赖的在线指纹模型训练源与运行方式不符合本赛当前使用边界。SIRIUS 的纯质量分解算法与这些预训练 web 模型是不同组件。 | 不使用受禁谱源权重；可独立采用可离线、无学习权重的公式枚举算法或合法 MIST-CF。来源边界见旧轮规则核查及[官方训练数据说明](https://v6.docs.sirius-ms.io/advanced-background-information/#training-data)。 |

## 5. 本次调研对“大幅提高”的具体含义

前沿资产给出的机会不是“把 0.492 换成论文 0.58”。可验证的大幅提升路径是：

1. **提高真实候选并集的召回**：新版 MIST 从较宽 PubChem 窗口挑出原 FPNet 丢掉的结构；MS-GPT 补充池外结构。
2. **在统一候选集上训练面向排名的融合**：避免每个新模块只动旧前25的中间位置，也避免单模块置信度跨尺度硬升位。
3. **适配公式与多谱证据**：同分子多 CE/adduct 的互补信息可以改变排序；当前文献大多按单谱报告，必须按本赛分子级度量重新验证。
4. **把昂贵能力离线变成轻量权重与缓存**：训练不放在最终9h运行里，最终按分子不确定性分配生成/前向预算。

这些都是研究假设。先证明新增信号能提高真实候选召回或纠正底座错误，再投入训练和提交；不能从某一个宽松离线面板的准确率直接给公榜增益区间。
