# MS2 同分异构体判别：前沿与开源方案调研

调研日期：2026-11-18。范围：只回答一个问题——**在已有候选池、不重训、断网、单 T4、6 小时内**，有没有现成模型能给「同一分子式、相近骨架」的候选结构打出比现有管线更好的分。
本文只做资料与源码/权重页阅读，没有下载大文件、没有跑训练或推理。

## 0. 结论摘要

1. **可以直接用的只有「冻结的候选-谱打分器」**：MIST（谱+公式→4096 位指纹概率）、ICEBERG/GLACIER（结构→碎片图/预测谱）、CFM-ID 4（结构→预测谱）。三者都有公开权重、可离线、许可可过、单 T4 成本可控。
2. **最强的新信号是 MIST 指纹概率**。它不是「再训一个指纹模型」，而是 MSG/CANOPUS 预训练、带公式注意力的概率指纹；GEMS 的消融显示，同式候选的排序瓶颈正是这套指纹（精确指纹排序把 NPLIB1 Hit@1 从 29.64% 抬到 56.66%）。【事实，[GEMS](https://arxiv.org/abs/2610.23617)】
3. **谱-谱模型（MS2DeepScore、Spec2Vec、DreaMS、MSBERT）不能直接给任意候选打分**：它们需要每个候选有一条参考谱。我们的候选多数没有参考谱，所以这类模型只能做库检索/补位，不能做同分异构体判别。本队 DreaMS kNN 已经实测 0.492 中性。
4. **SIRIUS 6 / CSI:FingerID / COSMIC 在方法上最对口，但不可用**：CSI:FingerID 的训练集含 NIST，且官方 web 服务只许学术非商用；奖金赛按商用处理，直接排除。ICEBERG 的默认 NIST'23 权重同理，只能用它的 MassSpecGym 权重。
5. **2024–2026 真正专门做同分异构体的新工作**：GEMS（异构图上的学习搜索，2026-10）、LSM-MS2（声称异构体识别 +30%）、MBGen（扩散生成，声称 isomer differentiation）。GEMS 未发布权重、LSM-MS2 商业闭源、MBGen 代码未核；都不能直接落地。

## 1. 筛选口径：为什么只看「候选-谱打分」

同分异构体判别有四类任务，信息流完全不同：

| 任务类型 | 输入 → 输出 | 能否直接给已有 25 个候选打分 | 代表 |
| --- | --- | --- | --- |
| 谱→指纹 | 谱（+公式）→ 指纹概率 | **能**（候选结构算指纹，比一致度） | MIST、CSI:FingerID |
| 谱→结构 | 谱→SMILES/图 | 不能直接；只能生成新候选或算似然 | MS-BART、MS-GPT、GEMS、MBGen |
| 谱-谱相似 | 谱+谱→相似度/Tanimoto | 不能；候选侧需要参考谱 | MS2DeepScore、Spec2Vec、DreaMS、MSBERT |
| 候选-谱打分 | 谱+候选结构→分数 | **能，原生支持** | CFM-ID、ICEBERG、MetFrag、碎片树对齐 |

本队的误差结构决定了只需要第四类：072d 的 150 分子仪器面板上，召回@25 = 150/150，MRR@25 = 0.8286，48 个「真值不在榜首」里 47 个榜首与真值同分子式，42 个名次 2 也是同分异构，Tanimoto 中位 0.471。【本队，`rounds/072-scale/progress.md`】所以问题不是「找不到」，而是「同式候选之间选错」。

> 判据：一个方法值得进短名单，必须同时满足 (a) 能对任意候选结构输出分数；(b) 有公开权重；(c) 断网可跑；(d) 许可允许比赛/奖金场景；(e) 有同分异构体证据；(f) 单 T4 上 400×25 不超过约 1 小时。缺一项就只进「风险/排除」清单。

## 2. 方法清单：到底预测什么、有没有权重

### 2.1 MIST（MIT / Coley group，2023 起，2026 新版）

- **预测什么**：输入谱峰 + 分子式，输出 **4096 位 Morgan 指纹（radius 2，无手性）的逐位概率**；公式感知注意力 + 迭代指纹预测。新版官方实现 72.3M 参数，输出 `probabilities.npy`，可 CPU/GPU。MIST 本身只出指纹；要做结构解析还需外接 MolForge、MS-GPT、FRIGID 等解码器。【事实，[coleygroup/MIST README](https://github.com/coleygroup/MIST)】
- **训练数据**：MassSpecGym 真谱（`msg-noaug`）、MSG 真谱 + refined-4M 合成谱（`msg-aug`）、CANOPUS/NPLIB1（`canopus-*`）。【事实，同上】
- **公开权重**：是。Zenodo 23303289 发布 `mist_msg_noaug.ckpt`、`mist_msg_aug.ckpt`，各 289.2 MB；另有 CANOPUS 两条与 GLACIER 生成器。【事实，[Zenodo 23303289](https://zenodo.org/records/23303289)】
- **许可**：代码 MIT；权重页未展示明确许可值，NOTICE 要求逐资产核对 provenance/license。【事实，[NOTICE](https://raw.githubusercontent.com/coleygroup/MIST/main/NOTICE.md)】
- **限制**：当前 `ION_LST` 只有 7 种正离子，没有 `[M-H]-`/formate/Cl；`timsTOF` 不在显式 instrument map。负离子和 timsTOF 会被归到 unknown/不支持。【事实，071 轮源码核查】
- **同分异构体能力**：指纹是局部原子环境，取代位置/环化方式不同 → 位模式不同，原则上可分辨构造异构；但**不是**为异构体专门训练的，手性不编码。间接证据很强：GEMS 全程用 MIST 指纹做候选检索、编辑导航与最终排序；固定候选池上把预测指纹换成真值指纹，NPLIB1 Hit@1 29.64→56.66、MassSpecGym 30.22→45.12，说明「指纹一致度」本身就是有效的同式排序信号，误差主要来自指纹预测而非打分函数。【事实，[GEMS](https://arxiv.org/abs/2610.23617) §5.6】
- **检索侧证据**：MassSpecGym 官方审计 v2 Table 3（公式已知）MIST 4M-aug R@1/R@20 = 40.31%/73.02%。【事实，[arXiv 2606.19624v2](https://arxiv.org/html/2606.19624v2)】

### 2.2 SIRIUS 6 / CSI:FingerID / COSMIC

- **预测什么**：SIRIUS 先从 precursor 质量分解出分子式，再为每个公式算碎片树；CSI:FingerID 用碎片树预测分子指纹并给数据库候选打分/排序；COSMIC 是 2021 年提出的**置信度分数**（不是独立结构生成器），校准 CSI:FingerID 注释正确的概率；SIRIUS 6 还含 combinatorial fragmentation 与 MSNovelist。
- **训练数据**：官方文档明说训练覆盖多仪器类型；本队 069 轮核查其指纹模型训练集含 **NIST + MassBank + GNPS**。【事实，[SIRIUS 文档](https://v6.docs.sirius-ms.io/advanced-background-information/)、069 轮核查】
- **公开权重/运行**：模型随 SIRIUS 发布，本地 CLI/Java 可离线跑（conda-forge 已有 6.5.4）。【事实，[conda-forge sirius-ms](https://anaconda.org/conda-forge/sirius-ms)】
- **许可**：客户端 AGPL-3.0；CSI:FingerID/CANOPUS/MSNovelist 的官方 web 服务「仅学术研究与教育」，再分发/商用受限，商业需 Bright Giant 订阅。【事实，[sirius README](https://github.com/sirius-ms/sirius)、[Wikipedia 许可摘要](https://en.wikipedia.org/wiki/SIRIUS_(software))】
- **同分异构体能力**：CSI:FingerID 的设计目标就是给同式数据库候选排序，COSMIC 进一步给置信度；方法上最对口。【事实，[COSMIC 2021](https://doi.org/10.1038/s41587-021-01045-9)】
- **判决**：**NIST 条款 + 学术-only 服务 → 不进比赛**。SIRIUS 的纯质量分解（规则算法）可以单独借用，但那不提供异构体判别信号。

### 2.3 CFM-ID 4.0

- **预测什么**：**结构 → 预测 ESI-MS/MS 谱**（竞争碎裂 + 机器学习强度），再把实验谱与每个候选的预测谱做相似度，输出候选排序；正/负离子都支持。
- **训练数据**：Metlin（非 NIST）。【事实，069 轮核查 + [ACS 2021](https://pubs.acs.org/doi/10.1021/acs.analchem.1c01465)】
- **公开权重/运行**：是，Docker 镜像 `wishartlab/cfmid`，可离线；官方也有 web server。【事实，[Docker Hub](https://hub.docker.com/r/wishartlab/cfmid)、[NAR 2022](https://academic.oup.com/nar/article/50/W1/W165/6591530)】
- **许可**：代码 LGPL-2.1（069 轮核查；镜像内资产需再确认）。主办方明确放行「CFM ID 4 is fine」。【事实，Kaggle discussion 743774，转引自 `ops/remote-workflow.md`】
- **同分异构体能力**：位置异构/骨架异构的碎裂路径与强度确实不同，预测谱能提供独立于指纹的证据；但 CFM-ID 4 比 3.0 计算量大很多，论文没有专门的异构体基准，实际判别力弱于 ICEBERG/GLACIER。【事实（计算量、无专门基准）+ 🔍（相对强弱）】
- **成本**：单候选单次预测约 1–5 s CPU；400×25 = 10k 次 → 3–14 h，4 核并行 ~1–3.5 h。**全量不可行，只能 top-3 或预计算库内候选。**🔍

### 2.4 MetFrag（含 in-silico fragmentation 打分）

- **预测什么**：候选结构 → 规则化 in-silico 碎片（可选 ML 碎片注释），把碎片 m/z 与实验峰匹配，输出候选分数；不训练谱→结构模型。
- **权重**：无（规则法）。**许可**：LGPL-2.1+。**运行**：Java CLI，离线。【事实，[MetFrag home](https://ipb-halle.github.io/MetFrag)】
- **同分异构体能力**：诊断碎片/中性丢失能分开一部分位置异构，但打分主要看峰是否命中，忽略强度与竞争碎裂；对 Tanimoto 0.47 的近似异构体对通常给不出稳定区分。【事实（方法）+ 🔍（对本赛的预期）】
- **现状**：本管线已有 `frag_rescore.py`（MetFrag-lite，Apache-2.0），再引入完整 MetFrag 的边际收益低。【本队，069 轮】

### 2.5 MS2DeepScore / Spec2Vec / MS2DeepScore-based embedding

- **预测什么**：MS2DeepScore 是 Siamese 网络，输入**两条谱**，输出结构 Tanimoto 预测；Spec2Vec 是峰级 word2vec 嵌入，谱-谱余弦。【事实，[MS2DeepScore 2021](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00558-4)、[2.0/跨离子模式 2026](https://www.nature.com/articles/s41467-026-69083-y)】
- **权重**：是。Zenodo 13348683（`ms2deepscore_model.pt`）、17826815（dual-mode，≥2.6）、18255344（2.5.3 代码）；训练集 GNPS/MoNA/MassBank/MSnLib，不含 NIST。【事实，[Zenodo 13348683](https://zenodo.org/records/13348683)、069 轮】
- **许可**：代码 Apache-2.0；权重记录多标 CC BY 4.0（本次 Zenodo 页面打不开，需以记录页为准）。【事实（代码）+ 待核（权重）】
- **同分异构体能力**：**弱**。它预测的是结构相似度，不是同一性；已知局限是不区分立体异构（综述明确列出），同式近似异构体的 Tanimoto 本来就只有 ~0.47，模型 RMSE ~0.15，排序信噪比很低。更关键：候选侧没有参考谱，**无法直接给 25 个候选打分**。所谓 “MS2DeepScore embedding” 只是把谱编码成向量做 kNN，仍是谱-谱，不改变这个限制。【事实（任务形式、RMSE、局限）+ 🔍（对本赛）】

### 2.6 DreaMS（含 embedding 与 kNN）

- **预测什么**：谱 → 1024 维自监督嵌入（masked peak + retention order 预训练），用于谱-谱相似、kNN 库检索、分子网络；GeMS 百万级谱预训练，DreaMS Atlas 2.01 亿谱。【事实，[GitHub](https://github.com/pluskal-lab/DreaMS)、[Nat Biotech 2025](https://www.nature.com/articles/s41587-025-02663-3)】
- **权重**：是，Zenodo 10997887，`ssl_model.ckpt` 1.4 GB、`embedding_model.ckpt` 1.2 GB；代码 MIT、权重 CC BY 4.0；主办方书面放行。【事实，069 轮核查 + [Zenodo](https://zenodo.org/records/10997887)】
- **同分异构体能力**：嵌入编码的是「结构相近」而非「结构相同」，对同式近似异构体天然不敏感；本队 072a 用 DreaMS kNN 做补位/融合，公榜 0.492 中性。**没有候选侧参考谱时不能给候选打分。**【本队 + 事实（论文定位）】

### 2.7 MSGNN / MS-BART / SpecGNN / DeepMASS / MSBERT 一类

- **MS-BART**（2025-10）：谱 → SAFE 结构序列的 BART 生成模型，代码 `OpenDFM/MS-BART`、权重在 Figshare；需要配 MIST 编码器；是 de novo 生成，不是候选打分。README 自身还列了待修路径/beam 参数，谱源需核。【事实，[arXiv 2510.20615](https://arxiv.org/abs/2510.20615)、[GitHub](https://github.com/OpenDFM/MS-BART)】
- **MSBERT**（2024）：BERT 式谱嵌入（mask learning + contrastive），用于库检索/相似度；仓库内含 `model/MSBERT.pkl`，可直接加载；许可未在页面明示（待核）。同 MS2DeepScore，需要候选参考谱。【事实，[ACS Anal Chem 2024](https://pubs.acs.org/doi/10.1021/acs.analchem.4c02426)、[GitHub](https://github.com/zhanghailiangcsu/MSBERT)】
- **DeepMASS v2**（2026）：Spec2Vec 嵌入 + HNSW 检索 + 指纹化学空间定位，给植物代谢物做结构注释；GUI 随包发布，权重/许可未核；排序依赖「与预测结构邻居的接近度」，不是逐候选谱打分，对同式近似异构体不敏感。【事实，[Plant Communications 2026](https://www.cell.com/plant-communications/fulltext/S2590-3462(26)00284-1)、[GitHub](https://github.com/hcji/DeepMASS2_GUI)】
- **MSGNN**：公开文献里这个名字多指**代谢通路预测**的多尺度 GNN，与 MS2 无关；MS2 侧的 GNN 谱预测代表是 MassFormer、GrAFF-MS、3DMolMS（`ms-pred` 内置基线）。不要把 MSGNN 当 MS2 模型。【事实，[PubMed 38127612](https://pubmed.ncbi.nlm.nih.gov/38127612/)、[ms-pred](https://github.com/coleygroup/ms-pred)】
- **SpecGNN**：JACS 2025 的 TranSpec/SpecGNN 是**振动光谱（vibrational spectra）**的双向结构↔谱模型，不是 MS2；名字相同但任务不同，别浪费时间。【事实，[JACS 2025](https://pubs.acs.org/doi/pdf/10.1021/jacs.5c05010)】

### 2.8 碎片树（FT）对齐

- **预测什么**：不做机器学习。给定谱 + 分子式，用规则/组合优化推出一棵碎片树（节点是碎片离子式，边是中性丢失）；两棵树做局部对齐，得到结构相似度；也可用于候选结构的理论碎裂树与实验树的一致性打分。
- **权重**：无。**许可**：算法公开；SIRIUS 实现是 AGPL，自己实现可行。**成本**：DP/ILP，逐对秒级，10k 对即小时级。【事实，[Fast alignment of fragmentation trees](https://academic.oup.com/bioinformatics/article/28/12/i265/267717)】
- **同分异构体能力**：**经典上最强的一类证据**。碎片树能区分结构异构体，论文用前列腺素异构体对做过展示；但依赖高质量碎片树，对低质量/低强度谱和 timsTOF 的噪声敏感，且没有可直接下载的「异构体判别权重」。【事实，[Fragmentation trees for structural characterisation](https://europepmc.org/article/MED/22956319)】

### 2.9 专门做 isomer discrimination 的新工作（2024–2026）

- **GEMS**（2026-10，Cornell）：把「已知分子式的结构解析」建模为**异构图上的学习搜索**；边是保分子式、保价态的 2-switch 键编辑；策略网络把候选指纹位映射到原子/键，比较 MIST 预测指纹，beam search 多起点；最终按指纹一致度 + 多起点共识排序。NPLIB1 目标从 PubChem 初始化中剔除后 Hit@1/Hit@10 = 29.64%/50.81%，MassSpecGym = 30.22%/41.33%；803 张 NPLIB1 端到端 3.3 min（对比 FRIGID 26.2 h）。**代码/权重「upon publication」才发布，现在拿不到。**【事实，[arXiv 2610.23617](https://arxiv.org/abs/2610.23617)】
- **LSM-MS2**（2025-10）：大规模谱基础模型，声称在「挑战性异构体」识别上比现有方法高 30%，复杂生物样本多 42% 正确注释；但它是商业产品（Matterworks/Pyxis），**没有公开权重**，论文许可还是 CC BY-NC-ND。**不可用。**【事实，[arXiv 2510.26715](https://arxiv.org/abs/2510.26715)】
- **MBGen**（AAAI 2026）：多体增强扩散，摘要声称「enabling accurate de novo generation and isomer differentiation」；属于生成路线，代码/权重未核，推理成本高。**观察，不落地。**【事实，[arXiv 2602.01643](https://arxiv.org/abs/2602.01643)】
- **FOAM**（2026）：从检索/生成候选出发做保分子式随机变异 + crossover，用 ICEBERG 预测谱选择；是「搜索 + 前向谱打分」，不是新判别模型；其价值是提示我们：**前向谱打分 + 局部编辑**可以作为候选扩展，但本队召回已 100%，不需要扩展。【事实，[GEMS 引用 FOAM](https://arxiv.org/abs/2610.23617)】

### 2.10 ICEBERG 2.1 / GLACIER（补充分数器，短名单要用）

- **预测什么**：ICEBERG 是两阶段（碎片图生成 + 强度预测），**结构 → 碎片图/预测谱**；GLACIER 是单阶段多断点碎裂，速度更快。二者都输出可比较的预测谱，按熵相似度/余弦给候选排序。
- **权重**：MassSpecGym 训练的 ICEBERG/GLACIER 权重公开（Dropbox）；NIST'23 权重需 NIST 许可，**不能用**。代码开源，`ms-pred` 内置 structural elucidation notebook：对 PubChem 同式候选逐个预测谱，再按 entropy similarity 排序。【事实，[ms-pred README](https://github.com/coleygroup/ms-pred)、[ICEBERG 2.0 bioRxiv](https://www.biorxiv.org/content/10.1101/2025.05.28.656653v1)】
- **同分异构体能力**：碎片级预测，天然对「取代位置/环化方式」敏感；官方明说用于同式候选的结构解析。比 CFM-ID 新、快、准（ICEBERG 2.1 比 2.0 约 3× 加速）。【事实（定位/速度）+ 🔍（对本赛判别力）】
- **限制**：MSG 权重以正离子为主，负离子/加合物覆盖需实测；碰撞能缺失时要边缘化或网格平均。

## 3. 可行性表

> 成本列均为 **400 分子 × 25 候选 = 10k 次候选打分** 的粗估；🔍 表示按参数量/批大小/官方 demo 时间推算，未实测。所有「离线」均指模型与权重已打包进 notebook 输入，不依赖运行时联网。

| 方法 | 任务类型 | 训练数据 | 公开权重 | 许可证 | 离线 | 单 T4 估算 | 真能区分同分异构体？ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **MIST MSG/CANOPUS** | 谱+公式→4096 位指纹概率 | MSG 真谱；MSG+4M 合成；CANOPUS | ✅ 289 MB×2 | 代码 MIT；权重许可未明示 | ✅ | **5–20 min** | **间接但强**：GEMS 全套排序依赖它；真值指纹消融 29.6→56.7 |
| SIRIUS 6 / CSI:FingerID / COSMIC | 公式+碎片树+指纹→候选排序/置信 | NIST+MassBank+GNPS | 随软件 | AGPL；web 学术非商用 | ⚠️ 本地可，模型受限 | 1–7 h+ | **直接**（设计目标含同式候选），但许可/NIST 排除 |
| CFM-ID 4.0 | 结构→预测谱；候选-谱相似 | Metlin | ✅ docker | LGPL-2.1（待核） | ✅ | 全量 3–14 h；top-3 20–100 min | **弱–中**：位置异构谱有差异，无专门基准 |
| MetFrag 2.5.6 | 结构→规则碎片；峰匹配分 | 无（规则） | 无 | LGPL-2.1+ | ✅ | 8–50 min | **弱–中**：诊断碎片可分部分异构；管线已有 lite |
| MS2DeepScore 2.x | 谱-谱→Tanimoto | GNPS/MoNA/MassBank/MSnLib | ✅ 小模型 | Apache-2.0；权重 CC BY（待核） | ✅ | 嵌入 <1 min，但需候选谱 | **否/弱**：预测相似度，不判同一性；立体异构已知不行 |
| Spec2Vec | 峰嵌入；谱-谱余弦 | 公开谱库 | ✅ | Apache/CC BY | ✅ | 轻 | **弱** |
| DreaMS | 谱→1024-d 嵌入；kNN | GeMS 百万谱 | ✅ 1.4/1.2 GB | MIT / CC BY 4.0 | ✅ | 400 谱 <1 min | **弱**：本队 072a 中性；无候选谱无法打分 |
| MS-BART | 谱→SAFE 结构 | MSG 等 | ✅ Figshare | 待核 | ✅ | 15–70 min 生成 | **间接**：能生成异构体，非候选打分 |
| MSBERT | 谱→嵌入；检索 | 公开谱库 | ✅ repo pkl | 待核 | ✅ | 轻 | **弱**：同 MS2DeepScore |
| DeepMASS v2 | 谱→邻居→指纹空间排序 | 百万谱 | GUI 随包 | 待核 | ✅ | 中 | **弱**：结构近邻，不是同一性 |
| 碎片树对齐 | 谱+公式→树；树-树对齐 | 无 | 无 | 自实现；SIRIUS AGPL | ✅ | 小时级 | **直接（经典）**：前列腺素异构体案例 |
| **GEMS** | 公式+谱→异构图搜索/排序 | PubChem 异构对 + 谱 | ❌ 发表后 | 论文 CC BY 4.0 | — | 5–15 min（若有权重） | **直接**：专为同分异构设计 |
| LSM-MS2 | 谱→嵌入/鉴定 | 百万谱 | ❌ 商业闭源 | 论文 NC-ND | ❌ | — | 声称 +30% 异构体识别 |
| MBGen | 谱→结构扩散 | NPLIB1/MSG | 未核 | 待核 | ⚠️ | 长 | 声称 isomer differentiation |
| **ICEBERG 2.1 / GLACIER** | 结构→碎片图/预测谱 | MSG（另有 NIST 权重） | ✅ MSG Dropbox | 代码开源；NIST 权重需许可 | ✅ | **10–60 min** | **中–强**：碎片级，官方用于同式候选排序 |
| MS-GPT | 谱→结构生成 | MSG | ✅ adapter | Apache-2.0 | ✅ | 2.8 s/谱×采样 | **间接**：生成可出异构，非打分 |

## 4. 排序短名单（≤3）

排序依据：新增信息量 × 落地成本 × 许可安全。三项都只改「同式候选内部的排序」，不动召回，不动公式层。

### 4.1 第一优先：MIST 指纹概率做同式候选重排

**接口**：需要候选结构 ✅；需要分子式 ✅（由 precursor/adduct 推断，不用真值）；输出是每个候选一个标量分数，直接进同式组内排序。

**为什么是它**：现有管线已经有一个在比赛数据上训练的指纹头（FPNet/N-1c），而 MIST 是**在 MSG/CANOPUS 上预训练、带公式注意力、输出逐位概率**的另一套指纹；训练数据、损失、输出语义都不同，具备互补性。更关键的是，GEMS 已经证明「指纹一致度」是同式候选排序的有效分数，且瓶颈在指纹预测质量——MIST 是这条线上公开可用、最贴近该用法的冻结模型。【事实 + 🔍】

**集成方式**：

1. **输入**：查询谱峰（m/z、intensity）+ precursor m/z + 加合物；先用质量窗/现有公式推断得到 1–2 个候选分子式（**不要用真值公式**）。
2. **MIST**：每个（谱, 公式）跑一次，得到 `q ∈ [0,1]^4096`；只对正离子、`ION_LST` 支持的加合物启用，其余走底座。
3. **候选侧**：对每个候选结构算 Morgan r2 4096 位（无手性），得到 `x ∈ {0,1}^4096`。
4. **分数**：`s = Σ_i [x_i log q_i + (1-x_i) log(1-q_i)] / 4096`（逐位 Bernoulli 对数似然，长度归一）。同式候选共享公式，长度一致，可直接比。
5. **融合**：在**同分子式组内**做 z-score，`final = base_score + w · z`；`w` 取 0.05–0.3，只在「组内 ≥2 个候选且组内最高分与 base top1 不同」时生效；限制单次最多上移 2 位，保护 base top1 的 margin。
6. **验证**：只在 072d 的 150 分子仪器面板上评（48 个错误、无库命中），按分子 bootstrap 报 ΔMRR、fix/harm；**不要**在 60 分子重排面板上调参（底座 MRR 0.9228，只有 7 个错误，撑不住）。
7. **成本**：400 谱 × ~1.5 公式 ≈ 600 次前向；72.3M 参数、batch 16，T4 上 5–20 min（含公式特征构建）。🔍

**预期**：面板上每把 10 个 rank2 翻到 rank1，ΔMRR = 10×0.5/150 = **+0.033**；若修复 20–40% 的 rank2 错误（8–17 个），面板 +0.027–0.057。公榜误差构成不同，不能线性外推，合理量级 +0.01–0.03。🔍

**止损**：若 fix/harm < 1.5，或 ΔMRR ≤ 0，或只在「公式预测正确」的 47 个里修不动，停止，不要加权重硬调。

### 4.2 第二优先：ICEBERG 2.1（MSG 权重）做前向谱打分

**接口**：需要候选结构 ✅；需要分子式 ❌（可选做同式过滤）；需要加合物与碰撞能（CE 缺失可边缘化）；输出是「预测谱 vs 实验谱」的相似度标量，进同式组内排序。

**为什么是它**：它与指纹是两条独立证据——指纹看「局部环境是否存在」，前向谱看「这个结构能不能解释这些峰和强度」。ICEBERG 的官方 structural elucidation 流程就是「对同式候选逐个预测谱，按 entropy similarity 排序」，与我们的需求一一对应。ICEBERG 2.1 比 2.0 快约 3×，MSG 权重公开，主办方放行 GLACIER/ICEBERG。【事实】

**集成方式**：

1. **输入**：候选 SMILES + 加合物 + 碰撞能（有则用；缺失时在 CE 网格上预测后取最大相似度，或按训练分布边缘化）。
2. **输出**：预测碎片峰（m/z、intensity）；用 `ms_entropy` 的 entropy similarity 与实验谱比，另加「实验 top-10 峰的覆盖率」作为辅助特征。
3. **范围**：只对同式组内 base top-5 打分；10k → 2k 次预测，T4 batch 8，约 10–30 min。🔍
4. **融合**：与 MIST 分数分别 z-score 后线性组合（先单独测，再测两者同时）；`w` 同样小步。
5. **验证**：同一 072d 面板；分层看正/负离子、加合物、CE 是否有标注。
6. **成本**：2k 次预测约 10–30 min；全量 10k 约 1 h。🔍

**预期**：与 MIST 互补；若两者同时有效，融合后可能多修 5–10 个 rank2。**不要**用 NIST'23 权重（许可）。🔍

### 4.3 第三优先：CFM-ID 4（top-3，或库内候选预计算）

**接口**：需要候选结构 ✅；需要分子式 ❌；需要加合物（正/负都支持）；输出预测谱→相似度标量，进同式组内排序。

**为什么是它**：许可最干净、主办方明确放行、正负离子都支持；作为 ICEBERG 的对照/备份，或在 ICEBERG 负离子覆盖不足时补位。

**集成方式**：

1. 仅对同式组内 base top-3 预测（400×3 = 1.2k 次），单核约 20–100 min，4 核约 5–25 min；🔍
2. 如果候选池主要来自固定参考库，可在**离线阶段**对库内候选预计算预测谱并打包；但 275k 全库按 1–5 s/条需要数十 CPU-天，不现实。只有候选 union 在 1–2 万条以内、且可大规模并行时才值得做。🔍
3. 打分用 entropy similarity/cosine，与 MIST、ICEBERG 做 late fusion；
4. 不作为第一优先：精度和速度都不如 ICEBERG，单独用大概率只贡献很小增量。

## 5. 风险与排除清单

### 5.1 只有理论、没有公开权重

| 方法 | 状态 | 处理 |
| --- | --- | --- |
| GEMS | 论文明确「upon publication」发布代码与 checkpoint | 跟踪；现在不能进实验 |
| LSM-MS2 | 商业产品，无公开权重，论文 NC-ND | 排除；只能作为「异构体可提升 30%」的上限参考 |
| MBGen | 生成路线，代码/权重未核 | 观察；即使有权重，生成成本也超预算 |
| DeepMASS v2 | GUI 随包，但不是候选打分器 | 排除 |

### 5.2 许可证/数据来源不允许

- **SIRIUS 6 / CSI:FingerID / MSNovelist**：AGPL 客户端 + web 服务「学术非商用」；指纹模型训练含 NIST。奖金赛按商用处理，**排除**。【事实，069 轮 + SIRIUS README】
- **ICEBERG/GLACIER 的 NIST'23 权重**：需要 NIST 许可；比赛禁止 NIST 及其训练模型（GNPS-NIST14-MATCHES 是明确例外）。**只能用 MassSpecGym 权重**。【事实，ms-pred README + `ops/remote-workflow.md`】
- **MS-BART**：权重谱源需核（是否含 NIST/禁止库）；在核实前不打包。
- **MIST 权重**：代码 MIT，但权重页未展示明确许可；保留原始 Zenodo 元数据与 NOTICE，不用「MIT 全包」表述。
- **MS2DeepScore 权重**：Zenodo 记录许可本次未能打开，打包前确认 CC BY 4.0 或等价条款。

### 5.3 需要联网 / 显存/时间超预算

- **SIRIUS web 服务**（CSI:FingerID/CANOPUS/MSNovelist）必须联网 → 直接排除；本地 CLI 虽可离线，但模型许可仍是问题。
- **GEMS**：Graphormer + beam search（K=5, P=B=16, D=3）在 16 GB T4 上可能吃紧；且无权重的现实问题更大。
- **CFM-ID 全量 10k**：3–14 h，超 6 h 预算；只能 top-3 或预计算。
- **MS-BART/MS-GPT 生成**：按谱生成再打分，成本随采样数线性增长；MS-GPT 100 samples 已 2.8 s/谱（RTX 6000 Ada），T4 更慢。
- **碎片树对齐**：逐对 DP，10k 对小时级；且需要可靠公式与碎片树，工程量大。

### 5.4 任务不匹配（最容易踩的坑）

- **MS2DeepScore / Spec2Vec / DreaMS / MSBERT / DeepMASS**：都是「谱-谱」或「谱-库」方法，候选侧必须有参考谱；我们的 25 个候选多数没有参考谱 → **不能直接给候选打分**。把它们当同分异构体判别器会重复 072a 的中性结果。
- **MSGNN**：同名的是代谢通路 GNN；**SpecGNN**：同名的是振动光谱模型。两者都不是 MS2 同分异构体工具。
- **谱→结构生成**（MS-BART、MS-GPT、GEMS、MBGen）：能产生新结构，但不能原生地给「已有候选」排序；除非做候选似然/编辑搜索，否则不解决 top1/top2 互换。

### 5.5 数据域风险

- **离子模式**：MIST 只支持 7 种正离子；本赛含负离子/timsTOF，覆盖不全。应在实验里显式报告「可支持谱占比」，不能把不支持当模型无信息。
- **天然产物域移**：MSG/CANOPUS 以代谢物为主，本赛是天然产物；2026 年的系统评估指出，指纹预测在 scaffold split 下会大幅退化，最近邻检索甚至能追平深度模型——这与本队 10 次指纹重训失败一致，支持「不重训、只用冻结预训练 + 轻校准」的策略。【事实，[Nearest-neighbour baselines](https://arxiv.org/abs/2610.02249)、[Why ML still can't crack small molecule structures](https://www.nature.com/articles/s42255-026-01544-6)】
- **公式不确定性**：MIST/GEMS/FT 都假设公式已知；比赛没有真值公式。必须用 precursor/adduct 推公式，并让公式后验参与分数，否则会把「公式错」误判成「模型弱」。

### 5.6 评分口径风险

- 只在**同分子式组内**重排；跨公式重排会破坏已正确的质量层（072c 已证明 mass prior 无空间）。
- 保护 base top1：若 base top1 与组内最高分差距小于阈值，不动；避免 082c 式「误升位」。
- 只看 MRR@25，不看 R@1；面板上 fix/harm 必须按分子配对统计。

## 6. 建议的最小验证（不新增测试体系，手动执行）

1. **冻结面板**：复用 072d 的 150 分子/681 谱/25 候选与 base 排序，落盘 `formula`、`adduct`、`base_rank`、`candidate_smiles`。
2. **单模型 A/B**：MIST 分数单独加入，只对正离子可支持谱；在面板上算 ΔMRR、fix/harm、按公式正确/错误分层。预算 ≤20 min。
3. **通过后再叠 ICEBERG**：top-5 前向谱，预算 ≤60 min；若 A 已正、B 不再增，停。
4. **公榜只推一版**：`w` 在面板上选定后冻结，不在公榜上扫；推送前做一次 smoke（12 个可见分子）确认 `reranked` 数量与 `skipped` 原因。
5. **明确失败信号**：fix/harm < 1.5、ΔMRR ≤ 0、或提升只来自 1–2 个分子 → 判为噪声，停止。

## 7. 主要来源

- MIST 官方实现与权重：[github.com/coleygroup/MIST](https://github.com/coleygroup/MIST)、[Zenodo 23303289](https://zenodo.org/records/23303289)；原论文 [Nat Mach Intell 2023](https://doi.org/10.1038/s42256-023-00708-3)
- GEMS：[arXiv 2610.23617](https://arxiv.org/abs/2610.23617)
- LSM-MS2：[arXiv 2510.26715](https://arxiv.org/abs/2510.26715)
- MBGen：[arXiv 2602.01643](https://arxiv.org/abs/2602.01643)
- ICEBERG/GLACIER：[github.com/coleygroup/ms-pred](https://github.com/coleygroup/ms-pred)、[ICEBERG 2.0 bioRxiv](https://www.biorxiv.org/content/10.1101/2025.05.28.656653v1)
- CFM-ID 4：[ACS Anal Chem 2021](https://pubs.acs.org/doi/10.1021/acs.analchem.1c01465)、[Docker](https://hub.docker.com/r/wishartlab/cfmid)
- MetFrag：[ipb-halle.github.io/MetFrag](https://ipb-halle.github.io/MetFrag)
- MS2DeepScore：[Zenodo 13348683](https://zenodo.org/records/13348683)、[跨离子模式 2026](https://www.nature.com/articles/s41467-026-69083-y)
- DreaMS：[GitHub](https://github.com/pluskal-lab/DreaMS)、[Zenodo 10997887](https://zenodo.org/records/10997887)、[Nat Biotech 2025](https://www.nature.com/articles/s41587-025-02663-3)
- MS-BART：[arXiv 2510.20615](https://arxiv.org/abs/2510.20615)、[GitHub](https://github.com/OpenDFM/MS-BART)
- MSBERT：[ACS Anal Chem 2024](https://pubs.acs.org/doi/10.1021/acs.analchem.4c02426)、[GitHub](https://github.com/zhanghailiangcsu/MSBERT)
- DeepMASS v2：[Plant Communications 2026](https://www.cell.com/plant-communications/fulltext/S2590-3462(26)00284-1)、[GitHub](https://github.com/hcji/DeepMASS2_GUI)
- 碎片树对齐：[Fast alignment of fragmentation trees](https://academic.oup.com/bioinformatics/article/28/12/i265/267717)、[Identifying unknowns by aligning FT](https://doi.org/10.1021/ac300304u)
- SIRIUS 6：[文档](https://v6.docs.sirius-ms.io/)、[GitHub](https://github.com/sirius-ms/sirius)、[COSMIC](https://doi.org/10.1038/s41587-021-01045-9)
- 指纹预测的系统性困难：[Nearest-neighbour baselines](https://arxiv.org/abs/2610.02249)、[Nat Metab 2026](https://www.nature.com/articles/s42255-026-01544-6)
- MassSpecGym 官方审计：[arXiv 2606.19624v2](https://arxiv.org/html/2606.19624v2)
- 本队内部依据：`rounds/072-scale/progress.md`（072d 召回/MRR/47-of-48 同分异构）、`rounds/069-new-signals/research/069-frontier-models.md`（许可与权重核查）、`ops/remote-workflow.md`（NIST 条款与主办方放行）
