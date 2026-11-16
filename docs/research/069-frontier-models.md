# 069 前沿模型调研：哪一个能作为「新信号」把 0.492 推到 0.50+

调研日期：2026-11-16（北京时间）。范围：2023–2026 年可实际接入的 MS/MS 结构解析模型与工具。只做资料核查，未运行内核、未提交、未改动 068a。

**结论先行**：能同时满足「带来现有通道没有的信息、许可干净、9 小时断网可跑」这三条的，只有一条路线——**DreaMS 谱嵌入 + MSAlign 式「谱–分子对比对齐」候选重排**。它的下限是 +0.000，现实预期 **+0.003～0.006**，乐观 +0.008；单靠它一步到 0.50 的概率不高。其余候选（MS2Mol、MSNovelist/CSI:FingerID、CFM-ID 4、MIST+MolForge、DiffMS/FRIGID）要么拿不到权重，要么被 NIST 条款排除，要么与现有通道同向、历史上已证伪。

---

## 1. 规则与许可基线（先固化可引用的事实）

许可判断只认两类出处：比赛官方页面/规则，以及 host 在 discussion 里的书面裁定。

### 1.1 比赛规则里与模型相关的三条

| 条款 | 内容 | 出处 |
| --- | --- | --- |
| 2.5.a.1 | 获奖方案需以 OSI 认可的开源许可（MIT 等）授权，且不得限制商业使用 | [rules §2.5](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules) |
| **2.5.a.3** | **若使用了许可不兼容的输入数据或预训练模型，不必对该数据/模型本身授予开源许可** | 同上（关键豁免条款） |
| 2.6.a / 2.6.b | 外部数据必须「公开可得、对所有参赛者同等可得、无成本或极低成本」；外部数据与模型默认允许，除非 host 明确禁止 | 同上 |

第 2.5.a.3 条决定了策略：**CC-BY-NC 的权重、AGPL 的代码都能进提交**（只要不在获奖包里再分发它们），真正硬的门槛是「不公开可得」——这正是 NIST 被禁的原因。

### 1.2 host 的书面裁定（比任何二手总结都可靠）

| 对象 | 可用 | 原话 / 出处 |
| --- | --- | --- |
| `train.parquet` 及其训练产物 | 是 | "can be treated effectively as open-source and any model trained on it is fine, redistributed artifacts are fine" — [745841](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745841) |
| CC-BY-NC 类开源数据 | 是（2026-10-17 起） | "solutions can use external data that is open sourced with noncommercial provisions" — [750330](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/750330) |
| MassSpecGym 及其训练模型 | 是 | "we consider MassSpecGym and models trained on it and released with permissive licenses as fair game" — [742991](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742991) |
| GLACIER / ICEBERG | 是 | "MassSpecGym is allowed, so is GLACIER"；"GLACIER/ICEBERG, DreaMS and ChEBI/LIPID MAPS are all fine" — [744338](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744338)、[745841](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745841) |
| **DreaMS 权重** | 是 | "DreaMS weights are allowed" — [745604](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745604) |
| CFM-ID 4 | 是 | "CFM ID 4 is fine" — [743774](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743774) |
| GNPS-NIST14-MATCHES | 是（谱库匹配 + 训练） | [745832](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745832) |
| Enveda-180 Zenodo 数据 | 是 | [745148](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745148) |
| **NIST 及其派生（含模型权重）** | **否** | "NIST is not permitted ... the data including derivatives such as model weights cannot be redistributed"；"any solution using or trained on NIST will be disqualified" — [750330](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/750330)、[744470](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744470) |

派生出的四条硬规则，后面每个模型都按它判：

1. 权重必须**公开可下载**（免费注册/接受条款算可接受，但要能离线打包）。
2. **任何训练集里含 NIST 的权重不能用**，包括「在 NIST 上预训练、只在公开集上微调」的 checkpoint。
3. CC-BY-NC、CC-BY-SA、AGPL 可用，但获奖包里不能重新分发这些资产。
4. 代码许可与权重许可是两件事，要分开写。

---

## 2. 当前管线已经用掉了什么（先排除重复劳动）

从 `notebooks/068a/enveda-068a-n1c-lam035.ipynb` 内嵌的冻结管线（`FROZEN_B64`，59 个 cell，277k 字符代码）核对得到：

| 通道 | 实际做法 | 信息类型 | 备注 |
| --- | --- | --- | --- |
| 候选池 | 约 711k = train 的 275k 结构 + COCONUT，带 FPNet 指纹与碎片表 | 结构库 | 所有重排都在这个池内 |
| Engine 1 | FPNet（A+B+fold-safe）指纹 logits，v4n LightGBM 排序器取 top 60 | 谱 → 指纹 | 核心通道 |
| 物理碎片校核 | `frag_rescore.py`，无训练的 MetFrag-lite（Apache-2.0） | 碎片一致性 | 只做门控/微调 |
| Engine 2 / 前向 | ICEBERG + GLACIER（ms-pred 的 **MassSpecGym 权重**，非 NIST'23） | 结构 → 预测谱 | 已有 |
| PubChem 通道 | FPNet logits z → 105M PubChem ±10 ppm 窗口 → pass1 ECFP4 粗排 5000 → pass2 全指纹 → 取 25 个不在池内的 | 谱 → 指纹 | 由 `probe_core2` 实现 |
| N-1c | 测试集内部 modified cosine 分子网络重排，τ=0.6、≥6 peaks、λ=0.35 | 谱–谱 | 065a/068a 的唯一改动 |
| 外部谱库 | MassBank-harmonized 等，TAU≥0.80 | 谱–谱 | 高精度低覆盖 |
| **DreaMS** | `hengck23-dreams-enveda-casmi26` 资源，**只用于「空槽填充」**：把质量兼容的 DreaMS 结构塞进候选不足的位置 | 谱 → 结构（kNN） | **没有分数，不进重排** |

三个关键事实：

- **DreaMS 目前只当补位器**，pipeline 里没有任何一处用它给候选打分。它的结构候选来自 Enveda-180 与 MassSpecGym 两个库的嵌入（该 Kaggle dataset 里是 `Enveda180_DreaMS_embeddings.npy` 4.7 GB / `MassSpecGym_DreaMS_embeddings.npy` 946 MB），**没有覆盖比赛自己的 2.5M 训练谱**。
- 现有 FPNet 学的是**指纹回归**（谱 → 4096 位指纹，用点积给候选打分），不是「候选对比」目标。
- 外部谱库/analog 通道与 FP 通道是两种独立信息；DreaMS 若接入，属于第三种（学习式谱–结构相似度），不是替换前两者。

---

## 3. 对比表

「9h 可行性」按最终 notebook 断网、T4×2、9 小时、2.5M 训练谱预算判。最后一列只写有出处的文献数字；对本赛公榜的推测统一放在 §6，且明确标注为推测，不混进表里。

| 模型 / 工具 | 年份 | 输入 → 输出 | 预训练权重 | 许可（代码 / 权重） | 硬件 | 9h 断网可行性 | 与现有管线互补性 | 文献证据（原文数字） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **DreaMS**（SSL / 相似度） | 2024–2025 | 峰表 + precursor → 1024-d 嵌入 | 是：Zenodo 10997887（`ssl_model.ckpt` 1.4 GB、`embedding_model.ckpt` 1.2 GB） | MIT / **CC BY 4.0** | GPU 友好，CPU 亦可 | ✅ 2.5M 谱嵌入需离线预算（估 20–40 min T4） | **高**：与 FP 回归、谱库余弦都不同 | 零样本嵌入余弦已超 MS2DeepScore；微调后库检索与 analog 检索 AUROC 均超 MS2DeepScore / modified cosine / spectral entropy（[Nat Biotech 2025 Fig.4a–c](https://www.nature.com/articles/s41587-025-02663-3)） |
| DreaMS-Mol | 2026α | 谱 → 直接分子 | **未发布** | — | — | ❌ | 潜在最高，但拿不到 | 作者 2026-01 称「今年计划发布，目前只有第一版」（[issue #32](https://github.com/pluskal-lab/DreaMS/issues/32)） |
| **MSAlign**（DreaMS×MolDeBERTa 对齐） | 2026 | 谱 + 候选 SMILES → 相似度 | 是：Zenodo 23372807（MassSpecGym 上训练） | **MIT** 代码 / 权重随论文发布；MolDeBERTa 为 HF gated | 需 CUDA | ✅（可离线打包） | **最高**：现成的「候选对比」打分器 | MassSpecGym formula split（K=256, ±10 ppm）：R@1 **47.5**；+分数融合 **58.2**；R@20 **87.0 / 92.8**；对照 MIST 41.4、bin 谱+指纹 34.3（[arXiv 2605.19752](https://arxiv.org/abs/2605.19752) Table 3/4） |
| MS2Mol | 2023 | 谱 → SMILES | **否**（论文承诺发布，未兑现） | 未发布 | — | ❌ | — | EnvedaDark：top-1 结构 62.1% 有意义相似、21.4% 近似匹配；top-25 87.6% / 36.8%（[ChemRxiv](https://doi.org/10.26434/chemrxiv-2023-vsmpx-v4)） |
| MSNovelist | 2022 | CSI:FingerID 指纹 → SMILES | 仅随 SIRIUS 6 提供 | AGPL-3.0 / 学术 | CPU，~5 min/谱 | ❌ | 低 | 25% top-1 精确（3,863 GNPS 谱）（[Nat Methods](https://www.nature.com/articles/s41592-022-01486-3)） |
| **SIRIUS / CSI:FingerID** | 2019–2026 | 谱 → 公式 + 碎片树 + 指纹 | 随 SIRIUS 发布 | AGPL-3.0 / **学术非商用** | Java，CPU 重 | ❌（许可 + NIST） | — | 训练集含 **NIST + MassBank + GNPS**（[SIRIUS 文档](https://v6.docs.sirius-ms.io/advanced-background-information/)） |
| CFM-ID 4 | 2021 | 结构 → 预测谱（正/负模式） | 是（随源码） | **LGPL-2.1** / 同 | CPU，<1 min/分子/次 | ⚠️ 400×25=10k 次预测 ≈ 数天 | 低（与 ICEBERG 同向） | 训练数据为 **Metlin**（非 NIST）（[ACS 2021](https://pubs.acs.org/doi/full/10.1021/acs.analchem.1c01465)） |
| MetFrag | 2016→ | 谱 + 候选 → in-silico 碎片匹配分 | 无（规则法） | 仓库未声明许可 | Java，CPU | ⚠️ 逐候选昂贵 | 已被 MetFrag-lite 覆盖 | — |
| MS2DeepScore | 2021–2024 | 谱 + 谱 → 结构 Tanimoto 预测 | 是：Zenodo 13897744 | **Apache-2.0 / CC BY 4.0** | CPU/GPU 轻 | ✅ | 中：可替换 N-1c 的 modified cosine | 权重训练于 **GNPS + MoNA + MassBank + MSnLib**（>500k 谱），不含 NIST |
| Spec2Vec | 2021 | 谱 → 峰嵌入 / 谱–谱相似度 | 是（MS2Query 包内，Zenodo 12606475） | Apache-2.0 / CC BY 4.0 | CPU 轻 | ✅ | 低（DreaMS 严格更强） | 同上，公开谱库训练 |
| **MIST**（+MolForge） | 2023–2025 | 峰-分子式对 → 4096 位 Morgan 指纹；MolForge 指纹 → SMILES | 是：Zenodo 11580401 / DiffMS 15122968 | **MIT / CC BY 4.0**；MolForge **CC BY-NC 4.0** | GPU | ✅ | 中：新增「生成候选」，但同族信号 | 修正批推理 bug 后 MassSpecGym top-1 **10.73%**、top-10 **14.48%**（原报 31.75%/40.55% 为 bug 虚高）（[复现仓库](https://github.com/harrylaucngd/MIST-MolForge)） |
| DiffMS | 2025 | 谱 → 扩散生成分子 | 是 | MIT | GPU | ✅ | 低 | MassSpecGym top-1 2.3%、top-10 4.25% |
| FRIGID | 2025 | 谱 + ICEBERG 引导 → SAFE 序列 | 是 | Apache-2.0 | GPU | ✅ | 已试过，0 命中 | 本项目 001–064 记录：4000 次生成对给定分子式真值命中 0 |
| ICEBERG 2.1 / GLACIER | 2024–2026 | 结构 → 预测谱 | 是（MSG 权重 MIT；NIST'23 权重需 NIST 授权） | MIT | GPU | ✅ | 已有，可提速换更大池 | 2.1 约 3× 加速（[ms-pred](https://github.com/coleygroup/ms-pred)） |
| Spectraverse | 2025–2026 | 公开谱库（488,630 谱 / 44,237 分子）+ 候选集 | 数据 | **CC BY 4.0** | — | ✅ | 训练数据补充，非模型 | [Anal. Chem. 2026](https://pubs.acs.org/doi/full/10.1021/acs.analchem.5c06256) / [Zenodo](https://zenodo.org/records/17870921) |
| Enveda PRISM | 2024 | 谱 → 表示/性质 | **否**（专有） | 专有 | — | ❌ | — | 仅公司博客 |

---

## 4. 逐个评估

### 4.1 DreaMS / DreaMS-DR —— 唯一被 host 点名放行、且信息类型是新的

- **是什么**：7 层 transformer（116M 参数），在 GeMS（GNPS/MassIVE 挖出的 2,400 万张高质量谱）上做掩码峰重建 + 保留时间序自监督；输出 1024 维嵌入。微调版本（`embedding_model.ckpt`）是**对比学习后的相似度模型**，正是库检索/analog 检索用的那个。
- **权重与许可**：代码 MIT（[GitHub](https://github.com/pluskal-lab/DreaMS)），权重 **CC BY 4.0**（[Zenodo 10997887](https://zenodo.org/records/10997887) 明写 "License: Creative Commons Attribution 4.0"）。CC BY 允许商用、允许再分发（署名），比 MIT 代码更宽松的部分在于不要求开源衍生代码。host 另有书面放行。
- **关键证据**：论文 Fig.4a 零样本嵌入余弦在预测 Morgan Tanimoto / MCES 上**已超过 MS2DeepScore**（后者是专门用成对谱训练出来的）；微调后 Fig.4b/4c 在库检索与 analog 检索（AUROC，MCES≤k）上超过 MS2DeepScore、modified cosine、spectral entropy，且在低质量谱上退化更小。Fig.4e：DreaMS 微调预测 Morgan 指纹与 **MIST 指纹模型打平**（在 MIST 的化合物库检索基准上）。
- **「DreaMS-DR」的核查结果**：公开检索（GitHub issues、作者访谈、Zenodo、HF）**没有**名为 DreaMS-DR 的独立发布物。作者口径里正在做的是 **DreaMS-Mol**（直接出分子，alpha 版，计划 2026 年内发布，目前未发）。因此「DR」不能作为可接入组件写进方案；本文按「DreaMS 嵌入 + 自训 denoising/对比头」处理。
- **9h 可行性**：模型小（1.2–1.4 GB checkpoint），2,300 张谱在 M1 上 <1 分钟（官方文档），T4 上 2.5M 谱估 20–40 分钟。**但必须离线预计算**成 dataset（2.5M×1024 fp16 ≈ 5.1 GB；按 275k 结构聚合后 ≈ 563 MB），不要放进提交 notebook 现算。
- **互补性**：高。它给的是「谱 → 结构空间的距离」，与 FPNet 的「谱 → 指纹位」、与谱库的「谱 → 谱余弦」都不重合。

### 4.2 MSAlign —— 目前唯一「现成、MIT、且在候选检索上量化更强」的打分器

这是本次调研最重要的发现，也是 §6 主方案的骨架。

- **做法**：冻结 DreaMS（谱侧）与 MolDeBERTa（分子侧），只训两个轻量 MLP 投影，用 **candidate-based InfoNCE**（负样本 = 同质量窗口内的候选分子）对齐；再做分数级融合（4 个对齐分数 + 3–4 个质量分数）。
- **量化结果**（MassSpecGym formula split，K=256 个 ±10 ppm 的 PubChem 候选，正是「同分子式异构体排序」这个难点）：

  | 方法 | R@1 | R@5 | R@20 |
  |---|---:|---:|---:|
  | DeepSets（指纹回归） | 4.7 | 13.0 | 27.5 |
  | JESTR | 13.9 | 32.5 | 57.0 |
  | Emb-Cos | 42.7 | 65.2 | 80.4 |
  | **MSAlign（DreaMS×MolDeBERTa）** | **47.5** | 71.3 | 87.0 |
  | MSAlign + 分数融合 | **58.2** | 80.7 | 92.8 |
  | MIST | 41.4 | 60.8 | 76.2 |
  | FLARE | 32.8 | 61.8 | 82.3 |

- **两个对我们直接有用的消融**：
  - 训练目标：回归（ms→mol）R@1 **16.7** vs candidate 对比 **47.5**（Table 5）。这与我们「FPNet 是回归目标」的现状直接对位。
  - 分子侧表示：Morgan 指纹 R@1 38.6 vs MolDeBERTa 47.5（Table 4）。也就是说，**换掉分子侧的表示，比换谱侧模型更值钱**。
- **许可**：代码 MIT（[仓库](https://github.com/KrzakalaPaul/MSAlign-NeurIPS2026)），预训练 checkpoint 在 Zenodo 23372807，训练数据是 MassSpecGym（host 已放行）。MolDeBERTa 是 HF gated（免费、需接受条款），可离线打包。
- **风险**：MassSpecGym 上训练的 checkpoint 直接搬到本赛（timsTOF、天然产物、10 种加合物）会掉点；正确做法是**用同一套代码在 train.parquet 上重训投影层**（冻结编码器，成本极低，论文报 <2 min/epoch）。

### 4.3 MS2Mol —— Enveda 自家模型，但拿不到

- 论文（ChemRxiv 2023，CC BY-NC 4.0）明确写 "Code ... and a version of MS2Mol trained on open data ... will be made publicly available via github repository at the time of publication"。核查结果：GitHub 全站只有第三方复现（`commons-research/MS2Mol`），**Enveda 官方仓库与权重均不存在**。
- 训练集含 Enveda 内部谱（论文明确区分 MS2Mol 与 "MS2Mol PA"），即便权重泄露也不可再分发。
- 结论：**问 3 的答案是否定的**——公开的 Enveda 模型/权重没有；PRISM 是专有 foundation model；公开的只有 Enveda-180 数据集（host 允许，但那是数据不是模型）。
- 顺带说明：MS2Mol 的数字（EnvedaDark top-1 62.1% 有意义相似 / 21.4% 近似匹配）是**结构相似度**指标，不是本赛的 MRR@25，也不能直接换算。

### 4.4 MSNovelist + SIRIUS/CSI:FingerID —— 双重不可用

- **许可**：SIRIUS 自 4.4.27 起 AGPL-3.0，但其官方声明 "The SIRIUS web services (CSI:FingerID, CANOPUS, MSNovelist) hosted by the Böcker group are for academic research and education use only"，且 "(Re-)distribution ... for commercial purposes is prohibited"。奖金赛按商用场景处理。
- **数据**：CSI:FingerID 的 SVM 集成 **训练于 NIST + MassBank + GNPS**（[官方文档](https://v6.docs.sirius-ms.io/advanced-background-information/)）。按 host 的「含 NIST 的模型权重一律不合格」，这条直接出局——不需要再讨论许可。
- MSNovelist 独立仓库已归档，官方说明「只能用 SIRIUS 6 内的版本」，而 SIRIUS 6 的指纹来自 CSI:FingerID，等于继承同一个 NIST 问题。
- 结论：**排除**。这也回答了问 2 的一半：不能用 CSI:FingerID 替换 FPNet。

### 4.5 CFM-ID 4 —— 许可干净，但工程上不成立

- 训练数据是 **Metlin**（正/负模式 QTOF，10/20/40 eV），不是 NIST；代码 LGPL-2.1；host 明确 "CFM ID 4 is fine"。
- 问题是速度：官方文档 "a molecule with a MW < 1000 Da will take less than one minute to predict"（CPU）。我们的场景是 400 分子 × top-25 候选 = 10,000 次预测 ≈ 数天；即使只算 top-3 也要 ~5 小时/4 核，且会与 ICEBERG 抢同一块时间预算。
- 信息上与 ICEBERG/GLACIER 同向（都是「结构 → 谱」前向打分），边际价值低。**仅在 ICEBERG 对某类结构系统性失效时**才值得做小规模兜底。

### 4.6 MetFrag —— 已被 MetFrag-lite 覆盖

- MetFragRelaunched 需要 Java 21，逐候选做 in-silico 碎片化，成本高于现成的 `frag_rescore.py`（无训练、Apache-2.0、已内嵌在管线里）。
- 仓库 README 未声明许可证（无 LICENSE 徽章），若将来要分发需先确认。
- 结论：不新增。

### 4.7 MS2DeepScore —— 可用，但被 DreaMS 严格压住

- Apache-2.0 代码 + CC BY 4.0 权重，训练于 GNPS/MoNA/MassBank/MSnLib（>500k 谱），**不含 NIST**，许可干净。
- 但 DreaMS 论文 Fig.4a–c 显示：零样本 DreaMS 嵌入已超 MS2DeepScore，微调后差距更大。MS2DeepScore 唯一优势是「谱–谱」直接可用、无需再训。
- 结论：如果只想用一天时间做最小改动，可以用它替换 N-1c 的 modified cosine 做 A/B；但作为「新信号」它不够新。

### 4.8 Spec2Vec —— 同上，且更弱

word2vec 式的峰嵌入，模型在 MS2Query 的公开库里（GNPS + MassBank + MoNA + Brungs 库，CC BY 4.0）。DreaMS 在同类任务上全面更强，Spec2Vec 只在 CPU 极紧张时有意义。**不推荐**。

### 4.9 MIST（MassSpecGym 权重）+ MolForge —— 生成路线，但数字被高估过

- MIST 代码 MIT，MassSpecGym 微调权重 CC BY 4.0（Zenodo 11580401）/ MIT（DiffMS Zenodo 15122968）。**注意**：原论文版 MIST 用 csi2022（含 NIST），只能用 MassSpecGym 版 checkpoint。
- **必须知道的修正**：Neo et al.（NeurIPS 2025 AI4Mat）报的 "MIST+MolForge top-1 31.75%" 是**批推理 bug 造成的虚高**；修正后 MassSpecGym top-1 **10.73%**、top-10 **14.48%**（[复现仓库](https://github.com/harrylaucngd/MIST-MolForge)）。任何引用 31% 的方案都要打折。
- 对本赛：生成候选只在「真值不在池中」（class 3）时有用；本项目 001–064 已试过 FRIGID（同为 Apache-2.0 生成路线），4000 次生成对给定分子式真值命中 **0**。修正后的 10.7% 是在 MassSpecGym 分布上，搬到 timsTOF 天然产物只会更低。
- 结论：**P2 备选**，不作为主攻。

### 4.10 其他 2025–2026 基础模型（核查后不进入方案）

| 名称 | 状态 | 判断 |
| --- | --- | --- |
| DiffMS（2025） | MIT，权重公开 | MassSpecGym top-1 2.3%，对 MRR 无意义 |
| FRIGID（2025） | Apache-2.0，权重公开 | 本项目已证伪（0 命中） |
| SpecBridge（2026-01） | **已撤稿**（作者自述 "problem in the preprocessing/evaluation pipeline"） | 不能作为基线引用；只能作为「对齐类方法容易踩评测坑」的反例 |
| Spectraverse（2026） | CC BY 4.0，488k 谱 / 44k 分子，含候选集 | 可作为**额外训练数据**（补充加合物/极性覆盖）；不改变模型选择 |
| Enveda PRISM | 专有 | 无法接入 |
| DreaMS-Mol | 未发布 | 值得跟踪，本赛用不上 |

---

## 5. 三个特别问题的直接回答

### 5.1 「用预训练谱嵌入对 275k train structures 建 kNN、给 test 投结构票，是否比现有 library/analog 通道更强？」

**不是「更强」，是「补位」；而且现在这条通道根本没被真正用起来。**

- 现有 library 通道是**同结构参考谱命中**：精度极高、覆盖极低（测试集 class 1 才可能命中）。
- analog 通道是 modified cosine 传播：覆盖略高，但跨碰撞能/跨仪器就崩（本项目 052 已记录弱命中误插 −0.010）。
- DreaMS kNN 覆盖最广（跨仪器、跨 CE 稳健），但**精度低于同结构命中**——它给出的是「结构相近」，不是「结构相同」。
- 因此正确用法是：**只在 library 通道无命中/低分时接管 2–25 名的排序**，第 1 名保持 leader-safe；而不是替换 library 通道。
- 还有一条被忽略的事实：现有 DreaMS 资源只覆盖 Enveda-180 + MassSpecGym，**没有**比赛自己的 2.5M 训练谱。用 train 谱建 kNN 是「域内」的，理论上更贴测试分布，这是尚未利用的信号。

### 5.2 「用预训练 fingerprint 模型替换或集成现有 FPNet，能否突破 0.492？」

**单纯换指纹模型：不能。换目标函数 + 换分子侧表示：有可能，但那是另一件事。**

三条证据：

1. **替换已被证伪**。本项目 001–064 记录：自训 FPNet 在离线 Tanimoto 上比公开网络好 **+26.5%**，7 次替换在公榜得到 **0.418–0.443**（底座 0.456）。离线的指纹质量与公榜不同向。
2. **候选级对比目标 vs 回归目标，差距是量级的**。MSAlign Table 5：回归 R@1 16.7，candidate InfoNCE 47.5。我们现在的 FPNet 正是回归，这才是可改进的地方。
3. **分子侧表示比谱侧模型更值钱**。MSAlign Table 4：Morgan 指纹 38.6 → MolDeBERTa 47.5（同一谱编码器）。

CSI:FingerID 被 NIST 条款排除（§4.4）；DreaMS 的指纹头**没有随权重发布**（Zenodo 只有两个 ckpt），自训一个 DreaMS 指纹头 ≈ 再做一个 FPNet，属于同一类信号，按第 1 条预期无效。所以问 2 的答案是：**不要换模型，要换损失函数与分子侧编码器**。

### 5.3 「有没有公开的 Enveda 自家模型/权重？」

**没有。** MS2Mol 的官方代码与权重从未发布（只有第三方复现）；PRISM 是专有 foundation model；公开的只有 Enveda-180 数据集（Zenodo，host 已放行）与 MS2Mol 论文本身（CC BY-NC 4.0）。任何声称「Enveda 官方 MS2Mol 权重」的来源都需要先验证真伪。

---

## 6. 接入方案

总原则：**先在 2–25 名加一个新通道，不动第 1 名**（本项目历史里 top-1 改动的期望为负：057 −0.005、058 −0.008、041 −0.066，所以 065a 起一直用 leader-safe），并且**所有重计算离线预计算成 dataset**，提交 notebook 只做推理。

### P0（主攻，1 个新信号）

**目标**：给每个（分子, 候选结构）一个「对齐空间余弦」分数 `s_align`，作为 mid-rank 融合的第 4 个通道。

| 项 | 具体 |
| --- | --- |
| 谱侧输入 | `test.parquet` / `train.parquet` 的 `ms2_mzs`、`ms2_normalized_intensities`、`precursor_mz`、`adduct`、`collision_energy_ev` |
| 谱侧编码 | DreaMS 冻结编码器（`embedding_model.ckpt`，CC BY 4.0）→ 每谱 1024-d；一分子多谱取强度加权平均（或按极性分组后平均，与现有 `merge_spectra` 口径一致） |
| 分子侧输入 | 候选池 711k 的 SMILES（已有）+ 105M PubChem 窗口内的 SMILES（已有） |
| 分子侧编码 | MolDeBERTa-base-123M（768-d）或 ChemBERTa-13M（768-d，许可需按 HF 卡片核对）；离线预计算，711k ≈ 15–30 min T4 |
| 训练 | 冻结两个编码器，只训 2 个 MLP 投影（约 4M 参数）+ candidate-based InfoNCE，K=64–256 个 ±10 ppm 池内候选；训练集 = `train.parquet`（2.5M 谱 / 275k 结构），**按 Murcko/InChIKey14 分组留出** |
| 运行位置 | 训练：Kaggle 远端一次性 kernel（或复用 bgunshui/scorer-train 这类训练内核）；推理：068a 之后的 069a 提交 notebook |
| 输出 | 每分子每候选 `s_align`；对 2–25 名做 z-score 后与现有 N-1c 分数并列融合，权重先取 `λ_align = 0.15`，只调这一个旋钮 |
| 时间预算（估算） | DreaMS 2.5M 谱 20–40 min；MolDeBERTa 711k 15–30 min；MLP 20–40 min；推理秒级 → 合计 < 2 h，可塞进 9 h |
| 产物 | `dreams_train_emb.npy`（275k×1024 fp16 ≈ 563 MB）、`mol_emb_pool.npy`（711k×768 fp16 ≈ 1.1 GB）、`align_head.pt`（~16 MB）→ 私有 Kaggle dataset |
| 验收 | molecule-disjoint 本地面板上看 **recall@25 不降** 且 **MRR@25 上升**；两者冲突时不接受 |

预期（**推测，无本赛证据**）：+0.003～0.006；乐观 +0.008（对应约 20 个分子各前进 3–5 位）。理由：MSAlign 相对回归/指纹基线的 R@1 提升是 10–30 个百分点量级，但我们的融合权重只给到 0.15–0.25，且底座已经把大部分容易的分子排对了。

### P1（召回，class 3 兜底）

- 用同一个对齐空间，在 ±10 ppm 的 PubChem 窗口（或 105M 全库的 ANN）里取 **top-3 未入池** 结构，只填 23–25 名。
- 硬约束：必须先验证「插入结构的 Tanimoto 分布」与现有真值分布不冲突；052 的 −0.010 就是因为弱命中误插。
- 备选：MIST(MSG)+MolForge 生成 5–10 个候选打散后并入，但按 §4.9 的修正数字，优先级低于上面的检索。

### P2（工程加速，不新增信息）

- ICEBERG 2.0 → **2.1**（约 3× 加速，MassSpecGym 权重 MIT），把前向重打分从 top-25 扩到 top-60。这是「用更多算力换更宽的排序」，不是新信号，但成本低。
- 若 P0 证明对齐通道有效，再把 ICEBERG 预测谱也投影进对齐空间做二次融合。

### 明确不做

- 不用 MSNovelist / CSI:FingerID / SIRIUS（NIST + 学术许可）。
- 不用 MS2Mol（无权重）。
- 不在 notebook 内现算 2.5M 谱嵌入。
- 不做「单旋钮扫参」冲 0.50（alexchilton 59 臂：最大 +0.006）。

---

## 7. 风险与审计清单

| 风险 | 说明 | 缓解 |
| --- | --- | --- |
| **NIST 污染** | ms-pred 的 **NIST'23 权重**需 NIST 授权；MIST 原论文 ckpt 用 csi2022；CFM-ID 早期版本、CSI:FingerID 均含 NIST | 只用：ms-pred 的 **MassSpecGym** 权重、MIST 的 **MSG** ckpt、CFM-ID 4 的 Metlin 版；提交前逐个 ckpt 追训练集并写进 notebook 的许可表 |
| **评测陷阱** | MassSpecGym-in-the-Wild 审计发现 26 篇里 17 篇有泄漏/捷径/实现 bug（[arXiv 2606.19624](https://arxiv.org/abs/2606.19624)）；SpecBridge 因管线 bug 撤稿 | 自建对齐模型必须 molecule/InChIKey14 分组留出；候选集构造也必须在留出内；不碰 visible test |
| **域移** | 测试集 100% Bruker timsTOF、天然产物、10 种加合物；DreaMS 预训练以 Orbitrap 为主 | 先在 train 内切出 timsTOF 子集做迁移检查；不行就降权而不是硬上 |
| **噪声** | 隐藏测试仅约 400 分子；+0.008 相当于少数分子的位次变化 | 单次提交分不清 +0.003；P0 先只调一个 λ，且准备好回滚 |
| **许可证** | DreaMS 权重 CC BY 4.0（需署名）；MolDeBERTa gated；MolForge CC BY-NC | 获奖包里不重新分发这些资产，只引用；notebook 内写全 attribution；满足 rules 2.5.a.3 |
| **时间** | 现有管线 TIME_GUARD 7.5 h | 新通道全部离线；notebook 内只加载 npy + 一次 matmul |
| **历史先例** | 本项目 7 次 FP 替换全退步、多数单旋钮中性或负 | 先做 A/B，只接受 recall@25 不降前提下的 MRR 提升；不因为离线指标好看就上 |

---

## 8. 证据强度与未核实项

| 结论 | 强度 | 依据 |
| --- | --- | --- |
| DreaMS 权重 CC BY 4.0、host 放行 | **已核实** | Zenodo 页面 + host 帖 745604 |
| CSI:FingerID 训练含 NIST | **已核实** | SIRIUS 官方文档原文 |
| MS2Mol 无公开权重 | **已核实（截至 2026-11-16）** | 论文承诺 + GitHub 全站检索仅第三方复现 |
| MIST+MolForge 31.75% 是 bug 虚高 | **已核实** | 复现仓库给出修正后 10.73% |
| MSAlign 表格数字 | **已核实（论文）** | arXiv 2605.19752 Table 3/4/5 |
| DreaMS-DR 不存在独立发布 | **已核实（检索范围内）** | GitHub/Zenodo/HF/访谈均无此名 |
| 2.5M 谱 DreaMS 嵌入的 T4 耗时 | **估算** | 由官方「M1 上 2308 谱 <1 min」外推 |
| P0 对公榜 +0.003～0.008 | **推测** | 无本赛证据；仅由 MSAlign 相对增益 + 本项目融合权重经验外推 |

未核实、需要下一步确认的：

1. `hengck23-dreams-enveda-casmi26` 里 Enveda180 嵌入对应的谱数（4.7 GB）与是否含 timsTOF 子集——决定 P0 是否能直接复用其索引做迁移检查。
2. MSAlign 预训练 checkpoint 在「10 种加合物 + 负离子」上的实际表现（论文以 [M+H]+ 为主，Spectraverse 上 52% 非 [M+H]+）。
3. MolDeBERTa 的 gated 条款是否允许把权重打进私有 Kaggle dataset（大概率允许，但需读条款原文）。
4. ICEBERG 2.1 的 MassSpecGym 权重在 065a 管线里的替换成本（是否与现有 `probs` 接口兼容）。

---

## 9. 参考链接汇总

- 比赛：[数据说明](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data)、[规则](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules)、[许可裁定 750330](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/750330)、[可用清单 748270](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/748270)
- DreaMS：[GitHub（MIT）](https://github.com/pluskal-lab/DreaMS)、[权重 Zenodo（CC BY 4.0）](https://zenodo.org/records/10997887)、[Nat Biotech 2025](https://www.nature.com/articles/s41587-025-02663-3)、[DreaMS-Mol 状态](https://github.com/pluskal-lab/DreaMS/issues/32)
- MSAlign：[arXiv 2605.19752](https://arxiv.org/abs/2605.19752)、[GitHub（MIT）](https://github.com/KrzakalaPaul/MSAlign-NeurIPS2026)
- Enveda：[MS2Mol ChemRxiv](https://doi.org/10.26434/chemrxiv-2023-vsmpx-v4)、[MS2Mol 技术博客](https://enveda.webflow.io/posts/ms2mol-a-transformer-encoder-decoder-model-for-exploring-the-dark-chemical-space-of-nature)、[PRISM](https://enveda.com/prism-a-foundation-model-for-lifes-chemistry/)
- SIRIUS / MSNovelist：[SIRIUS GitHub（AGPL-3.0）](https://github.com/boecker-lab/sirius)、[训练数据说明](https://v6.docs.sirius-ms.io/advanced-background-information/)、[MSNovelist GitHub](https://github.com/meowcat/MSNovelist)、[Nat Methods 2022](https://www.nature.com/articles/s41592-022-01486-3)
- 前向模型：[CFM-ID 源码（LGPL-2.1）](https://bitbucket.org/wishartlab/cfm-id-code)、[CFM-ID 4.0 论文](https://pubs.acs.org/doi/full/10.1021/acs.analchem.1c01465)、[ms-pred（MIT）](https://github.com/coleygroup/ms-pred)、[MetFragRelaunched](https://github.com/ipb-halle/MetFragRelaunched)
- 相似度/检索：[MS2DeepScore（Apache-2.0）](https://github.com/matchms/ms2deepscore)、[Spec2Vec/MS2Query 权重 Zenodo](https://zenodo.org/records/12606475)
- 生成：[MIST（MIT）](https://github.com/samgoldman97/mist)、[MIST MSG 权重 Zenodo](https://zenodo.org/records/11580401)、[DiffMS 权重（MIT）](https://zenodo.org/records/15122968)、[MolForge（CC BY-NC 4.0）](https://github.com/knu-lcbc/MolForge)、[MIST+MolForge 修正复现](https://github.com/harrylaucngd/MIST-MolForge)、[FRIGID（Apache-2.0）](https://github.com/coleygroup/FRIGID)
- 数据/基准：[MassSpecGym（MIT）](https://github.com/pluskal-lab/MassSpecGym)、[Spectraverse（CC BY 4.0）](https://zenodo.org/records/17870921)、[MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)、[SpecBridge（已撤稿）](https://arxiv.org/abs/2601.17204)
- 本赛现状：[已放弃研究总览](../research-archive.md)、[068 进展](../progress/068-n1c-lam035.md)
