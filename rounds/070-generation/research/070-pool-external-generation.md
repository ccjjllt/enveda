# 070 池外候选生成调研（hidden class-3 de novo 方向）

调研日期：2026-11-17（北京时间）。只读调研：web 检索 + Kaggle CLI 只读查询（`datasets list`、`kernels list`、`datasets metadata`），**未推送任何 kernel、未提交、未在本机跑训练或大内存任务**。

背景：主池窗口中位仅 107 个结构；真值不在池中时一切重排无效。本队 FRIGID 4000 次生成对给定分子式的真值命中为 0（见 `archive/research-archive.md` §2/§3 与 `rounds/069-new-signals/progress.md` 引用的 001–064 记录）。本页回答：还有哪些生成路线可用，它们的权重/许可/可跑性/适配性分别是什么，以及下一步最优先做哪一个实验。

结论先行（均为“查到什么”，不是疗效断言）：

- 可直接装进断网 notebook 的生成链只有一条是全链路可核查的：**MIST（MassSpecGym 权重）预测指纹 + MolForge 指纹→SMILES**。两端权重都公开可下载，许可分别是 MIT（MIST 代码）/ CC BY 系（MIST MSG 权重）与 CC BY-NC/SA（MolForge，见 §2.2 的许可原文矛盾点）。
- **MS2Mol 无官方权重**（只有第三方复现），**MSNovelist 独立仓库已归档**（作者指向 SIRIUS 6，而 SIRIUS/CSI:FingerID 被 NIST 条款排除），**DiffMS/FRIGID 有权重但文献 top-1 数字很低且依赖已知分子式**。
- Kaggle 上**没有** MIST/MolForge/DiffMS/FRIGID 的现成 dataset 或生成类公开 notebook（只读搜索返回空或无关结果）；池扩散方向只有谱库/候选集资产（ISDB、COCONUT、LOTUS、PubChemLite），它们都不能创造库外结构。

---

## 1. 规则基线（沿用已核实结论，不重复论证）

- 奖金赛要求获奖方案 OSI 开源许可授权，但**规则 2.5.a.3**：用了许可不兼容的预训练模型不必对该模型本身再授权（见 `rounds/069-new-signals/research/069-frontier-models.md` §1 与 `ops/remote-workflow.md` §比赛要点）。
- Host 书面裁定：**任何训练集含 NIST 的权重一律不可用**（discussion 750330/744470）；GNPS-NIST14-MATCHES 是明确例外，仅限库匹配与训练。DreaMS、GLACIER/ICEBERG、CFM-ID 4、MassSpecGym 及其训练模型已获放行。
- 最终 notebook 断网、T4×2、单次上限 9 小时；外部权重必须能离线打包进 Kaggle dataset。

---

## 2. 逐条路线：权重 / 许可 / 可跑性 / 预期增益 / 失败风险

### 2.1 MIST（谱 → 4096 位 Morgan 指纹）+ MolForge（指纹 → SMILES）

- **权重**：
  - MIST 代码 MIT（`samgoldman97/mist`，`main_v2` 分支，实查 2026-11-17）。checkpoint 分两类：原论文 csi2022 版（含 NIST，不可用，只能用 MassSpecGym 版）；MassSpecGym 版在 Zenodo `11580401`（MIST 发布时）与 DiffMS 的 Zenodo `15122968` 均有收录；`harrylau/msms-models`（HF）同时镜像了 `checkpoints/mist_vmassspecgym` 与 `checkpoints/mist_vdiffms`。本次检索到但**未下载验证大小与文件清单**。
  - MolForge checkpoint 不在 Zenodo，放在 **Google Drive**（`predict.py` 文档要求下载 `1zl6HBdwYsnA4JcnOi1o6OmcrRDB5iySK` 等 Drive 文件到 `saved_models/`）。Drive 直链能否在 Kaggle notebook 内下载**本次未验证**；稳妥做法是先在有网环境下载后打成私有 Kaggle dataset 再断网引用。
- **许可**：
  - MIST 代码 MIT；MIST MassSpecGym 权重随论文以 CC BY 系发布（Zenodo 记录页标注，069 调研已核实为 CC BY 4.0 口径；DiffMS Zenodo `15122968` 标注 MIT 口径收录）。
  - MolForge 许可存在**两处不同表述**（均实查于 2026-11-17 的仓库页面）：README 顶部徽章为 `CC BY-NC 4.0`，页底 License 节写的却是 `CC BY-SA 4.0`（LICENSE.md）。按规则 2.5.a.3，NC/SA 类权重仍可进提交（获奖包不重分发即可），且 2026-10-17 起 host 明确允许 NC 类外部数据（discussion 750330）。但引用时需在 notebook 许可表里同时注明这两处，以 Code 署名为准的是 LICENSE.md 的 BY-SA。
  - MIST 原论文 csi2022 checkpoint 因训练含 NIST **不可用**，只能用 MSG 版——用错版本即触 NIST 红线，这是本路线最硬的合规门控。
- **可跑性（断网 9h）**：
  - 推理侧是 transformer 编码 + MLP，400 分子、每分子多张谱合并后推理一次，T4 上为分钟级；MolForge 自回归解码每候选秒级，400×10 候选可在预算内。预计合计 < 1h（含 ICEBERG 前向校验见 §4）。
  - 前处理依赖：MIST 需要峰-分子式对标注（subformula assignment）。`main_v2` 已内置内部 subformula 模块，不再强依赖 SIRIUS（仓库 changelog 原文），但仍需 MAGMa 式碎片标注流程；其输入还有 precursor 分子式——**MIST 本体不预测分子式**，需另接 MIST-CF（见 §2.6）或沿用管线已有公式。
  - 打包体积：MIST checkpoint 百 MB 量级；MolForge checkpoint 体积本次未查到（Drive 未打开验证），是打包前必须确认的数字。
- **预期增益**：只转述文献数字，不外推本赛。
  - MIST+MolForge 在 MassSpecGym 上的修正后数字为 **top-1 10.73%、top-10 14.48%**（FRIGID 论文附录 A.3：把推理 batch 从 24 降到 1 后指纹 Tanimoto 0.812→0.457、top-1 重建 31.75%→10.73%；此前 31.75% 是批推理 bug 虚高）。MassSpecGym in the Wild 审计（17/26 篇有泄漏/捷径/实现 bug）与此同源。
  - 该 10.73% 是“给定正确分子式 + MassSpecGym 分布”下的重建准确率，不是本赛 MRR@25，也不是 timsTOF 分布。本赛无任何已核实数字。
- **失败风险**：
  1. 用错 csi2022/NIST 权重 → 直接违反 host 禁令（合规一票否决，需在 notebook 许可表逐 ckpt 写清训练集来源）。
  2. MIST 预测指纹是噪声指纹，MolForge 在其自带基准上 ECFP4 Tanimoto 精确度 92–95% 是对**干净指纹**的重建率，对噪声指纹的退化本次未查到数字。
  3. 同族信号冗余：MIST 指纹与管线 FPNet 同为“谱→指纹”，069 已证实换指纹模型不迁移公榜（自训 FPNet 离线 +26.5%，公榜 0.418–0.443 退步）。
  4. H+ 为主：历史 MIST 以 `[M+H]+` 为主训练；`main_v2` changelog 称“加了多加合物/instrument covariates 选项、训含其他加合物的数据集很直接”，但**MSG 公开 checkpoint 是否含 10 加合物/负离子未知**，需实测（见 §4 门控）。

### 2.2 MolForge 单独（指纹 → SMILES/SELFIES 重建器）

- **权重/许可**：见 §2.1。补充：论文 Ucak et al., J Cheminformatics 2023（DOI 10.1186/s13321-023-00693-0）；训练库为 ChEBI 系公开结构（本次未逐项核对 NIST 成分——MolForge 是纯结构→结构重建器，不吃谱，其训练与 NIST 谱库无已查到的关联；采用前仍需读其论文数据节确认）。
- **可跑性**：CPU/GPU 均可，`predict.py --fp --model_type --input --checkpoint --decode greedy/beam`，单候选秒级。
- **预期增益**：干净指纹下 ECFP4→SMILES Tanimoto 精确度 ~92–95%（仓库 README 表格）；噪声指纹下无数字。
- **失败风险**：只能重建“指纹对得上”的结构；若 MIST 指纹错了，MolForge 会自信地生成错误结构。必须配 §4 的校验门控，不直接进 25 槽。

### 2.3 MS2Mol（Enveda 自家谱→结构直出）

- **查到什么**：官方代码与权重**从未发布**（论文 ChemRxiv 2023 承诺“发表时公开”，截至 2026-11-17 的 GitHub 全站检索只有第三方复现 `commons-research/MS2Mol`；Enveda 官方组织下与 DreaMS 相关的 `enveda/DreaMS-Enveda` 仓库已于 2026-03-04 被所有者归档）。Enveda-180 数据集公开了（host 已批准），但那是数据不是模型。
- **结论**：此路线**无可用权重**，终止。不再花额度验证第三方复现权重（来源真实性无法确认，且论文训练含 Enveda 内部谱，即便拿到也涉再分发问题）。

### 2.4 MSNovelist + SIRIUS/CSI:FingerID

- **查到什么**：
  - MSNovelist 独立仓库（`meowcat/MSNovelist`）README 原文：“Use SIRIUS 6 instead!”，旧版依赖的后端已下线，只能走 SIRIUS 6 内置版本。
  - SIRIUS 自 4.4.27 起 AGPL-3.0；其官方声明 CSI:FingerID/MSNovelist 在线服务仅限学术研究教育用途、禁止商业再分发；CSI:FingerID 训练集为 **NIST + MassBank + GNPS**（官方文档原文，069 已核实）。
  - 文献数字：MSNovelist 在 3,863 张 GNPS 谱上 top-1 精确 25%、召回 45%（Nat Methods 2022）；SIRIUS 公式注释本身可用，但与生成无关。
- **结论**：**双重不可用**——NIST 训练条款一票否决；AGPL+学术非商用与奖金赛冲突；且 CPU 约 5 min/谱 × 400 分子多谱在 9h 内不可跑。终止。

### 2.5 DiffMS（扩散生成，谱条件）

- **权重/许可**：代码 MIT（`coleygroup/diffms`，实查 2026-11-17 README 声明 MIT）；权重随 Zenodo `15122968` 发布（与 MIST MSG 权重同源收录）。
- **可跑性**：GPU 可跑，断网打包可行。
- **预期增益（文献原文，不外推）**：MassSpecGym top-1 **2.30%**、top-10 4.25%；NPLIB1 top-1 8.34%、top-10 15.44%（AAAI 2026 Many-Body 对比表）。Schneider 2026 综述定性：de novo 当前仍是 formula-conditioned 与 scaffold-conditioned 两范式并存，DiffMS 处 formula-conditioned 一侧。
- **失败风险**：2.3% 的 top-1 意味着每 400 个 hidden 分子期望命中约 9 个（MassSpecGym 分布下），搬到 timsTOF 天然产物只会更低；且大概率需要分子式先验（与 FRIGID 同类约束）。定位为 P2 备选，不做主攻。

### 2.6 FRIGID（扩散语言模型，谱+指纹条件，SAFE 序列）

- **权重/许可**：Apache-2.0（`coleygroup/FRIGID`，实查）；NPLIB1 top-1 14.16%、top-10 25.27%（仓库 Results 表）。
- **失败风险（本队已有证据）**：001–064 记录——4000 次生成对给定分子式真值命中 **0**；附录 A.4 显示其对公式不确定性有一定鲁棒，但“已知公式”仍是前提。NPLIB1 的 14% 与本赛 timsTOF/天然产物分布无关。本路线**已证伪过一次**，重启需先解释与上次 0 命中的差异（不同的公式来源？不同的解码预算？），否则不排优先级。

### 2.7 MIST-CF（分子式预测，MIST 的公式侧扩展）

- **查到什么**（2026-11-17 新查）：`samgoldman97/mist-cf` 独立仓库；权重 Zenodo `8151513`；CASMI2022 上公式+加合物准确率与 SIRIUS 提交持平（0.862 vs 0.868，predicted 304 vs 272）；instrument one-hot 区分 Ion Trap/Q-ToF/Orbitrap/FTICR——timsTOF 属 QTOF 大类，至少在 covariates 设计内；**但仅限正离子模式**（MIST-CF 只考虑正模式多加合物，负离子不在覆盖内）。
- **定位**：不是生成器，而是 §2.1/§4 所需的“公式先验”候选（替代 SIRIUS、规避 NIST）。公式 top-1 ~0.74（NPLIB1）意味着约 1/4 分子公式起手即错，门控必须容错（§4）。

---

## 3. timsTOF 适配性（本赛 100% Bruker timsTOF，7 类可见 adduct / 10 类隐藏 adduct）

| 路线 | timsTOF 相关证据（查到的原文） | 缺口 |
| --- | --- | --- |
| MIST 系列 | MIST-CF instrument one-hot 含 Q-ToF；`main_v2` changelog 称加合物/instrument covariates 已选项化 | MSG checkpoint 的加合物/仪器混合比未公开；timsTOF 子集表现无数字，需 §4 门控实测 |
| MolForge | 纯结构侧，不吃谱 | 无仪器相关性；风险只在噪声指纹 |
| DiffMS/FRIGID | 训练以 MassSpecGym/NPLIB1（Orbitrap/QTOF 混合）为主 | timsTOF 微调证据为零；底座已有 timsTOF ICEBERG（073a 公榜 −0.001，冗余教训） |
| MSNovelist/SIRIUS | 需高精度质量（20 ppm 内），timsTOF 精度满足 | 已被许可+NIST 双否决，与仪器无关 |
| 本赛已验证的 timsTOF 资产 | `hjyact/casmi26-timstof-iceberg`（MIT，221 MB，经 `datasets metadata` 实查：`tims_ft3`/`tims_ft2` 双 ckpt + ms-pred MIT 源码 + wheels；描述称双 ckpt 融合在两把排除验证结构的尺子上 +0.036/+0.017 MRR@25） | 073a 已用它做中名次重排，公榜 −0.001：信号被底座吸收。作为**生成候选的校验器**（结构→预测谱）仍可复用，这是它在本路线里的位置 |

---

## 4. Kaggle 只读检索结论（无生成类现成资产）

2026-11-17 用 `KAGGLE_API_TOKEN="$HOME/.kaggle/access_token_jiyujia"` 只读命令实查：

- `kernels list -s "FRIGID MIST DiffMS de novo generation"`：仅返回 1 个无关 notebook（`mahirahman011/submission-csv`，0 票）。**无** MIST/MolForge/DiffMS/FRIGID 相关的比赛 notebook。
- `datasets list -s "MIST fingerprint MolForge"`：**No datasets found**。生成模型权重在 Kaggle 上无现成 dataset，需自建私有 dataset 打包。
- 池扩散/候选集类 dataset（存在，但都不能创造库外结构）：
  - `hemantkumarbk/isdb-np-insilico-msms-library`（CC BY 4.0，元数据实查：298,084 CFM-ID 4 预测谱/148,935 LOTUS 结构；作者自报：训练子集谱相似重排 MRR 0.332→0.516，但加到公开 notebook 2–25 名固定权重后**公榜 −0.009**；仅 `[M+H]+`/`[M-H]-` 预测，无逐能量匹配）。本队 069a 已验证 0/400 插入，不再重复。
  - `xjoannax88/casmi-coconut-candidates`、`qwertykeypadapple/casmi26-lotus-candidates`、`franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints`（CC0 指纹，候选集/指纹预计算，可作检索池，不能生成）。
  - `dmitriigluzdov/casmi26-natural-product-panel-benchmark`（28 票）+ `casmi26-identity-disjoint-validation-folds`（9 票）：天然产物 panel 与**身份不交验证折**，可作 §4 的离线门控评估协议参考（按规范化结构整组留出，不碰 visible test）。
- 高票公开 notebook（`evgendvorkin/enveda-casmi-2026` 215 票、`dmitriigluzdov/casmi-26-from-spectra-to-structures` 179 票等）均为 EDA/基线/检索管线，标题与摘要层未见 de novo 生成模块；未逐个打开全文核查（只读配额与时间所限），采用标题+票数+lastRunTime 粗筛结论。

---

## 5. 最优先实验设计：MIST(MSG)+MolForge 尾槽生成（P0）

> 只写设计与预算，不写疗效断言。通过/失败标准全部是可审计的计数器。

### 5.1 输入

- 谱：`train.parquet`（2.5M 谱，自训/校准用）+ 测试谱（推理用）；adduct/极性/CE 按 `main_v2` covariates 口径传入；timsTOF 子集单独标记。
- 分子式先验：MIST-CF（正离子）或管线已有公式；**负离子分子式缺失时该分子跳过生成**（如实记录跳过数，不硬编）。
- 权重（全部离线打成 1 个私有 Kaggle dataset，notebook 断网只读）：
  - 2026-11-17 联网探针 kernel `jiyujia/enveda-mist-p0-probe` v1（CPU，零 GPU）已验证 Kaggle 内可直下全部 5 个文件，字节数与来源一致：
  - `mist_fp.ckpt` 103,416,120 B，sha256 `59562ea6…`（Zenodo 直链）；
  - `decoder_molforge.pth` 556,475,975 B，sha256 `67710e7b…`（OSF 直链）；
  - `combined_morgan4096_vocab_sp.model` 302,153 B（sha256 与 OSF API 记录一致）、`combined_smiles_vocab_sp_morgan4096.model` 239,188 B、`.vocab` 68,084 B。
  - 打包时按此清单 + MIST 代码 `main_v2` + MIST-MolForge 集成层组私有 dataset 即可，Drive/OSF 可达性风险已消除。

### 5.2 生成 + 四道门控（全部只动 23–25 名，1–22 名冻结）

1. G1 分子式一致：MolForge 输出分子式必须等于输入先验公式（±0 Da 整数质量 + 元素组成一致），否则丢弃。
2. G2 质量窗口：按测试 10 加合物精确质量 ±10 ppm 过滤。
3. G3 结构去重：与池内 107 中位候选做规范化 InChIKey14 去重，已在池中则丢弃（只保留**真池外**结构）。
4. G4 前向校验：timsTOF ICEBERG 预测该结构的谱，与 query 谱做 cosine；阈值不在本机拍脑袋定——在 train 内按结构留出的 timsTOF 子集上先测“真值 vs 同式 decoy”的分离度，再定阈值；阈值与分离度曲线一并写入审计 JSON。
5. 数量上限：每分子最多 3 个生成候选，只填 23–25 槽；无通过者则该分子空置（不拿弱候选凑数）。

### 5.3 审计（随 submission 同目录输出 JSON，仿照 082c/`e180lib_audit.json` 口径）

`gen_audit.json` 至少含：`molecules`、`molecules_attempted`（公式可用）、`molecules_skipped_noformula`、`candidates_generated`、`g1_formula_reject`、`g2_mass_reject`、`g3_dup_inpool`、`g4_iceberg_reject`、`slots_filled`、`elapsed_s`，以及阈值来源（留出分离度实验的 dataset/notebook ref）。

### 5.4 GPU 预算（jiyujia 约 17h 上限内；11-16 实测该账号已用 5.62h/30h，ccjjllt 已用尽，额度以提交日实查为准）

| 步骤 | 位置 | 估计 |
| --- | --- | --- |
| MIST-CF 公式 + MIST 指纹（train timsTOF 子集，门控阈值用） | 远端训练 kernel，T4×2 | ~1–2h（含 memmap 随机读慢启动） |
| MolForge 解码（train 子集 ×10 候选，beam=5） | 同上 | ~1h |
| ICEBERG 前向校验（候选数×谱，`tims_ft3` 单 ckpt 先行） | 同上 | ~1h（073a 实测 628 候选 92s；生成候选量级相近） |
| 阈值确定 + 小样本 smoke 审计 | 本机只读检查/远端 smoke | ~0.5h |
| 完整 hidden rerun commit（含底座 068a 全管线） | 远端 notebook | ~0.5h（068a commit run 约 22 分钟） |
| 预留回滚/复核 1 次 | — | ~1h |
| **合计** | | **~5–6h**，在 17h 内，剩余额度不动（等 082c 公榜后再决定是否花一次提交验证） |

### 5.5  Paper trail（可恢复性）

- 权重 dataset 建好后记录其 Kaggle ref + 版本号 + 每个 ckpt 的训练集来源（一句话：MassSpecGym/Metlin/自训其一，**出现 NIST 字样即停**）。
- 若 G1–G4 后 `slots_filled` 在 visible smoke 上为 0（如 069a 的 0/400），**不提交**，只记录“门控过严或信号缺失”，回头放宽 G4 阈值只允许一次且需留出分离度证据支撑。
- 若 hidden rerun 的审计显示某加合物系统性 `skipped_noformula`（如负离子），如实记录覆盖缺口，不扩展审计口径掩盖。

---

## 6. 明确不做（本次已查到依据）

- 不碰 MS2Mol（无官方权重）、MSNovelist/SIRIUS（NIST+AGPL/学术条款）、csi2022 版 MIST 权重（NIST）。
- 不把 DiffMS/FRIGID 排 P0：文献 top-1（2.3%/NPLIB1 14%）均在“已知公式+非 timsTOF 分布”下，且 FRIGID 在本队已有 4000 次 0 命中记录；重启需先书面解释差异。
- 不在 notebook 内现算任何全量嵌入；不扫 `LAM`/权重单旋钮（069 已证明 2–25 名叠加净效应为零或负）。
- 不用可见 test 命中数论证泛化（082a 的 164 命中是跨结构 analog，占位 test 与 train 重复，coverage/top-1 数字只在 e180 内部成立）。

---

## 7. 参考（本次实查）

- MIST：`github.com/samgoldman97/mist`（MIT，`main_v2`，2026-11-17 实查）；MIST-CF：`github.com/samgoldman97/mist-cf`，权重 Zenodo `8151513`，instrument one-hot 含 Q-ToF、正模式限定。
- MolForge：`github.com/knu-lcbc/MolForge`（README 徽章 CC BY-NC 4.0 / LICENSE.md 节 CC BY-SA 4.0，2026-11-17 实查并存；checkpoint 经 Google Drive 分发）。
- MSNovelist 归档页（`meowcat/MSNovelist`，“Use SIRIUS 6 instead!”）；SIRIUS 仓库（AGPL-3.0，自 4.4.27）。
- DiffMS：`github.com/coleygroup/diffms`（MIT，2026-11-17 实查）；FRIGID：`github.com/coleygroup/FRIGID`（Apache-2.0；附录 A.3/A.4 修正数字与公式鲁棒性）。
- MIST+MolForge 修正：FRIGID 附录 A.3（batch 24→1：Tanimoto 0.812→0.457，top-1 31.75%→10.73%）；MassSpecGym in the Wild 审计 arXiv 2606.19624；`harrylau/msms-models`（HF，同时镜像两个 MIST checkpoint）。
- Enveda 官方组织 `github.com/enveda`（`DreaMS-Enveda` 2026-03-04 已归档；无 MS2Mol 官方权重发布）。
- Kaggle 站内（jiyujia token 只读，2026-11-17）：`hjyact/casmi26-timstof-iceberg`（MIT，metadata 实查）、`hemantkumarbk/isdb-np-insilico-msms-library`（CC BY 4.0，metadata 实查，作者自报 −0.009）、`dmitriigluzdov/casmi26-natural-product-panel-benchmark`、`casmi26-identity-disjoint-validation-folds`；MIST/MolForge dataset 搜索零结果、生成类 notebook 搜索仅 1 无关结果。
- 本队证据：`archive/research-archive.md`（FRIGID 4000 次 0 命中；FP 替换退步；052 误插 −0.010）、`rounds/069-new-signals/research/069-frontier-models.md`（许可基线与 P0 对齐路线）、`rounds/069-new-signals/progress.md`（069a/072a/073a/076a 公榜 0.492/0.491/0.490；082c 待出分）。
