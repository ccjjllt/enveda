# 069 数据资产审计：068a 已挂什么、还能加什么

审计时间：**2026-11-16（北京时间）**。只读：未运行训练、未推送内核、未提交、未改动 068a。
方法：Kaggle MCP `get_dataset_info` / `list_dataset_files` / `list_dataset_tree_files` / 比赛 discussion；本地解出 068a 的 `FROZEN_B64`（59 cell / 277k 字符）逐条核对代码引用；本地 `train.parquet` 与下载的 GNPS parquet 做结构重叠核对。

与既有 069 文档的分工：[069-frontier-models.md](069-frontier-models.md) 判模型，[069-public-frontier.md](069-public-frontier.md) 判公开配方与资产清单，[069-retrieval-design.md](069-retrieval-design.md) 判重排设计。**本篇只做两件事：现有 10 个 dataset 的真实用途盘点，以及「加数据还能不能提分」的边际覆盖核算。**

---

## 0. 结论先行

1. **068a 实际挂载 102.5 GB，其中 68.8 GB（67%）从未被任何代码路径打开**——`casmi26-pubchem-popularity-prior` 里 12 个 `fingerprints-*.bin` 分片。管线是在质量窗口内**现算 ECFP4**，不读这些分片（`probe_core2.init_worker` + `pc_runner.py`）。
2. **候选召回已饱和，不是加库能救的**：主池 710,701 结构（train 275,810 + COCONUT）已含全部训练结构；PubChem 通道再搜 105.9M 结构；ChEBI/LIPID MAPS 与 PubChemLite 已作为独立候选库挂载。LOTUS/NPAtlas 的结构本来就在 PubChem 里，加进来不产生新召回。
3. **实测谱库的边际覆盖也接近零**。我实测 `casmi26-gnps-community-cc0` 的 506,939 张谱：**98.07% 的结构已在 train**，真正新增只有 **1,141 个结构 / 2,597 张谱**，另有 1,405 张是「已知结构 + 新加合物」。花 425 MB 换这点增量不划算。
4. **能产生新增益的不是「更多库」，是「同仪器的新信号」**。排序第一的是 `hjyact/casmi26-timstof-iceberg`（MIT，221 MB，timsTOF 微调的 ICEBERG；作者留出实验 +0.030/+0.018，双 checkpoint +0.036/+0.017 MRR@25），它接的是管线**已有**的前向校核通道，而不是新开一条召回。
5. **第二值得加的是验证集，不是打分资产**：`casmi26-e180-not-in-train`（CC BY 4.0，51 MB，3,719 个不在 train 的结构 × 14,471 张 timsTOF 谱）。这是目前唯一同仪器、结构未见、许可干净的留出集，能把「加了到底有没有用」从公榜赌博变成可测。
6. **明确不建议**：LOTUS 候选回填（063b 已试无增益，且 25 槽是满的）、GNPS 全量（上面第 3 条）、ISDB/CFM-ID 预测谱（公开实验 −0.009）、MassBank 的各种「合规重打包」（数据与已挂的同一份）。

---

## 1. 现有资产：规模、许可、真实用途

`dataset_sources` 共 10 个，另加 1 个 kernel source。大小为 `get_dataset_info.total_bytes`；「代码引用」以 068a 内嵌管线（`FROZEN_B64`）与 068a 自身 cell 为准。

| # | slug | 大小 | 许可（Kaggle 标注） | 关键文件 | 管线里的实际用途 | 状态 |
|---|---|---|---|---|---|---|
| 1 | `samartalwar/casmi-2026-spectral-library-massbankharmonized` | 47.9 MB | CC BY-NC-SA 4.0（逐条混合，含 NC） | `spectra.parquet` 43.9 MB / `structures_with_fingerprints.parquet` 3.0 MB | 068a cell 25 overlay 的 **Source B**：读 `spectra.parquet`，丢弃 `is_casmi_subset`，10 ppm + 0.002 Da 前体窗 + √强度 cosine ≥0.80 + ≥4 峰 → 直接提 rank 1 | ✅ 用（只用了 spectra；指纹文件未用） |
| 2 | `sanpier/casmi-spectrum-model-v1` | 55.9 MB | Apache 2.0 | `final_model.pt` 11.9 MB、`model.py`/`features.py`/`config.json` | Engine 1 神经打分（16×500 步版）。068a 用 SHA256 pins 定位，作为 **ensemble 的 public 一半**：`0.5*z(public)+0.5*z(ours)` | ✅ 用 |
| 3 | `dmitriigluzdov/casmi26-fold-safe-fpnet` | 1.07 GB | 比赛条款 + 脚本 Apache-2.0 | `fpnet_fold0.pt`、fold1–4、`MODEL_INPUTS.json`、split/审计 | **只用了 `fpnet_fold0.pt`**：PubChem 通道 FPNet 集成的第三模型（权重 1/3，`CASMI_FPNET_FOLD0`，SHA pin）。fold1–4 与训练脚本全程未读 | ⚠️ 用了 1/5 |
| 4 | `dmitriigluzdov/casmi26-natural-product-knowledge-table` | 136.0 MB | 非商用+署名（LOTUS CC BY 4.0 + NPAtlas CC BY-NC 4.0） | `natural_products.parquet` 9.6 MB、`external_reference_spectra/`、`candidate_joins.parquet` | ① `external_reference_spectra/external_{delta,record_quality,scorer_sid_links}.parquet` → **751 张外部谱 / 469 结构**追加进训练谱库（MassBank+MoNA+GNPS+MSnLib，已被扣掉与 train 近重复）；② `natural_products.parquet` 只用于诊断分组表 | ✅ 用（谱是真信号，主表只做展示） |
| 5 | `dmitriigluzdov/casmi26-natural-product-panel-benchmark` | 233.1 MB | 比赛条款 | `panel.parquet`、`prior_rule_summary.csv`、cached replays | **仅诊断展示**（250 分子规模统计、prior replay 表）。不进入任何排序 | ⚠️ 不参与打分 |
| 6 | `dmitriigluzdov/casmi26-pubchem-popularity-prior` | **80.42 GB** | Other（PubChem 条款 + 署名） | `pool_popularity.csv` 180.9 MB、`pc_*.npy`、`delta_*/growth_*`、**`fingerprints-000..011.bin` 68.8 GB** | `pool_popularity.csv` 供 068a 文档先验；`pc_mass/off/smiles.npy` + `pc_lsid/lpmid/np/lotus_refs/lsynonyms/src_mask.npy` 供 PubChem 通道。**12 个指纹分片零引用** | ⚠️ 用了 11.6 GB，废 68.8 GB |
| 7 | `dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets` | 20.08 GB | CC BY-NC-SA 4.0 | 见下 | 管线主体：`casmi26-v2-pool`（710,701 结构池 + `fp_bits`）、`fpnet-full1`、`v4b/v3` 模型与 LightGBM、`iceberg/glacier`、`pubchem_tier__pc_{mass,off,smiles}.npy`、第二引擎（`fp-models-v2`、`ranker-features`）、`coconut`/`chebi`/`pubchemlite` 候选、`hengck23-dreams`、`spec2mol` | ✅ 用（主体） |
| 8 | `takumuhata/casmi26-extlib` | 264.7 MB | CC0 | `ext_meta/ext_mz/ext_in/ext_mass`、`coconut_index.parquet` 41.9 MB | 068a cell 25 overlay 的 **Source A**（GNPS+MassBank+MoNA 合并）。`coconut_index.parquet` **零引用** | ✅ 用（索引文件未用） |
| 9 | `sanpier/catalogue-experiment-saner` | 25.6 MB | **Unknown** | `catalogue_experiment/*.pkl`、`ranking.py`、replay checkpoint | **065a–068a 四份 notebook 代码里零引用**（grep `catalogue` = 0） | ❌ 死重 + 许可不明 |
| 10 | `metric/rdkit-2026-3-3-wheel` | 148.9 MB | Unknown（wheel 本身 BSD-3） | cp310–cp313 四个 wheel | 装 RDKit 2026.03.3，保证与官方 metric 的 InChIKey14 一致 | ✅ 用 |
| K | `popwenman/casmi26-scorer-train-v80`（kernel source） | 未计 | 比赛条款派生 | `final_model.pt` + `config_used.json` | **自训打分模型的来源**：`_swap_to_ours()` 在 `/kaggle/input` 找带 `config_used.json` 的 bundle 并替换；最终打分 bundle 用它，public v1 只作 ensemble 的一半 | ✅ 关键 |

### 1.1 v4g 包内逐项用途（`ROLE` 表来自冻结管线自身）

| 子目录 | 大小 | 用途 | 是否读 |
|---|---:|---|---|
| `casmi26-v2-pool` | 1.69 GB | 710,701 结构主池 + `pool_fp/frag` + `train_fp_sel` | ✅ 核心 |
| `fpnet-full1` | 199 MB | 主引擎指纹模型 | ✅ |
| `v4b-models` / `v3-models` | ~1.6 GB | LightGBM 排序器 / PubChem 通道 FPNet A+B | ✅ |
| `iceberg` / `glacier` | 121 MB | 前向谱模拟与碎片校核 | ✅ |
| `chebi-lmsd` `coconut` `pubchemlite` | 890 MB | 第二引擎候选库 | ✅ |
| `pubchem_tier__pc_*` | 7.21 GB | PubChem 105.9M 通道（mass/off/smiles） | ✅ |
| `hengck23-dreams` | 5.71 GB | 空槽的 DreaMS 质量匹配补尾。**但代码找 `dreams_np_mass.npy`/`dreams_np_smiles.npy`，本包没有这两个文件** → 分支恒不触发 | ⚠️ 死分支 |
| `casmi26-spec2mol` | 67 MB | 谱→分子模型，全代码零引用 | ❌ |
| `chebi-lipidmaps-casmi26` | 17.7 MB | 旧版 ChEBI 候选，注释写「为可复现保留」 | ⚠️ 冗余 |

> 资产侧的净结论：**102.5 GB 里真正参与打分的约 27 GB**（v4g 主体约 14 GB + pubchem 的 11.6 GB + 谱库 0.4 GB），**约 75 GB 是死重或不参与打分**（68.8 GB 指纹分片、5.7 GB DreaMS、233 MB panel、其余零引用文件）。

---

## 2. 候选结构库：哪些已在管线、哪些没有

| 结构库 | Kaggle slug | 大小 | 许可 | 结构数 | 已在 068a？ | 结论 |
|---|---|---:|---|---:|---|---|
| train 结构 | 比赛数据 | — | CC BY-NC 4.0 | 275,810 | ✅ 主池 | 全部在池 |
| **COCONUT** | `prvsiyan/coconut-casmi26-candidates`（v4g 内）/ `xjoannax88/casmi-coconut-candidates` / `takumuhata` 的 `coconut_index.parquet` | 423 MB / 34.3 MB / 41.9 MB | CC BY-NC-SA / CC BY 4.0 / CC0 | ~495k | ✅ 主池 + 第二引擎 | 已饱和；`xjoannax88` 是更干净的 CC BY 版，只在对许可敏感时替换 |
| **PubChem 全量** | `casmi26-pubchem-popularity-prior` + v4g `pubchem_tier__pc_*` + `ahmedberatozer/casmi26-pubchem-tier` | 80.4 GB / 7.2 GB / 1.2 GB | Other | 105.9M | ✅ PubChem 通道 | 已饱和；**结构侧没有更大的库了** |
| **PubChemLite** | v4g `casmi26-pubchemlite-cc0-candidate-fingerprints` / `xjoannax88/casmi-pubchemlite-candidates` | 361 MB / 31.2 MB | CC0 / CC BY 4.0 | 347k | ✅ 第二引擎 | 已用 |
| **ChEBI + LIPID MAPS** | v4g `casmi-chebi-lmsd-structures` / `prvsiyan/chebi-lipidmaps-casmi26` / `sergiomalv/casmi-chebi-lmsd-structures` | 106 MB / 17.7 MB / 30.7 MB | CC BY-NC-SA / CC BY 4.0 | 98k | ✅ 第二引擎 | 已用（两份重复） |
| **LOTUS** | `qwertykeypadapple/casmi26-lotus-candidates` | 5.84 MB（`candidates_lotus.parquet` 6.99 MB） | **CC0** | 149,983 | ❌ 文件没挂 | 063b 试过无增益；且 068a 的 2c 回填只在「槽位未满」时触发，实测 25 槽是满的 → **结构上无效** |
| **NPAtlas** | `shankhasuvromullick/npatlas`（原始 TSV 33.7 MB） | 10.8 MB | MIT | ~33k | ❌ 独立库没挂 | 但 `natural_products.parquet`（已挂）已含 LOTUS+NPAtlas 153,964 结构（含 smiles/mono_mass/ik14）→ 数据其实在，只是没当候选池用 |
| **Enveda-180 结构** | `egehseyinca/enveda180-casmi26-reference-ccby4` 等 | 218 MB | CC BY 4.0 | 180 | ❌ | 小；主要用于谱，不是召回 |
| 联合池 | `n10705013/casmi26-union-pool` / `rishavae/casmi-2026-candidate-database` | 25.9 MB / 620 MB | CC BY 4.0 / CC0 | — | ❌ | 都是上面几家的再打包，无新增来源 |

**判断**：候选结构侧**没有可加的增量**。理由不是「库不够大」，而是：
1. 主池已含全部 train 结构 + COCONUT，PubChem 通道再覆盖 105.9M；
2. LOTUS 的结构绝大多数在 PubChem 内（知识表原文："most are also in PubChem"），NPAtlas 亦然 → 加库不增加可达结构；
3. 最终 25 槽是满的（[069-retrieval-design.md](069-retrieval-design.md) §0），回填类改动没有落点。

> 若只想花 6 MB 把死代码点亮：挂 `qwertykeypadapple/casmi26-lotus-candidates` 能让 `lotus_fills` 从 0 变成非 0，但**不要期待分数**；更省的做法是直接读已挂的 `natural_products.parquet`（列名不同，需改 3 行）。

---

## 3. 实测谱库：哪些已在管线、真实边际覆盖是多少

| 谱库 | 获取方式 | 大小 | 许可 | 规模 | 与 train 的关系 | 结论 |
|---|---|---:|---|---|---|---|
| **MassBank（harmonized）** | `samartalwar/...-massbankharmonized` | 47.9 MB | CC BY-NC-SA（逐条） | 139,744 谱 / 20,204 结构 | 大部分在 train（知识表称 99.4% 的合规公开记录已在 train） | ✅ 已挂，overlay Source B |
| **GNPS+MassBank+MoNA 合并** | `takumuhata/casmi26-extlib` | 264.7 MB | **CC0** | 合并库（`ext_meta` + CSR 峰阵） | 同上 | ✅ 已挂，overlay Source A |
| MassBank+MoNA+GNPS+MSnLib 去重增量 | 知识表 `external_reference_spectra/` | 4.7 MB | 逐条 CC0/CC BY | **1,384 谱 / 782 结构** | 已扣除与 train 近重复 | ✅ 已挂，751 谱进库 |
| **GNPS community（CC0）** | `aryanawasthiteykik/casmi26-gnps-community-cc0` | 425 MB | CC0 | 506,939 谱 / 59,039 结构 | **实测：98.07% 结构已在 train；仅 1,141 新结构 / 2,597 新谱；另 1,405 谱是已知结构的新加合物** | ❌ 不推荐（见 §3.1） |
| GNPS 直接贡献库（源快照） | `senanuretin/casmi26-gnps-cc0-expanded` | 13.8 MB | CC0 | 源清单 + 记录哈希 | 是上面那类数据的子集 | ❌ 冗余 |
| **MoNA 独立包** | 未找到 | — | — | — | — | 只能经 extlib / 知识表间接获得；独立打包需联网抓 MoNA 全量（混合许可），**不建议为竞赛重打包** |
| **GNPS-NIST14-MATCHES** | 无 Kaggle 数据集 | — | GNPS 仓库条款 | — | — | **host 允许**（[745832](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745832)），但「新建 NIST 派生数据集」被明确禁止 → **不要自己切一份重新上传**；现有 GNPS dump 里带就带 |
| Enveda-180（Zenodo） | `aryanawasthiteykik/casmi26-e180-not-in-train` / `royalacecat/...` / `shaoyuantang/...` / `egehseyinca/...` | 50.8 / 448.5 / 127.8 / 218.6 MB | **CC BY 4.0** | 14,471 谱 / **3,719 个不在 train 的结构**（timsTOF） | 结构完全不在 train | ⭐ 最有价值的**验证+外部谱**资产（host 已批准 Zenodo 发布，[745148](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745148)） |
| GNPS+MassSpecGym 增量 | `yehudaitkin/casmi-zenodo-top128-novel-spectra` | 5.5 MB | CC BY 4.0 | 11,561 谱 / 3,782 结构 | 作者声明已删掉与 train 重复的「结构-加合物」对 | ⭐ 每 MB 边际覆盖最高 |
| MassBank「合规子集」多份 | `xiangjiangzhou/...20261115` 97.8 MB、`markbarney/...202611` 15.7 MB、`navyadevadiga/...lib-own` 39 MB、`kizhowl/...original-permissive` 20.2 MB | — | 逐条 | 同一份 MassBank 的不同切法 | 与已挂数据同源 | ❌ 无新增 |
| ISDB-NP（CFM-ID 预测谱） | `hemantkumarbk/isdb-np-insilico-msms-library` | 1.10 GB | CC BY 4.0 | 298,084 预测谱 / 148,935 LOTUS 结构 | 预测谱，非实测 | ❌ 作者自报公开榜 **−0.009** |

### 3.1 GNPS CC0 的实测重叠（本地核对）

把 `gnps_cc0_trainschema.parquet` 的 506,939 行与本队 `data/raw/train.parquet`（2,539,608 谱 / 275,810 结构）逐 key 比对：

| 指标 | 数值 | 占比 |
|---|---:|---:|
| GNPS 结构数 | 59,039 | 100% |
| 结构已在 train | 57,898 | **98.07%** |
| 结构不在 train | 1,141 | 1.93% |
| 谱的结构不在 train | 2,597 | 0.51% |
| 已知结构 + 新加合物（结构,adduct 新组合） | 1,405 | 0.28% |
| 已知结构 + 同加合物已存在 | 502,937 | **99.21%** |

结论：**再加一份完整 GNPS，等于给已有结构的同一加合物再塞 50 万张同源谱**。对 identity search 的净增量是 2,597 张谱 / 1,141 个结构，而隐藏测试只有 ~400 个分子。425 MB 与索引时间换这点覆盖，期望值接近 0。

### 3.2 现有 overlay 的真实射程

- 可见 `test.parquet` 是 1,213 张谱 / 400 个分子；隐藏集卡面写「~1,500 谱 / ~400 分子」。
- 068a cell 25 的规则：前体 ≤10 ppm、同离子模式、√强度 cosine ≥0.80（代码常量 `TAU=0.80`）、≥4 匹配峰，且 `lib_max < 0.90` 才允许提升。
- 可见 test 是训练谱的拷贝，`lib_max≈1.0` → 几乎全部被 `LIB_TAU=0.90` 保护，**可见集 0 命中是预期行为，不能据此判断隐藏集覆盖**（[069-retrieval-design.md](069-retrieval-design.md) 里「0 hits」要按这个口径读）。
- 隐藏集能吃到 overlay 的只有「class 1 且训练库里没有同结构谱」的那部分；而知识表显示这类记录只占公开合规谱的 0.6%。**这就是为什么继续堆谱库不会动 0.492。**

---

## 4. 预训练权重 / 嵌入：能不能挂、许可干不干净

许可裁定以 host 书面回复为准：本篇直接核对了 [750330](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/750330)（NC 开放、NIST 禁止）、[741359](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741359)（欢迎帖修订）与 [749217](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/749217)（NIST 派生与 checkpoint 口径、GNPS-NIST14-MATCHES 例外）；其余条目引自社区汇总表 [748270](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/748270) 与 [069-frontier-models.md](069-frontier-models.md) §1.2。要点：CC-BY-NC 可用；MassSpecGym / MIST / DreaMS / GLACIER / ICEBERG / CFM-ID 4 / ChEBI / Enveda-180 均可用；**NIST 及其派生权重禁用**（GNPS-NIST14-MATCHES 例外）。

| 模型 | 权重来源 | 可否离线挂载 | 许可 | host | 断网可跑 | 结论 |
|---|---|---|---|---|---|---|
| **DreaMS** | v4g `hengck23-dreams-enveda-casmi26`（权重 386 MB + 嵌入约 6.5 GB，包 total_bytes 5.7 GB） | ✅ 已挂 | MIT | ✅ | ✅ GPU | **唯一被两篇 069 同时看好的新信号**；现状只是空槽补尾，且分支缺 `dreams_np_*.npy` 恒不触发 |
| **ICEBERG / GLACIER（MassSpecGym）** | v4g `casmi26-iceberg/glacier`；`prvsiyan/casmi26-forward-reference-stack` | ✅ 已挂 / 3.83 GB 可加 | MIT 代码 | ✅ | ✅ | 管线已在用 `msg_all` 版 |
| **timsTOF 微调 ICEBERG** | `hjyact/casmi26-timstof-iceberg` | ✅ 221 MB | **MIT** | 推断：MassSpecGym 派生 → ✅ | ✅（有 py3.13 wheel 依赖） | ⭐ 新信号 + 同仪器，**排序第一** |
| **MIST（MassSpecGym）** | `ahmedberatozer/casmi26-mist-msg-v1` | ✅ 104 MB | 权重 CC-BY-4.0 / 代码 MIT | ✅（2026-11-16 明确） | ✅ GPU | 可作第二套谱→指纹集成；集成成本高 |
| **MS2Mol** | 无 Kaggle 包；代码 MIT，权重发布状态需再核 | ❌ 需自建 | MIT（代码） | 未单独裁定（开源公开） | 需打包 + GPU | 工作量大，收益不明；**次选** |
| **CSI:FingerID / SIRIUS** | 随 SIRIUS 发布 | ❌ | AGPL + **学术非商用**；SVM 训练含 NIST | ❌ | — | **出局**（许可 + NIST 双重） |
| **CFM-ID 4** | 源码/权重公开；预测谱可由 ISDB 预计算 | ✅（ISDB 1.1 GB） | Apache-2.0 / CC BY 4.0 | ✅ | ✅ | 预测谱已有公开负结果（−0.009）；若要试，只当特征不当标签 |
| **MS2DeepScore** | Zenodo 13897744（Apache-2.0 / CC BY 4.0） | ❌ 需自建 | 干净 | 未单独裁定（开源公开） | ✅ CPU/GPU 轻 | 训练集为 GNPS/MoNA/MassBank/MSnLib，无 NIST；**可替代 N-1c 的 modified cosine**，但需打包 |
| **Spec2Vec** | 随 MS2Query 公开库 | ❌ 需自建 | CC BY 4.0 | 未单独裁定 | ✅ CPU | 同类任务弱于 DreaMS；**不推荐** |
| FPNet 系列 | v4g / fold-safe | ✅ 已挂 | 比赛条款 | ✅ | ✅ | 已在用 |
| FRIGID | `asifmuddin/frigid-checkpoints-casmi2026` | ✅ 3.25 GB | Apache 2.0 | MassSpecGym 派生 | ✅ | 体量大、未验证；**次选** |

**判断**：模型侧的增量是真实的，但**只有 DreaMS 与 timsTOF-ICEBERG 是「已挂/可直接挂 + 许可干净 + 接得上现有通道」**。CSI:FingerID 与任何含 NIST 的权重应当直接从候选里删掉。

---

## 5. 最值得加的 5 个（按期望值排序）

排序依据：`预期作用 × 可接入性 ÷ (体积 × 9h 风险 × 许可风险)`。预期增益是**估计**，不是实测；只有公榜出分才算数。

| 排名 | 资产 | 类型 | 大小 | 许可 | 预期作用 | 接入点 | 9h 预算风险 | 许可风险 | 预期增益（估） |
|---:|---|---|---:|---|---|---|---|---|---|
| **1** | `hjyact/casmi26-timstof-iceberg` | 前向模型（timsTOF 微调） | 221 MB | **MIT** | 谱–结构一致性重排；作者留出集 +0.030/+0.018，双 ckpt +0.036/+0.017 MRR@25 | cell 25 之后新增 25b，`ICE_W=0.3–0.5`、`ICE_TOPN=20`、`ICE_CKPT=tims_ft3,tims_ft2`；失败即原样返回 | 中：多跑 1–2 个 ICE 检查点，约 +10–25 min（T4）；wheel 需匹配镜像 Python | 低（MIT；微调数据=train，host 已认可 train 派生） | **+0.005 ~ +0.02** |
| **2** | `aryanawasthiteykik/casmi26-e180-not-in-train` | 实测谱 + 留出验证 | 50.8 MB | **CC BY 4.0** | 3,719 个不在 train 的结构 × 14,471 张 timsTOF 谱：①同仪器留出验证 ②overlay Source C | 离线验证脚本；或 cell 25 的 `refs` 追加第三来源（同样 10 ppm / 0.80 / ≥4 峰） | 低：验证 0；作 Source C 约 +2–5 min | 低（host 已批准 Enveda-180 Zenodo） | 直接提分 **0 ~ +0.005**；**验证价值高** |
| **3** | `yehudaitkin/casmi-zenodo-top128-novel-spectra` | 实测谱（去重增量） | 5.5 MB | **CC BY 4.0** | 11,561 谱 / 3,782 结构，作者已删掉与 train 重复的「结构-加合物」对 → 唯一干净的身份检索增量 | 与 #2 同一条 Source C 路径，按来源加权（每来源 cap 2 条） | 低：索引 1.2 万谱 <1 min | 低（GNPS/MassSpecGym 派生，均获准） | **+0.002 ~ +0.008**（依赖隐藏集 class 1 占比） |
| **4** | 启用已挂的 **DreaMS**（无需新数据） | 谱嵌入重排 | 0（已挂 5.7 GB） | MIT | 谱–结构对比嵌入，给 N-1c 之外的第二种「谱–谱/谱–结构」相似度；现有 N-1c 只是测试集内部网络 | cell 25 之后 25b：对 query 与 top-25 候选算嵌入相似度，与 `library_max`/`cos` 做 z-score 融合；先只动 rank 2–25 | 中：1.2k 可见谱 / 隐藏 ~1.5k 谱 GPU 推理约 +5–10 min；显存与主模型串行即可 | 低（host：「DreaMS weights are allowed」） | **+0.003 ~ +0.008**（[069-frontier-models](069-frontier-models.md) 同结论） |
| **5** | `ahmedberatozer/casmi26-mist-msg-v1` | 谱→指纹检索模型 | 104 MB | 权重 CC-BY-4.0 / 代码 MIT | 第二套独立谱→指纹信号，与 FPNet（A+B+fold0）集成，降低单模型系统性偏差 | Engine 1 打分的并列模型；或 PubChem 通道第四模型（需改 `ModelBank` 与 `fp_bits` 对齐） | 高：集成/对齐成本最高，+10–20 min，易触发 shape/bit 不一致 | 中（权重 CC-BY-4.0 可再分发，仍需署名） | **+0.002 ~ +0.01**（不确定最大） |

### 5.1 每条的落地要点

**#1 timsTOF ICEBERG**
- 为什么值得：管线现在的 ICEBERG/GLACIER 是 MassSpecGym `msg_all`（Orbitrap 为主），而测试**全是 Bruker timsTOF**。这是唯一的「同仪器」前向模型，且作者做了结构留出（400+400 查询）。
- 怎么接：`run_ice.py submission.csv` 只做排名置换，不增删候选；融合式 `1/(3+r_list) + w·1/(3+r_ice)`，与 068a 现有的 leader-safe 思路一致，失败即原样返回。
- 风险：包里 wheel 面向 Python 3.13（另有 RDKit cp313 fallback）；068a 的 docker 未必是 3.13 → 先只跑 `ice_rerank.py`，用镜像自带 torch，不装它的 wheel。先 smoke 12 个可见分子确认能载入 ckpt。

**#2 Enveda-180 留出**
- 这是本审计里**最被低估**的资产：项目现在只有 panel（250 个训练集内分子）和可见 test（训练谱拷贝）两种验证，都无法回答「对没见过的结构有没有用」。这 3,719 个结构**不在 train**，同仪器、同实验室来源，是判断任何新信号的第一道闸。
- 用法：把 `submission.csv` 的候选列表在 3,719 个结构上算 MRR@25 的「候选覆盖率」（不要求排名对，只问答案在不在列表里），比公榜 5 次/天的额度便宜得多。
- 注意：它是 drug-like（MW 中位 324），与测试的天然产物分布有偏移；只能当**上界/regression 检查**，不能当公榜代理。

**#3 novel spectra**
- 5.5 MB 换 11,561 张「结构-加合物不在 train」的谱，是全场性价比最高的身份检索增量。
- 接入时必须做**来源 cap**：同一结构同来源最多 2 条，否则 MassBank 式多记录会把某个候选抬过 0.80 阈值（[069-retrieval-design.md](069-retrieval-design.md) §风险）。

**#4 DreaMS**
- 不要新增数据：权重与嵌入都已在 v4g 里。要做的是**写一条真正的打分路径**，而不是现在的空槽补尾。
- 最小实验：只在 rank 2–25 上用嵌入相似度做 z-score 融合，rank 1 保持现有 leader 规则（与 066a/067a 的中性结果同构，风险可控）。

**#5 MIST**
- 只有在 #1–#4 都无效时才值得动。它的价值是「完全不同的训练目标（fingerprint + retrieval）」，但 `fp_bits` 位序、模型输出层、归一化都要对齐，工作量是前四条之和。

---

## 6. 明确不建议加的

| 资产 | 理由（有数） |
|---|---|
| `qwertykeypadapple/casmi26-lotus-candidates`（6 MB，CC0） | 063b 已试无增益；回填只在槽位未满时触发，而 25 槽是满的；LOTUS 结构绝大多数已在 PubChem 内，不增召回 |
| `aryanawasthiteykik/casmi26-gnps-community-cc0`（425 MB，CC0） | **实测 98.07% 结构已在 train**，净增 1,141 结构 / 2,597 谱；见 §3.1 |
| `takumuhata/casmi26-con-train-ext`（991 MB，CC0） | 181k 外部谱 + 训练 pack，覆盖与 extlib 同源；体积换不到新结构 |
| MassBank 各种「合规重打包」（4 份，共 ~170 MB） | 与已挂 MassBank 同源，只是许可切法不同；host 已允许 NC，切了不产生新谱 |
| `hemantkumarbk/isdb-np-insilico-msms-library`（1.1 GB，CC BY 4.0） | 预测谱不是实测；作者自报公开榜 −0.009（0.459→0.450） |
| `prvsiyan/casmi26-forward-reference-stack`（3.83 GB） | 与已挂 ICEBERG/GLACIER 同模型同权重，只是自带 wheel；重复 |
| `sanpier/catalogue-experiment-saner`（25.6 MB，Unknown） | 068a 代码零引用；许可 Unknown，不该出现在获奖方案里 |
| CSI:FingerID / SIRIUS / 任何 NIST 派生权重 | 学术非商用 + 训练含 NIST，host 明确「使用或训练于 NIST 一律取消资格」 |

---

## 7. 给下一轮的三条可执行动作

1. **先补验证，再动分数**：把 `casmi26-e180-not-in-train` 接成离线评测（候选覆盖率 + MRR@25），用 068a 现有 `submission.csv` 跑一次基线。这一步不花 GPU、不花提交额度，却决定后面三条要不要做。
2. **只做一个打分实验**：cell 25 后加 25b 接 timsTOF ICEBERG（`ICE_W=0.3`、`tims_ft3` 单 ckpt），smoke 通后提交一次。这是当前**证据最强、接入最浅**的新信号。
3. **顺手清理死重**：`sanpier/catalogue-experiment-saner` 从 `dataset_sources` 移除（零引用 + Unknown 许可）；若 attach 时间是瓶颈，`casmi26-pubchem-popularity-prior` 的 68.8 GB 指纹分片是唯一值得换掉的大件。二者都不改分数，只降低运行与合规风险。

> 所有「预期增益」都是基于公开留出实验与本队历史的估计，**没有一条经过公榜验证**。按项目约定，只有实际提交出分才记为有效。
