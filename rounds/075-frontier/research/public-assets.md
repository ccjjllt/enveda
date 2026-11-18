# Enveda CASMI 2026 公开资产盘点

**盘点时间**：2026-11-18
**比赛**：`enveda-CASMI26-molecule-id-mass-spectra`（competition id 143318）
**目的**：在「同分异构体判别」是唯一剩余空间的前提下，找出**断网 final notebook 能离线挂载**、且**规则允许**的真实权重与真实结构库。
**方法与边界**：全部结论来自 Kaggle MCP 的元信息/文件清单/讨论区正文，以及 Zenodo、GitHub 页面；**没有下载任何大文件**。`search_datasets` 显示的 `total_bytes` 有时与文件清单不符（见 `casmi26-fold-safe-fpnet` 标 0 B 实为 1 GB），凡冲突处以文件清单为准并在备注标明。

> ⚠️ 前置结论：`heliosli/casmi26-0-495-public demo`（0.495）是空壳——权重已删、catalogue 是 3,683 个合成分子，**本表不收录，也不计入可用资产**。

---

## 1. 资产总表

许可证列取自 `search_datasets` 的 `license_name`；「真实权重」= 文件清单里存在可加载的 `.pt/.ckpt/.npz` 张量文件（不是只有代码或只输出 CSV）；「可离线」= 断网 notebook 直接挂载即可加载，不需要运行期联网下载。

### 1.1 含真实预训练权重的数据集

| ref | 类型 | 大小 | 许可证 | 内容 | 真实权重 | 可离线 | 最后更新 / 备注 |
|---|---|---|---|---|---|---|---|
| `dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets` | 权重+候选集聚合包 | 10.74 GB | CC BY-NC-SA 4.0 | FPNet 三套、GLACIER、ICEBERG、LightGBM ranker、DreaMS、`casmi26-v4b-models` 全套代码，外加 COCONUT/PubChem/ChEBI/LIPID MAPS/PubChemLite 候选与 RDKit 安装包；每文件带作者、许可证、SHA256 | 是（`.pt`/`.npz` 数十个） | 是（整包挂载，自带 MANIFEST 与安装包） | 2026-10-28；135,755 次下载；作者强调「除一个自有谱→结构模型外，每个文件都是公开资源的原样副本」。**注意**：包内含 `casmi26-ranker-features`、`casmi26-simulated-ranker-rows` 等被 host 判定为「无生成器即不可复现」的中间产物（见 756893） |
| `dmitriigluzdov/casmi26-fold-safe-fpnet` | FPNet 5 折检查点 + 训练/评估脚本 | 元数据标 0 B，**文件清单实为 5×198.8 MB ≈ 1.0 GB** | Other（描述内声明） | `fpnet_fold0..4.pt`、`make_split.py`、`eval_ckpt.py`、`identity_audit.json`、`SPLIT_AUDIT.json`、逐折 `fivefold_retrieval_fold*.json` | 是 | 是 | 2026-10-21；每折留出一个 identity fold，是我们做「未见结构」自评最干净的公开基线 |
| `sanpier/casmi-spectrum-model-v1` | 谱→结构打分小模型 | 48.97 MB | Apache 2.0 | 3 个 11.9 MB `.pt`（best/last/final）、`model.py`、`features.py`、`train.py`、`prepare.py`、`inference.py`、`training_history.json`、`data_audit.json` | 是 | 是（训练+推理代码齐全） | 2026-10-23；25,535 次下载。作者自己在描述里写明「schema 兼容性修复不应被理解为对已发布检查点的独立复现」——即**未自证复现**，引用时按此处理 |
| `hengck23/hengck23-dreams-enveda-casmi26` | DreaMS 权重 + 预计算 embedding | 5.71 GB | MIT | `embedding_model_state_dict.pt`（386 MB，精简 DreaMS）、`Enveda180_DreaMS_embeddings.npy`（4.73 GB）、`MassSpecGym_DreaMS_embeddings.npy`（947 MB）、`dreams_minimal_model.py`、`run_demo.py` | 是 | 是（自带精简模型定义，不依赖 DreaMS 安装） | 2026-10-01；**最省算力的 DreaMS 用法**：不必自己跑 4.7 GB 谱，直接读向量 |
| `ahmedberatozer/casmi26-nx-model` | 同分异构体判别模型（NX） | 469.7 MB | Other（描述：代码 MIT，数据沿用源许可，非商用+署名） | `nx_full1_final.pt`（生产，253 MB）、`nx_fold0_final.pt`（基准，253 MB）、`nx_train.py`、`nx_rerank.py`、`nx_rules.py`、两份 history | 是 | 是（训练/推理/规则模拟脚本齐全） | 2026-10-25；**与我们缺口最对口**：InfoNCE 对比学习，正样本是 ±10 ppm 同质量池里的真实同分异构体。fold0 用 Gluzdov 的 fold-safe 初始化并留出 250 个 enveda-np 结构 |
| `ahmedberatozer/casmi26-v4b-models` | 双引擎主模型包 | 1.23 GB | Other | 55 个文件：`casmi/`（engine/ranker/library/fpnet/dreams_model/…）+ `fe_v4/` 全特征工程代码 + 权重 | 是 | 是 | 2026-09-26；35,002 次下载；训练代码另发在 `ahmedberatozer/casmi26-training-code` |
| `ahmedberatozer/casmi26-iceberg` | ICEBERG 谱模拟器 | 64.2 MB | Other | 检查点 + 运行代码 | 是 | 是 | 2026-09-26；15,715 次下载 |
| `ahmedberatozer/casmi26-glacier` | GLACIER 谱模拟器 | 56.3 MB | Other | 检查点 + 运行代码 | 是 | 是 | 2026-09-28；15,169 次下载 |
| `prvsiyan/casmi26-fp-models-v2` | 指纹模型（第二引擎） | 267.3 MB | CC BY-NC-SA 4.0 | `fp_single_s2.pt`、`fp_merged_m1.pt`（各 144.1 MB） | 是 | 是 | 2026-09-16；生成/训练脚本在 `prvsiyan/casmi26-generation-code`（Apache-2.0），因此**不属于** 756893 那类「无生成器」资产 |
| `dreaddevelopment/sylva-casmi-models` | 集成指纹模型 | 344.8 MB | CC0 | `sylva_model_0..2.pt`（各 87.2 MB）、`structs.parquet`、`coco.parquet`、`labels_packed.npy` | 是 | 是 | 2026-09-15；🔍 作者无公开战绩、无训练代码、无 notebook 引用链，来源与可用性未经第三方验证 |
| `thivnti/casmi2026-v2-assets` | 谱编码器 + 候选目录 | 197.0 MB | CC0 | `spectral_fingerprint_best.pt`（130.7 MB）、`catalog.parquet`、`morgan_packed.npy` | 是 | 是 | 2026-09-17；🔍 同 sylva，CC0 声明宽松但来源未核实 |

### 1.2 含真实结构库 / 谱库（无权重）

| ref | 类型 | 大小 | 许可证 | 内容 | 可离线 | 最后更新 / 备注 |
|---|---|---|---|---|---|---|
| `prvsiyan/coconut-casmi26-candidates` | COCONUT 候选 | 148.0 MB | CC BY-NC-SA 4.0 | 436,389 行：中性单同位素质量 + 6,930 bit 打包指纹 + 对齐标识符 | 是 | 2026-09-15；索引选择规则可复现（见 758233 回复） |
| `xjoannax88/casmi-coconut-candidates` | COCONUT 候选 | 29.2 MB | **CC BY 4.0** | 495k 天然产物，按官方 metric key 组织 | 是 | 2026-10-30；比 NC 版许可更干净，🔍 作者无战绩，行数多于 COCONUT 2.0 官方 436k，需抽样核对 |
| `xjoannax88/casmi-pubchemlite-candidates` | PubChemLite 候选 | 31.2 MB | **CC BY 4.0** | 347k 代谢物/药物/化学品，官方 metric key | 是 | 2026-10-30；补 candidate recall 用 |
| `dmitriigluzdov/casmi26-pubchem-popularity-prior` | PubChem 文档度先验 | 43.8 GB | Other | 1.06 亿结构 + 71.1 万公开候选池的 substance/PubMed/专利/MeSH/同义词/LOTUS 计数，按质量排序、逐行对齐 | 是（体积过大，需预筛） | 2026-10-27；作者实测「同质量里文献最多的那个就是答案」占 66%，是同分异构体判别的一个**非谱学**先验 |
| `dmitriigluzdov/casmi26-natural-product-knowledge-table` | 天然产物知识表 | 84.1 MB | Other | 152,717 个 NP：化学分类、物种、scorer key、PubChem 计数 | 是 | 2026-10-23；只能做先验/分组，不直接判别异构体 |
| `samartalwar/casmi-2026-spectral-library-massbankharmonized` | MassBank 谱库 | 38.0 MB | CC BY-NC-SA 4.0 | 139,744 条 MS/MS：precursor m/z、adduct、离子模式、CE、RT、全峰表 + 指纹 | 是 | 2026-09-17；整理版，无原始构建脚本 |
| `navyadevadiga/casmi26-massbank-lib-own` | MassBank 谱库（可重建） | 31.4 MB | CC BY-NC-SA 4.0 | 同 139,744 条，**附构建脚本** | 是 | 2026-11-14；与上一行同源，但可复现性更好，做独立谱验证优先选这个 |
| `sergiomalv/casmi-open-msms-spectra` | 多源开放谱库 | 212.1 MB | CC BY-SA 4.0 | 421k 条 MS2 / 60k 结构，逐行记录 source 与 licence（MassBank、GNPS2、MultiMS2） | 是 | 2026-10-07；**许可最干净**的一份，适合对外声明 |
| `sergiomalv/casmi-chebi-lmsd-structures` | ChEBI + LIPID MAPS 结构 | 30.7 MB | CC BY 4.0 | 98k 标准化结构 + 指纹 | 是 | 2026-10-07 |
| `aryanawasthiteykik/casmi26-gnps-community-cc0` | GNPS 社区谱库 | 336.0 MB | CC0 | 506,939 条谱 / 59,039 结构，62 个直贡库，已转成 train.parquet schema | 是 | 2026-11-13；直接 concat 进训练集 |
| `kozykappa/casmi26-formula-and-fragmentation` | 分子式/中性丢失表 | 12.1 MB | CC BY 4.0 | 1.25M 分子式、82 个中性丢失、MS/MS 特征工具 | 是 | 2026-10-10；规则层资产，非权重 |

### 1.3 公开 model（`list_models`）检索结果

`search_models` 在本会话的 MCP 里不存在；正确的读接口是 `tools.kaggle.list_models`。用它按 `mass spectrum` / `enveda` / `dreams` / `molecule` 检索，**没有任何与本比赛相关的 Kaggle Models 条目**（返回空或无关）。结论：本比赛的公开权重**全部以 dataset 形式发布**，不在 Kaggle Models 目录里。

外部（非 Kaggle）可离线预置的权重：

| 来源 | 内容 | 许可证 | 离线 | 备注 |
|---|---|---|---|---|
| DreaMS（`pluskal-lab/DreaMS`，权重 Zenodo 10997887） | GeMS 自监督预训练谱编码器，1024 维 | 代码 MIT | 是（预置 ckpt） | host 明确：「Officially released DreaMS checkpoints are fine to use」（754610）；「DreaMS weights are allowed」（745604） |
| MIST（Zenodo 23303289，16 GB） | `mist_msg_noaug/aug`、`mist_canopus_noaug/aug`（各 289 MB）+ `glacier_msg/canopus.ckpt`（各 54 MB）+ MassSpecGym compact 特征 | **MIT** | 是 | host 明确：「MassSpecGym and MIST models are fine」（757307）。`msg_synthetic.tar.gz` 12.1 GB 可跳过 |
| MassSpecGym（HF `roman-bushuiev/MassSpecGym`） | 谱+结构数据集 | 公开 | 是 | host：「we consider MassSpecGym and models trained on it and released with permissive licenses as fair game」（742991，经 748270 转引） |
| ICEBERG / GLACIER | 谱模拟器 | 随 MassSpecGym 生态 | 是 | host：「GLACIER/ICEBERG, DreaMS and ChEBI/LIPID MAPS are all fine」（745841） |
| Enveda-180 Zenodo release | 天然产物谱+结构 | 见 Zenodo | 是 | host：「Data in the Zenodo release of Enveda-180 are fine to use」（745148） |

---

## 2. 规则结论（外部数据边界）

### 2.1 主判决：750330（sticky，2026-10-17，31 赞，host David Healey）

原文要点（逐条转述，非改写）：

1. **放开非商用许可**：「we've decided to remove the restriction on noncommercial license open source data (for example, CC-BY-NC)… solutions can use external data that is open sourced with noncommercial provisions.」
2. **其余仍然禁止**：「External data that is not open source, is not freely available to all, or is proprietary remains against the rules.」
3. **NIST 明确禁止**：「NIST is not permitted, as it is not open or freely available and the data including derivatives such as model weights cannot be redistributed.」
4. **保留追责**：「we reserve the right to check all solutions for compliance and remove from the competition participants found in violation.」

该主题下 2 条回复均已删除（`is_deleted: true`），无可引用内容。

### 2.2 COCONUT：759916（2026-11-17，host Marie Killian）

提问者担心 COCONUT 虽为 CC0，但底层含 NPAtlas、KNApSAcK 等更严许可。host 一句话回复：

> 「COCONUT is allowed. If any dataset is prohibited by our rules, it would need to be excluded.」

即：**COCONUT 整体允许**，不需要逐个子库剔除；责任落在「若某数据集被规则禁止则须排除」。

### 2.3 host 逐项裁定汇总（748270 的表格 + 本次复核的原始帖）

| 数据/模型 | 允许 | host 原话 | 出处 |
|---|---|---|---|
| `train.parquet` 及其训练产物 | ✅ | 「can be treated effectively as open-source and any model trained on it is fine, redistributed artifacts are fine」 | 745841 |
| 仅因继承 train.parquet 而变 NC 的数据集 | ✅ | 「any datasets that are only noncommercial by inheriting from train.parquet are fine」 | 745072 |
| MassSpecGym 及其训练模型 | ✅ | 「MassSpecGym and models trained on it… fair game」 | 742991 |
| **MIST**（MassSpecGym 训练，MIT） | ✅ | 「**MassSpecGym and MIST models are fine**」 | **757307（本次直接读到）** |
| GLACIER / ICEBERG | ✅ | 「MassSpecGym is allowed, so is GLACIER」 | 744338 / 745841 |
| DreaMS（官方发布检查点） | ✅ | 「**Officially released DreaMS checkpoints are fine to use**」 | **754610（本次直接读到）** |
| ChEBI / LIPID MAPS | ✅ | 「GLACIER/ICEBERG, DreaMS and ChEBI/LIPID MAPS are all fine」 | 745841 |
| CFM-ID 4 | ✅ | 「CFM ID 4 is fine」 | 743774 |
| GNPS / MassBank / MoNA 新版本 | ✅ | 「Newer releases of public datasets are fine… CC BY-SA records are acceptable」 | 745228 |
| GNPS-NIST14-MATCHES（在 GNPS 内） | ✅ | 「GNPS-NIST14-MATCHES in GNPS is fine to use as…」 | 757307 |
| Enveda-180 Zenodo | ✅ | 「Data in the Zenodo release of Enveda-180 are fine to use」 | 745148 |
| PubChem 结构（检索用） | ✅ | 「PubChem structures for retrieval are acceptable」/「submitting subsets of PubChem is fine」 | 741857 |
| COCONUT | ✅ | 「COCONUT is allowed」 | 759916 |
| **NIST 本体及任何 NIST 训练产物** | ❌ | 「any solution using or trained on NIST will be disqualified」 | 744470 |
| **新建的 NIST 派生数据集** | ❌ | 「NIST would consider any newly created such datasets to be derivatives and in violation of their license」 | 745832 |
| SIRIUS / CSI:FingerID | ❌🔍 | host 未直接回答；训练集不公开且可能含 NIST。**推断**按 750330 第 2 条判为不可用 | 751373（host 未答） |

### 2.4 三条容易被忽略的「可复现性」硬约束

这几条比许可证更容易让人翻车，且直接决定第 4 节短名单：

- **756893**（host David Healey）：「Solutions that load intermediate data artifacts without being able to reproduce those artifacts are **not reproducible**.」
  → 只 load 别人的 `rank_train.npz`、`sim_rank_rows.npz` 这类无生成器的中间表，不算可复现。这是 `casmi26-attributed-v4g-reproduction-assets` 里少数几个子目录的真实风险点。
- **754610**（host）：「Must a final submission be reproducible end-to-end from the competition data and openly documented resources? **Yes.**」同时强调保留取消资格的权力。
- **758233**（host Marie Killian）：「There needs to be justification for the choice of indices.」
  → 指纹位索引列表必须能解释/重建。社区已在同帖给出可 bit-exact 重建的两份规则：`prvsiyan` 的 6,930 bit（`prepare_legacy_exports.select_fingerprints`）与 `ahmedberatozer` 的 10,226 bit（`casmi/structs.select_bits`），并已用 train.parquet + RDKit 复算通过。
- **749217**（host 回复 + 提问者自述）：approved 列表**不是上限**——「Competitors are not limited to the external datasets that have been explicitly approved by name… provided they comply」。但「NIST 派生是否穿透到 fine-tune」仍未有一句话裁决（🔍 悬空风险）。
- **751373**：MS2DeepScore 2.0 / FIORA-OS 是否可用，host **未回复**（3 条评论中只有提问者与一位参赛者）。🔍 只能按 750330 的通则推断：公开训练代码 + 公开训练数据 → 可用。

### 2.5 一句话结论

**许可维度**：CC0 / CC BY / CC BY-SA / CC BY-NC / CC BY-NC-SA 全开；只有 NIST 及其派生、专有、非公开三类别碰。
**工程维度**：真正的门槛不是许可而是**可复现**——最终提交要能从比赛数据 + 公开文档化资源端到端重建，权重本身最好带训练代码。

---

## 3. top 队伍公开物清单

**核对方法**（不猜）：
1. `search_notebooks(competition=…, pageSize=100, sortBy=VoteCount/DateCreated)` 拉全量公开 notebook，共 **100 个、63 位作者**；`nextPageToken` 为空，可视为全量。
2. 把 leaderboard 前 40 名的 `team_name` 做小写去符号归一化，与 63 位作者逐一比对：**匹配数 0**。
3. 逐个 `search_notebooks(user=<该队名>)` 抽查 6 个可用队名（goonew / pjmathematician / giantwizard / parfyon / savchenko / shehabalshehabi）：前 5 个返回 0 篇，`pjmathematician` 有 20 篇但与化学/质谱无关（PII、UBCO、GGUF 等）。
4. 讨论区 80 个主题的标题全部过了一遍，**没有**任何 top 队伍署名的 solution writeup；`get_writeup_by_topic` 无对应条目。

| # | 队伍 | 公开 notebook | 公开 writeup | 备注 |
|---|---|---|---|---|
| 1 | Shehab Alshehabi | 没有 | 没有 | 0 篇 notebook；仅在 745072、746302、756912 等帖发过评论 |
| 2 | democat | 没有 | 没有 | 同名用户 5 个，均无本赛 notebook |
| 3 | Oleksandr Parfionov | 没有 | 没有 | `parfyon` 0 篇 |
| 4 | Savko | 没有 | 没有 | `savchenko` 0 篇 |
| 5 | goonew | 没有 | 没有 | 0 篇 |
| 6 | 한 밭 대 학 교 컴 퓨 터 공 학 과 | 没有 | 没有 | 检索无命中 |
| 7 | Terry Luo | 没有 | 没有 | 同名 6 个用户，均无本赛 notebook |
| 8 | Jaret Arnold | 没有 | 没有 | — |
| 9 | Paweł Biernat | 没有 | 没有 | `pawebiernat` 无本赛作品 |
| 10 | tu tu tu du | 没有 | 没有 | 检索无命中 |
| 11 | BMW | 没有 | 没有 | 检索只命中汽车数据集 |
| 12 | pjmathematician | 没有 | 没有 | 有 20 篇公开 notebook，**全部与化学无关** |
| 13 | muse | 没有 | 没有 | 同名 6 个用户，均无 |
| 14 | JTA | 没有 | 没有 | — |
| 15 | grinders | 没有 | 没有 | 同名用户是另一位（Dmitry Zhukov） |
| 16 | HMC_Stevelee | 没有 | 没有 | 有 1 个公开 dataset `datapublic11`，与本赛无关 |
| 17 | GiantWizard | 没有 | 没有 | 0 篇 |
| 18 | Shuta TAKANO | 没有 | 没有 | 0 篇 |
| 19 | Sirui Zeng | 没有 | 没有 | 无本赛作品 |
| 20 | Johnny Lee | 没有 | 没有 | 同名 6 个用户，均无 |
| 21 | huytngo | 没有 | 没有 | — |
| 22 | pikachu | 没有 | 没有 | — |
| 23 | Aneesh & Yash | 没有 | 没有 | 队名非用户名，无法单独定位；但不在 63 位作者内 |
| 24 | Kevin E R MILLE | 没有 | 没有 | — |
| 25 | DragonDu | 没有 | 没有 | — |
| 26 | MY | 没有 | 没有 | 队名过短，仅能确认不属于 63 位作者 |
| 27 | LH-and-FPGA | 没有 | 没有 | — |
| 28 | Szymon Roziewski | 没有 | 没有 | — |
| 29 | l1aF | 没有 | 没有 | — |
| 30 | Amil Gentili | 没有 | 没有 | — |

**✓ 结论：前 30 名（实查前 40 名）没有任何一个队公开了 notebook 或 writeup。**

**⚠️ 口径说明**：该比对基于「队名 ≈ 用户名」的字符串归一化。多人合并队的队名（如 `Aneesh & Yash`、`MY`、`한밭대학교…`）与用户名不一致，理论上存在漏判；但两个独立路径（全量 notebook 作者集、逐用户检索）都指向「无」，且榜单头部与公开活跃作者集完全不相交，所以该结论可以按「无」采用。

**📌 反向观察（对研究有用）**：公开 notebook 生态里最活跃的 63 位作者，最高分是 `mathleteutd` 的 0.492 与 `alexchilton` 的 0.488，全部落在 top-30 之外。**公开前沿 0.494 与榜首 0.550 之间的 0.056，没有任何公开文字解释来源**——这与我们「剩余空间只在同分异构体判别」的判断一致，也说明无法从公开物里抄到那 0.056。

---

## 4. 「可以直接用」短名单（≤5）

按「离线可加载 + 规则干净 + 带生成/训练代码 + 对同分异构体判别有直接作用」四条筛，只留 5 个：

| # | 资产 | 一句话用途 |
|---|---|---|
| 1 | `dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets`（10.74 GB） | 一个挂载就把 FPNet / GLACIER / ICEBERG / DreaMS / LightGBM / 全候选集和训练好的 RDKit 装好，作为 068a 之外第二条独立引擎的**离线底座**；只需避开包内 `ranker-features`、`simulated-ranker-rows` 两个无生成器的子目录。 |
| 2 | `ahmedberatozer/casmi26-nx-model`（469.7 MB） | 唯一公开的**同质量池对比学习判别器**（±10 ppm 真实异构体作正样本），正好补我们 068a 判别力不足的那一段；`nx_rerank.py` 可直接把它的分数并进现有 ranker。 |
| 3 | `dmitriigluzdov/casmi26-fold-safe-fpnet`（实际 ≈1.0 GB） | 5 折 identity-disjoint 的 FPNet + 逐折审计文件，用来做**「未见结构」自评**，避免再拿可见占位 test 的命中率当泛化。 |
| 4 | `navyadevadiga/casmi26-massbank-lib-own`（31.4 MB，或 `sergiomalv/casmi-open-msms-spectra` 212.1 MB 取更干净许可） | 139,744 条 MassBank 谱**带构建脚本**，用于「已见结构」的独立谱验证与库检索加分项，与 identity-disjoint 口径严格分开。 |
| 5 | `xjoannax88/casmi-pubchemlite-candidates`（31.2 MB，CC BY 4.0） | 347k PubChemLite 候选按官方 scorer key 组织，补 candidate recall 且许可是全表最宽松的一档，适合写进最终的 external-data 声明。 |

**没进短名单但值得记一笔**：
- `sanpier/casmi-spectrum-model-v1` — 代码最完整（train+prepare+inference），但作者自述未自证复现，只能当 baseline 不能当证据。
- `dmitriigluzdov/casmi26-pubchem-popularity-prior` — 43.8 GB，是唯一「非谱学」的异构体先验（66% 命中率），值得单独做一次小样本验证再决定是否上最终管线。
- `dreaddevelopment/sylva-casmi-models`、`thivnti/casmi2026-v2-assets` — CC0 且有权重，但无战绩、无训练代码、无引用链，🔍 先当噪声处理。

---

## 附：引用索引

| 类型 | 地址 |
|---|---|
| 规则主帖 | https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/750330 |
| COCONUT 许可 | https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/759916 |
| host 裁定汇总表 | https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/748270 |
| MIST / MassSpecGym | https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/757307 |
| DreaMS 检查点 | https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/754610 |
| 中间产物可复现性 | https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/756893 |
| 指纹位索引 | https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/758233 |
| 第三方预训练模型 | https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/751373 |
| MIST 权重（Zenodo） | https://zenodo.org/records/23303289 （MIT，16 GB，2026-10-20） |
| DreaMS 代码/权重 | https://github.com/pluskal-lab/DreaMS （MIT）→ https://zenodo.org/records/10997887 |
| 聚合资产 | https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets |
| NX 异构体模型 | https://www.kaggle.com/datasets/ahmedberatozer/casmi26-nx-model |
| Fold-safe FPNet | https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-fold-safe-fpnet |
| DreaMS embedding | https://www.kaggle.com/datasets/hengck23/hengck23-dreams-enveda-casmi26 |
