# 可用模型与公开资产清单（071/075 合并精简版）

盘点基准：2026-11-19晚间。目标：在**不重训原生模型、断网、T4 ≤ 9h**约束下，筛选候选-谱判别模型与外部资产。最新框架比较、0.495证据和实验顺序见[本轮深度研究](next-frameworks.md)。公开代码、可下载权重、训练来源完整、实际提分分开判断。

## 1. 规则与许可硬边界（Host 裁定）

- **允许**：`CC0` / `CC BY` / `CC BY-SA` / `CC BY-NC` / `CC BY-NC-SA` 开源外部数据与预训练模型（`750330`）；`train.parquet` 及其派生、`MassSpecGym` 与 `MIST`（`757307`）、`DreaMS` 官方 ckpt（`754610`）、`GLACIER`/`ICEBERG`（`745841`）、`CFM-ID 4`（`743774`）、`ChEBI/LIPID MAPS`、`PubChem` 检索子集、`COCONUT`（`759916`）、`Enveda-180` Zenodo（`745148`）、GNPS 内既有的 `GNPS-NIST14-MATCHES`（`757307`）。
- **严禁**：`NIST` 本体、任何含 `NIST` 训练的权重及新建 `NIST` 派生数据集（`744470`/`745832`）；无生成代码/不可重建的纯中间特征表（`756893`，注意避开 `v4g` 包内的 `ranker-features`/`simulated-ranker-rows`）。
- **可复现性**：最终方案须可从比赛数据与公开文档化资源端到端重建（`754610`），指纹选位索引须有明确生成规则（`758233`）。

## 2. 优先验证的候选-谱打分短名单

候选没有参考谱时，单纯谱-谱相似度不能直接给该候选打分；但其表示可以用于候选对齐、参考谱增强或子结构证据。072a中性只约束该kNN融合实现。本轮新增FLARE局部匹配与CE差异评分，作为整体指纹/余弦之外的候选证据：

| 模型 | 任务与输出 | 公开资产与许可 | 关键限制与用法 |
| --- | --- | --- | --- |
| **新增 FLARE** | 峰子公式Transformer与原子GCN的双向局部匹配 | [作者HF源码/权重](https://huggingface.co/spaces/HassounLab/FLARE/tree/main)，83.2MB；README声明MIT；发布配置MSG，checkpoint内部训练清单待核 | 非全局对齐、非现成cross encoder；首轮正离子；禁用缺失预测公式时回退真值公式；无显式CE条件 |
| **1. 官方新版 MIST（首选）** | 谱+公式 → 4096-bit Morgan r2（无手性）概率；72.3M 参数 | `coleygroup/MIST`（MIT 代码）；Zenodo `23303289` 含 `mist_msg_noaug.ckpt` / `mist_msg_aug.ckpt`（各 289.2 MB；打包需保留 NOTICE 元数据） | `ION_LST` 仅支持 7 种正离子且无显式 `timsTOF` 映射；须用推断公式+质量后验组合（禁用真值公式）。MSG 官方审计 v2 Table 3 R@1/R@20 = 40.31%/73.02%。 |
| **2. ICEBERG 2.1 / GLACIER** | 结构+adduct+CE → 预测碎片谱 | [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred)；**仅限来源核实的MSG权重**（禁用NIST版） | 先比对底座checkpoint；2.1加速不自动构成新信号，`msg_all`/`msg_simulation`评估口径不同；比较逐谱、差异峰和CE条件分数，候选宽度由残差决定 |
| **3. MSAlign 框架（DreaMS × MoLFormer）** | 对比对齐谱与结构语言表示 | `KrzakalaPaul/MSAlign-NeurIPS2026`（MIT 框架）+ `ibm-research/MoLFormer-XL-both-10pct`（Apache-2.0，~187 MB）+ 官方 DreaMS | **禁用 MolDeBERTa**（CC-BY-NC-ND 含分发衍生限制）；须用 MoLFormer 自行在真实困难负例与 CE/adduct 条件下重训投影，且先独立编码每张谱再聚合。 |
| **4. CFM-ID 4.0（备选）** | 结构 → 预测 ESI-MS/MS 谱 | Metlin 训练；`wishartlab/cfmid`（LGPL-2.1，host 放行） | 支持正负离子，但单候选取 1–5s CPU，全量过慢，仅可作同式 top-3 补充。 |
| **5. MS-GPT（受控扩展）** | 谱+公式 → SAFE 结构生成 | `flyfish666/casmi26-ms-gpt-{msg20261031-profile, 20261031-public-assets}`（Apache-2.0 adapter/MLM + MIT MIST） | arXiv v2 修正指标仅 Top-1/Top-10 = 10.22%/13.21%（预测公式 6.83%/8.58%）；只在检索增量有效后作 50–100 题有限采样探索。 |

## 3. 公开分数与资产限制

- **公开 0.495 已复现**。Taeyang 团队公开提交 `58449454` 和 `jiyujia` 的原版 v2 复现 `58605458` 均为 **0.495**。三个 FR 派生训练输入的生成过程仍待核对。[原版 v2](https://www.kaggle.com/code/taeyangg4/casmi26-lb-0-495-network-fr-p5-tail?scriptVersionId=371003154) · [实验进展](../progress.md)
- **FR 的训练与外部评估**。作者在 Enveda 训练谱中按结构键分折，逐题插入真值，主要取同式负例，见到的难负例至多八个按 Morgan 相似度选出。验证 MRR 0.8892 来自这种构造名单。公开资料给出 `structure_keys.parquet`、`structdb_v1/entries.parquet` 和 `pool_chemistry.json` 的哈希与字段，尚未找到相同文件或生成代码。原比赛推理仅调整第 3–25 名。外部比较采用自然召回的同一份 PubChem 深度 50 名单。[作者训练包](https://www.kaggle.com/datasets/taeyangg4/casmi26-fr3to25-clean)
- **FraGNNet**。官方代码含碎片图与同式难负例采样。MassSpecGym 论文给了该模型的模拟结果；作者仓公开配置仍都是 NIST20，旧推理脚本从 W&B 按训练 run 下载，匿名查询找不到该项目的实测 MSG 权重。可研究用允许的真实绝对 CE 从零训练。[作者代码](https://github.com/FraGNNet/fragnnet) · [公开训练配置](https://github.com/FraGNNet/fragnnet/blob/main/config/README.md) · [MSG 权重讨论](https://github.com/pluskal-lab/MassSpecGym/issues/46)
- **禁用/撤稿/已证伪**：
  - `CSI:FingerID / MSNovelist`相关受禁训练权重不使用；SIRIUS纯质量分解算法与这些模型分开判断，不笼统禁止全部算法。
  - `MSFlow`（2026-03-12 撤稿，潜在泄漏）与 `SpecBridge`（2026-03-03 撤稿，评估缺陷）：旧摘要数字不可信。
  - `GEMS` / `LSM-MS2`：无公开权重或商业闭源。
  - `casmi26-nx-model`：已在 `075d`（ref `58519267`）实测 **0.492 → 0.381（−0.111）**，淘汰。
  - `AIMe/DeepMS2Reasoner`、`MS2KOSMOS`：论文明确NIST训练，MSG只是外部评估，无已核实的合法独立离线权重。
  - `MARASON`：碎片图匹配RAG框架值得研究，但原MSG流程含NIST模型生成的CE伪标签，无已核实可直接用的干净MSG权重。只列未来合规重建方案，不进入近期无训练清单。

## 4. 关键 Kaggle 辅助数据集参考

- **离线留出与面板参考**：`dmitriigluzdov/casmi26-fold-safe-fpnet`（5 折 identity-disjoint FPNet，~1.0 GB）、`trnhip/casmi26-e2r-assets`（2,180 结构留出脚本思路，但本身带筛选条件且只重训 Engine 2，不可直接充当完整 068a 独立评测）、`aryanawasthiteykik/casmi26-e180-not-in-train`（3,719 不在比赛 train 的 timsTOF 结构，CC BY 4.0；若曾用于自训须显式核对交集）。
- **许可干净检索/知识补充**：`sergiomalv/casmi-open-msms-spectra`（212 MB，CC BY-SA 4.0）、`navyadevadiga/casmi26-massbank-lib-own`（31 MB，带构建脚本）、`xjoannax88/casmi-pubchemlite-candidates`（31 MB，CC BY 4.0）、`dmitriigluzdov/casmi26-pubchem-popularity-prior`（含非谱学文献先验；注意其中 68.8 GB `fingerprints-*.bin` 在 `068a` 管线中零引用）。
