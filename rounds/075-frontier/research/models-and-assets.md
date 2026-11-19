# 可用模型与公开资产清单（071/075 合并精简版）

盘点基准：2026-11-18/19。目标：在**不重训原生模型、断网、T4 ≤ 9h**约束下，筛选合法可用的 MS2 同分异构体判别模型与外部资产。

## 1. 规则与许可硬边界（Host 裁定）

- **允许**：`CC0` / `CC BY` / `CC BY-SA` / `CC BY-NC` / `CC BY-NC-SA` 开源外部数据与预训练模型（`750330`）；`train.parquet` 及其派生、`MassSpecGym` 与 `MIST`（`757307`）、`DreaMS` 官方 ckpt（`754610`）、`GLACIER`/`ICEBERG`（`745841`）、`CFM-ID 4`（`743774`）、`ChEBI/LIPID MAPS`、`PubChem` 检索子集、`COCONUT`（`759916`）、`Enveda-180` Zenodo（`745148`）、GNPS 内既有的 `GNPS-NIST14-MATCHES`（`757307`）。
- **严禁**：`NIST` 本体、任何含 `NIST` 训练的权重及新建 `NIST` 派生数据集（`744470`/`745832`）；无生成代码/不可重建的纯中间特征表（`756893`，注意避开 `v4g` 包内的 `ranker-features`/`simulated-ranker-rows`）。
- **可复现性**：最终方案须可从比赛数据与公开文档化资源端到端重建（`754610`），指纹选位索引须有明确生成规则（`758233`）。

## 2. 唯一可行的候选-谱打分短名单

谱-谱方法（`MS2DeepScore`、`Spec2Vec`、`DreaMS`、`MSBERT`）在候选缺少参考谱时无法直接区分同式异构体（`072a` 已证中性）。必须使用**候选-谱打分**或**谱→指纹一致度**：

| 模型 | 任务与输出 | 公开资产与许可 | 关键限制与用法 |
| --- | --- | --- | --- |
| **1. 官方新版 MIST（首选）** | 谱+公式 → 4096-bit Morgan r2（无手性）概率；72.3M 参数 | `coleygroup/MIST`（MIT 代码）；Zenodo `23303289` 含 `mist_msg_noaug.ckpt` / `mist_msg_aug.ckpt`（各 289.2 MB；打包需保留 NOTICE 元数据） | `ION_LST` 仅支持 7 种正离子且无显式 `timsTOF` 映射；须用推断公式+质量后验组合（禁用真值公式）。MSG 官方审计 v2 Table 3 R@1/R@20 = 40.31%/73.02%。 |
| **2. ICEBERG 2.1 / GLACIER** | 结构+adduct+CE → 预测碎片谱 | `coleygroup/ms-pred`；**仅限 MassSpecGym 权重**（禁用 NIST'23 版） | 碎片级建模对位置异构敏感；仅对同式组 top-5 打分并按 entropy similarity 排序；CE 缺失时边缘化。 |
| **3. MSAlign 框架（DreaMS × MoLFormer）** | 对比对齐谱与结构语言表示 | `KrzakalaPaul/MSAlign-NeurIPS2026`（MIT 框架）+ `ibm-research/MoLFormer-XL-both-10pct`（Apache-2.0，~187 MB）+ 官方 DreaMS | **禁用 MolDeBERTa**（CC-BY-NC-ND 含分发衍生限制）；须用 MoLFormer 自行在真实困难负例与 CE/adduct 条件下重训投影，且先独立编码每张谱再聚合。 |
| **4. CFM-ID 4.0（备选）** | 结构 → 预测 ESI-MS/MS 谱 | Metlin 训练；`wishartlab/cfmid`（LGPL-2.1，host 放行） | 支持正负离子，但单候选取 1–5s CPU，全量过慢，仅可作同式 top-3 补充。 |
| **5. MS-GPT（受控扩展）** | 谱+公式 → SAFE 结构生成 | `flyfish666/casmi26-ms-gpt-{msg20261031-profile, 20261031-public-assets}`（Apache-2.0 adapter/MLM + MIT MIST） | arXiv v2 修正指标仅 Top-1/Top-10 = 10.22%/13.21%（预测公式 6.83%/8.58%）；只在检索增量有效后作 50–100 题有限采样探索。 |

## 3. 明确排除的模型与高分陷阱

- **公榜头部无公开物**：实查公开 100 篇 notebook / 63 位作者与前 40 名队伍零交集；0.500+ 无公开配方，`0.494` 以上无直接可抄源码（`073a` `0.495` demo 为空壳权重+合成库；`074a` Dante 复现仅得 `0.492`）。
- **禁用/撤稿/已证伪**：
  - `SIRIUS 6 / CSI:FingerID / MSNovelist`：含 NIST 训练与学术非商用条款。
  - `MSFlow`（2026-03-12 撤稿，潜在泄漏）与 `SpecBridge`（2026-03-03 撤稿，评估缺陷）：旧摘要数字不可信。
  - `GEMS` / `LSM-MS2`：无公开权重或商业闭源。
  - `casmi26-nx-model`：已在 `075d`（ref `58519267`）实测 **0.492 → 0.381（−0.111）**，淘汰。

## 4. 关键 Kaggle 辅助数据集参考

- **离线留出与面板参考**：`dmitriigluzdov/casmi26-fold-safe-fpnet`（5 折 identity-disjoint FPNet，~1.0 GB）、`trnhip/casmi26-e2r-assets`（2,180 结构留出脚本思路，但本身带筛选条件且只重训 Engine 2，不可直接充当完整 068a 独立评测）、`aryanawasthiteykik/casmi26-e180-not-in-train`（3,719 不在比赛 train 的 timsTOF 结构，CC BY 4.0；若曾用于自训须显式核对交集）。
- **许可干净检索/知识补充**：`sergiomalv/casmi-open-msms-spectra`（212 MB，CC BY-SA 4.0）、`navyadevadiga/casmi26-massbank-lib-own`（31 MB，带构建脚本）、`xjoannax88/casmi-pubchemlite-candidates`（31 MB，CC BY 4.0）、`dmitriigluzdov/casmi26-pubchem-popularity-prior`（含非谱学文献先验；注意其中 68.8 GB `fingerprints-*.bin` 在 `068a` 管线中零引用）。
