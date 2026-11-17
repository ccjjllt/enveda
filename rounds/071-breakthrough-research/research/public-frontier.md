# 071 公开前沿：真正新增的是可用模型与可审计面板，不是高分标题

核查日期：2026-11-17（北京时间晚间；Kaggle 快照约 13:30–13:45 UTC）。范围：Kaggle 最新公开 notebook、数据集、讨论原文，以及模型作者仓库。此次仅检索和读取源码/小型说明；没有启动内核、训练、推理或提交，没有下载大型权重。

## 1. 结论与优先顺序

**仍未核实到“绑定当前源码版本、全部输入可得、可以复现且 public >0.492”的新公开方案，但这不等于模型资产已经用尽。** 本次发现两项会改变下一步实验选择的事实：

1. **新版 MIST 的 MassSpecGym 权重现在真实可得，而且有完整训练配置、数据和哈希。** 070 的“MSG 权重不可得”针对旧 Zenodo/旧架构查找链，不能再作为终止 MIST 的理由。第一步应把官方 MIST 指纹分直接加到真实多源候选并集，避免先承担生成模型的复杂性。
2. **MS-GPT 的官方 MSG 编码器与配套 LoRA 已在今天公开镜像到 Kaggle。** 可将候选生成作为第二阶段；它与旧 FRIGID/P0 的失败不等价。但最新论文指标远低于搜索缓存中的旧宣传数字，不能承诺大幅提分。
3. **trnhip 的 E2b-R 完整源码可以作为候选级面板的起点。** 它确实把模拟 query 结构从 FPNet 训练中剔除，修复了“只删检索库、模型仍见过真值”的问题；但其现成训练表和权重不能直接当完整 068a 的独立测试集。

相对底座，最值得优先验证的是“新 MIST 的候选分是否能在真实 union 中救出底座错误”，然后才是“MS-GPT 是否新增真结构”。不是再把已有 25 名按一个未经标定的权重重排。

## 2. 比赛前沿核实范围

本次读取 `search_notebooks(DateRun, 30)`、`search_notebooks(ScoreDescending, 20)`、`search_datasets(casmi26, Updated, 30)`、`list_competition_topics(New)`、排行榜前 200 名；并拉取三个新/活跃 notebook 的源码。

- [官方公榜](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/leaderboard)最新读取的前五分数是 **0.549、0.540、0.536、0.532、0.523**。这些证明 0.492 不是比赛上限，不说明具体模型或各 class 的损失。
- [flexonafft/anchored-spectral-network](https://www.kaggle.com/code/flexonafft/casmi26-anchored-spectral-network)当前返回 v10，正文仍把自己的 overlay 描述为：以 0.492 为对照，对 rank-1 分子式的 PubChem 异构体用 fold-0 FPNet、文献和天然产物特征做 joint ranker，再按 OOF margin 插入 rank 2。**是可研究的方法，但本次没有拿到与当前源码绑定的 >0.492 提交证据。** 它的来源在 [isomer-ranker-train](https://www.kaggle.com/code/flexonafft/casmi26-isomer-ranker-train)。
- [zz3t3rn41/peaks-to-structures-v3](https://www.kaggle.com/code/zz3t3rn41/casmi26-peaks-to-structures-v3)当前正文声称 0.510+，主要新增跨加合物 N-1c、库锚权重、训练谱检索、旧 leader 的 rank-2 保留。源码可见，**未获得相应 0.510 提交证据**；也没有新的强谱编码器。标题和数学上“保留 rank 2 可得 0.5”均不能作为整个测试集 MRR 保证。
- [heliosli/0.495-public-demo](https://www.kaggle.com/code/heliosli/casmi26-0-495-public-demo)读取到 v2 和 FRIGID/M5 等 overlay。其配套公开数据集检索结果已明确写“runnable public demo artifacts; original competition weights excluded”。因此“有文件、可执行”不等于取得曾产生高分的权重。例：[frigid-e1](https://www.kaggle.com/datasets/heliosli/casmi26-frigid-e1)、[joint-ranker](https://www.kaggle.com/datasets/heliosli/casmi26-joint-ranker)、[j2-judge](https://www.kaggle.com/datasets/heliosli/casmi26-j2-judge)。

讨论列表没有 11-15 之后新建的方法帖，但旧帖在 11-17 有新回复；不能把“没有新主题”写成“没有新讨论”。本次没有穷尽所有私有团队、全部公开版本历史，也不能据此证明某种方法不存在。

## 3. 首选：新版 MIST 先做候选评分

主源：[coleygroup/MIST](https://github.com/coleygroup/MIST)、[artifact manifest](https://github.com/coleygroup/MIST/blob/main/src/mist/artifacts.json)、[Zenodo 23303289](https://zenodo.org/records/23303289)、[训练配置](https://github.com/coleygroup/MIST/tree/main/configs)。核查日期同上。

| 项目 | 核实结果 |
| --- | --- |
| 输出 | 4,096-bit Morgan、radius 2、无 chirality；MSG/CANOPUS 实现为 72.3M 参数 |
| 代码与训练入口 | MIT；公开 `mist-train`、四种 recipe、数据处理、训练配置与 checkpoint 选择规则 |
| 干净起点 | `msg-noaug`：只用 MSG 实测谱；峰 dropout / 强度扰动仍保留，noaug 指没有合成谱 |
| 可选增强 | `msg-aug`：MSG + refined 4M GLACIER 合成谱；需固定该 MSG generator，而非 NIST 版 GLACIER |
| 权重完整性 | manifest 给 `mist_msg_noaug.ckpt` SHA-256 `6f8df582df36aa4a984f120efee14ee0a62f3a7723caa770b856a6e704236727`；`mist_msg_aug.ckpt` 为 `9f251d772f68315ba6fe2dc75a3b2e1ec477d3f41ffbeaf5143749157b67b177` |
| 比赛口径 | [host 757307](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/757307)在 11-11 明确允许 MassSpecGym/MIST；11-16 对 GNPS-NIST14-MATCHES 也给出许可回复 |
| 未核实 | 本赛 timsTOF、负离子/复杂加合物上的真实候选排序；官方模型训练结构与计划面板的交集；T4 推理吞吐 |

**相对现有底座的增量**：这是具有分子式归纳偏置的谱→指纹模型，不是 076 的 DreaMS×Morgan 双 MLP，也不是失败的 FP→谱模型。独立性只能通过配对错误分布实测；这里不声称它已经比 FPNet 强。

**最小可执行实验设计（尚未执行）**：

1. 固定一个规范化结构留出面板；先计算与 MSG 训练结构/候选排序器训练结构交集，区分未见结构与已见结构独立谱。
2. 为同一 query 导出 Engine1 top60、Engine2 top40、PubChem proposal、fusion overflow 等**真实并集**，同时保留候选来源及各模型原始分。正确结构缺失的 query 必须保留在分母，不能人为补真值后把条件排序率当端到端效果。
3. 从 query/候选公式假设构造 MIST 输入。先分开报告“真分子式已知”的诊断上限与“只能用质量窗候选分子式”的部署结果；最终提交不能读隐藏标签。不能无条件把底座 leader 的公式当真值。
4. 候选用同定义 RDKit 4096 位指纹重算，与 MIST 概率做标定的候选分（例如 Bernoulli 对数似然和 soft-Tanimoto 作为两种预先指定对照）。**不要把 4096 位输出直接塞入底座 6930 位/选位指纹接口。**
5. 先报独立 MIST 的条件 MRR、oracle union recall，再报它与底座联合后的配对 ΔMRR、救回和破坏数量；阈值在 calibration 子集确定，最后在未用于选择的结构上评估。通过后再考虑将同分加入正式候选融合。

计算预算：此路径只加载已有权重，不需要完整预训练；本次没有可靠 T4 吞吐证据，不能沿用旧文档“几十分钟一定够”的估算。可将**首批 200 结构的远端测量上限设为 1 GPU 小时**作为研究预算，而非运行时间承诺；按实测数据处理/前向/候选评分分段外推 400 题并检查 9h 总预算。

## 4. 第二阶段：MS-GPT 的新 Kaggle 资产

| 公开镜像 | 本次核实的实际内容 | 增量与边界 |
| --- | --- | --- |
| [MSG profile v1](https://www.kaggle.com/datasets/flyfish666/casmi26-ms-gpt-msg20261031-profile)；2026-11-17 12:30Z | `mist/mist_official/mist_massspecgym.pt` 289,228,237 B；配套 rank16 LoRA 5,613,720 B；adapter/inference/training config；manifest/attribution/license | 官方 MSG 编码器与适配器确实存在，解决旧 P0 查找链的权重阻塞；仍需核对与 shared decoder 的版本匹配 |
| [shared assets v1](https://www.kaggle.com/datasets/flyfish666/casmi26-ms-gpt-20261031-public-assets)；2026-11-17 10:59Z | `models/mlm/ms_gpt_mlm.pt`、SAFE tokenizer/config、未修改推理源码、offline wheels；同时含 NPLIB1 MIST/r8 LoRA | 只取公共 decoder/tokenizer/源码/wheels，与上一行 MSG profile 配对；不能因整个镜像公开而误用 NPLIB1/legacy 路径 |

两个包都固定作者 commit `ec032f87dcaa54c6380176c2f959e11335440ae5`，描述明确不宣称 CASMI 成绩。数据集 license 为 Other，保留上游逐项许可；不能笼统重标 MIT。[MS-GPT 作者仓库](https://github.com/VIKI623/MS-GPT)与[作者 issue #1](https://github.com/VIKI623/MS-GPT/issues/1)给出代码及模型/adapter Apache-2.0 许可；MIST 子源码保留 MIT。shared decoder 为结构语料预训练，谱侧必须固定 MSG 模型与适配器。更详细的谱/结构训练链核对见本轮模型文献报告。

**分数需要纠正版本**：[arXiv v2，2026-10-29](https://arxiv.org/html/2607.23607v2)给 MSG top-1/top-10 **10.22%/13.21%**，不是缓存搜索摘要的 23.9%/28.7%。同一论文在 RTX 6000 Ada + 16 核 CPU 上的 100/1000 候选池耗时为 2.80/21.16 秒每谱，召回为 14.49%/18.77%；不能直接换算成 T4 时间，也不能换算成本赛分数。

相对底座的作用是**生成当前所有库/通道之外的连通结构**。建议只有在 MIST 指纹对真实 query 有信息、质量/公式候选可控时继续：对底座低置信 query 生成候选，按 InChIKey14 去重并加回 union，再由独立候选分做选择。先测新真值召回及其新增 MRR，不只测 SMILES 合法率或生成频次。已在列表中正确候选从 r 升到 1 的收益是 `1−1/r`，不是每个正确升位都 +1；同理，不能把某个文献 top-1 数字与固定 0.51 门槛比较就否定所有条件路由。

## 5. P0 面板框架：E2b-R 的真实留出范围

主源：[trnhip/casmi26-e2r-assets v3](https://www.kaggle.com/datasets/trnhip/casmi26-e2r-assets)，2026-11-17 13:24Z 更新，logical total 320,037,597 B，license **CC BY-NC-SA 4.0**。本次读取了 README、manifest、模拟表 driver、FP 数据准备、结构留出脚本、训练 launcher、重特征脚本和训练日志；未执行它们。

### 5.1 它确实修了 FPNet 的监督泄漏

- [`script_set_simholdout.py`](https://www.kaggle.com/datasets/trnhip/casmi26-e2r-assets?select=script_set_simholdout.py)明确承认首轮 FPNet 看过模拟结构，导致 pv 正例 fp top-1 达 87%，原公开表约 42%。修正为 `is_val := seeded_2_percent OR six_in_simulation_structures`，不是只在谱库中屏蔽 ID。
- `script_spec_simhold_report.json`记录 **2,180** 个模拟结构整组留出，val 谱由 49,333 增至 168,555，val 结构 7,640，训练谱 2,371,053。
- [`script_train_queue_simhold.sh`](https://www.kaggle.com/datasets/trnhip/casmi26-e2r-assets?select=script_train_queue_simhold.sh)实际传 `--spec ds/spec_simhold.npz`，以 merge_p 0.6/0 分别训练 merged/single 两个 FPNet。`script_run_train.py`锁定 public training loop 哈希，只对池指纹分块解包做内存补丁。
- [`script_refeaturize_pv.py`](https://www.kaggle.com/datasets/trnhip/casmi26-e2r-assets?select=script_refeaturize_pv.py)用这两个重训权重重算 31 列 pv 特征中的 15–20、25–30；其余库/analog/fragment 特征保留。它还修复 missing collision energy 引起 NaN logits 的问题，以 25.0 eV 为 fallback。

因此，“它只是删检索库，预训练 FPNet 仍见过 query”对于这两份**重训的 Engine2 FPNet**并不成立。但没有因此证明完整 068a 无泄漏。

### 5.2 它模拟什么，不模拟什么

[`script_simrows_driver.py`](https://www.kaggle.com/datasets/trnhip/casmi26-e2r-assets?select=script_simrows_driver.py)固定 seed=20261116、每结构最多 8 谱；从 NP/GNPS/RIKEN/MoNA/MS-DIAL/MassBank/pluskal_ms2 七来源按配额选 2,250 个 query 记录，去重后为上面的 2,180 个 ik14。

| 维度 | 实际实现 | 对 P0 的含义 |
| --- | --- | --- |
| class 1 | 从检索库删除 query 所属整个 `ingest_lib` | 更接近跨来源独立谱检索；仍需另做峰级近重复检查，避免跨库复制 |
| class 2 | 从检索库删除该 shard 所有 query ik14 的全部谱；结构保留在候选池 | 是未见谱结构候选排序；删除 shard 并集比每 query 单独删除更严格，会改变 analog 环境 |
| 候选池 | COCONUT 436,389 行 + train 结构，按质量排序，约 712,199 结构 | 不是完整运行时池：不含其额外 ChEBI/LMSD、PubChemLite，更不含 Engine1/PubChem 全量/生成来源 |
| query 过滤 | 必须真值在 10 ppm 窗口（为空则 30 ppm）内才留；一组恰有一个正例 | **已条件化为真值可达**，不能衡量候选召回，也不含真值缺失组 |
| ours 表 | 51 特征，base/libx/anapol/fragx，`zlog=None` | 这部分未用 FPNet logits；可复用特征构造，不能当新谱学信号 |
| pv 表 | 819 query 配额：全 NP + 569 seeded 其他；31 特征 | “obscure”原始定义未知，作者明确改为均匀抽样；与原训练表不保证逐行一致 |
| SMILES | driver 的构池保留 `smiles`；FP prep 的 pool_meta 只保存 keys | 要做 MIST/ICE/MSAlign 联合打分，应另导出 ik14↔规范 SMILES 与每组候选映射，现成 X/Y 表不够 |

### 5.3 为什么它还不是独立端到端评测

1. 这些 query 被放入 FPNet **validation**，训练日志反复用其 hard-negative top-1 选择 best checkpoint；这是开发验证集，不是未触碰的最终测试集。
2. `e2_rows_ours/pv` 本来是 **ranker 训练输入**。若在同表上拟合再评估，仍会泄漏。新一轮必须先按规范化结构分 calibration/ranker-train/final-eval，确保某个 ik14 在各 library/query 的多条记录都在同一侧。
3. 只修了 Engine2 的两份 FPNet，并未重训或证明 Engine1 FPNet、既有 LightGBM、神经 scorer、GL/ICE 等没有见过这些结构。不能拿完整068a在这些query上的高分当未见结构泛化。
4. 训练池中保留留出结构的**无谱分子指纹作为候选/负例**与看过该结构的监督谱不是同一回事；class2检索允许真值结构已在库中。报告必须明确此边界，不能笼统称所有结构完全未知。

**建议复用方式**：用它的 split/mask/feature-export 思路构建 P0；为完整 union 显式存每个来源、SMILES、模型训练交集。在尚不能证明底座全部模型留出时，分别报告“同 query 的增量决策对照”与“所有监督模型未见结构的独立评估”，不要混成一个总 MRR。新模型优先在与其训练语料不重叠的结构上做验证，避免一次重训全部旧管线。

### 5.4 计算证据与许可

- 公开 `log_train_m1.log`显示单模型 36.0M 参数、到 79,800 step 为 **15,968 秒，约4.44小时**，best hardneg-top1 0.564；manifest 的 single 为 0.524。**这是作者内部验证分，不是本赛公榜或未见结构最终测试分。**
- launcher 注释为约14GB VRAM cap，代码使用本机显存比例0.30；日志只写 CUDA，未给可复核GPU型号。不能把4.44h称为T4耗时。每模型 `--max_minutes 300` 是上限设置，也不是本项目实测。
- 表生成按 Kaggle CPU shard 运行；本次未得到完整 shard 总时长。FP prep 注释 peak RSS约2.5GB，此为作者说明而非本机测量。
- 数据集沿用 COCONUT/v4g 的 CC BY-NC-SA 4.0，不适合擅自把衍生物重标MIT。[host 756893](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/756893)在11-16明确要求公开权重的生成代码和获准数据也可得；本包价值在于提供生成链，但仍需记录所有来源，不应仅凭“train.parquet训练”一句话认定完成复现。

## 6. 当前不值得投入的分支

- 继续追 0.495/0.510 的公开标题：没有源码版本与提交分绑定就不当作实验先验。
- 将 demo 权重视为原高分权重：公开数据描述明确区分二者。
- 再尝试缺文件的旧 MIST-MolForge 链：改用作者当前 MSG release；不需要重复修补已经失效的历史目录假设。
- 从零预训练大生成器：已有 shared decoder，不应先重做其分子语言预训练。
- 以 MSFlow “45%”为强证据：[官方 arXiv](https://arxiv.org/abs/2602.19912)明确该预印本因潜在数据泄漏于2026-03-12撤回。

## 7. 建议给主计划的可审查交付

第一份产物应是 **固定结构面板上的候选来源表 + truth-in-union 分解 + MIST/底座配对排名**，不应先产生提交 notebook。若新 MIST 在实际union中未带来独立救回，就停止其生成扩展；若指纹有信息但真值不在union，再测MS-GPT新增召回。完整MSAlign是另一项应独立核验的分子表示升级，076的Morgan版本不能替代它的验证；本轮主研究另行核查其正式checkpoint。

所有建议都是下一步实验设计。此次没有得到任何新离线模型结果，也没有新公榜增益。
