# 069 公开前沿审计：公开可复现资产里有没有超过 0.492 的新信号

审计时间：**2026-11-16T02:32Z**（北京时间 10:32）。只读审计：未运行训练、未推送内核、未提交、未改动 068a。

本文与 [069-frontier-models.md](069-frontier-models.md) 并列：那一篇回答「该接哪个外部模型」，本篇回答「公开侧现在到底有没有已核实的更高分配方」。两篇编号相同是同日两次调研，内容不重叠。

---

## 0. 结论先行

1. **没有。** 在公开可复现资产里，截至 2026-11-16T02:32Z，**不存在任何已核实高于 0.492 的配方**。
2. 全部 336 个公开 notebook 中，作者本人公榜分最高的一个是 **OzanM. 0.499**（rank 32，ref 58393626，2026-11-15T16:53:52Z），但他那份公开 notebook 自述只到 **0.484**，且被 Rogers Johnson 在另一账号复现为 **0.488**。0.499 来自未公开的私有变体，配方不可见。
3. 0.492 这条线现在有**三个独立账号**复现：Rogers Johnson（0.492）、alex chilton（ref 58380172，0.492）、Дворкин（ref 58375394，0.492）。三者的共同改动都只是 N-1c `LAM` 0.25→0.35。
4. 标题写 0.494/0.498/0.5 的 notebook 全部是同一份 0.492 配方的 fork 或加了未复现的 learned reranker，作者实际公榜 **<0.487–0.492**，没有一个 >0.492。
5. 一个 0.492 队伍（Дворкин，rank 92）公开了 17 个受控实验，其 notebook 表格里最好只到 0.488（0.492 是随后未写入表格的那一臂）；alex chilton 公开 168 次提交、59 个单旋钮，最大增益 +0.006。两份独立记录都指向同一结论：**单旋钮到不了 0.50，需要新信号**。

---

## 1. 快照

| 项 | 值 | 来源 |
| --- | --- | --- |
| 队伍数 | 5,356 | `get_competition`，2026-11-16T02:32Z |
| **比赛截止** | **2026-12-14T23:59Z** | 同上 |
| 新参赛者截止 | 2026-12-07T23:59Z | 同上 |
| 指标 | MRR@25（榜面 3 位小数） | 榜单 + 公开 notebook |
| 公榜前 10 | 0.549 Shehab Alshehabi / 0.540 democat / 0.536 Oleksandr Parfionov / 0.528 Savko / 0.518 goonew / 0.518 Terry Luo / 0.510 Paweł Biernat / 0.510 tu tu tu du / 0.509 BMW / 0.509 muse | `get_competition_leaderboard`，同上 |
| 名次线 | rank 20 = 0.503；rank 27 = 0.500；rank 50 = 0.494；rank 100 = 0.492 | 同上（600 行） |
| 分布 | ≥0.500：**27 队**；≥0.492：**236 队**；≥0.488：**576 队** | 同上 |

> 注意：背景里写的"截止 2026-12-07"是**新参赛者截止**；正式截止是 **2026-12-14**。多出一周，但公开侧的可复现前沿不会因此变化。

与 2026-11-16T01:54Z 快照（见 `docs/progress/068-n1c-lam035.md`）相比：
- goonew 0.514 → **0.518**（升到第 5）；新面孔 `tu tu tu du` 0.510。
- ≥0.492 从 229 队增至 **236 队**；≥0.500 仍是 27 队。即 0.492 继续扩散，但 0.50+ 没有新增来源。

---

## 2. 公开 notebook 审计

### 2.1 方法

- 用 `search_notebooks` 按 `DateCreated` / `DateRun` / `VoteCount` 三个方向各取 4 页 ×40，去重后 **336 个公开 notebook**（比赛公开代码页的全部）。
- 把作者显示名与公榜队名做 **Unicode 规范化后精确匹配**（只保留字母/数字，去掉 `.`、空格、标点），命中 148 个。
  - 这里踩过一个坑并已修正：早期版本用 `[^a-z0-9]` 归一，"Дворкин Евгений Владимирович" 和 "한 밭 대 학 교 컴 퓨 터 공 학 과" 都被压成空串，产生了"作者 0.504/rank 14"的假匹配。改用 `\p{L}\p{N}` 后，Дворкин 的真实位置是 **rank 92 / 0.492**。下文所有数字都是修正后的。
- 对每个候选逐个 `get_notebook_info` 拉 `metadata` + notebook 正文，核对：作者公榜分、最近运行时间、正文是否真是新配方、依赖是否公开可复现。

### 2.2 结果

**作者公榜分 ≥0.488 或标题声称 ≥0.489 的全部 notebook：**

| notebook | 作者 | 标题/自述分 | 作者实际公榜（ref） | 是否新配方 | 可复现性 | 依赖 |
| --- | --- | --- | --- | --- | --- | --- |
| `analyticaobscura/casmi-2026` | OzanM. | 自述 **0.484** | **0.499**（58393626，11-15T16:53Z） | 否：0.487 底座 + ExtLib `TAU` 0.85→0.80 | 公开可跑，但只到 0.484；0.499 的配方未公开 | 10 个公开数据集，无私有 kernel |
| `rogersjohnson/casmi26-l4-lam035` (v1) | Rogers Johnson | 正文 0.488 + 单格 N-1c `LAM`0.35 | **0.492**（rank 80） | 否：0.488 的单旋钮 | 是，全公开 | 10 数据集 + `popwenman/casmi26-scorer-train-v80` |
| `rogersjohnson/casmi26-p1-lam038` … `p5-placebo` | Rogers Johnson | 无 | 0.492 | 否：`LAM` 0.38/0.40/0.42/0.45/placebo 五臂扫描（11-16 00:17–00:23 跑） | 是 | 同上 |
| `alexchilton/casmi26-0-492-what-moved-the-board-59-arms` | alex chilton | 0.492，168 次提交全表 | **0.492**（58380172，rank 122） | 否 | 是（跑的是 Rogers Johnson 原码） | 同上 |
| `zz3t3rn41/casmi26-peaks-to-structures-v2` (v57) | zz3t3rn41 | 标题 **"0.494+ SOTA Frontier"**，正文期望 0.5070 | **0.488**（正文自述 58333939，rank 281） | 否：`LAM`=0.38 + learned logistic reranker | 可跑但分不实 | 12 数据集 + 上游 kernel |
| `adityaraj0612/casmi26-exp306/307/308-*` | ADITYA RAJ | 0.492 | **0.492**（rank 207） | 否 | 是 | 同族 |
| `subsh2004/casmi26-imperial-gold-network-synergy-lb-0-492` | Subham Sharma | 标题 0.492+ | **0.492**（rank 131） | 否 | 是 | 同族 |
| `jtheyogi/enveda-casmi26-apex-0494-calibrated` | JK | 标题 **0.494** | 未进前 600（<0.487） | 否：0.492 + learned reranker | 不推荐 | 同族 |
| `jacobcascone/casmi26-0494-fork` | Jacob Cascone | 标题 **0.494** | 未进前 600 | 否 | 否（纯 fork） | 同族 |
| `k3rn3l/casmi26-v598-net492` | K3rn3l | 标题 0.492 | 未进前 600 | 否 | 否 | 同族 |
| `xiangqizhang/casmi26-net492cpu` | xiangqi zhang | 标题 0.492 | 未进前 600 | 否；**CPU、无 GPU**（11-16T02:16 新建） | 结构未验证 | 同族 |
| `heetp0/notebook0b6a5a21c6` (v29) | Heet Patel | "Sovereign V46 0.492 Lock" | **0.487**（rank 597） | 否 | 是 | 同族 |
| `huangkx666/enveda-p2s-lam035` | Leo | 0.488 | 未进前 600 | 否 | 是 | 同族 |
| `riochiang/casmi26-silver-0488-*`（6 个变体） | YUANHAO JIANG | 0.488 | **0.488**（rank 505） | 否 | 是 | 同族 + 14 数据集 |
| `evgendvorkin/spectra-to-structures-deep-eda-molecular-clues` (v39) | Дворкин | 无标题分；正文 17 个受控实验表，verified best **0.488** | **0.492**（58375394，rank 92） | 否；V37 `λ`=0.35 为唯一未结算的合理臂，归属属推断 | 是 | 10 公开数据集 |

（`heliosli/casmi26-0-489` 标题 0.489，正文仍是 2026-10-29 的 0.477 版本，作者未进前 600，不列入。）

### 2.3 三条值得单独记录的发现

**(a) 0.492 已被第三个账号独立复现（配方归属为推断）。**
`evgendvorkin/spectra-to-structures-deep-eda-molecular-clues` v39 的正文有一张 17 行实验表：V21–V22 复现公开底座 0.487，V29 = OzanM + N-1c `λ`=0.25 得 **0.488**（其"verified public best"），V23/V26/V32 等分别跌到 0.466/0.482/0.470；V37（`λ`=0.35）、V38（0.40）、V39（0.45）当时都标 Pending。该队公榜 **0.492**（ref 58375394）已结算，而 V37 的 `λ`=0.35 是三臂中唯一有外部先验的。**归属是推断，不是证明**：`list_team_public_submissions` 记的提交时间是 2026-11-15T07:48:56Z，而公榜 `submission_date` 是 12:46:42Z，两个端点不一致，不能据此锁定是哪一臂出的分。能确定的是：**第三个独立账号用同一份公开管线拿到了 0.492**（另两个：Rogers Johnson、alex chilton 58380172）。

**(b) 一个 0.492 队伍用 17 个受控实验没能超过 0.488。**
上表 17 行里，除 `λ` 扫描外，扩展 APEX 余弦阈值（0.80→0.78）、库锚定网络传播、碰撞能量碎片轨迹、神经引导候选回收、跨加合物中性丢失共识**全部中性或退步**（0.488 / 0.487 / 0.470 / 0.466）。与 alex chilton 的 59 臂结论一致。

**(c) 标题分与实际分系统性偏离，最大偏离 −0.006。**
`zz3t3rn41` v57 标题写 "0.494+ SOTA Frontier"、内文写 "Target: 0.5070+"，而它自己记录的官方 PB 是 **0.4880**（sub 58333939，rank 281）。`jtheyogi`、`jacobcascone` 标题 0.494，作者连前 600 都没进。**标题分不可用作证据。**

### 2.4 前 27 队有没有技术披露

- 前 27 队（≥0.500）中**没有**任何一支有可确证的公开 competition notebook：按队名与 336 个作者显示名做词元匹配，最高命中的是 rank 32 的 OzanM.。rank 43 的 `PL`（0.495）与 `kitopl` 的 notebook 是否同一主体**无法确证**（仅用户名相似），且那两份是 EDA/「Data Pitfalls That Break Your CV」类分析，不是配方。
- 讨论区也没有：163 个 topic 里，2026-11-14–11-16 新增的只有队列问题（759398 `Notebooks getting queued`、759394 `Submission Queued for 6 days`）、一个低分自述（759307 `SpectralHybridEncoder 0.057→0.089`）和规则澄清类，**没有一篇 ≥0.50 的方法披露**。
- 结论：0.500+ 来自私有管线。公开侧拿不到，也没有泄漏。

---

## 3. 公开资产审计

只列 2026-11 之后更新、或与本队当前管线相关的公开资产。大小为 `get_dataset_info.total_bytes`。许可为 Kaggle 页面标注。

| slug | 类型 | 大小 | 许可 | 068a 是否已用 | 潜在价值 |
| --- | --- | --- | --- | --- | --- |
| `samartalwar/casmi-2026-spectral-library-massbankharmonized` | 谱库 | 45.7 MB | CC BY-NC-SA | ✅ | 现有 ExtLib 主库 |
| `takumuhata/casmi26-extlib` | 谱库（GNPS+MassBank+MoNA 合并） | 252.4 MB | CC0 | ✅ | 现有 rank-1 overlay |
| `sanpier/casmi-spectrum-model-v1` | 谱→结构权重 | 53.3 MB | Apache 2.0 | ✅ | Engine 1 神经分 |
| `dmitriigluzdov/casmi26-fold-safe-fpnet` | 指纹权重（5 fold） | 1,022 MB | 自定义（可复现） | ✅ | 候选生成核心 |
| `dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets` | 管线复现包 | 19,152 MB | CC BY-NC-SA | ✅ | 冻结管线主体 |
| `dmitriigluzdov/casmi26-natural-product-knowledge-table` | NP 先验表 | 129.7 MB | Other | ✅ | LOTUS/COCONUT 知识表 |
| `dmitriigluzdov/casmi26-pubchem-popularity-prior` | 文档先验 | 76,690.8 MB | Other | ✅ | POP/SPEC 权重来源 |
| `sanpier/catalogue-experiment-saner` | 目录/索引 | 24.4 MB | Unknown | ✅ | 候选目录 |
| `qwertykeypadapple/casmi26-lotus-candidates` | 候选库（LOTUS） | 6.7 MB | CC0 | ❌（063b 试过，无增益） | 已放弃 |
| `bgunshui/casmi26-scorer-v2-model` | 神经打分权重 | 11.3 MB | Unknown | ✅（经 scorer v80） | 第二模型 |
| **`aryanawasthiteykik/casmi26-gnps-community-cc0`** | **谱库** | **405.4 MB** | **CC0** | ❌ | **506,939 张谱 / 59,039 结构，直接按 train schema；identity search 覆盖最大增量**（2026-11-13） |
| `aryanawasthiteykik/casmi26-e180-not-in-train` | 谱库 | 48.5 MB | CC BY 4.0 | ❌ | Enveda-180 中**不在训练集**的结构谱，可做独立验证集（2026-11-13） |
| `royalacecat/casmi26-enveda180-zenodo-ccby4` | 谱库 | 427.7 MB | CC BY 4.0 | ❌ | 同上，另一份打包（2026-11-13） |
| `shaoyuantang/enveda-180-ccby4-casmi26` | 谱库 | 121.9 MB | CC BY 4.0 | ❌ | 同上，离线版（2026-11-15） |
| `xiangjiangzhou/casmi26-permitted-massbank-20261115` | 谱库 | 97.8 MB | Other（逐条 CC0/CC BY/CC BY-SA） | ❌ | **只保留允许商用的 MassBank 记录，逐条保留原始许可与测定前体**（2026-11-15） |
| `markbarney/casmi26-massbank-open-202611` | 谱库 | 15.7 MB | CC BY 4.0 | ❌ | MassBank 开放子集（2026-11-10） |
| `navyadevadiga/casmi26-massbank-lib-own` | 谱库 | 39.0 MB | CC BY-NC-SA | ❌ | 139,744 张谱 + **构建脚本**（可复现性最好，2026-11-14） |
| `studyexchange/enveda-casmi26-s46-iceberg-assets` | 前向模型 | 64.0 MB | Unknown | ❌ | ICEBERG 2.1（MassSpecGym msg_all）权重 + wheels，MIT 代码（2026-11-10） |
| `prvsiyan/casmi26-forward-reference-stack` | 前向模型 | 3,654 MB | Other | ❌ | ICEBERG/GLACIER 离线栈 + pinned wheels |
| `yehudaitkin/casmi-dreams-minimal-asset-mirror` | 谱→结构检索 | 368.5 MB | MIT | 仅空槽填充 | DreaMS 两个 pinned 文件（2026-11-13） |
| `eleazarmaru/dreams-offline` | 谱→结构检索 | 2,939.6 MB | Apache 2.0 | ❌ | DreaMS 离线完整包（2026-11-10） |
| `hemantkumarbk/isdb-np-insilico-msms-library` | 预测谱库 | 1,051.9 MB | CC BY 4.0 | ❌ | 149k 天然产物 × 298k 条 CFM-ID 预测谱，质量索引（2026-11-06） |
| `aryanawasthiteykik/casmi26-spectrum-to-fingerprint-mlp` | 谱→指纹权重 | 179.2 MB | CC BY-NC-SA | ❌ | Morgan+MACCS MLP + 离线排序代码 |
| `prvsiyan/casmi26-self-trained-scorer-seeds` | 打分权重 | 22.7 MB | CC BY-NC-SA | ❌ | 自训 scorer 种子（16 个 notebook 在用） |
| `xhhuang/casmi26-frigid-clean-src` | CSI 工具源码 | 0.7 MB | Unknown | ❌ | FRIGID 源码 + ScaledMIST runner（2026-11-15） |
| `boomvector/casmi-2026-reranker-frozen-assets` | 重排资产 | 122.9 MB | CC0 | ❌ | 冻结重排特征（2026-11-15） |
| `harishdhanarajan/enveda-four-experts-pack-theobroma-v1-36` | 集成权重 | 2,170.8 MB | **Unknown** | ❌ | 四专家包；**许可不明，不建议用于获奖方案**（2026-11-10） |

**资产侧的判断：**

- 2026-11-13～11-16 的新增资产集中在**谱库扩容**（GNPS CC0 405 MB、Enveda-180 三份副本、MassBank 三份合规子集），**不是新模型**。
- 新模型侧只有 DreaMS 镜像、ICEBERG/GLACIER 前向栈、ISDB 预测谱、谱→指纹 MLP 四个方向，全部在 2026-11-13 之前就有；11-13 之后没有新增强权重。
- **没有任何一个公开 notebook 用上述新资产做出 >0.492 的分数**（作者公榜分见 §2.2）。资产「新」不等于「有增益」：063b 的 LOTUS 候选、多份 learned reranker、Дворкин 的 17 个受控实验都是反例。

---

## 4. 规则侧更新（影响后续可走的路）

| 议题 | 裁定 | 出处 |
| --- | --- | --- |
| **CC-BY-NC 类开源数据** | 2026-10-17 起**允许**（"solutions can use external data that is open sourced with noncommercial provisions"） | [750330](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/750330)，last comment 2026-11-15T09:23Z |
| MassSpecGym / **MIST 权重** | **允许**（"MassSpecGym and MIST models are fine"） | [757307](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/757307)，2026-11-16T00:53Z |
| GNPS-NIST14-MATCHES | 允许（可用于谱库匹配与训练） | 同上首条回复 |
| NIST 及其派生（含权重） | **禁止**，使用即取消资格 | 750330 / 744470 |
| 无法复现权重的中间资产 | **不合格**（"Solutions that load intermediate data artifacts without being able to reproduce those artifacts are not reproducible"） | [756893](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/756893)，2026-11-16T00:52Z |

第 5 条对我们特别重要：068a 依赖的 `dmitriigluzdov/casmi26-fold-safe-fpnet` 与 `sanpier/casmi-spectrum-model-v1` 都有公开训练代码/协议；但 `harishdhanarajan/*` 这类许可不明、无生成代码的包不能进最终方案。

---

## 5. 明确回答

> **在公开可复现资产里，有没有任何证据表明某种新信号能超过 0.492？**

**没有。** 具体到三个层次：

1. **分数层**：336 个公开 notebook 中，没有一个是「作者公榜 >0.492 且正文含该分数的配方」。唯一 >0.492 的公开 notebook 作者是 OzanM.（0.499），其公开正文止于 0.484（复现 0.488），0.499 无公开来源；另有 188 个作者显示名在公榜前 600 中找不到同名队，理论上存在改名导致的漏配，但前 27 队逐一核对均无公开 notebook。
2. **方法层**：两份独立的全量实验记录（alex chilton 168 次提交/59 臂，最大 +0.006；Дворкин 17 个受控实验，最好 0.488）都表明在现有通道上做单旋钮改动的上限约 +0.006，距 0.503（前 20 线）还差 0.011。
3. **资产层**：2026-11-13 之后新增的公开资产全是谱库/前向模型/检索权重的**扩容**，没有一个在公榜上产生 >0.492 的可复现结果；而 host 刚刚放开的 MIST/MassSpecGym 通道，公开侧也还没有人拿出成绩。

**唯一可执行且未被证伪的方向**（不含已验证的 0.492）：把 DreaMS 的「谱→结构 kNN」从当前的**空槽填充**升级为**参与重排的独立分数通道**（与现有神经分做逐分子 z-score 融合，只动第 2–25 名，第 1 名保持 leader-safe）。依据：DreaMS 权重明确允许（745604）、当前只用于填充无分数（068a 内嵌管线核对）、且它与 FPNet 指纹通道信息源不同。但这是**假设**，没有任何公开证据表明它能超过 0.492，必须自证。

---

## 6. 建议（3 条）

1. **锁定 0.492，停止在 `LAM`/门控/中间名次权重上继续扫。** ref 58410619（Rogers Johnson v1，`LAM`=0.35）出分即记为 0.492 并作为新底座；`LAM` 0.38/0.40/0.42/0.45/placebo 五臂正在被 Rogers Johnson 自己跑，重复没有信息量。
2. **下一条实验必须是新通道，且只做一个。** 首选 DreaMS 分数化重排（如上），次选 ISDB 预测谱（CFM-ID，CC BY 4.0）做前向一致性重排。两者都要求「先离线面板证明 >+0.002，再上公榜」，且必须记录面板 MRR 与公榜的偏差——离线指标误判公榜的例子至少 3 处（自训 FPNet 离线 +26.5% Tanimoto 却 7 次公榜全负；网络传播重排离线 −0.0033 但公榜 +0.001；FIORA-OS 混合离线 −0.0139），均见 rogersjohnson 的公开表。
3. **不要相信标题分，也不要再跟 fork。** 0.494/0.498/0.5 标题的 notebook 作者实际 <0.487–0.492；前 27 队无一公开配方。任何候选进决策前，先按本文 §2.1 的方法核一次作者公榜 ref。

---

## 7. 附：068a 与 ref 58410619 状态（2026-11-16T02:32Z）

| 项 | 状态 | 命令/来源 |
| --- | --- | --- |
| `jiyujia/enveda-068a-n1c-lam035` | `KernelWorkerStatus.RUNNING`，last run 2026-11-16T02:20:10Z | `kaggle kernels status`（jiyujia token） |
| ref 58410619（Rogers Johnson v1 隐藏测试重跑） | `SubmissionStatus.PENDING`，提交 2026-11-16T02:06:52Z | `kaggle competitions submissions` |
| 本队最佳 | **0.488**（ref 58380689，067a） | 同上 |
| 今日剩余提交 | 4/5 | 同上 |

068a 与 58410619 是**同一配方**（都是 065a/0.488 底座 + N-1c `LAM`=0.35），但是两个不同 kernel：58410619 重跑的是 Rogers Johnson 的公开 kernel v1，068a 是本队自有版本（10 个公开数据集 + `popwenman/casmi26-scorer-train-v80`）。两者出分应落在同一档（0.492）；谁先出分都可以用来确认配方的可复现性。

---

## 8. 证据来源

- 比赛与榜单：`mcp kaggle get_competition` / `get_competition_leaderboard`（600 行，2026-11-16T02:32Z）；`list_team_public_submissions`（team 16898092 / 16895960 / 16926646 等）。
- notebook：`get_notebook_info` 逐个拉取，含 `metadata` 与正文；重点：`analyticaobscura/casmi-2026`、`rogersjohnson/casmi26-l4-lam035`、`rogersjohnson/casmi26-p1..p5`、`alexchilton/casmi26-0-492-what-moved-the-board-59-arms`、`zz3t3rn41/casmi26-peaks-to-structures-v2` v57、`evgendvorkin/spectra-to-structures-deep-eda-molecular-clues` v39、`jtheyogi/enveda-casmi26-apex-0494-calibrated`、`jacobcascone/casmi26-0494-fork`、`k3rn3l/casmi26-v598-net492`、`xiangqizhang/casmi26-net492cpu`、`heetp0/notebook0b6a5a21c6` v29、`huangkx666/enveda-p2s-lam035`、`riochiang/casmi26-silver-0488-*`、`kitopl/casmi-2026-data-pitfalls-that-break-your-cv`。
- 数据集：`get_dataset_info`（大小/许可/更新时间），共 26 个 slug。
- 讨论区：`list_competition_topics`（sortBy New/Top，163 topics）+ `list_topic_messages`（750330、757307、756893）。
- CLI：`KAGGLE_API_TOKEN=$HOME/.kaggle/access_token_jiyujia kaggle kernels status` / `kaggle competitions submissions`。
- 本队记录：`docs/progress/068-n1c-lam035.md`、`notebooks/068a/kernel-metadata.json`。

**未做的事**：没有运行任何训练或推理；没有推送、fork 或提交任何 kernel；没有修改 068a 或任何代码文件。本文的每个分数都来自榜单或 notebook 作者自述，标题分一律单列为"标题/自述分"并附作者实际公榜对照。
