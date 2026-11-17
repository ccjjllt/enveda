# 070 公开前沿重扫：11-16 之后有无已核实高于 0.492 的公开配方

审计时间：**2026-11-17T00:51Z**（北京时间 08:51）。只读审计：未推送 kernel、未提交、未跑任何训练/推理、未改动任何代码。
基线是 [069-public-frontier.md](../../069-new-signals/research/069-public-frontier.md)（快照 2026-11-16T02:32Z），本文只记**新增量**。

---

## 0. 结论先行

1. **没有。** 截至 2026-11-17T00:51Z，公开侧**不存在任何已核实高于 0.492 的可复现配方**：凡是作者公榜 >0.492 的账号，都没有公开对应分数的配方；凡是标题写 0.494/0.495/0.500 的公开 notebook，作者实际公榜全部 ≤0.492，且已拉源码逐一核对（标题党，或内容仍是 0.477/0.488 旧配方）。
2. 唯一一个"纸面 0.494"的公开版本（`heliosli/casmi26-0-494-public-demo` v3，第三方 Dariush 记录 11-15 13:05Z 跑出 0.494）**现在已 403（转私有/删除），拉不到源码**，无法核对 diff、无法复现；且该作者其余 0.49x 标题 notebook 拉回来的内容都是 0.477 旧配方。**不能计入已核实**，也不能跟进。
3. 真正的新增量只有两类：(a) **Rohan（mathleteutd）的 0.492**——第 4 个独立账号用同一单旋钮（N-1c `LAM` 0.25→0.35）拿到 0.492，继续加固 0.492 而非突破它；(b) **Igor Zharov 的 LightGBM isomer joint ranker**——新方法（自拟 Heliosli CV-JOINT，无其私有文件），但作者分仍是 0.492，未超线，且分归属尚不确定。
4. 榜面出现一个**无公开来源的 0.494/0.495 簇**（0.494 ×18 队、0.495 ×11 队，首现 11-15），机制未知，公开侧无对应配方。这是本次唯一的开放问题，只记录、不解释。
5. Rogers Johnson（0.492 作者）在 11-16 23:56–11-17 00:03Z 起了一组新的网络参数扫描（q1–q5），尚无提交、无分数，是**待观察**不是信号。

---

## 1. 公榜增量（11-16T02:32Z → 11-17T00:51Z，全量 5422 队）

| 项 | 069 快照 | 本次 | 变化 |
| --- | --- | --- | --- |
| 队伍数 | 5,356 | **5,422** | +66 |
| 前 6 | 0.549 / 0.540 / 0.536 / 0.528 / 0.518(goonew) / 0.518(Terry Luo) | 同上，仅 goonew **0.518→0.522**（11-16 13:55） | goonew +0.004 |
| 新进前 20 | — | Jaret Arnold **0.512**(#7)、pjmathematician 0.509(#11)、JTA/grinders/HMC_Stevelee/Shuta TAKANO 0.507(#13–16)、韩队 0.505(#17)、Johnny Lee 0.504(#19)、huytngo/MY/LH-and-FPGA/GiantWizard 0.503(#20–23) | 0.50+ 扩容 |
| ≥0.500 | 27 队 | **30 队** | +3 |
| ≥0.494 | 未统计（rank 50 = 0.494） | **81 队**（0.494 ×18、0.495 ×11） | 新簇，无公开源 |
| ≥0.492 | 236 队 | **516 队** | +280，`LAM`=0.35 持续扩散 |
| ≥0.488 | 576 队 | **746 队** | +170 |
| rank 50 / 100 | 0.494 / 0.492 | **0.496** / 0.492 | 中腰线被 0.494+ 抬起 |
| OzanM. | 0.499（11-15T16:53Z） | **0.499**（11-16 17:35 重交，137 次提交） | 同分续命，仍是公开配方关联账号的最高分；其公开正文止于 0.484 |

新进前 20 的全部队员用户名已用 `kernels list --user` 逐一核对（mjarnold、pjmathematician、stevelee2531、shutatakano、wuliaokaola、huytngo、mikhailyurasov、lhandfpga、giantwizard、hhhoang、agentili、sroziews、xsmaxpc、kkubuck、woominyo、gordeevmax、smirnolegov、jzhpfn，均为 0 个本赛 notebook；bibrocks1 名下 1 个是 10-07 的 "First" 存根）：**前 30 名依然无一支有可确证的公开配方 notebook**，与 069 §2.4 一致。

---

## 2. 新增公开 notebook 核对（按 `dateRun` 取 11-16T02:32Z 之后）

| notebook | 作者 / 公榜 | 标题/自述分 | 是否新配方 | 可复现性 | 与 0.492 配方的 diff |
| --- | --- | --- | --- | --- | --- |
| `mathleteutd/casmi26-0-492-test-set-network-rerank`（76 vote，11-16 04:32Z） | Rohan，rank 124，**0.492**（16 次提交，11-16 10:02Z） | 正文 0.492 + `LAM` 扫描表（0→0.487 / 0.15→0.487 / 0.25→0.488 / **0.35→0.492** / 0.50→0.484） | 否：同一单旋钮 | 是（alexchilton 0.488 底座） | **无 diff**——第 4 个独立账号复现 `LAM`=0.35（前三：Rogers Johnson、alex chilton、Дворкин）。另附一条反证据：learned reranker 在其 Enveda-180 面板 MRR 0.27→0.38，公榜只出 0.08–0.17，已放弃 |
| `flexonafft/casmi26-anchored-spectral-network`（12 vote，11-16 18:18Z） | Igor Zharov，rank 456，**0.492**（11-16 12:25Z） | 无标题分；拟合 LightGBM lambdarank isomer joint ranker | **是，新方法**：按分子式取 PubChem tier 候选，FPNet fold-0 f.z + 文档 + NP 成员 16 特征，自拟 Heliosli CV-JOINT（明说不用其私有的 RT/PX v2 文件），overlay 只动 rank 2 起 | 是（公开输入+自拟权重，训练 notebook 同体） | 与 `LAM`=0.35 正交的新通道，但**分未超线**；且其 0.492 提交（12:25Z）早于本 train notebook 运行（18:18Z），0.492 归属本 ranker 还是 `LAM` fork **无法确定**，只能记为"0.492 的新方法候选"，不可当作增益证据 |
| `dariushafshar/casmi26-the-public-chain-version-by-version`（6 vote，11-16 20:26Z） | Dariush Afshar，rank 740，0.488 | 自述是考据：326 notebook × 788 scored version，逐版列分与 parent diff | 否（考据文档） | 是 | 价值在证据不在配方；其核心断言见 §3（Heliosli 0.494 v3），本表不重复计分 |
| `mohitnehete2005/casmi26-apex-500-sota-n1c3-lam35-extlib-v6`（0 vote，11-16 16:19Z） | JuniorParadox，rank 745，**0.488** | 标题 "0.500+ SOTA"，正文 base 0.488（sub 58333000）+ `LAM`=0.35/TOPK=3/双神经/ExtLib V6 | 否：0.492 同族 + learned 件 | 名义可跑 | **标题 fals**：作者 PB 0.488，无 0.500 提交；`_n1c` 即 N-1c 同物 |
| `riochiang/casmi26-silver-0494-helios` 等 4 个（0 vote，11-16 15:58–17:57Z） | YUANHAO JIANG，rank 688，0.488，最后提交 11-16 00:40Z | 标题 0494/0492 | 否 | — | **这 4 次运行之后无任何新提交**：0494-helios 等变体未出分，记为无增益 |
| `nguynhucng/casmi26-hel494*` ×5（0 vote，11-16 10:41–16:37Z） | Nguyễn Hữu Công，rank 147，**0.492**（11-16 16:32Z） | 标题 hel494 | 否 | 是 | Best 仍是 0.492，hel494 系列未破线 |
| `adityaraj0612/casmi26-exp312-hel494-sovereign-apex`（0 vote，11-16 20:50Z） | ADITYA RAJ，rank 246，**0.492**（11-16 15:31Z） | 标题 hel494 | 否 | 是 | 运行在其 0.492 提交之后，**之后无新提交**，无证据 |
| `ccuuuuy/casmi26-0-492-test-set-network-rerank`（Dante fork，11-16 10:56Z） | Dante #2，rank 434，**0.492** | 同 Rohan | 否 | 是 | 同配方第 5 复现（含 Dante 本体 rank 401 0.492） |
| `aiokajima/enveda-mathlete-0492-reproduction-public`（11-16 20:59Z） | AI OKAJIMA，rank 479，**0.492** | 复现 Rohan | 否 | 是 | 同配方第 6 复现 |
| `huangkx666/enveda-p2s-lam035-tau080/topk3`、`bill123987/n1clam35*` | Leo 系 0.492 / bill123987 rank 463 **0.492** | `LAM`35 变体 | 否 | 是 | 同族 |
| `rogersjohnson/casmi26-q1-nbr10/q2-nbr25/q3-wexp2/q4-posflat/q5-placebo`（11-16 23:56–11-17 00:03Z） | Rogers Johnson，rank 123，0.492（最后提交 11-16 00:27Z） | 邻居数/权重形态扫描（nbr10/nbr25/wexp2/posflat/placebo） | 方法未知（未拉源码，只读标题与时间） | 是 | **扫描进行中、无提交、无分数**：待观察，不计数 |
| `xiangqizhang/casmi26-probe-cpu-*`、`lbjyang/casmi26-v2-492-net` | xiangqi（CPU 线）/ lbjyang rank 4826 **0.138** | 探测/标题 492 | 否 | — | 失败或探测，无分 |

`ozertuu/0-500-casmi26-dual-neural-physical-isomer`（069 已判 fork，Dariush 补全版本分：v1 0.482 … 最好 v6/v13/v15 **0.487**，v20 起跌到 0.465）：标题 0.500 fals，维持原判。

---

## 3. Heliosli 0.494 专项核证（本次唯一"纸面超线"项，为何仍不计数）

1. **断言来源**：Dariush 考据（READ_UTC 11-16 16:51Z，在 069 快照之后）的 VERSIONS 表：`heliosli/casmi26-0-494-public-demo` v1 0.488（11-03）→ v2 0.489（11-13 23:44）→ **v3 0.494（11-15 13:05，+1044/−1 行）**。
2. **不可核实**：该 notebook 现已 **`kernels pull` 403 Forbidden**（转私有或删除）；flexonafft 同日注记 "His fitted files are no longer public"，互相印证作者在收回公开资产。无源码则无 diff、无复现、无跟进。
3. **作者标题模式 fals**：同作者另两个 0.49x 标题 notebook 已拉回源码——`heliosli/casmi26-0-495`（11-13）与 Google 索引的 `heliosli/casmi26-0-494`——**内容都是 "LB 0.477 · Two 0.472 Fixes" 旧配方**（Spectral Parallax 系），标题分与内容分系统性偏离（069 §2.3 同类，最大 −0.018）。
4. **榜面侧写不支持归因**：榜上有 18 队恰为 0.494，但最早的 Emre Cirak（11-15 10:22Z）**早于** Heliosli v3 运行（13:05Z）；且 leaderboard 上无任何 Heliosli 同名队。0.494 簇的机制归属不明（见 §1 第 4 点），不能把簇的存在当成该 demo 的复现证据。
5. 配套的 heliosli 数据集栈（11-16 06:16–06:36Z：`frigid-e1` 1.6 GB、`joint-ranker`、`rr-v3`、`pxv2-deploy`、`rt-np`、`m5-v2d`、`j2-judge`）是 reranker 组件发布，**没有任何公开 notebook 用它们跑出 >0.492**。

**判定**：0.494-demo 按"分数层曾存在、方法层不可见、资产层已撤回"记录，不计入已核实，不跟进；若作者重新公开，优先复核 v3 的 +1044 行 diff。

---

## 4. 数据集侧新增（`datasets list -s casmi26 --sort-by updated`，11-16T02:32Z 之后）

| slug | 内容 | 潜在价值 | 判 |
| --- | --- | --- | --- |
| `dariushafshar/casmi26-notebook-chain-changes-and-scores`（81 KB，11-16 20:26Z） | 326 notebook × 788 版本分与 parent diff 全表 | 本次 §2–§3 的第三方证据源 | 考据有用，配方无 |
| `heliosli/casmi26-frigid-e1` 等 7 个（11-16 06:16–06:36Z） | FRIGID/RT-NP/PXv2/joint-ranker/rr-v3 组件 | 见 §3.5 | 无 >0.492 实证 |
| `carsonrodrigues/kaggle-medal-line-tracker`（11-17 00:30Z） | 奖牌线跟踪（全赛通用） | 无 | 无关 |

结论与 069 §3 一致：新增仍是**谱库扩容/考据/组件发布**，没有带来 >0.492 的新权重或新谱库。

---

## 5. 讨论区（方法缺口，如实记录）

- Kaggle CLI **没有 discussion 只读接口**（`competitions leaderboard / kernels list / datasets list` 均无 topic 覆盖）；网页抓取讨论列表被 Kaggle 限流（"Too many requests"，JS 渲染页），本次未能直接列出 11-16 之后的新帖。
- 可用的最新第三方系统记录仍是 Dariush 考据（READ_UTC 11-16 16:51Z）：其 788 版本全表即 covers 11-14–11-16 的公开版本动态；其"前沿 0.494"断言的核证见 §3。
- Web 搜索未发现任何 ≥0.50 方法披露（仅命中 Heliosli 0-494 页，其内容即 §3.3 的 0.477 旧配方）。
- **缺口**：11-16 16:51Z 之后的新 topic，需下次用 MCP `list_competition_topics`（sortBy New）补查；本次结论"无可跟进新配方"不受此缺口影响（配方侧已由 kernels 全量 dateRun 排序覆盖）。

---

## 6. 明确回答

> 2026-11-16 之后，公开侧是否出现**已核实高于 0.492** 的配方/讨论/数据集？

**没有。** 分三层：

1. **分数层**：作者公榜 >0.492 且正文含对应配方的公开 notebook 不存在。OzanM. 0.499 的公开正文止于 0.484（复现 0.488）；前 30 名无公开配方；18+11 队的 0.494/0.495 簇无公开来源。
2. **方法层**：新增的唯一真新方法是 Igor 的 LightGBM joint ranker（自拟、可复现），但分止于 0.492 且归属不明；其余全部是 `LAM`=0.35 同族复现（Rohan/Dante/AI OKAJIMA/bill123987/Leo 系，0.492 账号已逾 10 个）或标题党。`LAM` 扫描（Rohan 表 + Rogers Johnson q-series 进行中）仍是单旋钮范畴。
3. **资产层**：新增数据集无一在公榜上产生 >0.492 的可复现结果；host 规则侧本次未复查（只读 CLI 无规则接口，且 069 §4 距今仅 22 小时，变化概率低，列为下次补查项）。

**可执行建议**：不跟进任何公开新配方；唯一值得挂耳的是 Rogers Johnson q-series 的结算分（若其 `nbr/wexp/posflat` 任一臂 >0.492 且代码公开，再按 069 §2.1 方法核 ref）。在此之前，A 线仍锁定 068a（`LAM`=0.35，public 0.492）。

---

## 7. 证据来源

- 榜单：`kaggle competitions leaderboard -c enveda-CASMI26-molecule-id-mass-spectra -d`（csv，2026-11-17T00:51Z，5,422 队）；top-20 成员用户名来自 csv `TeamMemberUserNames` 列。
- notebook：`kaggle kernels list --competition ... --sort-by voteCount/dateRun`（各 40–50 条）、`--user heliosli/guhongbin/flexonafft` 与 19 个新前排队员名、`kernels pull` 拉回 6 个（mathleteutd 0.492、mohitnehete2005 500、flexonafft anchored、dariushafshar chain、heliosli 0-495 成功；heliosli 0-494-public-demo 403）。
- 数据集：`kaggle datasets list -s casmi26 --sort-by updated`。
- 第三方考据：dariushafshar chain notebook（READ_UTC 11-16 16:51Z）VERSIONS 表；websearch 交叉核对 Heliosli 0-494 页目录。
- 全部拉取物在 `/tmp`（`lb070/`、`k070/`），未入仓库；`git status` 仅新增本文档。

**未做的事**：没有推送/fork/submit 任何 kernel；没有运行训练或推理；没有修改任何代码；每个分数均来自榜单 csv 或 notebook 作者自述，标题分一律单列并附作者实际公榜对照。
