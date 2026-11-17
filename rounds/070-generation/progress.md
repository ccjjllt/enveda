# 070 进展

## 2026-11-17 070a commit

- `jiyujia/enveda-070a-full25` v1 COMPLETE 但 overlay 未生效：`neural_scores.py:32 assert mid in query`——evidence 含 400 分子而 commit run 的 query 谱只覆盖 12 分子，首个缺谱分子即断言失败，无输出、submission 原样。根因是 scorer 要求 evidence 分子必须在 query 中（083a 因 evidence 源自 query 未触发）。
- 修复：evidence 只含 query 覆盖的分子（commit 12 个，hidden rerun 全 400），其余分子保持原样；v2 已推送。审计待拉取：若可见改动只有 ~10 列表或 top-1 零变化，按 kill 规则判死不提交。
- v2 COMPLETE：`neural_full25_audit.json`：**reranked 12，top1_changed 0**（8 个因 `lib_max>=0.90` 冻结 leader）；diff 确认 12 列表变化、top-1 零变化。**070a 按 kill 规则判死，不提交。** 神经分与底座顺序在头部几乎共线，换帅门控全军覆没。
- 070b（ICE+对齐联合）已实现并跑完：`jiyujia/enveda-070b-icealign` v1 COMPLETE；`ice_align_full25_audit.json`：**ice_scored 400，align_scored 400，reranked 12，top1_changed 0**（4 个因 lib 冻结）。双新证据全覆盖 400 分子仍只改 12 列表、零换帅，与 070a/072/073/076 同模式。**070b 同样判死，不提交。**
- 结论：集合内重排路线（5 个独立打分函数：DreaMS kNN / timsTOF ICE / 对齐头 / 管线神经 / ICE+对齐联合）全部零换帅。底座头部顺序已被现有证据锁定，`w_mid` 与 RRF 融合没有可挖增量。

## 2026-11-17 11:00-11:35 P0a：MIST(MassSpecGym)+MolForge 链验证

目标：在 train timsTOF `[M+H]+` 150 个结构上跑通 MIST 指纹预测 → MolForge 解码 top-10，测 top-1/top-10 命中率，作为 P0 尾槽生成的阈值与期望值依据。kernel `jiyujia/enveda-p0a-mist-chain`（T4，联网，只读 train.parquet）。

迭代记录（每版均有日志 `[p0a] STAGE`）：

| 版本 | 结果 | 根因/修复 |
| --- | --- | --- |
| v1 | subform FAIL | 子进程缺 `PYTHONPATH` → 修 |
| v2 | subform FAIL | MGF 缺 `ID=`，`KeyError: 'ID'` → MGF 补 `ID=<name>` |
| v3 | subform FAIL | `mist.utils.chunked_parallel` 缺失：`mmf/src/mist` 是 regular 包（有 `__init__.py`）遮蔽 `main_v2` namespace 包 |
| v4 | 诊断确认 | `mist diag /kaggle/working/mmf/src/mist/__init__.py` 证实遮蔽 |
| v5 | subform FAIL | 只留 main_v2 后 `ModuleNotFoundError: pytorch_lightning.loggers.base`（Kaggle PL 无旧 API） |
| v6 | subform FAIL | in-process stub 失败：`tune_utils` 硬 import `ray` |
| v7 | subform FAIL | 子进程隔离 + 剔除 `tune_utils` 仍因子进程无 PL stub 而失败 |
| v8 | **subform OK `n_json=150, rc=0`** | 拷包到 `/kaggle/working/mistfix` + 空 `__init__.py` + 剔除 `tune_utils`，runner 内先 stub `LightningLoggerBase`/`rank_zero_experiment` 再 `runpy.run_path` |

v8 已进入 `run` 阶段（MIST 编码 + MolForge beam=10 解码 150 分子，batch-size=1，阈值 t=0.172）。结果待拉取，判读口径：若 top-10 命中率明显高于 0（文献修正值 top-1 10.73%），P0 尾槽生成值得做；若为 0 或与 FRIGID 一样全灭，P0 终止。

## 2026-11-17 11:29 公开前沿快照（CLI `--download` 实查）

- 公榜 5429 队；≥0.492 **538** 队、≥0.494 **83** 队、≥0.495 **63** 队。
- Rogers Johnson 仍 **0.492 / 第 125 名**，`LastSubmissionDate 2026-11-16 00:27`——q1–q5（nbr10/nbr25/wexp2/posflat/placebo）跑完**未提交**，无新信号。
- 榜首仍 0.549。公开侧仍无已核实 >0.492 配方。

## 2026-11-17 11:50 审计口径校订（重要）

- 底座 `COMMIT_MODE='smoke'`：commit run 只排 **12 个可见分子**，其余 388 行写单候选占位；`IS_RERUN`（hidden scoring rerun）才排全部 400。证据：`kout068a/submission.csv` 槽位数 min/median/max = 1/1/25，388 行 <25；底座 cell 27 原文「the counts cover the 12 visible smoke molecules」。
- 因此 070a/070b 的 `reranked 12, top1_changed 0` 实际是 **12/12 个可审计分子**，不是 12/400。n=12 下 0 次换帅只能排除大效应（真实换帅率 20% 时漏检概率 6.9%），不能排除 5% 级小效应（漏检 54%）。
- 结论不变（070a/070b 仍不提交：改动集中在 2–25 名、MRR 增量本就是 1e-4 量级），但**今后任何 overlay 审计都必须注明 n=12 的统计功效**，不能再用「改动只有 12/400」当作判死证据。

## 2026-11-17 12:40 P0 终止（权重来源穷尽 + 算术上限≈0）

- v9 结果：`subform OK`，`run FAIL RuntimeError('no mist .pt; candidates=[]')`。DiffMS `zenodo 15122968` 的 `diffms_checkpoints.tar.gz` 只有 **6 个成员、无任何含 "mist" 的文件**；`msg_preprocessed.tar.gz` 是数据（magma/谱），`MassSpecGym_mist_data.zip`（Zenodo 11580401）经中央目录实查 **22,726 条目、0 个 .pt/.ckpt/.pth 权重**。MIST-MolForge README 声称的 `mist_msg.pt` 在公开 Zenodo 记录里实际不可得。
- 架构也不兼容：`mist_fp.ckpt`（Zenodo 11580401，103MB）是 MIST 旧架构（3 层 peak attn、fragment predictor 512 维），与官方 MSG 配置（2 层、2048 维）逐层 size mismatch，不能直接用。
- 决定性判据见 `research/070-pool-external-generation.md` §6：MRR@25 下尾槽生成上限 +0.0004～+0.0013，rank-1 升位盈亏平衡精度 51%（弱池子集 72%），MIST 修正 top-1 仅 10.73%。**P0 不进入提交，GPU 不再投入。**
- 遗留可复现资产：探针 kernel `jiyujia/enveda-mist-p0-probe`（5 个权重可下）、`p0a/p0a_chain.py`（含 subform 修复的完整链：拷包+空 `__init__`+剔除 `tune_utils`+runner stub 旧 PL API）。若将来拿到 `mist_msg.pt`，链只差替换权重文件。

## 2026-11-17 13:30 082d 标定版已提交（ref 58457692）

- 先做了决定性精度标定（`research/070-library-precision-calibration.md`，kernel `jiyujia/enveda-e180-precision-curve` v3）：归一化余弦 τ=0.70 → precision **0.887**、τ=0.80 → **0.955**、τ=0.85 → 0.975；analog 中位仅 0.484。rank-1 替换盈亏平衡精度 ~0.19，库匹配是其 4.7 倍。
- 由此发现 **082c 的门其实极松**：它的 `cos` 是未归一化 sqrt-强度内积（未除范数），τ=0.80 实际等价于"≥4 个匹配峰 + 质量窗"，可见集 106 个"命中"全是 analog。082d 改成归一化余弦后，同一 400 个 query 只有 **4 个** 通过 τ=0.72。
- 082d 审计（`jiyujia/enveda-082d-e180cal` v1 COMPLETE，55s CPU）：`refs 14471, hits 4, promotions 4`，四个升位 n_cons 均 2、cosine 0.754/0.963/0.868/0.792、`in_pool 0`、公式 C16H26N2O3 / C14H19BrN2O2 / C15H22N4O2 / C19H25N3O2。提交文件 400 行 / 唯一 / ≤25 槽。
- 已提交 **ref 58457692**（今日剩 3 单），PENDING。与 082c（ref 58450336，松门 106 升位）构成同日"松门 vs 标定门" A/B。
- 预期：082d 在隐藏集上的升位数 ≈ 隐藏集与 e180 的真实重叠数；若重叠 k 个，收益 ≈ +0.885k/400（k=20 → +0.044），误升损耗有界（每次 0.215/400）。若 082d 也中性，则隐藏集与 e180 无重叠，082 系与公开资产路线正式到顶。

## 2026-11-17 13:45 平台级排队故障（PENDING 原因已查明）

- 讨论区实证（Kaggle MCP `get_forum_topic` 原文，2026-11-17 实查）：
  - **759394 "Submission Queued for 6 days"**（2026-11-15T10:22Z，OverfitOracle）：提交可被卡 6 天；回复（sacuscreed）称平台当日警告过执行问题，且**被卡的提交自动从每日额度中剔除**。
  - **759398 "Notebooks getting queued. Although GPU is available"**（2026-11-15T10:53Z，JamshaidSohail）：GPU 有余量但 notebook 排队。
- 自证：公榜快照显示 **261 个队今天已评分**（最新 05:16Z），我方 `ccjjllt` 仍停在 11-16 04:32 的 0.492；082c（00:49Z）与 082d（05:33Z）均 PENDING。jiyujia GPU 余 14.37h、ccjjllt 0h；metadata GPU=on / internet=off / competition source 已挂，非配置问题。
- 策略调整：不再"等分再动"。把高期望变体尽快推入队列（多份已评分提交可在最终选择时择优；被卡的提交不占额度）。优先级：082e（τ=0.65 + 池内 e180 验证）> 082f（ICE 给 outsider 打分补召回）。

## 2026-11-17 14:07 082e / 082f v1 结果

- **082e**（`jiyujia/enveda-082e-e180tau065` v1 COMPLETE）：`hits 5, promotions 5, in_pool_promotions 0`（082d 为 4）。τ 0.72→0.65 只多 1 个升位；池内 e180 验证路径在可见集 0 命中（预期：可见 12 个分子是 train 结构，e180 是 not-in-train）。→ 与 082d 高度相关，**不单独提交**。
- **082f v1**（`jiyujia/enveda-082f-outsiderice` v1 COMPLETE）：`outsiders 240, ice_scored 12, gate_pass 0, promotions 0`——可见集 12 个分子 `lib_max` 全 1.0，`lib_max<0.90` 门 100% 拦截，无法暴露任何行为。
- 发现 v1 隐患：它把 `ice_rerank` 的 outsider 分与底座 `frozen_candidate_evidence.ice` 列跨源比较，尺度可能不一致（可见集被门挡住，测不出）。→ 已推 **v2**（同源双跑：池内 25 与 outsider 都用同一 ICE 跑；margin 0.15、MIN_OUT 0.40、cap 60、打印两侧 p50/p90 分布）。

## 2026-11-17 14:08 **082c 出分 0.468（−0.024）——松门库匹配实测证伪**

- ref `58450336` 由 PENDING 变 COMPLETE：public **0.468**（底座 0.492，−0.024），这是 11-16 以来第一个新出分结果。
- 反推精度：106 个升位、−0.024 = −9.6 rank 单位；设点火集底座 rank-1 正确率 c，`k·1.0 − (106−k)·c·0.5 = −9.6`：
  - c=0.43 → k≈10.9 → **precision 10.2%**；
  - c=0.25 → k≈3.2 → **precision 3.1%**。
  - 盈亏平衡是 11–19%，实测精度只有 3–10% → 大幅负分完全符合预期。
- 结论 1：**082c 的"未归一化 sqrt-内积 τ=0.80"确实等价于"≥4 匹配峰"**，可见集 106 个 analog 命中在隐藏集同样是 analog（精度标定预测成立）。
- 结论 2：隐藏集与 e180 的**真实重叠很低**（否则即使松门也会有一部分真命中把分数拉正）。082d 的严格门只剩 4 个升位，等于把同一假设压到最小暴露；其期望接近中性，不再是提分主力。
- 后续：把额度转向**池内召回**（082f2：ICE 给 `fusion_overflow`（池内 26+ 名）与 PubChem 提案打分），这是 28% 零分分子里"真值在池内但排不进 25"那部分唯一未测的机制。

## 2026-11-17 15:06 082f3 已提交（ref 58460262）

- 修正了 v2 的语义错误：`ice_scores.parquet` 的 `ice` 列是**重排名次**（1–25，越大越差），真正的分数是 `ice_tims_ft3`/`ice_tims_ft2`（0–0.91 余弦）。v2 拿名次当分数比较（因门全拦住未造成影响）。
- 082f3（`jiyujia/enveda-082f3-outsiderice` v1 COMPLETE，169s）：同源双跑 ICE（池内 25 与 outsider 都用同一模型），`margin 0.10`、`MIN_OUT 0.60`、`cap 40`、`lib_max<0.90`。
  - 可见集：`gate_pass 0, promotions 0`（12 个分子 lib_max 全 1.0，门正确拦住）。
  - 诊断（12 个可诊断分子）：`ice_out p50 0.663 / p90 0.871`；`delta = outsider最佳 − 池内最佳`：**p50 −0.123 / p90 +0.004**，`diag_pass_margin 0` → **outsider 平均打不过池内最佳**，机制在证据不足时不会乱动。
- 已提交 **ref 58460262**（今日剩 2 单），PENDING。这是"池内召回"这一唯一未测机制的有界押注（最多 40 个升位，理论上限下行 ~−0.02）。
- 当前队列：082d（58457692，标定库匹配，4 升位）、082f3（58460262，ICE 召回）；已知 082c = **0.468**。

## 2026-11-17 15:20 公开前沿重扫（11-17 增量）+ 系统性因素核查

**新增 notebook 全部不是已核实的 >0.492：**

| notebook | 标题声称 | 实际 |
| --- | --- | --- |
| `gloriachen0816/casmi26-0510-trainret-full`（05:43Z） | "LB 0.510 (attempt)" | 其团队「邓紫棋牛逼」公榜 **0.488**（11-16 09:06），标题是尝试不是结果 |
| `mathieuw/casmi26-0-494-new-public-dataset`（07:03Z） | "0.494 new public dataset" | 源码就是 heliosli 的 0.477 notebook，`mathieuw` 公榜 **0.487** |
| `heliosli/casmi26-0-495-public-demo`（04:40Z 重跑） | 0.495 | 源码仍是 0.477 配方（两处 0.472 修复合并），无 0.495 证据 |
| `liuzewen123/casmi26-h495-apex` | h495 | 公榜 **0.488** |

→ **公开侧仍无任何已核实 >0.492 的可复现配方**；0.494/0.495 簇依旧无源。讨论区自 11-15 起无新帖。

**trainret 机制评估（gloriachen0816 的新增 cell）**：索引 `train.parquet` 2.54M 张谱（前体 ±12ppm+0.003Da、同离子模式、0.01Da 分箱 sqrt-余弦），`cos≥0.97→rank1 / ≥0.90→rank2 / ≥0.80→rank4`。
- 判定**冗余**：底座 `frozen_candidate_evidence.library_max` 在可见 12 分子上全为 1.0，说明底座库匹配已完整覆盖训练谱库；隐藏集中"库内结构"同样被覆盖，"池外 de novo"没有训练谱可检索。故 trainret 只能覆盖底座已覆盖的分子。

**其他系统性因素（均无缺口）**：
- 可见 test 加合物：`[M+H]+ 959 / [M-H]- 193 / [M+CH2O2-H]- 31 / [M+Na]+ 22 / [M+NH4]+ 4 / [M+K]+ 2 / [M+Cl]- 2`，**400/400 分子的加合物都在底座 SH 映射内**，无系统性丢分。
- 仪器 100% timsTOF；碰撞能量在底座里几乎未被使用（24 处命中全是嵌入模块字符串），是唯一未利用信号，但对"库内已命中"分子增益有限。
- 运行时间：commit 12 分子 21.0min，其中 Engine 2 两个 ranker 644s、GLACIER+ICEBERG 190s；按 400 分子外推接近 9h 上限，与 notebook 自述「±0.002 来自时间守卫是否跳过可选阶段」一致——但 0.492 与 Rogers 独立复现一致，说明我方 rerun 基本完整。

## 2026-11-17 15:23 PENDING 时长已解释：等于 hidden rerun 自身耗时（约 5.2h）

- 公榜快照 07:23Z：今天 **387 队已评分**，最新一条 **07:19:14Z**（4 分钟前）；其中 15 队的最后提交在 05:30–06:30Z 也已出分。
- 同期我方 082d（05:33Z 提交）仍 PENDING 1.85h、082f3（07:06Z）0.25h。
- 结合 082c 实测（00:49Z 提交 → 约 06:0xZ 出分，**5.2h**）与 commit 耗时（12 分子 21min），结论：**我方 PENDING 时长 = 全 400 分子的 hidden rerun 时长（约 5h），不是队列拥堵或被单独卡住**。他人几分钟出分是因为其 notebook 轻。
- 出分 ETA：082d ≈ 10:45Z、082f3 ≈ 12:15Z（北京 18:45 / 20:15）。

### 门控决策表（分数到达后机械执行，今日剩 2 单）

| 082d 结果 | 含义 | 动作 |
| --- | --- | --- |
| > 0.492 | e180 高精度升位成立（082c 反推的 3–11 个真重叠被抓到） | 交 082e（τ=0.65，已 COMPLETE）做 τ 曲线第二点；保留 1 单 |
| = 0.492 | 严格门未点火（隐藏集与 e180 重叠≈0 或未过门） | 公开资产路线正式到顶；剩余额度与 14.37h GPU 转「训练排序器」或停止，交用户确认 |
| < 0.492 | 严格门也打进了 analog | 立即停止库路线，不再消耗额度 |
| 082f3 > 0.492 | ICE 池外召回成立（28% 零分分子的池内 26+ 名部分可救） | 按 `outsider_ice_diag` 的 Δ 分布放开门控（margin 0.10→0.06 或 cap 40→80）再交一版 |
