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

### 指标口径核实（解码 FROZEN_B64 原文）

```python
def metric_identity(smiles):
    key = canonicalize_smiles_to_ik14(smiles)
    return (key, heavy_atom_composition(smiles)) if key else None
```

- 判等 = **InChIKey14（连通性块）+ 重原子组成**，**不含立体化学**。
- 推论 1：底座按 `ik14` 去重 / 升位是**正确**的，不会误删正确立体异构体 → 排除了一条系统性丢分假设。
- 推论 2：任何立体异构体只要连通性正确即计分，因此生成/检索路线不需要立体精度（但精度瓶颈仍在，见 §6 的盈亏平衡推导）。
- 推论 3：25 个槽位应放 25 个互不相同的 ik14，底座已满足。

## 2026-11-17 15:30 决定性算术：ICE 无法改善 rank-1（否决宽窗 ICE 路线）

核实 `e004_candidate_inputs.json`（12 个可见分子）：底座真实候选窗口远大于 25——
`prefusion 25 / base_tail 60 / engine2 40 / pubchem 25 / fusion_overflow 15`，并集约 **165/分子**。这本来是"宽窗召回"的入口（082f3 只看 20 个 outsider）。

但把两个已测量数字并排：

| 量 | 值 | 来源 |
| --- | --- | --- |
| timsTOF 微调 ICEBERG 在**未见结构**上的 R@1 | **0.4395** | 069 ICE 泛化验证（实测） |
| 底座 rank-1 正确率（0.492 MRR / 0.72 recall@25 反推） | **≈0.43** | `069-pipeline-map` §4.2 |

**ICE 的 rank-1 精度与底座几乎相等**。rank-1 替换的盈亏平衡是 `p ≈ 0.9(1−c)`，c=0.43 → **p≈0.51**；ICE 只有 0.44，**低于盈亏平衡**。

推论（不需要再实验）：
1. 宽窗 ICE 重排（把 165 个候选交给 ICE 选 25）**期望为负**，不提交。
2. 073a（ICE 0.3 权重混合）实测 0.491 与此一致。
3. 082f3（ICE 给 outsider 打分升位）同一算术下先验偏弱，其价值只在于"margin 0.10 的门把精度抬到 0.51 以上"这一未验证假设；已在队列，等结果即可，不再做同类变体。
4. 任何"换一个谱学打分器来改 rank-1"的路线都要先证明该打分器 rank-1 > 0.51；目前没有任何公开打分器达到（MIST 0.107、ICE 0.4395、对齐头 0.316）。

**结论：0.492 之上只剩"训练一个 rank-1 > 0.51 的排序器"这一条路，其余机制均被算术封死。**

## 2026-11-17 15:38 实测否决"对齐头选 rank-1"（最后一个非库信号）

076a 的留出数字 R@1 0.6684 是在**训练分布内**（182,941 结构 / 13,803 个 ≥2 同式组）测的；它当年只被允许以 w0.3 重排 2–25 名、leader 冻结，**从未测试 25 选 1**。

本次直接测（kernel `jiyujia/enveda-align25-probe` v3，10.5s）：用 e180 未见天然产物 3,713 个结构、1,843 个分子式组，按公式组采样真值+干扰项，对齐头打分。

| 候选数 | R@1 | 随机 | n |
| --- | --- | --- | --- |
| 5 | **0.321** | 0.20 | 168 |
| 10 | **0.131** | 0.10 | 168 |
| 25 | 0.143 | 0.04 | 7（组太少，不可靠） |

- 在**未见结构**上，对齐头仅略高于随机；10 路只有 1.31×随机。**远低于底座 rank-1 0.43**。
- 与 ICE 的结论文档一致：ICE 的 0.4395 是在**同式组中位 5 个候选**上测的（R@1 per-candidate 更弱）。
- 因此三条非库信号全部低于 rank-1 替换的盈亏平衡（0.51）：MIST 0.107、timsTOF ICE 0.4395(5路)、对齐头 0.32(5路)/0.13(10路)。

**最终结论：0.492 底座在所有可用公开信号上已是 rank-1 最优；唯一可能有量级提升的路径是从头训练一个在未见结构上 25 选 1 精度 >0.51 的模型。这需要多天 GPU 与高方差投入，按项目约定交用户决定。**

### 隐藏加合物覆盖核实（解码 FROZEN_B64 原文）

`ADDUCTS` 表含 `[M+H]+ / [M+NH4]+ / [M+Na]+ / [M+K]+ / [M-H2O+H]+ / [M-2H2O+H]+ / [M+2H]2+ / [M]+ / [M-H2O]+ / [M+CH3OH+H]+ / [M+CH3CN+H]+ / [M-H]- / [M-H2O-H]- / [M+CH2O2-H]- / [M+Cl]-` 等，**官方列出的 10 种隐藏加合物全部在内**。→ 排除"源内失水加合物导致 class 1 丢分"的假设。

### 算力盘点（2026-11-17）

jiyujia **12.66h** 余 / zhuating 1.69h / qingyu1232 0.28h / ccjjllt 0（11-21 刷新）→ 合计约 14.6h T4，不足以有把握地训练一个在未见结构 25 路上 R@1 >0.51 的新模型。

## 2026-11-17 16:05 class 2 排序器训练（止损式）：闸门 FAIL，不提交

- 数据：`train.parquet` 过滤 **timsTOF + [M+H]+** = 589,076 谱 / 167,406 结构（全库 2,539,608 行 / 275,810 结构）。
- 模型：Morgan FP(2048) → MLP(1024,1024) → 4000 个 0.25 Da 分箱谱（softmax），cosine 损失，8 epochs，4,328 steps，1,054s（T4）。
- kernel `jiyujia/enveda-class2-fp2spec` v2 COMPLETE；产物 `class2_ranker_result.json` / `fp2spec.pt`。

| 评估 | 5 路 R@1 | 10 路 R@1 | n |
| --- | --- | --- | --- |
| 分布内（train 留出结构） | **0.820** | **0.736** | 250 / 110 |
| **分布外（e180 未见天然产物）** | **0.205** | **0.195** | 200 / 200 |
| 随机基线 | 0.20 | 0.10 | — |

- **判定：FAIL**（闸门 e180 5 路 >0.44）。分布内 0.82 → 分布外 0.205 ≈ 随机：模型学到的是训练骨架的记忆，不可迁移。
- 与两条既有证据一致：对齐头同样是"分布内 0.668 → 分布外 0.32"；公开记录 7 次自训替换全部掉分。
- 现有所有信号在 e180 5 路上的排序：**ICE 0.4395 > 对齐头 0.321 > FP2Spec 0.205 > 随机 0.20**。**ICE 仍是我们最强的 OOD 结构→谱信号**，而它也只有 0.44，低于 rank-1 替换的盈亏平衡 0.51。
- 按止损线：**不建 overlay、不消耗提交额度**，本次训练成本 1,054s GPU。
- 结论：要用训练突破 class 2，需要的不是"在 14.6h T4 上训一个 MLP"，而是一个 **GNN 级结构→谱模型（ICEBERG 量级）+ 数十 GPU 小时**；本轮预算内不可达。

### 训练路线的最后一块也封死了：ICE 权重已经在本赛数据上微调过

`hjyact/casmi26-timstof-iceberg` 数据集说明原文（2026-11-17 实查）：

> `ice_tree/sota/ice_ckpt/tims_ft3/best.ckpt` — ICEBERG intensity model **fine-tuned by us on the competition training spectra** recorded on timsTOF instruments; all structures of our validation rulers removed first

- 我们最强的 OOD 结构→谱模型（e180 5 路 R@1 **0.4395**）**已经是"在本赛 timsTOF 训练谱上微调过"的版本**，且作者做了正确的留出（验证集结构全部剔除）。
- 因此在同一份训练数据上再微调**不可能有增益**；要让 class 2 的 25 路 R@1 超过 0.51，需要的是**新数据或新架构**，不是再训一遍。
- 作者自报的留出增益（+0.030 Enveda-type / +0.018 COCONUT，相对 public ICEBERG）在我们底座的融合里已被吸收——073a 实测公榜 0.491（−0.001）就是证据。

## 结论：在现有数据与算力下，0.492 已到顶

证据链（全部为实测或算术，不是推测）：

| 路线 | 上限 / 实测 | 状态 |
| --- | --- | --- |
| 集合内重排（DreaMS kNN / ICE / 对齐头 / 神经 / ICE+对齐） | 公榜 0.492 / 0.491 / 0.490；可见审计 0 换帅 | 5 次实测，全部 ≤0.492 |
| 松门外谱库升位（082c） | 公榜 **0.468**（−0.024，反推精度 3–10%） | 实测证伪 |
| 高精度库升位（082d）/ ICE 池外召回（082f3） | 各自 ≤4–40 个升位，有界 | PENDING |
| 池外生成（MIST/MolForge） | 尾槽 +0.0004~0.0013；rank-1 盈亏平衡精度 0.51，MIST 0.107 | 算术封死 |
| 宽窗 ICE 重排（165 候选） | ICE 0.4395 < 0.51 | 算术封死 |
| 对齐头选 rank-1 | 未见结构 5 路 0.32 / 10 路 0.131 | 实测封死 |
| 从零训 FP→谱 | 分布内 0.82 → **分布外 0.205** | 实测封死（闸门 FAIL） |
| 微调 ICE | ft3 已在本赛 timsTOF 训练谱上微调过 | 数据层面封死 |

**剩余唯一未闭环：082d（58457692）与 082f3（58460262）的公榜分**，两者升位数少、有界，最好情况约 +0.005~0.02，最坏约 −0.02。

## 2026-11-17 16:38 082g 合并版已提交（ref 58462795）

- kernel `jiyujia/enveda-082g-combined` v1 COMPLETE（35 cells，两个 overlay 串联）：
  - 机制 1（e180 高精度升位，τ=0.72）→ 可见集 `hits 4, promotions 4`；
  - 机制 2（ICE 余弦 outsider 召回，margin 0.10 / MIN_OUT 0.60 / cap 40）→ 可见集 `gate_pass 0, promotions 0`（12 个分子 lib_max 全 1.0，门正确拦住），且正确读取 `submission_pre_e180d.csv` **排除机制 1 已改动的分子**（两押注落在互斥分子上）。
  - 提交文件 400 行 / 唯一 / ≤25 槽。
- 已提交 **ref 58462795**（今日剩 1 单），PENDING。
- 队列（按 hidden rerun ≈5h 外推）：082d ≈10:45Z、082f3 ≈12:15Z、082g ≈13:45Z。
- 三份提交的机制互斥性：082c（松门库，已 0.468）/ 082d（严格库，4 升位）/ 082f3（ICE 召回，≤40）/ 082g（前两者合并）。全部有界，最坏合计下行约 −0.03。

## 2026-11-17 16:42 收尾基线核实：0.492 共 3 份，ref 已锁定

`kaggle competitions submissions` 全量 50 份实测：

| ref | public | 时间(UTC) | 说明 |
| --- | --- | --- | --- |
| **58413244** | **0.492** | 2026-11-16 03:23 | 072a（068a + DreaMS kNN w0.3，实际与底座同分） |
| **58412029** | **0.492** | 2026-11-16 02:47 | **068a 本体**（065a + N-1c LAM=0.35） |
| **58410619** | **0.492** | 2026-11-16 02:06 | 公开配方 N-1c LAM=0.35 直提（ccjjllt） |
| 58413231 | 0.491 | 11-16 03:23 | 073a ICE |
| 58415444 | 0.490 | 11-16 04:32 | 076a 对齐头 |

- 最终选择时以 **58412029（068a 本体）** 为首选基线，另两份同分可作备份。
- PENDING：58462795（082g）、58460262（082f3）、58457692（082d）。
- 今日剩 1 单，按决策表只在 082d/082g > 0.492 时用于 082e。
