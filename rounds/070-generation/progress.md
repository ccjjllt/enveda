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
