# 069 进展：新信号筛选（timsTOF ICEBERG、DreaMS、PubChem 尾召回）

## 2026-11-16 算力与基线

- ccjjllt 周 GPU 已用尽（33.76h / 30h，0.00h 剩余，2026-11-21 08:00 北京刷新）。
- 用户授权使用 jiyujia 与 zhuating 的配额。实测：
  - jiyujia：5.62h 已用 / 30h，剩余约 24h；与 ccjjllt 同队，可读比赛提交。
  - zhuating：22.55h 已用 / 30h，剩余约 7.45h；未接受本比赛规则，不能挂比赛数据；当前有 2 个其他 GPU session，推送被 “Maximum batch GPU session count of 2” 拒绝。
- 068a 用 jiyujia 推送 `jiyujia/enveda-068a-n1c-lam035` v1，commit run 约 22 分钟，400 行 submission 审计通过（1–25 候选、400 唯一 molecule_id、top-1 相对 pre-N-1c 0 变化、2 个列表重排）。提交 ref **58412029**（2026-11-16 02:47 UTC），出分 **0.492**，与公开复现一致。
- 公开 0.492 kernel 提交 ref **58410619**：2026-11-16 15:10 北京出分，公榜 **0.492**，新最佳；ccjjllt 第 314/5376（2026-11-16T07:10Z 快照）。0.492 平台已扩散，仍未进入前 100。

## 069a：ISDB-NP 预测谱尾部插入（否证）

- 数据：`hemantkumarbk/isdb-np-insilico-msms-library`（298,084 张 CFM-ID 预测谱 / 148,935 LOTUS 结构，CC BY 4.0；host 已放行 CFM-ID 4）。
- 实现：068a 之后新增 overlay，对每个分子按 precursor ±10 ppm、同加合物/极性取 ISDB 候选，sqrt-intensity cosine；仅当 cosine ≥0.80 且结构不在当前 25 名时，替换最后 3 槽；leader 与 2–22 名不动。
- 运行：`jiyujia/enveda-069a-isdb-tail` v1，commit 约 26 分钟；`isdb_tail_audit.json`：molecules 400，**molecules_with_tail_insertion 0**，tail_slots_inserted 0，elapsed 28.6s。
- 解释：可见 test 是 train 的占位副本；0/400 说明预测谱与实测 timsTOF 的余弦在 0.80 门控下几乎没有命中，或 400 个可见分子不在 LOTUS。与 ISDB 作者“公开 notebook 固定权重 −0.009”一致。**不提交，不继续 LOTUS/ISDB 方向。**

## 072a：DreaMS Enveda180 kNN 中名次重排

- 数据：`hengck23/hengck23-dreams-enveda-casmi26`（DreaMS 权重 386 MB + 1,153,785 × 1024 Enveda180 嵌入 + index，MIT/CC BY 4.0）。
- 实现：`dreams_minimal_model.py` 计算 1,213 张可见 test 谱的 DreaMS 嵌入；对 1,153,785 条 Enveda180 嵌入分块 cosine top-100；按 inchikey14 取每候选最大相似度，`z(base)+0.3*z(sim)` 重排 2–25 名，leader 冻结。
- 运行：`jiyujia/enveda-072a-dreams-knn` v1，commit 完成；加载 ref (1,153,785, 1024)，query (1,213, 1024) 18s；kNN+重排 35.2s；`dreams_knn_audit.json`：**reranked 11/400**，top1_changed 0。
- 可见 test 的 max_sim 多为 1.0（占位 test 与 train 重复，kNN 找到同谱），只能证明通道可运行，不能证明隐藏集增益。提交 ref **58413244**，**公榜 0.492，与 068a 打平，无增益**。DreaMS kNN 中名次重排不作为新底座。

## 073a：timsTOF 微调 ICEBERG 中名次重排（当前最强新信号）

- 数据/代码：`hjyact/casmi26-timstof-iceberg`（MIT，221 MB，含 `ice_tree/kag/ice_rerank.py`、`tims_ft2/tims_ft3` checkpoint、ms-pred 代码、vendor DGL2 与 wheels）。
- 实现：068a 之后以子进程运行 `ice_rerank.py`；`CASMI_MAIN=ice_tree/main`，`ICE_TOPN=20`，`ICE_WEIGHTS=0.3`，`ICE_CKPT=tims_ft3,tims_ft2`；取 `sub_safe_w0.3.csv`（leader 保持），校验行数/ID/≤25 后原子替换。
- 运行：`jiyujia/enveda-073a-tims-iceberg` v1，commit 完成；`tims_iceberg_audit.json`：**applied true，rows 400，topn 20，weights 0.3，ckpt tims_ft3+tims_ft2，leader_changed 0，elapsed 92.5s**；ICE 预测 628/628 候选，lists re-ordered 12/400。首次差分位次分布：rank 2–6 共 11 个，rank 13 共 1 个。
- 作者留出证据：单 ckpt +0.030/+0.018，双 ckpt +0.036/+0.017 MRR@25。**提交 ref 58413231，公榜 0.491**：相对 068a 的 0.492 **−0.001**，未提分。
- 反向教训：ICE 在**结构未见、同仪器**留出上 R@1 0.4395/R@20 0.9997（`enveda-ice-val`），但管线底座已含 MassSpecGym ICEBERG/GLACIER + 神经重排，timsTOF ICE 的中名次信号高度冗余；12/400 可见重排在隐藏集净效应为负。**timsTOF ICEBERG 不作为新底座，不再调 w0.5/topN。**

## 074a：PubChem/outsider 尾 3 槽 rescue（no-op）

- 改动：`CUSTOM rescue` 0→3；`consensus.rescue_tail` 允许 PubChem/fusion/engine2 外侧候选按神经分数竞争最后 3 名；1–22 名不变。
- 运行：`jiyujia/enveda-074a-rescue3` v1，commit 完成；日志 `outsider_fills 0`，最终 `submission_sha256` 与 068a 相同（74a1b363…），400 行未变。可见 test 上无 outsider 被接纳；该改动是 no-op，不提交。历史 busyaprime 0.477 底座同改动 −0.006，与本次 no-op 一致，放弃。

## 076a：DreaMS×Morgan 对比对齐（训练完成，推理 kernel 运行中）

- 训练：`jiyujia/enveda-align-train` v1，数据 `hengck23` + RDKit 2026.3.3；182,941 个唯一结构、22,266 个分子式、13,803 个同式组（≥2）；训练 1,038,704 谱，留出 115,081 谱 / 10% 结构。
- 模型：DreaMS 1024→512→256 与 Morgan 2048→512→256 两个 MLP，同式硬负例 InfoNCE，batch 1024，1014 steps（约 38 分钟；前 200 步从 4.7 GB memmap 随机读取较慢）。
- **留出验证：n=1,529 个同式候选组，R@1 0.6684，R@20 0.9954**；`align_head.pt` 7.35 MB，`align_config.json` 记录同一指标。
- 推理：`jiyujia/enveda-076a-align` v1，overlay 用对齐分数只重排 2–25 名，w0.3，leader 冻结；`align_audit.json` reranked 12/400。提交 ref **58415444**，**公榜 0.490**：相对 068a 0.492 **−0.002**，未提分。
- 结论：对齐头在未见结构同式组上 R@1 0.316、median rank 2，但管线底座已经有 FPNet/谱库/前向/网络四类证据；对齐中名次重排的净效应为负。**DreaMS×Morgan 对齐不作为新底座**；若要继续，必须把它做成候选级强打分或替换现有指纹目标，而不是 2–25 名小权重叠加。

## 076a 泛化验证：Enveda-180 未见结构留出

- 数据：`aryanawasthiteykik/casmi26-e180-not-in-train`（3,719 个不在 train 的结构 × 14,471 张 timsTOF 谱，CC BY 4.0）。
- kernel：`jiyujia/enveda-align-val` v1，COMPLETE；`align_val_result.json`。
- 结果：622 个同式组（≥2）中 9,853 张谱可评；**R@1 0.3165，R@20 0.9889，median rank 2，median group 5，max group 31**。
- 解释：对齐头在**未见结构**上仍把真值排到中位第 2，R@20 近 0.99；rank-1 不如训练留出的 0.668，说明存在结构域移，但作为 2–25 名重排信号仍有价值。与 076a 可见 test 仅 12/400 重排一致：底座顺序已经较强，对齐主要修正少数列表。

## 075a：timsTOF ICEBERG 权重 0.5（与 0.3 同为 12/400）

- `jiyujia/enveda-075a-ice05` v1 COMPLETE；`tims_iceberg_audit.json`：applied true，topn 20，weights 0.5，leader_changed 0，elapsed 91.7s；lists re-ordered 12/400，与 073a w0.3 的数量相同。可见 test 不能区分权重，等 073a 公榜。

## ICE 泛化验证：Enveda-180 未见结构（强阳性）

- kernel `jiyujia/enveda-ice-val` v1，COMPLETE；用 3,719 个未见结构中 5,645 张受支持加合物的 timsTOF 谱构造合成 test；同式组 ≥2 的 3,463 张谱、22,825 个候选；tims_ft3+tims_ft2 预测 22,825/22,825，耗时 1,450s。
- **结果：R@1 0.4395，R@20 0.9997**（同式组中位 5 个候选，随机约 0.2）。
- 解释：timsTOF 微调 ICEBERG 在**结构未见、同仪器**的留出上仍有 44% top-1，接近作者留出 +0.036 的方向；这是 073a 提交的最强证据链。继续以 073a 为主候选。

## 081a：全加合物 N-1c 图（commit-only）

- 改动：N-1c 不再只连 `[M+H]+`/`[M-H]-`，对所有观测加合物建图，跨加合物用 precursor shift 的 modified cosine 兜底；`LAM=0.35`、`TAU=0.6`、leader 冻结。
- 运行：`jiyujia/enveda-081a-alladduct` v1 COMPLETE；`[n1c]` 日志：molecules 400，**with spectra 400（原 391）、with neighbors 118（原 114）、edges 113（原 109）、changed lists 2、top1 0、162s**。
- 可见 test 只多 4 条边、最终仍只改 2 个列表；隐藏 test 的 10 加合物（含水丢失）可能带来更多邻居，但可见证据很弱。等明日额度再决定是否用一次提交验证隐藏加合物假设。

## 082a：Enveda-180 not-in-train 外部谱库（hidden-specific，commit 中）

- 数据：`aryanawasthiteykik/casmi26-e180-not-in-train`（3,719 个不在 train 的 Enveda-180 结构 × 14,471 张 timsTOF 谱，CC BY 4.0，host 已批准 Zenodo 来源）。
- 改动：068a 之后新增 overlay；对 `library_max<0.90` 的 query，按结构精确质量 ±10 ppm、同极性、modified cosine ≥0.80、≥4 峰匹配；命中则把该结构升到 rank 1，已在列表则前移；否则插入。
- 预期：可见 test 是 train 副本，commit 审计预计 0 命中；只有 hidden test 含这些“库外但 Enveda-180 有谱”的结构时才会生效。若 073a/076a 公榜正向而额度允许，用一次提交验证这个 hidden-specific 假设。
- 实际 commit：`jiyujia/enveda-082a-e180lib` v1 COMPLETE；`e180lib_audit.json`：refs 14,471，**hits 164，promotions 164**，elapsed 59.8s；命中 cosine 0.95–0.99。
- 自检索精度核验（CPU kernel `jiyujia/enveda-e180-precision`，30s）：3,559 个多谱结构 / 14,311 个 query，**coverage 0.991，top-1 precision 0.812**。即在 e180 内部，谱库检索能正确找回同一结构；但可见 test 是 train 副本，164 个命中是**跨结构 analog**，不能直接当作真阳性。若 hidden test 与 e180 有重叠，082a 的上限很高；若无重叠，164 个 rank-1 替换有下行风险。
- 082c（多谱一致性门控）：`jiyujia/enveda-082c-e180cons` v1 COMPLETE；要求命中在 ≥2 张 query 谱上 cosine ≥0.75（或单谱 ≥0.95）。`e180lib_audit.json`：**hits 106，promotions 106**（从 164 降到 106），仍保留 rank-1 替换。
- 082b（TAU 0.95，rank-2）：`jiyujia/enveda-082b-e180safe` v1 COMPLETE；**hits 143，promotions 143**，leader 不动，只把命中结构放到第 2 名。082a/082b/082c 三个变体覆盖“激进 rank-1 / 严格 rank-2 / 多谱一致 rank-1”，明日按额度提交对照。

## 083a：Enveda-180 候选 + 管线神经打分（进行中）

- 思路：主池不含 3,719 个“不在 train”的 Enveda-180 结构；082a/b/c 只用谱库余弦。083a 把这些结构按 ±10 ppm 组成候选 evidence，调用管线自己的 `neural_scores.py`（popwenman v80）逐候选打分，再按 z-score 边际决定是否升到第 1 名或第 2 名。
- 这是**候选级**新信号，不是 2–25 名小权重叠加；若 e180 结构在 hidden test 中，神经打分比裸谱库余弦更可靠。
- kernel `jiyujia/enveda-083a-e180neural` v1，commit 中；审计 `e180_neural_audit.json`。若 scorer 失败则原样返回。

## 079a/080a：组合层（commit-only）

- 079a：068a + DreaMS kNN（11/400）+ 对齐（12/400），leader 冻结；`jiyujia/enveda-079a-dreams-align` v1 COMPLETE。
- 080a：068a + timsTOF ICEBERG + DreaMS kNN + 对齐，三层 leader 冻结；`jiyujia/enveda-080a-full` v1 COMPLETE。
- 可见 test 上组合层仍只改动约 12 个列表；具体隐藏增益未知，等单项公榜后再决定是否提交组合版本。

## 078a：PubChem-only 尾 5 槽插入（可见审计不可用）

- 改动：读取管线自己的 `e004_candidate_inputs.json` 的 `pubchem[mid].pc`（top-25 PubChem-only），在 `library_max<0.90` 时替换最后 5 名；1–20 名不动。
- 运行：`jiyujia/enveda-078a-pc-tail` v1，commit 完成；`pc_tail_audit.json`：**molecules_with_insertion 0，slots_inserted 0**，elapsed 0.1s。
- 原因：commit run 的 `e004_candidate_inputs.json` 只有 12 个 smoke 分子的 payload，而 submission 是 400 行；可见审计无法判断隐藏 rerun。真正的 hidden rerun 会有 400 个 payload。该改动未在可见 test 上验证，先不提交；等 073a/076a 分数与明日额度再定。

## 076b：对齐权重 0.60（与 0.30 同为 12/400）

- 运行：`jiyujia/enveda-076b-align06` v1，commit 完成；`align_audit.json`：reranked 12/400，w0.6，leader 0；与 076a 的重排数量相同，具体顺序可能不同。可见 test 不足以区分 0.3 与 0.6；等 076a 公榜后再决定是否提交 0.6。

## 074a 之外的组合计划

- 若 073a（timsTOF ICEBERG）与 076a（对齐）公榜均为正，做 077a 叠加：068a → ICEBERG（top20，w0.3）→ 对齐（w0.3）；两者作用于不同证据，顺序固定，leader 始终冻结。
- 若 073a 正向但 076a 中性，优先调 `ICE_W` 0.5 / `ICE_TOPN` 10–40，只提交一次最优。
- 若两者都中性，下一阶段转池外召回/生成；不再扫公开单旋钮。

## 公榜结果汇总（2026-11-16 晚）

| ref | 版本 | 新信号 | public | 相对 068a |
| --- | --- | --- | ---: | ---: |
| 58410619 | 公开 N-1c LAM=0.35 | 无（公开复现） | **0.492** | 0 |
| 58412029 | 068a | 无（自有复现） | **0.492** | 0 |
| 58413244 | 072a | DreaMS Enveda180 kNN | **0.492** | 0 |
| 58413231 | 073a | timsTOF 微调 ICEBERG | **0.491** | −0.001 |
| 58415444 | 076a | DreaMS×Morgan 对齐 | **0.490** | −0.002 |

结论：**三个公开可复现的新信号都没有超过 0.492**；底座 068a 的融合已经覆盖了这些证据，2–25 名小权重叠加的净效应为零或轻微负。与 alexchilton 59 臂分析的结论一致：单通道/小权重不够，真正的缺口在**池外候选生成**与**更强的排序模型**。

## 待办

1. 等 068a/072a/073a 公榜出分；只有实际出分才记录为有效。
2. 074a commit 审计；若 rescue 名额被大量使用且没有明显破坏头部，考虑用今日最后 1 次提交。
3. 076a 训练完成后看验证 R@1；达标则推推理 kernel，否则放弃。
4. 若 073a 正向，优先做 `ICE_W`/`ICE_TOPN` 小网格（commit-only，公榜只提交最优一个）。
5. 若 073a/072a/076a 都中性，下一阶段转向池外召回/MIST-MolForge 生成，不再扫公开单旋钮。
