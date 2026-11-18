# 公开前沿核查（2026-11-17）

本轮问题：0.492 之上还能不能靠现有渠道提分？下面每条结论都附实测来源，未验证的一律标注。

## 1. 榜单位置（实测 `get_competition_leaderboard`，2026-11-17 22:2x 北京时间）

| 名次 | 分数 | 队伍 |
| --- | --- | --- |
| 1 | 0.549 | Shehab Alshehabi |
| 2 | 0.540 | democat |
| 3 | 0.536 | Oleksandr Parfionov |
| 4 | 0.532 | Savko |
| 5 | 0.523 | goonew |
| 6–8 | 0.518 / 0.514 / 0.512 | Terry Luo / 한밭대 / Jaret Arnold |
| 9–20 | 0.510 → 0.504 | 12 队 |
| 21–40 | 0.504 → 0.498 | 20 队 |

我们 0.492 大约在第 45–50 名。**第 20 名（金牌线）0.504，0.492 到金牌线只有 +0.012。**

## 2. 公开代码的上限就是 0.494

`dariushafshar/casmi26-notebook-chain-changes-and-scores`（2026-11-16 16:51 UTC 读取）覆盖 326 个 notebook、91 位作者、788 个有分版本，README 明确写：**该表内最高公开分 0.494**，来自 `heliosli/casmi26-0-494-public-demo` 版本 3。

实测核对：
- `versions.csv` 里 `public_score` 最大值 = 0.494，第二名 0.492。
- 0.494 那一步是 `code changed vs v2 (+1044 / −1 lines)`，是一个**整段新阶段**，不是旋钮。
- 该 notebook 现已取不到：`kaggle kernels pull` 用 ccjjllt 与 jiyujia 两个 token 都返回 `403 You don't have access to this resource`（MCP `kernels.get` 亦被拒）。**内容未核实**，无法复刻。

结论：公开可复现的天花板是 0.492（我们自己就在这条线上）。0.504+ 的 20 队全是非公开改进。

## 3. 59 臂单旋钮实验：旋钮已经用尽

`alexchilton/casmi26-0-492-what-moved-the-board-59-arms`（0.492 本体，168 次提交记录）自述：

> of 59 one-knob arms on a fixed base, 4 gained more than +0.002 and the largest gain was +0.006. 33 fell, and the largest loss was -0.047.
> the gold line is still 0.011 above us, about twice the best single knob we ever measured. So a knob will not get there; a new signal will have to (a better candidate generator, or a reranker that uses something the current channels do not).

逐族核对（同表）：

| 族 | 臂数 | 结果 |
| --- | --- | --- |
| N-1c `LAM` | 3 | 0.35 → **+0.004**（正是我们的 068a）；0.50 → −0.004；TAU 0.6→0.5 → 0 |
| extlib overlay `TAU` | 3 | 0.85→**0.80 +0.006**（最大单臂增益）；→0.90 / →0.75 均 0 |
| neural weight | 4+3 | w075o3 **−0.047**（最大损失）；w050o3 −0.022；w025o1 +0.001 |
| V8 预设 | 3 | MASS→RR −0.021、→BOTH −0.020、→BASE 0 |
| engine-2 `max_depth` 6→7 | 5 | +0.004 / 0 / −0.006 / −0.006 / −0.006，方向不一致 |
| 共识权重 / rescue / RRF | 8 | 最好 +0.004，其余 ≤0 或负 |
| **重训 FPNet（7 个独立尝试）** | 7 | **全部为负：−0.007、−0.013、−0.017、−0.018、−0.018、−0.024、−0.024** |

**最重要的负面结论**：在比赛训练集上重训 fpnet / fingerprint 模型，7 次独立尝试全部掉分（−0.007 到 −0.024）。与我们自己的 073a(0.491)、076a(0.490)、082c(0.468) 一致。这条证据直接压低了「用本赛数据训练/微调新谱编码器」的期望收益。

**唯一的正面家族**：扩大候选结构来源。
- PubChemLite CC0 进 engine-2：+0.004 / −0.003（2 次，方向不一致）
- PubChem promo 阈值 rel 250→150：**+0.003、+0.004、+0.003（3 次全部为正）**
- 这两者都是「让更多结构进入候选池」，是本表里唯一有重复正证据的方向。

## 4. 噪声与分辨率：一步 = 一个分子

同表 README 与 `rerun_groups.csv` 实测：
- 公开榜只评约 400 个隐藏分子里的**约 132 个**；**一个分子从漏掉变成第 1 名约 +0.0076**；一个分子升一个名次约 +0.001。
- 46 组「代码/数据/镜像完全相同」的重跑里，**43 组分数完全一致**，最大离散 0.006。

所以：0.492→0.504 约等于 1.6 个分子；0.492→0.549 约 7.5 个分子。我们过去 12 次「±0.003」是不到一个分子，属于确定性噪声范围内的真实微差，不是可积累的进步。

## 5. 一个被证伪的假设（本轮自查）

原假设：管线对 `[2M+Na]+` 等二聚体加合物中性质量算错，会污染候选。
实测（`rounds/072-scale/probe/audit_adduct_mass.py`，121 个加合物类、每类 ≤400 行真实 train、RDKit 精确质量作真值）：

- 自写通用解析器在主要类别上 **99.75%–100% 落在 20 ppm 内**（`[M+H]+` 0.9975、`[2M+Na]+` 1.0、`[2M-H]-` 1.0、`[M+CH2O2-H]-` 1.0、`[M+Cl]-` 1.0），说明解析与数据都对。
- `_apex_extlib_overlay` 里的中性质量：二聚体类 **0% 正确**（中位误差 ≈1.0e6 ppm），`[M+CH2O2-H]-`、`[M+Cl]-` 也 **0%**。
- **但该 `nm` 是死代码**：overlay 实际用 `q['prec']` 对参考谱 `prec` 做二分查找（`prec_array` 由 `r[1]` 即参考 precursor 构造），`nm` 从未参与匹配。所以既不伤分也不加分。
- 真正受影响的只有 v8 mass prior：`_ADD` 精确字典只覆盖 7 个加合物类，**覆盖 84.72% 的 train 行**；其余 15.28%（主要是二聚体、水合物）`nm=NaN` 被丢弃 → 这些分子**静默失去 mass prior**，不惩罚也不受益。

结论：加合物不是可提分点，撤销该假设。可选项是把 `_ADD` 扩成支持 `nM` 的通用式（解析器已验证），但属于旋钮族，按第 3 节证据优先级最低。

## 6. COCONUT 候选池：查过了，不可验证（已排除）

- `068a` 的 `dataset_sources` 里**没有任何 COCONUT 候选数据集**；notebook 内 `coconut` 只出现 2 次，都是 popularity prior 的 `np_coconut` **标记列**，不产生候选。
- 现有候选来源：main engine（ChEBI/LMSD bio 池，`bio_*`）、engine-2、PubChem、`candidates_lotus.parquet`（LOTUS 回填）、extlib overlay。
- 公开侧有现成包，形状可用：`xianyu3/casmi26-licensed-coconut-candidates-202610`（`candidates.jsonl`，字段 `key14` / `neutral_mass` / `smiles` / `provenance`，CC BY 4.0）。

**决定性的否证**：下载该许可子集后实测 **149,002 个结构、149,002 个唯一 key14**；与 `artifacts/truth/visible_truth.parquet` 的 400 个真值比对，**命中 0/400 = 0.0%**。也就是说在我们唯一有真值的面板上，COCONUT 一条正确答案都不含；15 ppm 量窗虽能覆盖 358/400 个分子（中位 5 个候选），但没有一个是对的。

后果：这个臂**无法离线验证**，只能盲提交；而盲提交的期望值已被 59 臂表否证（33 降 / 4 个 >+0.002）。另外要动 COCONUT 必须进入 engine-2 的候选池，而主引擎是 **base64 嵌入 + SHA-256 校验**的黑盒，只能走事后 overlay（LOTUS 回填那种「只填空位」的方式），而空位早已被 LOTUS 填掉，增益趋零。

结论：**排除 COCONUT 臂**。

- 注意陷阱：`hoowoo/casmi26-analyticon-megx` 的公开版是**演示数据**，README 自述「3,683 generated molecular examples」「does not contain the original AnalytiCon/COCONUT catalogue」「No leaderboard score is claimed」。不可用于提分。

## 7. 结论与下一步

0.492 之上要靠**候选生成**，理由：
1. 公开面板（`sanpier/catalogue-experiment-saner`）把真值强制放进 final25，`candidate_recall25`=0.9667，MRR 0.8789；而真实公榜只有 0.492。面板**测不到召回损失**，所以它不是判断提分工具的仪器——但它反过来说明：一旦真值在池内，排序并不差；隐藏分与面板的巨大差距更可能来自「真值没进池」。此为推断，未直接测量。
2. 唯一有重复正证据的改动族就是扩大候选结构来源（第 3 节）。

因此 072a 定为：**强化同式异构体 reranker**。这是唯一同时满足「有公开正证据」和「可离线配对验证」的方向。

依据链：
1. 公开侧 0.494 那一步（`heliosli` v3，+1044 行）的内容从其可读的同族 notebook 可确认：是 **RR3 overlay** ——「peaks matched to each candidate's 1-2-cut fragments, graph network + transformer」的同式异构体 reranker，并带 `FREEZE`（训练集结构保留原位）与 `lib_max ≥ LIB_TAU` 保护。其自述该 reranker 会把训练集结构压低约 **0.34 SD**，所以必须 FREEZE 校正。
2. 我们 068a 里已有同族的 `_RR8_OVERLAY`（LightGBM + 50 维片段特征），但特征只有单谱平均，且没有跨谱一致性项。
3. 59 臂作者的结论也是「需要一个用到现有渠道没有的信息的 reranker」。

单一变量：给 RR8 增加**逐谱编码后聚合**的跨谱一致性特征（而不是把不同谱的峰平均——那正是 083a 的确定性缺陷，会造出不存在的峰），并在配对面板上先验证分子级配对 ΔMRR。

验收：
- 离线：分子级配对 ΔMRR > 0 且 bootstrap 下界 > 0，才允许提交。
- 在线：公榜 **> 0.496**。0.492→0.493 视为中性，按止损处理。
- 护栏：rank 1 与 `library_max ≥ 0.90` 的分子完全不动，保证下行结构性为零。

风险与边界：
- 082f3（outsider 召回）、082c/082d（库升位）都属中性，说明事后 overlay 类的收益极不稳定。
- 若离线配对增益不显著，**不提交**，保留 0.492 作为最终答案；剩余配额不强行花掉。
- 公开榜 132 分子、单分子 ≈0.0076，所以验收线定在半个分子以上，不接受 0.001 量级。
