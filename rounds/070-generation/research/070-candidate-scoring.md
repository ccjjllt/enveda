# 070 候选级强打分：对整列 25 候选逐个打分并重排

日期：2026-11-17（北京）。状态：**只设计 + 只读核对，不写实现代码、不推送、不提交、不跑大任务**。
实现等 082c 出分后再做（082c ref 58450336，2026-11-17 08:48 已提交，见 `rounds/069-new-signals/progress.md` §2026-11-17）。

## 1. 为什么做这个方向

072a（DreaMS kNN，0.492 打平）、073a（timsTOF ICEBERG 中名次，0.491 −0.001）、076a（DreaMS×Morgan 对齐，0.490 −0.002）
三连证伪了一个共同模式：**leader 冻结 + 2–25 名小权重（w=0.3）z 叠加**，在可见 test 上只改动 11–12/400 个列表，
在隐藏集净效应为零或负。底座 068a（065a + N-1c `LAM=0.35`，public 0.492）的融合已经覆盖了这些证据，
中名次小权重没有增量信息。

alexchilton 59 臂分析的结论一致：单旋钮到顶（最大 +0.006），缺的是**更好的候选生成或使用当前通道没有的信息的 reranker**。
069 管线解剖（`rounds/069-new-signals/research/069-pipeline-map.md` §4）进一步指出：在现有 25 名集合内排序已接近饱和，
但 072/073/076 的失败**只证伪了"小权重叠加"这种用法**，没有证伪这些打分函数本身——它们从未被允许决定 rank-1，
也从未以"逐候选绝对分数"而非"与原顺序的 z 融合"方式使用过。

本轮假设：**同一个打分函数，全列强打分（允许换帅、有门控）≠ 中名次小权重叠加**。
证伪标准同样硬：任一实验公榜 ≤0.492 即放弃该实验，不调权重、不扩 topN，直接转向池外召回/生成。

## 2. 可接入点（只读核对，均已核实）

底座入口：`rounds/068-n1c-lam035/068a/enveda-068a-n1c-lam035.ipynb`（32 cells，`COMMIT_MODE='smoke'`，
`CUSTOM = dict(w_top=0.25, w_mid=0.75, rescue=0)`，cell 5）。

| 接入点 | 位置 | 现状行为 | 强打分改造位 |
| --- | --- | --- | --- |
| 神经打分 | cell 12 `MODULES['neural_scores.py']`，`score(bundle, checkpoint, evidence, queries, output)` | 对 `frozen_candidate_evidence.parquet` 的每行候选、`effective_query_spectra.parquet` 的**全部**观测谱打分；输出 `neural_cos / neural_fp / neural_score / neural_std / n_spectra / n_adducts / neural_ok`，其中 `neural_score = mean_spectra(cos/temperature + fp_score_weight·fp_cos)`；cell 15 再做 public/v80 两路 `0.5·z + 0.5·z` 融合；`consensus.leader_safe_order` 只做 RRF 重排 | overlay 直接读 `neural_candidate_features.parquet`，按 `neural_score` 绝对值重排**全 25 名**（含 rank-1），不再与原顺序做 RRF 融合 |
| N-1c | cell 24 `_n1c(work, inp, LAM=0.35, TAU=0.6)` | 只用 `[M+H]+ / [M-H]-` 谱，modified cosine（`n≥6, s≥TAU`），候选得分 `z(-log rank) + LAM·z(Σ w⁴·Σtop5 p·Tanimoto)`；跑在 `submission.csv` 写出之后；异常则原样返回 | 强打分 overlay 统一放在 `_n1c()` **之后**、提交校验之前，保持"N-1c 先、强打分后"的固定顺序 |
| ICE | 073a `ice_rerank.py` 子进程，`ICE_TOPN=20, ICE_WEIGHTS=0.3, CKPT=tims_ft3+tims_ft2`，leader 保持（`tims_iceberg_audit.json: leader_changed 0, lists re-ordered 12/400, 92.5s`） | 只重排 top-20、w0.3 叠加、leader 冻结 | 同一 `ice_rerank.py` 入口，改 `TOPN=25`、取 ICE 原始预测分做全列排序（见实验 B） |
| 对齐头 | 076a `align_head.pt`（7.35 MB）+ `align_config.json`，DreaMS 1024→512→256 与 Morgan 2048→512→256 同式硬负例 InfoNCE；未见结构同式组 R@1 0.316 / median rank 2（`rounds/069-new-signals/progress.md` §076a 泛化验证） | w0.3 叠加重排 2–25 名，leader 冻结，可见 12/400 重排 | 同一 embedding + head，对 25 候选逐个算对齐分，做全列排序（见实验 B） |
| 失败语义 | cell 24 尾部 `try: _n1c() except: ... '[n1c] FAILED -> submission unchanged'`；083a 同理"scorer 失败则原样返回" | 已确立"overlay 失败恢复上一版"的写法范式 | 两个实验共用同一范式（§5） |

关键约束（来自 069-pipeline-map §2.9/§4.1）：068a 的 25 名候选集合被 frozen 0.465 冻结（`rescue=0`），
本轮两个实验**都不引入新候选**，只对整列 25 个逐个打分并重排——这是与 082 系（池外召回）的明确分工：
082 负责"集合"，070 负责"集合内的最强排序"。

## 3. 实验 A：神经全列强打分（070a）

**一句话**：把管线自己的神经打分从"RRF 融合权重"升级为"逐候选绝对分数 + 门控换帅"。

### 3.1 输入

- `submission.csv`（N-1c 之后）：`molecule_id, smiles`（`;` 分隔 25 槽）。
- `neural_candidate_features.parquet`：cell 15 已产出的 public/v80 融合特征
  （`neural_score / neural_cos / neural_fp / neural_std / neural_ok / n_spectra / n_adducts`），与 25 槽逐行对齐。
- `frozen_candidate_evidence.parquet`：`library_max / baseline_rank / processed`（门控用）。
- 不新增任何数据集、不新增模型权重、不跑 Engine。

### 3.2 打分函数

对同一 `molecule_id` 的 25 个候选，按融合后的 `neural_score` 降序排列（`stable`，并列保持原顺序）。
`neural_score` 本身已是"全部观测谱平均的 `cos/temperature + fp_w·fp_cos`"，是一个跨谱、跨加合物的
候选级绝对分数——这正是 072/073/076 的 z 叠加没有用过的形态。备选键（仅当主实验中性后才考虑，
**不在本轮实现**）：`neural_cos` 单键、`neural_score - 0.5·neural_std`（不确定性惩罚）。

### 3.3 门控（leader 保护规则）

rank-1 换帅必须**同时**满足以下四条，否则该分子 leader 冻结、只重排 2–25 名：

1. `library_max < 0.90`（有强库支持的题不动；与 APEX 外谱 overlay 的 `LIB_TAU=0.90` 对齐）。
2. `neural_ok` 全真且 `n_spectra ≥ 2`（单谱题的神经分不可靠，不换帅）。
3. 边际：`neural_score[best] - neural_score[leader] ≥ 0.10`（分数是跨谱均值，
   0.10 约为可见 commit 上分子内 std 的 1σ 量级；取 1σ 而非 2σ 是为了让可见审计有足够的换帅样本可读，
   公榜若负则直接放弃、不调参）。
4. 一致性：新 leader 的 `neural_cos` 与 `neural_fp` **同时**高于原 leader
  （防止单一键驱动的错换；与 `parallax_physical.decision` 的双键门控同构）。

2–25 名无门控，直接按 `neural_score` 排序（这本身就是对 CUSTOM `w_mid=0.75` RRF 的替换，
预期可见改动数远大于 11–12/400——若可见改动仍只有 ~10 个列表，说明神经分与原顺序几乎共线，
实验在 commit 审计阶段即判死，**不提交**)。

### 3.4 审计 JSON（`neural_full25_audit.json`）

```json
{
  "molecules": 400,
  "reranked": 0,
  "top1_changed": 0,
  "top1_frozen_by_lib": 0,
  "top1_frozen_by_spectra": 0,
  "top1_frozen_by_margin": 0,
  "top1_frozen_by_consistency": 0,
  "margin_p50_changed": 0.0,
  "elapsed_s": 0.0,
  "fallback": false
}
```

另存 `neural_full25_moves.csv`（`molecule_id, old_top1, new_top1, margin, gate`），供 commit 人工核对前 5 个换帅样本。

### 3.5 失败即原样返回

```text
overlay 前 copy submission.csv -> submission_pre_full25.csv；
try: 读特征 -> 门控重排 -> 校验（400 行、每行 1–25 槽、molecule_id 唯一且与输入一致）-> 原子替换；
except 或校验失败: 恢复 submission_pre_full25.csv，audit 写 fallback=true，流程继续（后续 overlay 照常跑）。
```

### 3.6 GPU/时间预算

纯 pandas/numpy 排序，CPU 秒级（<60s，含审计写出）；不占 GPU；放在 N-1c 之后，
`TIME_GUARD_HOURS=7.5` 之内。若超时守卫跳过，`fallback=true`。

## 4. 实验 B：ICE 全 25 + 对齐头联合强打分（070b）

**一句话**：把两个"未见结构留出上有 rank-1 证据"的打分函数（ICE R@1 0.4395、对齐头 R@1 0.316）
从"中名次小权重"升级为"全列双证据联合排序"，leader 同样只在双证据一致时才换。

做 B 的理由（不是重复 073/076）：073a 只用了 ICE top-20 + w0.3 且冻结 leader，
而 ICE 在未见结构留出上的强项恰恰是 **R@1 0.4395**（`enveda-ice-val`，3,463 张谱/22,825 候选）；
076a 的对齐头在未见结构同式组 median rank 2。两个信号的 rank-1 能力从未被允许作用于 rank-1。

### 4.1 输入

- `submission.csv`（N-1c 之后）+ `effective_query_spectra.parquet`（query 全部观测谱）。
- ICE：`hjyact/casmi26-timstof-iceberg` 的 `ice_rerank.py`，`CKPT=tims_ft3+tims_ft2`（与 073a 同），
  改 `TOPN=25`，输出 25 候选原始预测分（cosine/softmax 前的 logit 或其 rank 分，取脚本原生输出列）。
- 对齐：`align_head.pt` + query 谱的 DreaMS 嵌入（076a 推理 kernel 已验证的链路）与 25 候选的 Morgan 指纹，
  经 head 得对齐分。
- 不新增数据集；权重文件均为已挂载/已验证。

### 4.2 打分函数

分子内 z-score 后等权相加：`S = 0.5·z(ice) + 0.5·z(align)`，按 S 降序排全 25 名（`stable`）。
等权是起点：两个证据正交（预测谱 vs 谱-指纹对齐），073/076 的失败都发生在"各自 w0.3 与原顺序融合"，
联合排序刻意**不混入原顺序**，让实验回答一个干净问题——"双新证据自己排全列，能不能赢过底座顺序"。
（若公榜中性，允许的一次后续才是学权重；本轮不做。）

### 4.3 门控（leader 保护规则）

rank-1 换帅必须**同时**满足：

1. `library_max < 0.90`（同实验 A）。
2. 双证据一致：ICE 与对齐各自的 top-1 是**同一个**候选（任一证据不同意即冻结 leader，只重排 2–25）。
3. 边际：联合分 `S[best] - S[leader] ≥ 0.5σ`（σ 为该分子内 S 的 std；0.5σ 比实验 A 更松，
   因为双证据一致本身已是强门控）。
4. 物理兜底：新 leader 必须通过 `frag_rescore.score_molecule` 的"解释度不低于原 leader"检查
   （cell 14 已有函数，纯 CPU；任一新 leader 解释度更低即冻结）。

### 4.4 审计 JSON（`ice_align_full25_audit.json`）

```json
{
  "molecules": 400,
  "ice_scored": 0,
  "align_scored": 0,
  "reranked": 0,
  "top1_changed": 0,
  "top1_frozen_by_lib": 0,
  "top1_frozen_by_disagree": 0,
  "top1_frozen_by_margin": 0,
  "top1_frozen_by_fragment": 0,
  "ckpt": "tims_ft3+tims_ft2",
  "elapsed_s": 0.0,
  "fallback": false
}
```

另存 `ice_align_full25_moves.csv`（`molecule_id, old_top1, new_top1, S_margin, ice_top1, align_top1`）。

### 4.5 失败即原样返回

与实验 A 同一范式（`submission_pre_icealign.csv` + 校验 + 恢复），外加两条：
ICE 子进程非零退出即整实验 fallback；任一证据缺失（某分子 ICE 0 预测/对齐 embedding 失败）
则该分子整列不动（不做部分重排），`ice_scored/align_scored` 如实记录。

### 4.6 GPU/时间预算

- ICE 全 25：073a 实测 top-20 628/628 候选 92.5s；全 25 约 +25% 候选，**预算 300s**（T4）。
- 对齐 head：MLP 前向，400×25，CPU/GPU 均秒级，**预算 300s**（含 DreaMS query 嵌入；若超则 fallback）。
- `frag_rescore` 只对换帅候选（预期 ≤ 几十个分子 × 2 SMILES）调用，**预算 600s**。
- 合计新增约 **20 分钟**，068a 整 kernel 约 2.7h 墙钟、9h 上限内余量充足（069-pipeline-map §5 预算口径）。

## 5. 两个实验的共用工程约束（实现时照此写，不在本轮写）

1. 顺序固定：`CUSTOM 提交 -> Variant M -> N-1c -> [070a | 070b] -> 提交校验`；070a 与 070b **互斥**
   （同一 kernel 只跑一个，避免双重换帅不可归因）。先做 070a（零新增依赖，失败面最小）。
2. 校验门：行数 400（commit smoke 下为 12）、每行 1–25 槽、`molecule_id` 唯一且与输入一致、
   `molecule_id` 顺序不变；任一失败即恢复预 overlay 文件。
3. 可见 commit 的判死线（二者任一成立则**不提交**，等 082c 出分后重议）：
   - 可见改动数与 072/073/076 同量级（~12/400）——说明"强打分"实际没强起来；
   - `top1_changed = 0` 且门控冻结数 >95%——说明门控过严，公榜方差为零、提交浪费额度。
4. 公榜判定：任一实验相对 068a（0.492）**>+0.001 即新底座**；**≤0.492 即放弃该实验**，
   不调阈值、不换权重、不扩 topN（073a/076a 的教训：可见 test 区分不出 0.3/0.5/0.6，调参是浪费）。
5. 配额纪律：实现只在 082c 出分（约 2026-11-17 晚）后启动；082c 若正向，070 让路（优先 082a/b 对照）；
   GPU 用 jiyujia 剩余额度（约 24h），单 kernel 预算 ≤3h。

## 6. 与 083a 的分工（不重复）

083a（已推送待审计）是"e180 候选 + 管线神经打分"：**向集合里加新人**（池外 3,719 结构按 ±10 ppm 组 evidence，
调 `neural_scores.py` 逐候选打分，再定升 rank-1 还是 rank-2）。070a 用的是**同一打分函数**，
但作用对象是已有的 25 名、不加新人。若 083a 正向而 070a 中性，说明神经分的价值在"召回"不在"重排"；
若反之，说明价值在"重排"——两者正交，可归因。这也是 070a 排在 070b 之前的原因：
它与 083a 共用打分函数，对照最干净。
