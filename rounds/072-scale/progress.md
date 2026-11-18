# 072 进展

## 2026-11-17 晚 起点

配额与提交实测：

| 项 | 实测值 |
| --- | --- |
| ccjjllt GPU | 33.76h/30h，剩 **0.00h** |
| jiyujia GPU | 18.59h/30h，剩 **11.41h** |
| qingyu1232 / zhuating | 0.28h / 1.69h |
| 刷新 | 2026-11-21T00:00Z |
| 今日提交 | 5/5 已用，08:00 北京重置 |

提交历史实测（`kaggle competitions submissions --csv`）：

| ref | 内容 | 状态/分数 |
| --- | --- | --- |
| 58465327 | 083a ICE×DreaMS 对齐门 | **0.488（−0.004，负面）** |
| 58462795 | 082g 合并版 | **0.492**（11-18 才出分，中性） |
| 58460262 | 082f3 ICE outsider recall | 0.492 |
| 58457692 | 082d e180 库升位 | 0.492 |
| 58450336 | 082c Enveda-180 库 | 0.468 |
| 58415444 | 076a DreaMS×Morgan 对齐 | 0.490 |
| 58413244 | 072a DreaMS kNN rerank | 0.492 |
| 58412029 | **068a 本体（当前最佳）** | **0.492** |

**新发现（操作层）**：code competition 的提交要靠 GPU 重跑评分。082g/083a 提交后长时间 `PENDING`，而 ccjjllt 的 GPU 配额为 0。11-18 复查：**082g 最终出分 0.492**（提交→出分约 13 小时），说明评分链路没有坏、只是极慢；083a 仍 `PENDING`。因此 jiyujia 剩下的 11.41h 同时是「实验预算」和「提交预算」，且一次提交的反馈周期以小时计，不能按「3 次跑 + 无限提交」估算。

## 已完成的核查

### 1. infra kernel 报错根因（已定位并修复）

`jiyujia/enveda-072-infra` v1 是 `KernelWorkerStatus.ERROR`。日志实际内容是：
`FileNotFoundError: /kaggle/input/enveda-CASMI26-molecule-id-mass-spectra/test.parquet`。
比赛数据挂在 `/kaggle/input/competitions/` 下，v1 用错了路径。068a 的 `COMP` 解析（notebook 第 299 行附近）本来就是对的。
v2 改为与 068a 相同的解析方式后重新推送：`Kernel version 2 successfully pushed`。该 kernel 不启用 GPU，不消耗配额。

顺带确认：默认镜像**没有 rdkit**（`PKG rdkit MISSING`），torch 是 CPU 版；068a 靠挂 `metric/rdkit-2026-3-3-wheel` 数据集解决。

### 2. 可见 400 分子真值面板（已完成，之前轮次）

`artifacts/truth/visible_truth.parquet`：400/400 唯一、1213/1213 谱精确匹配、400 个不同 InChIKey、RDKit 往返 400/400、precursor ppm ≤5.97。峰值内存 1.46 GB。

**但该面板不能当排序仪器**：可见 test 是 train 的 1:1 拷贝，因此每个分子的 `library_max`=1.0，管线必然拿第 1 名。用它跑 2.7h GPU 只能测「确定性缺陷」，测不到排序改进。据此**撤销**了原本计划的「072a 可见 400 全流程 + 证据导出」——信息价值太低，不值 2.7h。

### 3. 公开前沿与 59 臂证据

见 `research/public-frontier-1117.md`。核心三条：
- 公开代码上限 **0.494**（`heliosli` v3，+1044 行整段新阶段），且该 notebook 已取不到（两个 token 均 403）。我们自己 0.492 就在公开天花板上。
- 59 个单旋钮臂：只有 4 个超过 +0.002，最大 +0.006；**7 次重训 FPNet 全部掉分（−0.007 ~ −0.024）**。
- 唯一有重复正证据的方向是**扩大候选结构来源**（PubChemLite +0.004；promo 阈值 rel150 +0.003×3）。
- 分辨率：公榜约 132 分子，单分子 ≈0.0076，同代码重跑 43/46 完全同分。过去 12 次「±0.003」不到一个分子。

### 4. 加合物假设（已证伪，撤销）

`probe/audit_adduct_mass.py` 对 121 个加合物类、每类 ≤400 行真实 train 做了中性质量审计：
- 自写通用解析器在主要类别 99.75%–100% 落在 20 ppm 内 → 数据与解析都对。
- `_apex_extlib_overlay` 的中性质量对二聚体/甲酸/氯加合物 0% 正确，但该 `nm` 是**死代码**：overlay 用 `q['prec']` 对参考谱 `prec` 二分匹配，`nm` 从未参与。既不伤分也不加分。
- 真正受影响的是 v8 mass prior：`_ADD` 只覆盖 7 类，**覆盖 84.72% 的 train 行**；其余 15.28% 静默失去 mass prior（不惩罚也不受益）。

产物：`artifacts/adduct/adduct_neutral_mass_audit.csv`、`artifacts/adduct/summary.json`。

### 5. 离线面板复算（与作者结果一致）

`sanpier/catalogue-experiment-saner` 的 `labelled_candidate_scores.parquet`（120 分子 × 25 候选，含 `label` 与各引擎 rank）：
- 我按 rank 复算 MRR，与作者 `ranking_summary.csv` **完全一致**（baseline 0.8789 / main_only 0.9417 / combiner_all 0.9233 / pubchem_only 0.2074）。
- 该面板 `library_max` 在真值行上均值 **0.617**（不是 1.0），是真·留出面板。
- 但其 `report.json` 写明 `candidate_policy: same final25 for every ranking`（真值被强制留在池内，recall 0.9667），并且自述 `30-query test is exploratory; do not choose a submission from its winner`。**它测不到召回损失**，故不能用来判断「加候选池」是否有效。

## 判断

0.492 不是上限，但**旋钮不是路径**——这与 59 臂作者的独立结论一致（"a knob will not get there; a new signal will have to (a better candidate generator, or a reranker that uses something the current channels do not)"）。且「用本赛数据重训」已被 7 次独立尝试证伪。

面板 recall 被强制为 0.97、MRR 0.879，而真实公榜 0.492，说明差距主要不在「池内排序」，更可能在「真值没进池」（推论，未直接测量）。唯一有重复正证据的改动族也是扩候选来源。

## 072a 已执行：跨谱一致性特征（离线结论）

代码：`probe/rr8_cross_spectrum_experiment.py`（建特征）、`probe/eval_rr8_cross_spectrum.py`（训练+配对评估）、`probe/eval_shipped_rr8.py`（评测已发布模型）。`probe/rr8mod/` 是从 068a 抽出的 `frag_rescore.py` 与 `rr8_features.py`（原文未改）。

改动只有一处：`frag_rescore.score_molecule(spectra, ...)` 内部对多张谱取**均值**；改为逐谱调用后聚合出 4 个跨谱一致性特征（`m1_min`、`m1_std`、`m1_mean`、`m1_win`＝该候选在多少比例的谱上不劣于榜首）。**不对原始 m/z 峰做任何平均**（083a 的缺陷就是不存在的峰）。

### 关键发现 1：该面板无法检验已发布 RR8

120 个分子的真值结构 **119/119 全都在 train.parquet 里**，因此 RR8 的 `FREEZE`（训练集结构保留原位）把所有移动全部锁死，ΔMRR **恒等于 0**。这不是特征无效，是仪器与护栏不相容。隐藏测试分子是未见结构、不受 FREEZE 限制，所以真正有信息量的条件是 **freeze off**。

### 关键发现 2：已发布 RR8 在隐藏条件下是中性

载入 068a 里 `_RR8_MODEL_Z` 原模型（LightGBM，102 特征，200 棵树），按原 `model_features` 重建输入，120 个分子：

| 条件 | ΔMRR | up/down | bootstrap 95% | 被 `lib_max≥0.90` 跳过 |
| --- | --- | --- | --- | --- |
| freeze on（面板原状） | +0.0000 | 0/0 | [0, 0] | 18/120 |
| **freeze off（隐藏条件）** | **−0.0071** | 2/5 | **[−0.0292, +0.0139]** | 18/120 |

即 RR8 在这批留出分子上既不加分也不显著掉分，与我们 068a 的 0.492 稳定复现一致。

### 关键发现 3：新特征确实带来信息，但不足以翻正

训练 60 个 train 分子（1500 行），评估 60 个 calib+test 分子：

| arm | MRR(base→arm) | ΔMRR | bootstrap 95% | AUC（候选级） |
| --- | --- | --- | --- | --- |
| A 已发布均值特征 | 0.9228→0.7705 | −0.1523 | [−0.2204, −0.0875] | 0.828 |
| A2 重算均值（对照） | 0.9228→0.7705 | −0.1523 | [−0.2213, −0.0856] | 0.828 |
| **B +跨谱一致性** | 0.9228→0.8743 | **−0.0485** | [−0.0908, −0.0147] | **0.919** |
| **B − A 配对** | — | **+0.1039** | **[+0.0321, +0.1779]** p>0=0.998 | — |

5 折按分子分组 CV（120 分子）复核：A −0.1103、B −0.0278、**B−A +0.0825 [+0.0460, +0.1208] p>0≈1.000**，up/down 25/5。A2−A 恒为 0，说明重算无噪声。

### 结论与是否提交

- **特征层验证通过**：候选级 AUC 0.83→0.92，配对 ΔMRR 在两个独立划分下 bootstrap 下界均 >0。指令里的前置条件（配对 ΔMRR>0 且下界>0）**满足**。
- **但重训本身不交付**：两个 arm 都**劣于不重排**（B 仍 −0.028~−0.049）。原因是只用 60 个分子训练严重欠拟合；随后用 3,657 条合成三元组重训也仍未胜出（见下节）。**不把重训模型推上去。**
- **路线由此定位到真正的缺陷**：掉分主要来自 RR8 可以换掉榜首。加 leader-safe 护栏后已发布模型转正（+0.0042），这才是 072b 交付的东西。
- veto 方案（跨谱不一致就否决 RR8 的移动）已试：阈值 0.4/0.5/0.6/0.75/1.0 最好只到 −0.0029，CI 仍含 0，无实用价值，放弃。

### 关键发现 4：RR8 少了 leader-safe 护栏（这是可交付的改动）

`_RR8_OVERLAY.rerank_list` 的可行集**包含 position 0**，也就是 RR8 可以替换榜首。而公版 `_PUBLIC_RERANKER.propose` 第一句就是 `if original[0] != order[0]: action='protected_previous_top1'; return order`，比赛里的 alexchilton 0.472 leader-safe、romanrozen consensus 的 `w_top/w_mid` 也都是同一约定。**只有 RR8 漏了。**

在 120 个留出分子上用已发布 booster 实测（`probe/eval_leader_safe.py`）：

| 变体 | ΔMRR | up/down | bootstrap 95% | p(>0) |
| --- | --- | --- | --- | --- |
| 已发布 RR8，原样 | −0.0013 | 4/4 | [−0.0235, +0.0196] | 0.457 |
| **已发布 RR8 + 榜首保护** | **+0.0042** | 3/1 | **[−0.0001, +0.0099]** | **0.953** |
| 重训 A（3657 三元组），原样 | −0.0504 | 5/17 | [−0.0864, −0.0160] | 0.001 |
| 重训 A + 榜首保护 | +0.0000 | 2/3 | [−0.0060, +0.0060] | 0.514 |
| 重训 B（+跨谱），原样 | −0.0699 | 3/20 | [−0.1066, −0.0371] | 0.000 |
| 重训 B + 榜首保护 | −0.0032 | 2/5 | [−0.0114, +0.0041] | 0.195 |

结论：**掉分主要来自「换掉榜首」，不是来自特征**。加一行护栏后，已发布模型从 −0.0013 变为 +0.0042，方向为正、下行结构性受限。

### 072b 已推送

`rounds/072-scale/072b/`：068a 原样 + `rerank_list` 可行集排除 position 0，其余常量（`V8='MASS'`、`RR8_ALPHA=1.0`、`RR8_LIB_TAU=0.9`、`COMMIT_MODE='smoke'`）、数据集、镜像全同。

护栏自检（本机，300 次随机列表）：原实现 **32%** 的列表会改动 position 0，本版 **0%**，且候选多重集每次都完全保持。
`Kernel version 1 successfully pushed`。

### 规模化重训（已做，未胜出；前置条件已核实）

已用 3,657 条同式三元组重训两臂（`probe/build_rr8_train_set.py`、`probe/train_rr8_scaled.py`）：
A 臂 −0.0571、B 臂 −0.0647（均劣于未重排），加榜首保护后 A +0.0000、B −0.0032。
即**合成同式 decoy 训练出来的模型打不过已发布 booster**，重训路线在现有数据构造下不成立。

可行性与成本（供后续判断）：
- 结构目录 `panel/identity_map.json` = **291,476** 条 SMILES→inchikey14；**51,291** 个分子式，其中 **2,148** 个有 ≥25 个同式结构。
- 覆盖：400 个 train 派生结构中 **248/400 = 62.0%** 的式子能凑出 ≥25 个同式候选。
- 特征化实测 **0.18 s/分子**；本机可完成，不需要 GPU。

## 未决 / 风险

- 082g、083a 长期 `PENDING` 且无 GPU 配额，实际已死；不要把它们当「待出分」。
- 取不到 `heliosli` 0.494 的代码（两 token 均 403），只能从其同族可读 notebook 反推做法。
- 剩余 11.41h 同时是提交预算，且提交后需要 GPU 重跑才出分，不能按「3 次跑 + 无限提交」估算。
- 若 072a 离线为负，本轮结论就是「0.492 是当前可复现上限」，剩余配额不强行花掉。

## 072b 运行状态（11-18 00:30 北京时间）

- `jiyujia/enveda-072b-leadersafe-rr8` v1 已推送，状态持续 `KernelWorkerStatus.QUEUED` 约 45 分钟，GPU 配额仍为 11.41h（排队不消耗）。
- 同时段证据：082g 的评分重跑在约 13 小时后才完成并给出 0.492，说明**评分链路正常但极慢**；083a 仍在 PENDING。072b 的 GPU 会话排队与提交评分排队叠加，反馈周期可能以 10 小时计。
- 待办：排队转 RUNNING 后，确认 commit run 产出合法 `submission.csv`（可见 12 分子全部 `library_max=1.0`，RR8 对它们整体跳过，所以 smoke 输出应与 068a 完全一致——差异只会在隐藏重跑里出现），然后用 `create_code_competition_submission` 提交，验收线 >0.496。

## 072b 后续核查：N-1c 是否也缺 leader-safe（无法离线判定）

`_n1c` 同样对整表重排（`argsort(-(z(base)+LAM*z(N)))`，并自带 `top1 changed` 计数），与 RR8 是同一缺陷类。但用 120 分子面板无法判定：

```
molecules with spectra 115, with neighbours 4, edges 2
[n1c] protect_top1=False delta=+0.0000 top1_changed=0
[n1c] protect_top1=True  delta=+0.0000 top1_changed=0
```

N-1c 靠测试集内部的分子网络传播，120 个分子只连出 **2 条边**，网络几乎为空，两臂都不动任何列表。真实测试有约 400 个分子、邻域密度完全不同。**结论：N-1c 的 leader-safe 改动无法用现有面板验证，不在本轮提交。**

## 072b 逐分子核对（提交前尽调）

护栏版 RR8 在 120 个留出分子上的逐分子增量，按真值原始名次分组：

| 真值原始名次 | 分子数 | ΔMRR 合计 | 说明 |
| --- | --- | --- | --- |
| 1 | 105 | **0.000** | 护栏保证完全不动（这也是本改动的下行边界） |
| 2 | 7 | 0.000 | 未触发 |
| 3 | 3 | **+0.333** | 3/3 全部 3→2，每例 +0.167 |
| ≥5 | 4 | +0.160 | 一例 7→3 (+0.190)，一例 14 名轻微 −0.016 |

合计 +0.508 RR 单位 / 120 分子 = **+0.00423**。改善集中在「真值第 3 名」这一档且 3/3 全中，唯一损失只有 −0.016，收益/损失高度不对称。这比单一分子偶然升位更可信。

## 072b v1 失败原因（已修，v2 已推送 18:12 UTC）

v1 在 21 分钟后 `KernelWorkerStatus.ERROR`。日志并非环境问题，而是**源码哈希锁**：

```
6 assert hashlib.sha256(_RR8_FEATURES.encode()).hexdigest() == 'aec9012e...'
7 assert hashlib.sha256(_RR8_OVERLAY.encode()).hexdigest() == 'e83b7986c46d0ef02bc6cfde17e891c9e7ee66a0eed8c716f5e3474e5663ccf7'
AssertionError
```

**更正**：我先前说「`_RR8_OVERLAY` 没有哈希锁、只有模型文本有」是错的。`_RR8_OVERLAY` 确有一把 SHA-256 锁，改动后必须同步更新期望值。

修法：把期望值改成补丁后文本的 SHA-256 `3337fc31cff40f04…`，并复核四个字符串锁全部一致：

```
_physical      OK  eeacdcf0afb173a8
_post          OK  cd1594b3bbfa75ce
_RR8_FEATURES  OK  aec9012e72324095
_RR8_OVERLAY   OK  3337fc31cff40f04
all four string pins verified
```

v1 未产生 `submission.csv`，因此**没有消耗任何提交**（只花了约 0.24h GPU）。
v2 已推送：`Kernel version 2 successfully pushed`。

## 结论：072b 是空操作，不提交（关键更正）

v2 `COMPLETE`，400 行提交文件合法（`id order matches sample_submission: True`、`list length 1..25`、无空无重复）。但日志里的阶段报告是决定性的：

```
SUBMISSION READY: ... | V8 MASS | mass lead {"status": "ran", "leader_moves": 0} | RR {"status": "off in this preset"}
```

代码门槛（068a 第 198–199 行）：

```python
MASS_ON = V8 in ('MASS', 'BOTH')
RR8     = V8 in ('RR',  'BOTH')
```

而 `V8 = 'MASS'`（第 196 行）⇒ **RR8 = False，整个 RR8 阶段根本不执行**。我改的 `_RR8_OVERLAY.rerank_list` 在 0.492 的配置里是死代码。

**因此 072b 的隐藏重跑输出必然与 068a 逐字节相同，提交只会把 0.492 再评一次。不提交。**

这也与公开 59 臂表一致：`V8: MASS→RR = −0.021`、`MASS→BOTH = −0.020`——**打开 RR 是掉分的**，所以 068a 关掉它是正确的。

更正本轮此前的两处错误认识：
1. 我曾说 `_RR8_OVERLAY` 没有哈希锁——错，它有（v1 就是被这把锁拦下的）。
2. 更重要：我把 072a/072b 的目标定在 RR8 上，但 RR8 在 0.492 配方里**是关闭的**。整个 reranker 方向对本配置无效。

本轮 GPU 花费：v1 约 0.24h（被哈希锁拦下，无提交）+ v2 约 0.53h = **约 0.77h**，剩 **10.64h**。未消耗任何提交。

### 0.492 配方里真正在跑的阶段（据 v2 日志）

| 阶段 | 状态 | 日志证据 |
| --- | --- | --- |
| mass prior (V8 MASS) | on | `molecules 400, candidates 688, penalised 447, lists_touched 394` |
| mass-lead 规则 | ran | `leader_moves 0` |
| N-1c 网络重排 | on | `LAM=0.35 TAU=0.6 ... with neighbors 114, edges 109, changed lists 2, top1 changed 0, 81s` |
| spectral parallax / merge | applied | `merged 400, leader_moves 0, ranks_reordered 12` |
| extlib overlay | on | — |
| **RR8** | **off** | `RR {"status": "off in this preset"}` |

即：真正在动的只有 mass prior（触碰 394 个列表）与 N-1c（改 2 个列表）。这两个的旋钮都已在公开 59 臂表里被逐一扫过并停在最优（LAM 0.35 = +0.004 为峰值，0.50 转负；extlib TAU 0.80 = +0.006 为峰值；V8 改预设全为负）。

## 072c：mass prior 重标定（Kaggle CPU kernel，已完成）——无余量

`jiyujia/enveda-072c-massgrid`，`KernelWorkerStatus.COMPLETE`，CPU-only 不耗 GPU。500 组网格（CENTER×FREE×FULL×W）：

```
molecules 120 | where the prior can act 10
MRR without prior 0.9128 | with shipped params 0.9144 | delta +0.00159
grid size 500 | best global delta +0.00159
shipped vs best: +0.00159 vs +0.00159 -> improvement available +0.00000
mass-error AUC 0.6449 | truth median 1.36 ppm | decoy median 1.44 ppm
```

出厂参数（C=-1.5, F=3, FULL=8, W=0.15）**已在网格最优点**，500 组无一组更好。可提升量 **+0.00000**。

原因：能动的分子只有 **10/120**（真值名次≥2、`library_max<0.9`、ppm 可算）。prior 确带信息（AUC 0.645；真值 3 ppm 内 97.4% vs 诱饵 77.3%），但真值/诱饵的 `|ppm-CENTER|` 中位数只差 0.08 ppm，鉴别力太弱。

详细：`research/072c-mass-prior-recalibration.md`。

**决定**：旋钮路线关闭，不推 GPU、不提交（参数未变，提交只会再评一次 0.492）。**0.492 是本配方族的可复现上限。**

## 本轮总账

| 项 | 值 |
| --- | --- |
| 起始 / 结束 GPU | 11.41h → **10.64h**（用掉 0.77h：072b v1 0.24h + v2 0.53h） |
| 提交消耗 | **0 次**（今日额度完整） |
| 公榜分数 | **0.492**（068a，ref 58412029）不变 |
| 新增 Kaggle 运行 | infra v2（COMPLETE）、072b v2（COMPLETE，经证实为空操作）、072c massgrid（COMPLETE） |
| 结论 | 0.492 = 本配方族上限；继续提分需新模型，不是旋钮 |

## 083a 出分：0.488（−0.004）

`58465327`（ICE×DreaMS 对齐门，作者声称「精度 1.00 @ 11% 覆盖、零下行」）最终 **0.488**，相对 068a 的 0.492 **掉 0.004**。

这否证了「零下行」的说法：该门在实际隐藏集上不是零成本，而是净负。与 082c（0.468）、082d（0.492）、082f3（0.492）、082g（0.492）合并看，整个 082/083 事后升位族 5 次全部 ≤0.492，无一提升。

## 072d：留出候选生成仪器跑通（召回 100%，召回假设被否证）

`jiyujia/enveda-072d-instrument`，T4，`COMPLETE`，2.17h GPU。150 个 train 分子（681 张谱）的参考谱从训练库整分子移除（库从 275,810 个结构降到 275,660，差正好 150），用 0.492 管线重建 top-25。

```
n=150  recall@25 = 1.000 (150/150)  MRR@25 = 0.8286  top1 = 0.680
rank hist: 1 -> 102, 2 -> 42, 3 -> 2, 4/5/9/18 各 1
仅「榜首已错」的 48 个分子: MRR = 0.4642
```

- **候选召回不是瓶颈**：没有库命中时真值仍 100% 进池 ⇒「加新候选源」方向关闭（无空间可回收）。
- **损失集中在榜首**：榜首错时 MRR 仅 **0.4642**，与真实公榜 **0.492** 几乎相同；推断隐藏集榜首正确率远低于此处的 68%。
- 42 个分子停在名次 2（21 个 RR 单位），是唯一还有空间的位置，但需要重排在**未见化学型**上有效——RR8 做不到且本配置不启用。
- 边界：这是「留出谱」，不是「留出化学型」；类似物谱和结构目录仍在，所以 0.829 无法代表隐藏集。

详细：`research/072d-heldout-instrument.md`。内核 POST 单元的打分有 bug（SMILES 未转 InChIKey 就比较），已用下载的 submission.csv 重算并保存 `instrument_report_corrected.json`。

### 本轮总账（更新）

| 项 | 值 |
| --- | --- |
| GPU | 11.41h → **8.47h**（用掉 2.94h：072b 0.77h + 072d 2.17h） |
| 提交消耗 | **0 次** |
| 公榜最佳 | **0.492**（068a，ref 58412029）不变 |
| 已关闭方向 | 旋钮（59 臂 ≤+0.006）、mass prior 重标定（+0.00000）、RR8/reranker（本配置不启用）、**加候选源（召回已 100%）** |
