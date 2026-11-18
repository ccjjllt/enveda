# 075 进展

## 2026-11-18 13:35 UTC 启动

**在途**：074a（Dante 0.494 复现）ref `58505232`，10:21 UTC 提交，13:31 仍 `PENDING`、`totalBytes 0`（隐藏重跑尚未产出文件）。GPU 剩 **6.57h**，今日提交剩 3 次。

## 三条并行研究线（subagent 已派出）

| 线 | subagent | 产出文件 | 目标 |
| --- | --- | --- | --- |
| A | MS2 同分异构 SOTA | `research/ms2-isomer-sota.md` | 盘点 MIST / SIRIUS-CSI:FingerID / CFM-ID / MetFrag / MS2DeepScore / DreaMS / MSGNN / FT 对齐等，确认**公开权重**、许可证、离线可跑性、单 T4 成本 |
| B | 未使用信号审计 | `research/unused-signal-audit.md` | 冻结管线里「算过但没进排序」的逐候选数值，哪些能在同分异构体之间区分 |
| C | 公开资产与规则 | `research/public-assets.md` | 真实权重数据集 / models / 允许的外部谱库 / top 队伍有无公开物 |

## 本地准备的两条发现（轻量审计，无训练）

### 1. 本机 RDKit 的 Murcko 骨架不可用

```
MurckoScaffold.GetScaffoldForMol -> RuntimeError: bad any cast   (6/6 分子)
MurckoScaffold.MurckoScaffoldSmiles -> RuntimeError: bad any cast
rdkit 2026.09.2 / Python 3.12
```

⇒ 「按骨架整组留出」的化学型分组**不能在本机做**，必须在 Kaggle（RDKit 2026.03.3）上算。这也解释了为什么 075b 仪器必须走远端。

### 2. 训练库的结构多样性：类似物家族很稀疏

在 20,167 个「≥2 张谱」的结构里抽 5,000 个，用 Morgan2(2048) + leader 聚类（Tanimoto ≥0.65）：

```
clusters 4,921 / 5,000
size 1: 4,844 | 2-4: 77 | 5-19: 0 | >=20: 0 | 最大簇 3
```

⇒ 至少在这个阈值下，**结构库高度分散，类似物家族极少**。这对 072d 的解读是个补充：那次 MRR 0.8286 不太可能主要由「类似物谱泄漏」解释，更可能是查询谱本身在管线内被模型重建出了正确结构。真正的化学型留出仍需 075b 在远端验证。

产物：`artifacts/kout/075/chemotype_groups.json`（骨架统计，因 Murcko 失败为空）、`artifacts/kout/075/fp_leader_clusters.json`。

## 2026-11-18 13:40 UTC 三条调研回收 + 找到唯一公开的「同分异构判别器」

### 调研结论（三份文档已落盘）

- **A（SOTA）**：能同时满足「候选-谱打分 + 公开权重 + 断网 + 许可」的只有 MIST 指纹概率、ICEBERG/GLACIER 前向谱、CFM-ID 4。SIRIUS/CSI:FingerID（含 NIST）排除；MS2DeepScore/Spec2Vec/DreaMS 是谱-谱方法，候选侧没有参考谱，不能直接给候选打分。
- **C（公开资产）**：本赛公开权重全部以 dataset 发布（Kaggle Models 目录没有）。规则已放宽：CC-BY-NC 允许，**NIST 及其派生物一律禁止**，COCONUT/MassSpecGym/MIST/DreaMS 明确允许。top-30 队伍**没有任何公开物**。
- **B（未使用信号）**：管线里最大的浪费是 Engine 1 的逐候选 LightGBM 分数（算了随即丢弃）、ICE 的 47 分钟 GPU（主列表不用）、物理碎片分被压成 5 个阈值。

### 关键发现：`ahmedberatozer/casmi26-nx-model`

一个**专门做同分异构判别**的公开模型（469.7 MB，MIT 代码）：FPNet 谱编码器 + 分子塔，InfoNCE 对比学习，正样本是 ±10 ppm 同质量池里的**真实同分异构体**。附带 `nx_rerank.py`（只在同一分子式组内重排，候选集不变）与作者自己的 fold-safe 规则模拟：

```
H 集（fold-safe，never-seen molecules）   base 0.7851 -> sfz_w1 0.8072  (+0.0221) -> window+3nx 0.8174
H_T 集                                    base 0.8142 -> sfz_w1 0.8361  (+0.0219) -> window+3nx 0.8437
```

规则就是 `z(-当前名次) + w·z(NX)`，只在**同一分子式组内**换序，w=1 最优。这正是我们「47/48 错误是同分异构体」的靶心。

### 已推送的验证实验：075c

`jiyujia/enveda-075c-nx-instrument` v1（13:41 UTC）——不跑管线、不重训，只给 **072d 留出仪器的 150×25 候选**打 NX 分数（fold0 与 full1 两个检查点），输出 `nx_scores.csv`，本地再做同式组 z 重排并算 MRR。

- 仪器候选已内嵌（base64，150 行），查询谱在 kernel 内从 `train.parquet` 第 1 个 row group 按 inchikey14 过滤重建。
- 泄漏声明：这 150 个是 train 结构，`nx_full1` 训练时见过它们，所以这次结果**只能用来否证**（若连见过分子都改不动，就彻底没戏）；fold0 是相对干净的参照。
- 若 fold0 在同式组上确有增益，下一步才是「跑一次 0.492 管线 + nx_rerank」并提交。

### 在途

- 074a（Dante 0.494 复现）ref `58505232`：13:41 仍 PENDING（3h20m）。
- 075c：已推送，等运行结果。

## 2026-11-18 13:56 UTC 075c 结果：NX 在仪器上大幅提升

`jiyujia/enveda-075c-nx-instrument` v2 `COMPLETE`（v1 因数据集挂载路径失败：本镜像在 `/kaggle/input/datasets/<owner>/<slug>/`，已改成递归 glob）。输出 `nx_scores.csv`：150 分子 × 25 候选 × 2 检查点 = 7,500 行。

本地用 `probe/nx_eval.py` 做同式组 z 重排（`z(-rank) + w·z(NX)`），结果：

| 模型 / w | MRR@25 | top1 | 名次>1 的分子数 |
| --- | --- | --- | --- |
| 基线（w=0） | 0.8286 | 0.680 | 48 |
| **fold0 w=0.5** | 0.9211 | 0.860 | 21 |
| **fold0 w=1.0** | **0.9394** | **0.900** | **15** |
| **fold0 w=1.5** | **0.9402** | **0.900** | 15 |
| fold0 w=2.0 | 0.9191 | 0.867 | 20 |
| full1 w=1.0 | 0.9170 | 0.847 | 23 |

- 基线 0.8286 与 072d 原始记录完全一致，说明评估脚本无误。
- 33 个分子的榜首被纠正（0.680 → 0.900）。
- **fold0 优于 full1**：全量训练的模型反而更差，这与「纯记忆」假设相反，是个弱正面信号。

**泄漏声明（必须记住）**：这 150 个是 train 结构，fold0 只留出了 identity fold 0，所以绝大多数在两者训练集内。这个 +0.11 不能当作隐藏集预期；作者自己的 fold-safe 模拟是 +0.022，那个更可信。本次结果只用于证明「NX 在这类同式组上确有判别力，且集成路径可用」。

## 2026-11-18 13:58 UTC 075d：把 NX 接到 0.492 管线上，已推送

`jiyujia/enveda-075d-nx-rerank` v1 = **068a 0.492 管线原样 + 末尾追加一个 NX 单元**：

- 读 `/kaggle/working/submission.csv`，对 400 个分子重建查询谱，fold0 检查点打分，**只在同分子式组内**按 `z(-rank) + 1.0·z(NX)` 重排，候选集不变；
- 原文件备份为 `submission_baseline_075d.csv`，审计写 `nx_rerank_audit.csv`；
- **整体包在 try/except 里**：任何异常都保留基线提交，最坏情况退化为 0.492，不会因集成失败而崩掉；
- 为减小干扰，`fp_bits.npy`（41 KB）单独重打包为私有数据集 `jiyujia/casmi26-nx-bits`，没有挂 1.3 GB 的 pool 包。

commit run（smoke，12 分子）将先验证集成是否跑通；通过后提交触发 400 分子隐藏重跑。

## 2026-11-18 15:56 UTC 075d v2 smoke：门控生效，但阈值读成了管线自己的 0.78

```
NX: submission rows 400 molecules 400
NX: query rows 1213 molecules 400
NX: library gate 0.78 lib entries 400
NX RERANK {"status":"applied","weight":1.0,"lib_gate":0.78,"molecules":400,
           "active":0,"changed_rows":0,"changed_top1":0,"failures":0}
```

- `active 0` 是**正确的**：12 个可见分子 `library_max = 1.0`，全部被门控跳过，所以这次没有动任何满分榜首（v1 是 `changed_top1 1`）。
- 但门值不是我写的 0.9，而是管线自己的 `LIB_HIT_GATE = 0.78`——两者语义不同：0.78 会把「有较强类似物库匹配」的分子也一并保护起来，0.78–0.9 这一段 NX 就永远碰不到。
- 改用显式 `NX_LIB_GATE`（默认 0.9，作者默认值），推送 **v3**。

另一个操作教训：**commit run 还没产出 `submission.csv` 时提交会失败**：

```
400 Submission not allowed: Internal Error: Could not get Notebook Run information.
    Did not find provided Notebook Output File.
```

必须等 commit run 完成后再提交。已改成「等 COMPLETE + 日志出现 `NX RERANK` 才自动提交」。

## 2026-11-18 17:20 UTC 075d v3 已提交：ref 58519267

```
NX: library gate 0.9 | NX RERANK {"status":"applied","weight":1.0,"lib_gate":0.9,
    "molecules":400,"active":0,"changed_rows":0,"changed_top1":0,"failures":0}
Submission ref: 58519267   (今日剩 2 次)
```

commit run 里 `active 0` 符合预期（12 个可见分子 `library_max = 1.0` 全被 0.9 门控挡住），评分路径只在隐藏重跑里执行。

同时在本地把检查点组合也试了（免费，用已有 `nx_scores.csv`）：

| 配置 | MRR@25 / top1（w=1.0） |
| --- | --- |
| fold0 | **0.9407 / 0.900** |
| full1 | 0.9170 / 0.847 |
| fold0+full1 均值 | 0.9281 / 0.867 |

⇒ **不做集成，单用 fold0 最好**；提交的正是这个配置。

判据：隐藏重跑的分数 >0.492 才算 NX 在未见结构上兑现；≥0.50 则说明作者 fold-safe 的 +0.022 基本可迁移。
