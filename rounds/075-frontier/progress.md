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
