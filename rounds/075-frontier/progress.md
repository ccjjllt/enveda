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
