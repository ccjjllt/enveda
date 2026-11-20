# Enveda CASMI 2026 研究项目

比赛：[Enveda CASMI 2026 — Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)，指标 **MRR@25**。默认使用中文记录。

## 当前进展（2026-11-20）

- **当前最高公榜为 0.495**。`jiyujia` 复现 Taeyang 公开方案原版 v2，Kaggle 提交 `58605458` 已完成。本仓的 068a 配置得到 **0.492**（`58412029`）。[公开方案](https://www.kaggle.com/code/taeyangg4/casmi26-lb-0-495-network-fr-p5-tail?scriptVersionId=371003154) · [最新实验](rounds/075-frontier/progress.md)
- **已封闭方向**：`001–075d` 的配方旋钮、`RR8` 重排/重训、mass prior 重标定（`+0.00000`）、e180/ICE/对齐门事后升位（0/6）、`class2` MLP（OOD 0.205）及 `075d` NX 异构体判别器（公榜 **0.381**，`−0.111`）均已终止，结论见 [已放弃研究总览](archive/research-archive.md)。
- **当前优先研究是 PubChem 同分子式候选召回**。在 110 道 MassBank 外部题上，将搜索深度扩到 50 后，PubChem 分支 MRR 从 **0.367** 升到 **0.382**，前 25 命中从 59 增到 65。扩到 100 时为 **0.386**、68 题。FLARE、MIST 与实测 CE 直接重排在固定前 25 上均降低 MRR。[075 实验进展](rounds/075-frontier/progress.md)

## 目录与入口

```text
enveda/
├── AGENTS.md                        # 协作与清理约定
├── archive/research-archive.md      # 001–075d 已放弃方向总览与恢复方式
├── rounds/
│   ├── README.md                    # 当前保留轮次索引
│   ├── 068-n1c-lam035/              # 0.492 基线（plan + progress + 068a/）
│   └── 075-frontier/                # 前沿计划、进展与精简调研（已合并 071）
├── ops/                             # remote-workflow.md、063/{chain,watch}.sh
├── data/
│   ├── raw/                         # 比赛原始 Parquet（2.8 GB，不入 Git）
│   └── references/                  # 历史核对 JSON（684 KB，入 Git）
└── artifacts/                       # 已清空（不入 Git）
```

> 历史快照标签：`pre-cleanup-2026-11-16`、`pre-cleanup-069-2026-11-16`、`cleanup-070-dead-2026-11-17`、`pre-cleanup-2026-11-19`（`7cf7681`）。
