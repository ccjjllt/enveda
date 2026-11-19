# Enveda CASMI 2026 研究项目

比赛：[Enveda CASMI 2026 — Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)，指标 **MRR@25**。默认使用中文记录。

## 当前现状（2026-11-19）

- **当前最佳基线**：`068a`（N-1c `LAM=0.35`）public **0.492**（自有 ref `58412029` / 公开 ref `58410619`）。
- **公开前沿**：11-19早间榜单快照第1名0.550、第20名0.506。晚间新核Taeyang平台公开提交0.495，Alex/Arsen公开自报复现；源码与FR/P5/D-L1推理资产可读，但FR训练三项派生输入生成链尚未闭环，本队未复现。它与旧073a演示版不同，详见[075框架研究](rounds/075-frontier/research/next-frameworks.md)。
- **已封闭方向**：`001–075d` 的配方旋钮、`RR8` 重排/重训、mass prior 重标定（`+0.00000`）、e180/ICE/对齐门事后升位（0/6）、`class2` MLP（OOD 0.205）及 `075d` NX 异构体判别器（公榜 **0.381**，`−0.111`）均已终止，结论见 [已放弃研究总览](archive/research-archive.md)。
- **当前优先研究（[075-frontier](rounds/075-frontier/plan.md)）**：不重训原生模型，固定等输入、整组留出的075b面板；新增FLARE峰—原子局部匹配、adduct/CE条件下差异碎片评分，与MIST MSG比较互补。072d定向150题不能证明召回饱和；MARASON原MSG流程含NIST派生CE，仅作以后合规重建的备选。

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
