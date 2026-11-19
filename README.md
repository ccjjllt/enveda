# Enveda CASMI 2026 研究项目

比赛：[Enveda CASMI 2026 — Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)，指标 **MRR@25**。默认使用中文记录。

## 当前现状（2026-11-19）

- **当前最佳基线**：`068a`（N-1c `LAM=0.35`）public **0.492**（自有 ref `58412029` / 公开 ref `58410619`）。
- **公榜位置**：第 1 名 `0.550`、第 20 名金牌线 `0.506`、第 30 名 `0.502`；前 30/40 名无公开 notebook。公开可复现前沿为 `0.492–0.494`（`074a` Dante 复现仅得 `0.492`，`073a` `0.495` demo 为空壳）。
- **已封闭方向**：`001–075d` 的配方旋钮、`RR8` 重排/重训、mass prior 重标定（`+0.00000`）、e180/ICE/对齐门事后升位（0/6）、`class2` MLP（OOD 0.205）及 `075d` NX 异构体判别器（公榜 **0.381**，`−0.111`）均已终止，结论见 [已放弃研究总览](archive/research-archive.md)。
- **唯一未封闭前沿（[075-frontier](rounds/075-frontier/plan.md)）**：不重训原生模型，专注 MS2 同分异构体判别——建 `075b` 化学型整组留出面板、零成本审计 `068a` 未用信号、单臂评估官方新版 `MIST MSG` / `ICEBERG 2.1 MSG` / `MSAlign (DreaMS×MoLFormer)`。

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
