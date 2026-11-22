# Enveda CASMI 2026 研究项目

比赛：[Enveda CASMI 2026 — Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)，指标 **MRR@25**。默认使用中文记录。

## 当前进展（2026-11-22）

- **本项目最高公榜为 0.498**。`jiyujia` 复现 goodpjw 公开原版，Kaggle 提交 `58695195` 已完成。单独改成按加合物合峰的提交 `58695253` 得 **0.488**。此前 Taeyang 原版为 **0.495**（`58605458`），本仓 068a 配置为 **0.492**（`58412029`）。[公开方案](https://www.kaggle.com/code/goodpjw2008/enveda-casmi-stack-aprior-uniq-lb-0-498/versions/3) · [提交列表](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/submissions)
- **已封闭方向**：`001–075d` 的配方旋钮、`RR8` 重排/重训、mass prior 重标定（`+0.00000`）、e180/ICE/对齐门事后升位（0/6）、`class2` MLP（OOD 0.205）及 `075d` NX 异构体判别器（公榜 **0.381**，`−0.111`）均已终止，结论见 [已放弃研究总览](archive/research-archive.md)。
- **当前优先改进 PubChem 粗筛后的结构覆盖**。在 96 道独立 E180 实测题上，等量加入原首式的粗筛外结构，再用 ICEBERG 与 GLACIER 截回 25，命中 **59/96 → 80/96**。允许在原首式内换首时，MRR **0.46920 → 0.63240**。正在另 121 个此前未用的结构上复核。[075 实验进展](rounds/075-frontier/progress.md)

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
