# Enveda CASMI 2026 研究项目

比赛：[Enveda CASMI 2026 — Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)，指标 **MRR@25**。默认使用中文记录。

## 当前进展（2026-11-23）

- **本项目最高公榜为 0.498**。`jiyujia` 复现 goodpjw 公开原版，Kaggle 提交 `58695195` 已完成。单独改成按加合物合峰的提交 `58695253` 得 **0.488**。此前 Taeyang 原版为 **0.495**（`58605458`），本仓 068a 配置为 **0.492**（`58412029`）。[公开方案](https://www.kaggle.com/code/goodpjw2008/enveda-casmi-stack-aprior-uniq-lb-0-498/versions/3) · [提交列表](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/submissions)
- **已封闭方向**：`001–075d` 的配方旋钮、`RR8` 重排/重训、mass prior 重标定（`+0.00000`）、e180/ICE/对齐门事后升位（0/6）、`class2` MLP（OOD 0.205）及 `075d` NX 异构体判别器（公榜 **0.381**，`−0.111`）均已终止，结论见 [已放弃研究总览](archive/research-archive.md)。
- **121 题整线对照已完成**。0.498 原版与首式扩池版的离线 MRR 为 **0.57611 → 0.57675**，前 25 均命中 **103**，首位均 **52**。此前分支新增的 15 个命中中有 14 个已被原整线找回。[075 实验进展](rounds/075-frontier/progress.md)
- **47 道独立负离子题完整质量窗检索已完成**。原分支并集漏掉 18 题，7 个真结构在窗内。等预算融合新增命中 **0**，MRR@25 持平 **0.410616**。停止扩大当前配对训练。[逐题回执](data/references/075-negative-window-comparison-2026-11-23.json) · [自主训练研究](rounds/075-frontier/research/training-and-retrieval.md)

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
