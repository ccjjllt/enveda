# Enveda CASMI 2026 研究项目

比赛：[Enveda CASMI 2026 — Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)，指标 **MRR@25**。默认使用中文记录研究。

## 当前状态（2026-11-16）

- **最佳：065a public 0.488**（ref `58329677`）。066a/067a 均为 0.488，无增益。
- **公开 0.492 方案已提交**：Rogers Johnson `rogersjohnson/casmi26-l4-lam035` v1（N-1c `LAM=0.35`），ref **58410619**，2026-11-16 10:06 北京，状态 **PENDING**；出分后以实际公榜为准。
- **下一刀：068a 已备好，未推送。** 065a 只改 N-1c `LAM` 0.25→0.35，对应已核实的公开 0.492 配方；周 GPU 已用尽，**2026-11-21 08:00 北京**刷新后推送。
- 当前只走 Kaggle 远端 GPU；CPU 路线在 062a 已证明不可行。无运行中的 enveda 内核，无后台任务。
- **公榜已明显抬高**（2026-11-16 09:54 北京，5355 队）：前 10 为 0.507–0.549，前 20 线 **0.503**；27 队 ≥0.500，55 队 ≥0.494，229 队 ≥0.492。ccjjllt 0.488 第 **313**。0.492 只是第 100–230 名区间，不是竞争前沿。
- **公开 notebook 没有已核实的高于 0.492 的配方**：0.494/0.498/0.5 标题多为同一份 0.492 配方或其变体，作者公榜 0.487–0.492；前 10/27 队没有公开 competition notebook。alexchilton 的 59 臂公开分析（4 个增益 >+0.002，最大 +0.006，33 个退步）结论是单旋钮到不了前 20，需要新信号。

## 导航

- [已放弃研究总览](docs/research-archive.md)：001–064 的研究概况、实际结果、放弃原因、删除范围与恢复方式。**接手先读这一页。**
- [当前计划 068](docs/plan/068-n1c-lam035.md) / [当前进展 068](docs/progress/068-n1c-lam035.md)：0.492 配方与推送顺序。
- [065 计划](docs/plan/065-network-n1c.md) / [065 进展](docs/progress/065-network-n1c.md)：当前最佳 0.488 的来源。
- [066](docs/plan/066-libcos-midrank.md) / [067](docs/plan/067-own-hn-midrank.md)：两个中性试验，不作为底座。
- [比赛规则、数据与远端工作流](docs/remote-workflow.md)：Kaggle CLI、提交、配额与数据说明。
- [文档索引](docs/README.md)、[脚本说明](scripts/README.md)、[notebook 入口](notebooks/README.md)。

## 当前目录

```text
enveda/
├── AGENTS.md                  # 协作约定
├── docs/
│   ├── research-archive.md    # 已放弃研究总览（001–064）
│   ├── remote-workflow.md     # 比赛/数据/Kaggle 操作
│   ├── README.md
│   ├── plan/065-068.md        # 仅保留当前 A 线
│   └── progress/065-068.md
├── notebooks/
│   ├── 065a/ 066a/ 067a/ 068a/  # 当前 A 线候选
│   └── README.md
├── scripts/
│   ├── 063/                   # chain.sh / watch.sh
│   └── README.md
├── data/
│   ├── raw/                   # 原始比赛数据，不入 Git
│   └── references/            # 精简来源与历史核对，入 Git
└── artifacts/                 # 已清空；新产物仍不入 Git
```

001–064 的明细文档、旧脚本、旧 notebook、`src/` 与 5.9 GB 本地产物已于 2026-11-16 清理；清理前快照见 Git 标签 `pre-cleanup-2026-11-16`。研究结论没有丢失，统一收敛到[已放弃研究总览](docs/research-archive.md)。

## 工作约定

接手先读对应计划与进展，结束后更新进展。当前 A 线只改已得分底座上的单个旋钮，先 smoke 审计、再提交；代码成功、远端完成、公榜提分分开记录。失败实验的结论写入总览或进展，不保留无用的运行产物。原始数据与凭据不入 Git，不创建测试文件或 CI 门控。完整约定见 [AGENTS.md](AGENTS.md)。
