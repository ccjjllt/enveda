# Enveda CASMI 2026 研究项目

比赛：[Enveda CASMI 2026 — Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)，指标 **MRR@25**。默认使用中文记录研究。

## 最新研究入口（2026-11-18 上午）

[072 提分空间排查](rounds/072-scale/progress.md)：最新结论——**0.492 是本配方族上限，已用机制性证据闭合**。072e 首次在 0.492 配方里真正启用 RR8 重排 + z 保证金榜首晋级，公榜 **0.487（−0.005，ref 58487796）**；与 082c/082d/082f3/082g/083a 合并，升位族 **0/6 全部 ≤0.492**。072d 留出仪器（删掉 150 分子参考谱）显示召回 **150/150**、MRR 0.8286；离线再核：48 个排序错误中 **47 个是同分异构体**（分子式完全相同），mass prior 网格可提升量 **+0.00000**。⇒ 候选不缺、质量层不错，残差 100% 在 MS2 同分异构判别，而现有重排在该层无判别力（AUC 0.712）。越线需要新的谱→结构模型，属新一轮独立预算。

[071 提分路线深度研究](rounds/071-breakthrough-research/research-report.md)与[实验设计](rounds/071-breakthrough-research/plan.md)：当前已验证最佳为068a **0.492**。优先研究新版官方MIST候选评分、MSAlign式结构表示与统一候选排序，MS-GPT作有限生成探索。已确认083a在DreaMS编码前平均原始峰序列的输入缺陷；e180校准与对齐头监督训练是否重叠仍待核对。“0.492已到顶”“0.51统一精度门槛”“一致性门零下行”的旧推断不再作为决策依据。最新提交读回与核查范围见[071实际进展](rounds/071-breakthrough-research/progress.md)。以下11月16日内容保留为历史快照。

## 当前状态（2026-11-16）

- **新最佳：public 0.492**（ref `58410619`，公开 N-1c `LAM=0.35` 复现，2026-11-16 15:10 北京出分），ccjjllt 第 **314/5376**。相对 0.488 为 **+0.004**；0.492 平台已扩散，仍只在前 100–230 名区间。
- 自有 `jiyujia/enveda-068a-n1c-lam035` v1（065a + N-1c `LAM=0.35`）ref `58412029` 也已出分 **0.492**，与公开复现一致；自有权重版本可复现。
- **069 新信号已出分，均未超过 0.492**：DreaMS kNN 0.492（ref `58413244`，打平）、timsTOF ICEBERG 0.491（ref `58413231`，−0.001）、DreaMS×Morgan 对齐 0.490（ref `58415444`，−0.002）。对齐头在 3,719 个未见结构留出上 R@1 0.316、R@20 0.989、中位 rank 2；ICE 在未见结构上 R@1 0.4395、R@20 0.9997，但中名次叠加到 0.492 底座均为中性或负。**公开可复现的新信号没有超过 0.492。**
- **算力**：ccjjllt 周 GPU 已用尽；用户授权 jiyujia（约 24h，本次实际使用）与 zhuating（当前被其其他任务占用）。全部在 Kaggle 远端 GPU 运行，本机不跑训练。
- **今日提交额度 5/5 已用完**，2026-11-17 08:00 北京重置。明日优先提交 hidden-specific 的 **082c**：Enveda-180 not-in-train 外部谱库（106 个多谱一致命中，自检索 top-1 precision 0.812）；082a（164 命中，激进）与 082b（TAU 0.95，rank-2）作对照。无运行中的 enveda 内核时以 `kaggle kernels list --user jiyujia --sort-by dateRun` 核对。
- **公榜**（2026-11-16 12:22 北京快照，5364 队）：前 10 为 0.507–0.549，前 20 线约 0.503；ccjjllt 0.488 第 **335**。0.492 只在第 100–230 名区间，不是竞争前沿。
- **公开 notebook 没有已核实的高于 0.492 的配方**：0.494/0.498/0.5 标题多为同一份 0.492 配方或其变体，作者公榜 0.487–0.492；前 10/27 队没有公开 competition notebook。alexchilton 的 59 臂公开分析结论是单旋钮到不了前 20，需要新信号。

## 导航

- [已放弃研究总览](archive/research-archive.md)：001–064 的研究概况、实际结果、放弃原因、删除范围与恢复方式。**接手先读这一页。**
- [轮次索引](rounds/README.md)：065–069 计划/进展/notebook/专题调研，一轮一目录。
- [当前计划 069](rounds/069-new-signals/plan.md) / [当前进展 069](rounds/069-new-signals/progress.md)：timsTOF ICEBERG、DreaMS kNN、对比对齐、PubChem 尾召回的实验与审计。
- [068 计划](rounds/068-n1c-lam035/plan.md) / [068 进展](rounds/068-n1c-lam035/progress.md)：0.492 配方与推送顺序。
- [065 计划](rounds/065-network-n1c/plan.md) / [065 进展](rounds/065-network-n1c/progress.md)：0.488 的来源。
- [066](rounds/066-libcos-midrank/plan.md) / [067](rounds/067-own-hn-midrank/plan.md)：两个中性试验，不作为底座。
- [比赛规则、数据与远端工作流](ops/remote-workflow.md)：Kaggle CLI、提交、配额与数据说明。
- [远端脚本说明](ops/README.md)。

## 当前目录

```text
enveda/
├── AGENTS.md                  # 协作约定
├── rounds/                    # 研究主体，一轮一目录
│   ├── 065-network-n1c/        # plan.md + progress.md + 065a/
│   ├── 066-libcos-midrank/    # 同上结构
│   ├── 067-own-hn-midrank/
│   ├── 068-n1c-lam035/        # 当前最佳 0.492，入口 068a/
│   ├── 069-new-signals/       # plan + progress + research/ + 待提交 082c/
│   └── README.md              # 轮次索引
├── archive/
│   └── research-archive.md    # 已放弃研究总览
├── ops/
│   ├── remote-workflow.md     # 比赛/数据/Kaggle 操作
│   ├── 063/                   # chain.sh / watch.sh
│   └── README.md
├── data/
│   ├── raw/                   # 原始比赛数据，不入 Git
│   └── references/            # 精简来源与历史核对，入 Git
└── artifacts/                 # 已清空；新产物仍不入 Git
```

001–064 的明细文档、旧脚本、旧 notebook、`src/` 与 5.9 GB 本地产物已于 2026-11-16 清理；清理前快照见 Git 标签 `pre-cleanup-2026-11-16`。069 已终止变体（072a/073a/076a 等）的本地 notebook 与 /tmp 废弃下载约 520 MB 已于同日清理，快照见 `pre-cleanup-069-2026-11-16`。研究结论没有丢失，统一收敛到[已放弃研究总览](archive/research-archive.md)。

## 工作约定

接手先读对应计划与进展，结束后更新进展。当前 A 线以 0.492 为底座；069 已证明 DreaMS kNN / timsTOF ICEBERG / 对比对齐的中名次叠加都不提分，下一步只做 hidden-specific 的 Enveda-180 谱库或更强的候选生成/排序模型。先 smoke 审计、再提交；代码成功、远端完成、公榜提分分开记录。失败实验的结论写入总览或进展，不保留无用的运行产物。原始数据与凭据不入 Git，不创建测试文件或 CI 门控。完整约定见 [AGENTS.md](AGENTS.md)。
