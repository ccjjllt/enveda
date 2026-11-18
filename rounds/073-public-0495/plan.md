# 073 复现公开 0.495

日期：2026-11-18。背景：072e 在 0.492 配方族上回 0.487（−0.005），该族判决上限。复查公榜与公开 notebook（见 `../072-scale/research/public-frontier-1118.md`）后发现唯一一份声称高于 0.494 的可读公开方案是 `heliosli/casmi26-0-495-public-demo`（0.495），且它出自与我们不同的配方线。

## 目标

1. 验证这份 0.495 notebook 能不能在我们账号下忠实复现（数据集全部可读：14 个 `sanpier`/`dmitriigluzdov`/`heliosli` 数据集逐个查过）。
2. 若复现成功，用一次提交验证公榜是否真为 0.495；这是把 0.492 底座换成 0.495 底座的前提。

## 假设

- 0.495 的增益来自**神经 RRF 共识**：榜首权重 0.25（leader-safe）、2–25 名权重 0.75，融合式 `1/(3+rank)`；随后跑一个**未训练的 MetFrag-lite 碎片解释度重打分**，合并时只采纳它的榜首移动。
- 这条与 068a 的「mass prior + N-1c + parallax」不是同一条路径；065a/066/067 做过同族 mid-rank 叠加（0.488），但从未测过 `w_mid=0.75` + 物理碎片检查 + 只保留榜首移动。

## 设计

- 原样复制源码，只改 `id`/`title`/可见性，docker 镜像与数据集列表保持与原作者一致，以最大化保真度。
- 不修改任何逻辑、不调参、不动 `COMMIT_MODE='smoke'`。
- 入口：`073a/enveda-073a-0495-repro.ipynb`（slug `jiyujia/enveda-073a-0495-repro`，T4，私有）。

## 判据

- 代码成功：kernel `COMPLETE` 且产出合法 `submission.csv`（400 行、id 与 sample 一致、每行 25 个候选）。
- 远端完成：日志出现完整管线各阶段。
- 公榜：≥0.494 视为复现成立；0.495 为完全复现；≤0.492 说明该 notebook 的声明不可迁移。

## 成本

GPU 约 2.6–2.8h（当前剩 8.11h），提交 1 次（今日剩 4 次）。失败路径成本仅为一次 kernel 运行。

## 结果（2026-11-18 08:15 UTC）

**假设被否证，且不是「复现失败」，而是「没有可复现的东西」。** `heliosli` 的 7 个数据集都是 `PUBLIC-DEMO-1116-R1`：真实竞赛权重被删除、替换为初始化 demo 权重，catalogue 换成 3,683 个合成分子，数据集描述明确写「Not a reproduction of 0.494 / no claimed leaderboard score」。Kaggle 也拒绝提交（non-versioned datasets）。

结论：公开可复现前沿 = **0.494**，0.495 是演示标签。本方向关闭。
