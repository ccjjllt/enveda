# 074 复现并提交最高分公开方案（Dante 0.494）

日期：2026-11-18。背景：073 证明 `heliosli` 的 0.495 是 demo 空壳（权重删除、catalogue 合成）。复查公榜后，唯一**作者真实持有高于 0.492**的公开 notebook 是 `ccuuuuy/casmi26-0-492-test-set-network-rerank`（作者 Dante，公榜 **0.494**）。

## 目标

用这份 notebook 做一次真实提交，验证它能否把我们的 0.492 推到 0.494。

## 这份 notebook 是什么

- 主线：`alexchilton` 的 0.488 peaks-to-structures 网络 + `LAM 0.25→0.35`（该组合作者实测 **0.492**，与我们 065a/068a 同源）。
- 在此之上叠了：
  - `CUSTOM(w_top=0.25, w_mid=0.75)` 神经 RRF 共识（leader-safe）
  - `PHYSICAL=True` 未训练 MetFrag-lite 碎片检查，`FRAG_ON_V2=True` 只采纳榜首移动
  - `ENABLE_K14_DEDUP=True` 候选池内 InChIKey-14 去重
  - 「Final layer (baidalinadilzhan)」文档先验在名次上的加权
- 作者公榜 **0.494**，比我们的 0.492 高 0.002；notebook 自述的 LAM 曲线停在 0.492，说明 0.494 来自上面这几层的叠加。

## 设计

- 原样复制源码，唯一改动：`COMMIT_MODE 'fast' → 'smoke'`（让 commit run 真正执行各阶段并写出合法 400 行文件，约 0.5h GPU；隐藏重跑本来就是全量）。
- 数据集与镜像保持与原作者一致（10 个数据集 + `popwenman/casmi26-scorer-train-v80`，全部为真实资产，不是 073 那种 demo）。
- 私有 kernel，slug `jiyujia/enveda-074a-dante-0494-repro`。

## 判据

- 代码成功：kernel `COMPLETE`，各阶段无 assert 失败，`submission.csv` 400 行。
- 公榜：≥0.494 = 复现成立并提分；0.492 = 该 notebook 实际只到我们现有水平；<0.492 = 叠加层有害。

## 成本

GPU 约 3.2h（commit 0.5h + 隐藏重跑 2.7h），提交 1 次（今日剩 4 次）。
