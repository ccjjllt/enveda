# 055 计划：严格外部谱库 Rank-1（TAU≥0.85 / K≥4）

## 背景
- 最佳仍 051 = alexchilton Neural Ensemble 公榜 **0.480**（020 v30）。
- 052 在 051 上叠 xl TAU=0.55 K=3 → 公榜 **0.470**（误插；hit_cosines≈0.55–0.57）。
- 054/054b 复现 romanrozen 0.482：v34 占位误交 0.459；v35 smoke 已交 ref 58062849 PENDING。
- 公开 ozertuu「0.500+」notebook 的 Dual-Band ExtLib 单元实为 **stub**（只 load+assert，无 cosine 提升）；标题分数未实测。

## 假设
把 052 已验证可跑的 xl 叠加，阈值提到 ozertuu 文案水平（cosine≥0.85、匹配峰≥4），可大幅降低假阳，在 051（或 054b 若≥0.480）底座上净增益。

## 实验设计
1. 底座优先 **051**（已证实 0.480）；若巡检时 054b 公榜 ≥0.480 则改挂该底座。
2. 复用 `ccjjllt/casmi26-extlib-052` + 052 overlay 代码；改 `TAU=0.85`、`K=4`（或 LIB_TAU 同步提高）。
3. `COMMIT_MODE='smoke'`；推送 020 下一版本；COMPLETE 后 `-k/-v` 提交（避开 output 下载）。
4. 离线先在可见 test / 051 submission 上审计 hits/changed；若 hits=0 则不提交，改扫 TAU∈{0.75,0.80}。

## 不做什么
- 不提交 TAU0.55 变体；不把 ozertuu stub 当 SOTA。
- 本轮不做硬负例重训（GPU/时间另排）。

## 许可
xl 库多为 CC-BY/CC0；底座含 sanpier → **A 线冲牌**。
