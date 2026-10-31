# 046-sanpier：A 线（sanpier 模型，冲分保奖牌）计划（2026-10-31）

用户决定两线并行：A 线用 sanpier/casmi-spectrum-model-v1（许可 unknown，**不参与奖金**，只冲奖牌）；B 线自研 model3（许可干净，045b 队列照旧）。最终 2 个选定提交：A 一个、B 一个。

## 046A：复现公开 0.477 配方（在我们 036/v19 管线上）
- 底座：notebooks/036/enveda-036-v15-neural.ipynb（v19 = sanpier rerank W0.25，公榜 0.465）。
- leader-safe：rank1 由 W_TOP=0.25 融合决定；rank 2–25 用 W_MID=0.75 重排（alex chilton / Roman Rozen）。
- 碎片 rank-1 规则（Roman Tamrazov Spectral Parallax，MetFrag-lite）在 W_MID=0.5（V2）序上跑，只保留 leader 改动：`[frag_leader, V2_leader] + 0.75 序其余`（busyaprime merge）。
- 门槛对齐公开版：lib<0.87、≥3 谱、rank2–6 同分子式、frag_new≥0.27、margin≥0.095、old margin≥0.028、能量组投票≥0.70、神经 margin/std≥0.23、neural_cos ≥ leader+0.003、neural_fp ≥ leader−0.012、引擎 top5 支持≥1。
  与公开版差异：没接 ICE/GL simulator_vote（公开版允许 supports<1 但 sim≥1 通过、sim≤−1 拒绝）——我们只用 supports≥1，更保守。
- 构建：`notebooks/046/build.py [W_MID]`。
- 预期公榜：0.472–0.477（底座与公开同源；若只拿到 leader-safe 部分约 0.471–0.472）。

## 后续深度优化（按优先级）
1. **simulator_vote 补齐**：把 ICE/GL 分数传进碎片规则，完全对齐公开 0.477 门槛（可能 +0.001~0.003 的 leader 改动召回）。
2. **融合权重**：在 sanpier 上扫 W_TOP∈{0.20,0.25,0.30}、W_MID∈{0.6,0.75,0.9}、W_FRAG∈{0.5,0.75}；公榜每天 ≤1 次 A 线，优先只动一个参数。
3. **model3 + sanpier 集成**：神经分 = z(sanpier) + α·z(model3)（α∈{0.3,0.5}）或两个 rank 的 RRF；model3 外部池验证 0.932，与 sanpier 误差相关性低时期望 +0.003~0.008。需要先在 panel 上离线对比两模型 top1 互补率。
4. **自建同分子式异构体重排器（heliosli rr 思路）**：heliosli 0.475 靠私有 rr v4。我们用自己的 train 同分子式组 + 044 外部池难负例，训 LightGBM/小 MLP 重排器，特征 = sanpier/model3 分、碎片解释比、ICE/GL、引擎各源 rank、PubChem 热度；只在 rank1–6 同分子式内重排。这是冲金（≥0.482）的主要筹码。
5. 碎片相邻交换、尾部 rescue：公开实验为 0 或负，最低优先级。

## 额度与节奏
- GPU 周额度 30h（周六 08:00 重置），单次全量约 0.5–1h T4；两线合计每天 ≤2 次提交（Kaggle 日界 08:00 北京）。
- 新建独立 kernel ccjjllt/enveda-046-sanpier 失败：新 kernel 无法固定旧 docker 镜像，落到 Python 3.13，ICEBERG/GLACIER 需要的 RDKit cp312 wheel 装不上（v1、v2 均 ERROR，约 1 分钟）。因此 A 线也推到 ccjjllt/enveda-020-e006-repro，靠时间错开（046A 今天跑，045b 明天 08:05 推），按版本号提交互不影响。
