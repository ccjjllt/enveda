# 024 — 留出 NP 的 Engine1 指纹模型复核

日期：2026-10-21。接手`5673ecb`，承接023。对应[024进展](../progress/024-leak-free-engine1.md)。

## 问题

023 发现 R 遮蔽下 Engine1-only MRR 0.963，远高于完整流程 0.455。Engine1 加载的 `casmi26-fpnet-full1/fpnet_full1.pt` 记录的 val np=0.918，而 v4b 自带的 `fpnet_0.pt`（训练代码默认留出 np/nplib/syn/plusk/twin）val np=0.578：full1 几乎肯定训练时见过 NP 结构。因此 023 中「保护 Engine1 生成的第一名」在 C 上的收益（τ=0.3 +0.030，不门控 +0.164）以及 R 上关闭 PubChem 提升的 +0.052 都可能被训练重叠放大。**换成留出 NP 的 fpnet_0 后，这些结论还成立吗？**

## 假设

- fpnet_0 下各遮蔽的绝对 MRR 会下降，Engine1 之后的合并阶段相对更有价值。
- 若保护规则在 fpnet_0 下 C 仍为正、B/R 不退化，则规则不依赖泄漏，可进入 v14。

## 实验

1. 验证器加开关 `E024_FP0`：Engine1 不加载 fpnet_full1，回退到 notebook 自带的 `fpnet_0.pt`；其他（Engine2、PubChem 通道、ranker）不变，复用 021 的 Engine2/PubChem 输出。
2. 在 C、R、B 上各跑一次完整流程（跳过 cell≥27，去模拟器），同时扫 PubChem 提升关闭；导出每个分子 Engine1 第一名、是否生成、lib_max。
3. 事后在 base 与 promo_off 输出上测试保护规则 τ=0.3/0.5；若时间允许，再测 lib_max 门控的扩大生成。
4. 逐分子配对、bootstrap 95% CI。

## 成功标准

- 规则在 C 上正、R/B 零损失（wins/losses 无 loss 或 CI 不含负），才构建 v14 = v13＋规则并提交一次。
- 否则只记录，不提交。

## 算力

box CPU；每个遮蔽约 50 分钟。不占 Kaggle GPU。
