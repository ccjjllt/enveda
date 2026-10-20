# 020 — 公开 E006＋PubChem Promo（rel>150）原样复现

日期：2026-10-19。接手`e2c298e`，承接019（特异碎片规则为负结果，收尾）。用户目标仍为金牌与奖金，并已批准把公开方案原样 fork 为私有 kernel 并提交一次。对应[020进展](../progress/020-e006-reproduction.md)。

## 问题

v9r2/v10 停在 public 0.323，距 10 月 18 日快照第五名 0.474 差约 0.15。012 已读过的公开源码中，[seyitkaangunes E006＋PubChem Promo rel>150](https://www.kaggle.com/code/seyitkaangunes/casmi26-e006-pubchem-promo-rel-150-lb-0-446) 作者页面显示 0.446，是完整的双引擎方案（Engine1 v4n 引擎＋Engine2 COCONUT/ChEBI 池＋PubChem 通道＋ICEBERG/GLACIER）。问题是：**这份分数能否在本账号、本数据版本下原样复现？** 若能，就以它代替 v10 作为后续改进的底座，比在 0.323 上继续叠加更划算。

## 假设

- 作者分数绑定到 V1 版本与 16 个输入数据集；原样 fork 不改代码应得到同分（±0.001）。
- 失败若出现，更可能来自运行环境（GPU 分配、镜像、Python 版本），而非方法本身。

## 实验

1. `kaggle kernels pull` 源 notebook 与 metadata，逐字节核对 notebook sha256；只改 `id`/`title`/`is_private`，保留 16 个 dataset source、T4、关网。
2. push 一次，确认交互运行（自带 12 分子 smoke 模式）通过 E004 preflight、走完全部 cell；只看日志，不在可见占位 test 上声称效果。
3. 运行 COMPLETE 后用该版本提交一次，记录 ref、版本号与 public。
4. 运行环境失败时只改环境（镜像、GPU 设置），不改代码；每次改动与失败原因写进进展。

## 成功标准

- public 与作者 0.446 一致（同三位小数）即视为复现成功，后续以该 fork 为基线。
- 否则记录差异来源，不把作者分数当本项目分数。

## 算力

只用 Kaggle T4；交互 smoke 约 0.2–0.5 GPU 小时，隐藏集重跑由比赛环境承担。本轮不训练、不改方法。

## 需继承的结论

- 012：单纯扩大候选库会伤害 FP-only 排序，广候选需配套更强排序；E006 正是这种配套方案。
- 上游数据集未写明许可证，领奖需另行核实来源与许可。
