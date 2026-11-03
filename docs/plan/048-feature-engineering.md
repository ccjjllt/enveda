# 048 计划：特征工程（谱-候选物理特征）

目标：给自研 model3（B 线，license-clean）加手工谱-候选特征，提高同分异构体区分能力；A 线以后可通过集成复用。

1. 复现 044 holdout（rng 20261025，2% 分子），重建同分子式候选集，CPU 跑 model3 得 emb/ll/tani 分数（`src/fe048/build_val.py`）。
2. 特征（`src/fe048/frag_feats.py`，纯 RDKit 自实现）：1 刀/2 刀（非环键对、同环开环）断键碎片质量 ±H；峰解释覆盖率（计数/强度加权/√强度/top10）、中性丢失解释、碎片数；分子描述符；查询内 rank/z/与最大值差。
3. 融合方式对比：LGB 直接排序、LGB 以 base 为 init_score 残差、线性 listwise（softmax）残差（`rank_cv.py`、`resid_cv.py`）。
4. 独立 holdout（val2，另 1500 查询、63 候选）检验（`fit_holdout.py`）；离线明显提升才推 Kaggle。
