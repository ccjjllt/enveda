# 039 计划：自研谱-分子双塔模型（替代许可证不明的 sanpier 权重）
- 目的：神经重排保持获奖资格；只用 train.parquet + 自写代码（src/own_spec/）。
- 模型：峰 token（mz / 中性丢失 Fourier 特征 + 强度）6 层 Transformer d=384，CLS 融合 adduct/仪器/CE/前体；分子塔 Morgan 计数指纹 4096 → MLP；InfoNCE（可学习温度）+ 指纹 BCE 辅助头。
- 针对同分子式异构体：每个 batch 一半来自同分子式分组（每组≤8 个），使 in-batch 负样本含大量异构体。
- 验证：按 inchikey14 留出 2%；每个留出分子以训练库中同分子式结构（≥5 个）为候选，报告 isomer MRR（不同 fp 权重）。
- 算力：Kaggle T4×2（DataParallel，BS 1024，AMP），时间预算 7.3h 余弦退火；每 30 分钟评估+存档。Colab 当前无接入，故用 Kaggle。
- 产出：/kaggle/working/own_spec/{model.pt,feats.py,model.py,train_result.json}，之后做成私有 dataset，写推理 cell 替换 036 cell。
