# 040 计划：v15 + 自研神经重排
- 基线：036 notebook（v15 + sanpier 神经重排 W=0.25，公榜 0.465）。v20（W=0.40）0.453，故保留 W=0.25。
- 改动：最后一个 cell 换为 notebooks/040/own_neural_cell.py（039 自研模型，OWN_FP_W=0.5），dataset 去掉 sanpier/casmi-spectrum-model-v1，加 ccjjllt/enveda-own-spec。
- 目的：去除许可证不明的第三方权重，保住获奖资格；分数目标 ≥0.464（不低于 v15）。
- 判定：≥0.465 则作为新主线；0.464 左右亦采用（合规优先）；明显下降则排查 fp 权重/W。
