# 036 计划：同分子式异构体 —— 公开 notebook 调研 + 神经重排

## 公开 notebook 调研（2026-10-25）
| notebook | LB | 关键点 |
|---|---|---|
| sanpier/pubchem-neural025-lb-0-466 | 0.466 | 0.465 基线（候选级 PubChem promotion 修正）跑完后，对最终 25 个候选用 sanpier/casmi-spectrum-model-v1（谱 transformer + 指纹头）打分，RRF：1/(3+基线名次)+0.25/(3+神经名次)；候选集合不变 |
| lavinwins/casmi-26-lb-top-1 | ? | 同基线 + 把主引擎 top1 提回首位（protect_main_top1），无神经 |
| avazaku evidence-lab / guhongbin v50x | 0.465 | 同族基线变体 |
我们的 v15（0.464）已含 candidate_promotion_evidence 修正，与 0.465 基线同族；神经重排是唯一可直接移植、且有公开 LB 正证据（+0.001）的新点，正好作用于同分子式异构体的最终排序。

## 方案
v15（030 notebook）末尾追加一个 cell：读 submission.csv 与 te，用 final_model.pt 打分，W=0.25/offset=3 融合，失败则不动。数据集加 sanpier/casmi-spectrum-model-v1。

## 本地验证局限
模型用全部 train 训练，NP250/train 面板都会泄漏，无法诚实离线评估；本地仅做冒烟（12 分子全部打分，top1 变 1 个）。判断依据是公开 LB 证据。
