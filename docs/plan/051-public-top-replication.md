# 051 公开榜首方案拆解与复现

## 公开 notebook 对比（2026-11-05）
| notebook | 公榜 | 与我们 046A(0.468) 的差异 | 可复现 |
|---|---|---|---|
| heliosli/casmi26-0-482 | 0.482 | 0.468 底座 + rr3 同分子式异构体重排器(1-2 键断裂碎片 + GNN/Transformer，训练集结构"冻结"校正 0.34SD 偏差，lib_max>=0.9 不动) + xl 外部新版谱库叠加(MSnLib 2026-07 / GNPS 2026-10 / MassBank 2026.03 中不在 train 的结构，cosine>=TAU 且匹配峰>=K 时直接放第 1；主办方在讨论 #750684 允许) | 否：两个 heliosli 数据集私有(metadata 中为空) |
| alexchilton/casmi26-lb-0-480-neural-ensemble | 0.480 | 自训更久的谱-结构打分器(kernel 输出 casmi26-scorer-train-v80)替换 sanpier；W_MID 0.75；MetFrag-lite 物理碎片检查在 V2 上改 leader，旧 leader 钉在第 2；InChIKey14 去重；2-25 名加文献先验 POP 0.10/SPEC 0.05(lib>=0.87)/NP 0.02；空位用外部候选补 | 是，全部公开 |
| bgunshui/casmi26-480-neural-ensemble、zz3t3rn41/peaks-to-structures-v2 | ~0.480 | alexchilton 变体(zz3t3rn41 另加 LOTUS 候选) | 是 |

## 许可
- sanpier/casmi-spectrum-model-v1：unknown；dmitriigluzdov/* 与 sanpier/catalogue-experiment-saner：未标注；alexchilton scorer 为 notebook 输出(未单独声明)。→ 只能用于 A 线冲牌，不可领奖。
- metric/rdkit wheel：官方，可用。

## 本轮
- 把 alexchilton 0.480 原样推到 ccjjllt/enveda-020-e006-repro v30，k_watch2.sh 自动提交(致谢作者)。
- 050(v29) 未提交：top1 与 046A 完全相同。

## 提分方向（按期望收益排序）
1. 自建外部新版谱库叠加(xl)：MassBank 2026.03(021 已有基础)、GNPS 2026-10、MSnLib 2026-07，排除 train 结构，cosine 阈值在 timsTOF 训练分子上定假阳性率。期望 +0.005~0.015；库本身多为 CC-BY/CC0，A、B 两线都能用。
2. 在 0.480 底座上叠加我们的 model3 碎片解释特征作为同分子式重排器，并加"训练集结构冻结"校正(model3 也偏向训练结构)。期望 +0.003~0.008。
3. B 线移植 alexchilton 的无模型技巧：InChIKey14 去重、文献先验、旧 leader 钉第 2、外部候选补空位。期望 +0.002~0.005，许可干净。
4. 自训更久/更大的谱-结构打分器(参照 scorer-train-v80)，与 sanpier 做集成。期望 +0.002~0.006。
5. LOTUS 候选扩充(zz3t3rn41)。期望 +0~0.003。
