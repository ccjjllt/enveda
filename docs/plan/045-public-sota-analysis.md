# 045 公开 SOTA 实现分析（2026-10-28）

只用 `kaggle kernels pull` 拉了源码（未下载输出），放在 `/tmp/pubk`。分析的 6 个 notebook：

| notebook | 公榜 | 本质 |
|---|---|---|
| busyaprime/lb-0-477-two-0-472-fixes-merged-the-right-way | 0.477 | 0.468 leader-safe + w_mid 0.75 + 碎片检查(只取 V2 序上的 rank-1 改动) |
| pjmathematician/casmi26-e303-pub477-gpu | 0.477 | 与上面 **逐字节相同**（md5 一致），纯复制 |
| rajrajak99/casmi26-apex-v1300-sota-pxm-0477 | 0.477 | 同一配置，删了说明文字，COMMIT_MODE='fast' |
| alexchilton/casmi26-lb-0-472-leader-safe-ranks-2-25-at-0-75 | 0.472(fork) | 0.468 只把 w_mid 0.50→0.75 |
| heliosli/happy-to-make-the-dataset-public-if-it-helps | 0.475 | 0.468 + 私有数据集的同分子式异构体重排器 rr v4（数据集不公开，无法复现） |
| romanrozen/0-468-casmi-from-peaks-to-structures | 0.468→现 0.472 | V4(w_mid 1.0) + rr3 + 碎片检查 + 尾部召回 |

## 1. 与我们的具体差异

底座完全同源：它们内嵌（SHA-256 固定）的 "public 0.465 pipeline" 就是我们 v465 fork 的那条线（Engine1/2、PubChem 通道、ICEBERG/GLACIER）。差异 **全部在最后 25 个候选的重排层**：

1. **Leader-safe 融合（最大收益，+0.006~0.007）**
   - 我们（040/043/044）：`1/(3+b) + W/(3+n)`，W=0.25 对全部名次 = 他们的 V1（0.466）。
   - 他们：rank 1 仍用 w_top=0.25 选出；rank 2–25 用 w_mid 重排。w_mid 0.50→0.468，0.70/0.75/0.80→0.472，1.00→0.470。
   - 这解释了我们 v20（W0.40 全局，0.453）为何掉分：动了 rank 1。我们 v22“top-1 保护”0.398 明显是实现问题，不是思路问题（他们的保护规则是 rank1 由 w_top=0.25 的融合决定，不是冻结 baseline）。
2. **碎片检查 Spectral Parallax（romantamrazov，未训练的 MetFrag-lite）**：≤2 键断裂碎片解释的强度比例。
   - Rank-1 规则：leader 库匹配 <0.87 且 ≥3 张可用谱时，rank 2–6 中与 leader 同分子式、碎片解释明显更多、神经分更高、且有引擎 top5 或 ICE/GL 支持的候选可升到 rank 1。
   - 0.477 的关键：该规则在 V2(w_mid 0.5) 序上运行，只保留它的 leader 改动；旧 leader 放第 2，其余按 0.75 序。直接叠加在 0.75 序上只有 0.472（候选窗口变了，引入错误 leader）。相邻交换和末位替换被关掉。
3. 其他试过无效：RRF offset 3→1（-0.002）、尾部 rescue（0）、GL_LAM 1.5（-0.02）、自训 FPNet 替换（全输）。
4. heliosli 0.475 依赖私有 rr 数据集，不可用。

## 2. 依赖与许可证（`kaggle datasets metadata` 读取）

| 数据集 | 许可 | 用途 | 风险 |
|---|---|---|---|
| sanpier/casmi-spectrum-model-v1 | **unknown** | 神经打分模型 | 高——正是我们已替换掉的模型；所有 0.47+ 公开方案都用它 |
| sanpier/catalogue-experiment-saner | **unknown** | 目录/辅助 | 高 |
| dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets | **CC-BY-NC-SA-4.0** | FPNet、LightGBM、ICE/GL、DreaMS、候选集打包 | 中高：非商业+相同方式共享 |
| dmitriigluzdov/casmi26-fold-safe-fpnet / panel-benchmark / knowledge-table / pubchem-popularity-prior | other（声明“竞赛数据条款：非商业+署名”，NPAtlas CC BY-NC） | 先验与表 | 中 |
| 碎片检查、leader-safe 代码 | 公开 notebook（Kaggle 默认 Apache-2.0）；frag_rescore.py 标 Apache-2.0 | 代码 | 低 |

注意：我们自己的主线也用 prvsiyan/casmi26-fp-models-v2（CC BY-NC-SA，见 v4g 清单）及多个 ahmedberatozer 数据集，NC 风险本就存在，与他们同级；**唯一额外红线是 sanpier 的 unknown 模型**。结论：不引入任何新数据集，只移植两段 **算法代码**（leader-safe 融合 + 碎片检查），神经分数一律用我们自己的 model3/ensemble，许可面不变。代码移植需在 notebook 中注明来源（Roman Tamrazov / alex chilton / busyaprime）。

## 3. 能否与 model3 结合

可以，而且是直接替换：
- leader-safe 只需要每个候选的 baseline rank 与 neural rank，我们 040 cell 第 66 行改成 rank1 用 w_top、2–25 用 w_mid 即可（~10 行）。
- 碎片检查的“神经分更高”门槛改读 model3 分数；其余输入（谱、ICE/GL、引擎 top5）我们管线都有。
- 风险：w_mid 最优值是用 sanpier 模型标定的。model3 在外部池验证 0.932 > 旧模型，理论上可承受更大 w_mid，但必须离线在 panel/外部池上扫。

## 4. 045 计划

1. **045a（优先，最小改动）**：044 notebook + leader-safe，w_top=0.25，w_mid=0.75，model3。离线扫 w_mid∈{0.5,0.75,1.0}（用 044 的外部池验证 + 自建同分子式 panel）选值。替代周六原定的纯 044 提交（或作为第 2 次）。
2. **045b**：移植碎片检查（仅 rank-1 规则，在 w_mid 0.5 序上跑，leader 改动合并到 045a 序上；关闭 swaps/末位），门槛读 model3。
3. 额度：每次全量约数小时 T4；周六重置后 30h 足够 2–3 次。每日最多 1–2 次提交。
4. 目标：复现到 ≥0.472（045a）→ ≥0.477（045b），且无 unknown 许可模型；之后再看 rr 式异构体重排（自训）冲金线 0.471 以上的余量。
