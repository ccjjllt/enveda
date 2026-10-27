# 044 计划：自研重排模型的「候选池硬负例」微调

## 为什么选这个方向
- 036~043 神经重排 val 涨（0.7583→0.7636 + 集成）但公榜一直 0.464~0.465，说明 **val 与测试分布不一致**：
  - 旧 val 的负例是「训练库内的同分子式结构」（均为训练过的分子），而测试时重排的是基线给出的 ≤25 个候选，来自 PubChemLite / COCONUT 等外部库，大多数从未被分子塔见过。
  - 旧损失只有 in-batch InfoNCE（1024 个负例中同分子式仅占少数）+ 指纹 BCE；推理用的分数 `cos*scale + fp_w*指纹对数似然` 从未被整体训练过，fp_w 只是事后扫出来的。
- v22（top1 保护）0.398、033（E1 候选注入）0.450 都证明改召回/门控风险大；硬负例只改重排打分，候选集合不变，风险可控，且直接针对「同分子式异构体」这一核心难点。

## 方案
- 热启动：042 的 model2.pt（val 0.7636@fp1.0），lr 2e-4，余弦退火，warmup 1000 步，seed 2。
- 外部候选池：franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints（385,760）+ prvsiyan/coconut-casmi26-candidates（436,389）的 SMILES，
  用与推理完全相同的规范化（去立体 + 规范互变异构体 + Morgan r2 4096 计数）重算指纹和 RDKit 分子式；与训练库重复的 inchikey14 去掉。
  指纹表（训练库 + 外部池，约 4.4GB uint8）整体放 GPU0，避免每步 CPU→GPU 拷贝。
- 新损失（每步）：从 batch 中取 384 个锚点，每个锚点在「同分子式池」（训练分子（不含留出）+ 外部池）随机取 64 个候选，
  用当前模型无梯度打分，取最高的 24 个作为硬负例，与正例一起按 **推理同款分数（fp_w=1）** 做 25 选 1 listwise CE。
  总损失 = InfoNCE + 20×指纹 BCE + 1.0×硬负例 CE。
- 验证：同一 2% 留出（seed 20261025）。
  - isomer val（与 039/042 同口径，门槛 0.7636）；
  - **pool val（新）**：正例 + 至多 63 个外部池同分子式候选，模拟测试时的候选来源。
- 算力：Kaggle T4×2（DataParallel 谱塔+分子塔），BUDGET 10.3h（指纹预处理约 1.5~2h，外部池指纹最长 110 分钟截断），训练约 8h。
- 产出 own_spec/model3.pt + train_result3.json（含 init_val/hist）+ ext_pool_meta.npz（外部池分子式/键，后续轮次可复用）。

## 上线门槛与流程（/workspace/k044_watch.sh 自动执行）
1. 训练 COMPLETE → 下载 model3.pt / train_result3.json / 日志到 /workspace/k044out。
2. 门槛：isomer val 最佳 ≥ 0.7676（比 0.7636 高 ≥0.004）**且** pool val 最佳 ≥ 初始 pool val + 0.01；否则不上线（NOT_PROMOTED）。
3. 过门槛：model3.pt 加入私有 dataset ccjjllt/enveda-own-spec 新版本 → 推送 notebooks/044/enveda-044-own-hn.ipynb（= 043 notebook，重排 cell 换成仅用 model3、fp_w 取 isomer+pool 之和最大者，W0.25）→ k_watch 要求日志含 "NEURAL OK" 与 "044 HN 1" 才提交。
