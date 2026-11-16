# 069 计划：在 0.492 底座上筛新信号（timsTOF 前向、DreaMS、PubChem 尾召回）

## 背景

- 公开已核实最高仍是 **0.492**：065a + N-1c `LAM=0.35`。本队 068a 已在 jiyujia 账号复现并提交（ref **58412029**），公榜待出。
- 公开 notebook 没有已核实高于 0.492 的配方；前 10/27 队没有公开 competition notebook。alexchilton 的 59 臂单旋钮分析：最大 +0.006，33 个退步，结论是需要新信号。
- 公开侧新增资产里没有新的强权重；可用的新信号集中在：
  1. **timsTOF 微调 ICEBERG**（`hjyact/casmi26-timstof-iceberg`，MIT，221 MB）：作者结构留出 +0.030/+0.018，双 checkpoint +0.036/+0.017 MRR@25。
  2. **DreaMS 谱嵌入**（`hengck23/hengck23-dreams-enveda-casmi26`，MIT/CC BY 4.0）：现有管线只用于空槽补尾，没有给候选打分。
  3. **PubChem 通道尾召回**：主池窗口中位仅 107 个结构，PubChem 窗口中位约 9,400；通道只插 5 个固定槽，剩余 20 个 PubChem-only 候选未用。
  4. ISDB-NP / LOTUS / GNPS：候选结构侧已饱和，ISDB 覆盖仅约 4% 训练结构，实测尾部插入 0/400，不作为主攻。
- 算力：ccjjllt 周 GPU 已用尽；用户授权 jiyujia（约 24h）与 zhuating（约 5–7h，当前被其他任务占用）。所有训练/推理在 Kaggle 远端，不在本机跑。

## 目标与假设

目标：在 0.492 之上找到一个可复现、许可干净、9h 断网可跑的新信号；单条目标 +0.003～+0.02，多条可叠加但先分别验证。

假设：
- H1：测试 100% timsTOF，而管线 ICEBERG 是 MassSpecGym（以 Orbitrap 为主）；timsTOF 微调前向模型能改善同分子式异构体的中名次排序。
- H2：DreaMS 学习式谱嵌入对跨 CE/跨仪器更稳健；把它作为候选分数（kNN 或对比对齐）能补充现有 FPNet 回归与谱库余弦。
- H3：PubChem 通道的 5 个固定槽丢弃了过多候选；把少量高置信 PubChem-only 候选放进尾部可提升 class 2 召回。
- H4（否证）：ISDB-NP 预测谱与实测 timsTOF 的余弦分布偏低，0.80 门控下无插入；不作为提分通道。

## 实验设计

| 轮次 | 改动 | 预期 | 判定 |
| --- | --- | --- | --- |
| 068a | 065a + N-1c `LAM=0.35` | 0.492 | 安全基线；公榜出分即锁定 |
| 069a | 068a + ISDB-NP 预测谱尾部插入（TAU 0.80，尾 3 槽） | 0～+0.003 | 审计 0/400 插入 → 不提交 |
| 070a | 068a + `enveda-np-examples` 源内 direct-search 升位 | 0～+0.005 | 与 H1/H2 重叠，低优先 |
| 071a | 068a + `enveda-180`/NP timsTOF 源内 direct-search 中名次重排 | +0.002～+0.008 | 与 073a 同向，低优先 |
| 072a | 068a + DreaMS Enveda180 kNN（top100，w0.3，leader 冻结） | +0.003～+0.008 | 审计 11/400 重排；提交 ref 58413244 |
| 073a | 068a + timsTOF 微调 ICEBERG（tims_ft3+ft2，top20，w0.3） | +0.005～+0.02 | 审计 12/400 重排、leader 0；提交 ref 58413231 |
| 074a | 068a + `CUSTOM rescue=3`（PubChem/outsider 尾 3 槽） | 0～+0.005 | 先 commit 审计，历史同改动 −0.006，谨慎 |
| 075a | 073a 的 `ICE_W` 0.3→0.5 / topN 10–40 | ±0.002 | 等 073a 公榜后再决定 |
| 076a | DreaMS×Morgan 对比对齐（同式硬负例 InfoNCE），中名次 w0.3 | +0.003～+0.008 | 训练 kernel `jiyujia/enveda-align-train`；验证 R@1 达标才推 |

## 实现步骤

1. 068a 用 jiyujia 推送并提交，锁定 0.492。
2. 069a/072a/073a/074a 用同一 068a notebook 加独立 overlay cell；每个 overlay 失败即原样返回，不破坏 `submission.csv`。
3. timsTOF ICEBERG：`ice_rerank.py` 以子进程运行，`CASMI_MAIN` 指向数据集 `ice_tree/main`，`ICE_CKPT=tims_ft3,tims_ft2`，只取 `sub_safe_w0.3.csv`。
4. DreaMS：`dreams_minimal_model.py` + `embedding_model_state_dict.pt` 计算 test 嵌入；Enveda180 嵌入 mmap 分块 kNN；leader 冻结。
5. 对齐：冻结 DreaMS，训练 1024→512→256 与 2048→512→256 两个 MLP，同式硬负例 InfoNCE；按 InChIKey14 分组留出 10% 结构，报告 R@1/R@20。
6. 每次记录：kernel 版本、输入、GPU、时长、审计 JSON、ref、公榜分。代码成功、远端完成、公榜提分分开记录。
7. 只提交公榜有正证据的版本；失败版本保留 kernel 记录但不占最终提交。

## 许可

- timsTOF ICEBERG：MIT；微调数据为 train，host 已认可 train 派生。
- DreaMS 权重：CC BY 4.0，host 已书面放行（discussion 745604）。
- Enveda180 嵌入/index：MIT（hengck23 包）；底层 Enveda-180 已获 host 批准。
- 不使用 CSI:FingerID / MSNovelist / 任何含 NIST 的权重。
- 获奖包里不重新分发 CC-BY-NC/AGPL 资产；notebook 内保留 attribution。

## 继承结论

- 单旋钮不够；top-1 改动风险高，所有新通道先只动 2–25 名，leader 冻结。
- 离线指标与公榜多次反号；只有实际提交出分才算有效。
- ISDB/LOTUS/GNPS 不是候选结构侧的增量；不要在这条线上继续投入。
- 真正缺口在池外召回/class 3 生成；本轮先验证同仪器前向与学习式谱嵌入，若都中性，再考虑 MIST/MolForge 生成路线。
