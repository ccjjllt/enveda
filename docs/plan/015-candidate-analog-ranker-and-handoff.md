# 015 — 候选级analog排序实证与交接

日期：2026-10-17（北京时间）。用户要求继续深度研究、明确下一步并写好交接文档。接手commit`ca1abba`，工作树干净。对应进展`docs/progress/015-candidate-analog-ranker-and-handoff.md`及正式交接`docs/handoff/015-candidate-analog-ranker.md`。

## 背景与目标

014证明013只有query级支撑，存在本谱未排除和单峰/完整单体峰背景，不能直接提交；固定碎片在R3也未确认迁移。当前最短有效动作是建立真正候选—参考结构证据，在同一原v9排序基线上完成固定OOF模型比较。此轮不重新扫描论文榜单、不下载大模型、不迁移TPU。

本轮目标：完成可复用候选特征、固定Logistic/HGB排序、参考缺失两域和谱信息负对照；把真实结果与可执行下一步写成完整交接。现有NP/G1/R3均已用于研究，保留250也已揭盲，不能把本轮OOF写成全新独立确认或encoder未见结构泛化。

## 预先固定的实验契约

### 数据、候选与两场景

- 只用NP250和G1研究250；query取最大原始峰数代表谱，复现原v9 FP＋ICE完整top100；候选为v9最终前50，同式槽位内重排，其余位置保持。
- 参考沿用原stride50约50681谱，按8192行batch读取；raw→canonical映射来自005身份表。不是全库召回上限；固定采样不声称随机样本。
- B_seen：全局排除这500个query的全部谱精确峰hash，允许同结构其他谱；这只建立精确重复排除，不自动等于来源独立。
- A_absent：在B基础上全局排除500个目标canonical的全部参考。此为预先冻结的评价干预，不是推理时已知真值，更不证明FP/ICE预训练未见。
- 场景过滤先于每query选参考；同极性、中性质量±200Da，最多最近2000谱；不跨极性硬对齐。

### 谱相似与候选传播

- 固定floor=.002、top256、sqrt强度；去前体±1.5Da，已知n>1时另去完整单体离子±1.5Da。该清洗是新的开发假设，记录删峰/空谱与信号保留份额，不能直接宣称通用正确。
- 只对两端单体、|z|=1、同极性启用位移，Δ=precursor_query−precursor_ref；其他条件仅direct。direct与shift候选配对边联合按强度乘积贪心一对一，tol=.01Da，不沿用013整谱二选一。
- quality=`cosine**3.2 * min(matched_count/3,1) * (1-largest_pair_fraction)`；数值截到合法范围。先按reference canonical取最高quality，再取top80不同结构。保存来源/原行/adduct/hash/匹配数和单峰贡献。
- Morgan r2/2048位、统一规范化结构。每个候选的analog传播排除reference canonical等于自身；自身直接匹配作为独立direct_quality，来自完整2000参考池，不用truth比较。
- 基线四特征：fp_z、ice_z、ice_present、v9_reciprocal_rank；新增六特征：direct_quality、analog_max_tanimoto、analog_weighted_tanimoto、analog_max_product、analog_top5mean_product、analog_best_spec_quality。空证据给0，不加入来源名/候选身份/面板名称等特征。
- 固定负对照：同query参考集合与Tanimoto不变，仅打乱reference quality，seed20261017；direct_quality保持不变，用于区分谱权重贡献与结构/直接检索先验。它不是删除全部谱信息的全无信息对照。

### 训练与评价

- 同NP/G1共享模型，同一个canonical的B/A副本必须同折。按SHA256(`20261017|015|canonical`)排序round-robin五折，不按行随机划分。
- 每query候选等权、query总权重相同，两场景各1/2；候选前50无真值的query保留负例。不用class_weight或阈值扫描。
- Logistic固定C=1、max_iter=1000、lbfgs，StandardScaler只拟合训练折。
- HGB固定max_iter=150、learning_rate=.05、max_leaf_nodes=15、min_samples_leaf=30、l2_regularization=1、early_stopping=False、random_state=20261017，单线程。
- 五方案：LR/HGB×baseline-only/+analog，另HGB+analog-shuffled。输出按同式前50槽位重排，平分按原位置；同时比较相对原v9及相对相同模型baseline-only，避免把学习排序自身改动归因analog。
- 每域/场景报告MRR@25、Top1/Top25得失、逐例rank、20000次分子bootstrap；不把两个场景的同分子当独立样本。OOF为开发诊断，若信号正向仍需新结构组或独立来源确认与离线端到端实现。

## 实施与预算

- `src/analog_evidence.py`、`scripts/015/build_015_candidate_features.py`：先probe，特征计算目标≤15分钟CPU、内存适合M5/32GB，产物`artifacts/runs/015-analog/features/`。
- `scripts/015/run_015_candidate_ranker.py`：项目内`.venv-015`复用离线缓存安装scikit-learn/scipy，不改全局环境；产物`artifacts/runs/015-analog/ranker/`含OOF、折、输入SHA及明确实验性的全数据拟合模型。
- 代码、运行、离线指标和public分别记录；不创建测试文件/框架/CI，不新开GPU作业，不自动提交未经验证的模型。
- 正式交接包含当前分数/版本、研究结果、数据使用边界、可复用路径、准确命令、停止无效路线的依据、下一次submission条件与远端操作限制。

## 主实验完成后的追加归因（事后诊断，非原预注册方案）

五方案结果显示NP B有增量、NP A/G1未确认；shuffle仍保留direct和参考身份，无法分离直接谱库检索与analog。因此在不修改原五方案、折、特征、参数和权重的前提下，追加LR/HGB各自“4基线＋direct_quality”五折消融，单独保存`artifacts/runs/015-analog/direct-only-ablation/`。只比较原v9、同模型baseline、direct-only与原组合，固定20000次分子bootstrap。不做超参/阈值扫描、不新增全量拟合模型、不按结果部署。实现`scripts/015/analyze_015_direct_ablation.py`，实际结果进入同名进展。
