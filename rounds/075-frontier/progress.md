# 075 进展

更新日期：2026-11-20（北京时间）。

## 2026-11-20 实验进展

- 用 `jiyujia` 提交了一次 Taeyang 公开方案原版 v2。Kaggle 提交 `58605458` 已完成，实得公榜 **0.495**，较 `068a` 的 **0.492** 增加 **0.003**。标称 0.496 的公开演示缺少原始训练权重，标称 0.498 的公开变体未查到对应平台实分。[Taeyang 原版 v2](https://www.kaggle.com/code/taeyangg4/casmi26-lb-0-495-network-fr-p5-tail?scriptVersionId=371003154) · [Kaggle 提交](https://www.kaggle.com/competitions/enveda-casmi26-molecule-id-mass-spectra/submissions)
- [075b 整组留出 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-queries) 在断网 Kaggle CPU 完成，用时约 34 分钟。80 个不同骨架的正离子结构、211 张谱，参考谱排除清单覆盖 1,301 个原始训练标识。[仅质量排序分支 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-mass-queries) 的共同输入也有 80 个结构、211 张谱，候选数中位数 25，Recall@25 为 0.625，MRR@25 为 0.20685。FLARE、MIST 与 CE 首轮在质量候选上评分。
- [ICEBERG 2.1 MSG](https://www.kaggle.com/code/jiyujia/enveda-075b-ce-fragments) v1 在断网 Kaggle CPU 跑完。三个结构的 15 eV 预测与 35 eV 预测的谱熵相似度为 0.913–0.933，与 55 eV 预测为 0.607–0.664。该结果确认所用权重随 CE 响应。异构体名次和公榜收益仍待测。
- [FLARE/MIST 评分 v2](https://www.kaggle.com/code/jiyujia/enveda-075b-flare-mist) 与[排序比较 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-comparison) 已在断网 Kaggle CPU 完成。固定质量候选及同分子式组的占位后，FLARE 得 MRR@25 **0.37084**（+0.16399，14 题升到第一、1 题丢掉第一），MIST 得 **0.18803**（−0.01882）。71 个未匹配到公开 MSG 结构标识的样本上，FLARE +0.15980。真值在候选的 134 张谱中，单公式推断命中 88 张。FLARE 用发布权重，图卷积用 PyTorch 复现作者计算，尚未与原版 DGL 逐值比对。还需要真实前 25 候选上的结果。
- [FLARE 逐同式组公式 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-flare-group-formula) 在相同输入得 **0.35974**（+0.15289，12 题升到第一、1 题丢掉第一）。先用单公式版本验证真实候选。该轮各模型的评分阶段耗时为 MIST 12.3 秒、FLARE 单公式 37.9 秒、逐组 43.8 秒，均为 Kaggle CPU。
- [068a 真实候选导出 v2](https://www.kaggle.com/code/jiyujia/enveda-075b-pipeline) 用时约 61 分钟，Kaggle 断网 T4×2。排除 14,406 条训练参考谱，外谱查询窗另排除 34 条。80 题实际前 25 的 Recall@25 **0.98750**、MRR@25 **0.94848**；各来源合计 7,365 个题–候选组合，并集召回仍为 0.98750，剩余可排序空间 **0.03902**。查询来自比赛训练集，既有权重的训练曝光尚未消除。v1 在 15 分钟处因物理打分源码和校验值不同退出，修复后 v2 成功。
- [独立实测谱比较 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-independent-spectra) 在 Kaggle CPU 完成。同加合物、不同来源且排除近重复后，正确结构有独立参考谱的 8 题均由谱库排在第一。FLARE 将其中 3 题降位，该子集 MRR 从 **1.000 降为 0.716**。这 8 题不足以选定谱库保护阈值。
- [FLARE 跨 CE 一致排序 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-consensus) 在相同质量候选上比较一个固定规则：同一加合物至少两个实测 CE，且所有可比较谱都支持新首位，才改首位。同式组其余候选仍可重排。MRR 从普通 FLARE 的 **0.37084 降到 0.36459**，升首 14→13，丢首仍为 1。该规则在这 80 题未减少丢首，独立参考谱上另测。
- [FLARE 差异谱峰加权 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-flare-peaks) 复用相同公式与权重，按同式候选之间的峰–原子最大匹配分差给谱峰加权，保留原原子–谱峰得分。80 题质量候选上得 **0.36471**（普通 FLARE **0.37084**），升首 13 对 14、丢首均 1。真实候选首轮沿用普通 FLARE。
- [独立谱扩样 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-independent-ce-agreement) 在断网 Kaggle CPU 约 4 分 33 秒完成。跨来源富集的 120 个新结构、391 张正离子谱，正确候选有独立参考谱的 102 题。全部 120 题的 MRR@25 为实测谱库 **0.80700**、直接 FLARE **0.70247**（升首 4、丢首 23）、跨 CE 一致 **0.75143**（升首 2、丢首 11）。跨 CE 一致仍会改错受实测参考谱支持的首位，后续须区分有可靠参考的检索题和无参考的结构判别题。
- [独立谱首位配对 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-independent-pairs) 对 27 次改变首位正确性的变动复算。拟替换者缺独立谱时为 0 修正、8 误动。双方都有独立谱时为 4 修正、15 误动，原首位减拟替换者的谱库余弦差中位分别为 **0.040** 和 **0.626**；两组范围重叠，修正仅 4 题，暂不定阈值。
- [ICEBERG 2.1 评分 v2](https://www.kaggle.com/code/jiyujia/enveda-075b-ce-fragments) 在 Kaggle CPU 完成 5,161 组候选–能量条件（2,849.6 秒，0 失败），[质量候选比较 v3](https://www.kaggle.com/code/jiyujia/enveda-075b-comparison) 的 80 题上，观测 CE 余弦最高 **0.40921**（+0.20236，17 升首、0 丢首，77 题覆盖）；固定 CE 熵 **0.39284**、观测 CE 熵 **0.39875**、纯跨 CE 差异 **0.33267**、熵加差异 **0.40038**。真实候选下一轮对照观测 CE 余弦和中位固定 CE 余弦，以分离碰撞能量与相似度度量的影响。
- [独立谱等权倒数名次融合 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-library-fusion) 复用扩大 S 的 120 题和 FLARE 分数，按同式组的谱库名次和 FLARE 名次等权融合。Kaggle CPU 约 62 秒。MRR@25 **0.79887**，升首 2、丢首 4，相对原谱库 **0.80700** 的配对差为 **−0.00813**（95% 区间 −0.02925 至 +0.01306）。直接 FLARE 为 0.70247，升首 4、丢首 23。真实前 25 候选会同时比较倒数名次融合与直接重排。
- [真实候选比较 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-real-comparison) 读回固定 80×25 候选，原排序 0.94848。普通 FLARE **0.35650**（0 升首、58 丢首）、跨 CE 一致 **0.46430**（0/45）、等权倒数名次融合 **0.81793**（0/19）、MIST **0.17336**（0/70）。71 个未匹配公开 MSG 结构标识的题也均下降。停止这几种直接接入配置；继续分解原排名来源，并准备比赛训练集外的官方测试折。[真实候选 ICEBERG 对照 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-real-ce) 已启动，仅比较观测 CE 和固定 CE 下同一种余弦。
- [真实候选 ICEBERG v1](https://www.kaggle.com/code/jiyujia/enveda-075b-real-ce) 在断网 T4×2 完成 5,786 组条件，预测耗时 138.6 秒。80 题上，固定中位 CE 余弦 **0.57201**（1 升首、45 丢首），逐谱实测 CE 余弦 **0.60104**（1/42）；等评分条件下实测 CE +0.02903。直接重排相对原排序 −0.34745，停止该接入方式。
- 对 068a 候选运行读回，已确认加载 [Popwenman v80](https://www.kaggle.com/code/popwenman/casmi26-scorer-train-v80)，其公开训练入口读比赛 `train.parquet`。归因资产和冻结代码指向主 FPNet 的全训练数据版本。当前 80 题的排序高分须按旧权重可能已见结构解释。外部测试准备使用 MassSpecGym 原始官方 test 折，排除 Enveda train 和 MSG train/val 的规范化别名，仅使用有原始记录的 CE。[测试能量回填讨论](https://github.com/pluskal-lab/MassSpecGym/issues/71)
- 对原 80 题分解，Engine 1 单独已排对首位 73 题（MRR@25 **0.94625**），Engine 2 为 69 题（**0.91167**），最终 74 题（**0.94848**）。Engine 1 的 `library_max` 中位数为 0.48760，只有 10 题达 0.78。v80 训练结果记录最终用全部接受的比赛数据重训，含原验证身份。80 题的逐结构训练匹配仍待核对。这批结果不能作为独立泛化分。
- 已启动原始官方 test 折的外部数据准备，筛除比赛 train 和 MSG train/val 规范化同结构，优先保留与 Enveda 不同的骨架。候选由 068a 自然产生，正确结构缺失的题继续计入分母。FLARE 随附分折代码在 `split_pth=None` 时直接使用数据的 fold，权重未附实际训练数据 SHA 或 ID。MIST 的公开训练流程把 test 留出。[Kaggle 外部谱 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-external-msg-test) · [MIST 发布记录](https://zenodo.org/records/23303289)
- 官方 MSG v1.5 MIST-CF 的 Kaggle CPU 比较 v2 已完成，联网下载作者代码与权重。真结构进入 068a 前 25 的 207 张谱上，两法共享同一候选分子式列表。现有 FLARE 上游碎片覆盖评分为 Top1 **162/207**、Top5 **203/207**，MIST-CF 为 **133/207**、**202/207**。保留现有公式打分。这个实验只比较名单内排序，独立分子式枚举与召回尚未测量。[Kaggle 对照](https://www.kaggle.com/code/jiyujia/enveda-075b-mist-cf)
- MSG v1 外部测试的 [Kaggle 准备](https://www.kaggle.com/code/jiyujia/enveda-075b-external-msg-test) 从 14,066 张正离子谱读到 3,044 个标准化结构记录，扣除身份重合并保证一骨架一结构后，最终只选出 **8** 个可用骨架。程序因原先最低 80 个的约束退出，未产出新面板。筛选日志未分别给出结构重合及谱可用性各自的损失。已启动 [MassBank 新记录提取](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-202611-source)，采用 2026.11 与 4 月 15 日再发的 2026.03 两版 accession 差，保留逐记录许可与实测正离子 MS2，再按 Enveda 训练结构排重。下一轮保存实际筛选计数及符合条件的小样本。[MassBank 2026.11](https://zenodo.org/records/24082491) · [MassBank 2026.03](https://zenodo.org/records/19590880)
- MassBank 新谱提取 v1 读回两份官方 ZIP 的校验值。5,707 个新增 accession 中，得到 **3,686 张**有许可、可解析的正离子 `[M+H]+` MS2，来自 **363 个**原始 SMILES。1,099 张明确标有单一 eV。原始 SMILES 尚未排重或核骨架，前体 m/z 按档案原字段保留，其中可能含计算值。[Kaggle 新谱提取](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-202611-source)
- MassBank 外部谱准备 v1 已完成。遍历 277,566 个 Enveda 结构记录并与 MSG train/val 排重后，筛出 **110 个结构、170 张谱**。每个规范化骨架保留一个结构，39 个结构的通用 Murcko 骨架未见于 Enveda；按同来源、同仪器计，20 个结构有多个实测碰撞能量。参考库排除清单含 **10,291 个**训练标识。121 张谱的前体 m/z 与理论值相差不到 0.1 ppm，因此保留档案值时不能假定它们都是实测质量。[Kaggle 外部谱准备](https://www.kaggle.com/code/jiyujia/enveda-075b-external-massbank-202611)
- 068a 已在 Kaggle 断网 T4×2 上启动外部谱候选生成。固定参考库排除清单与原生产排序，分别统计原始候选召回、前 25 召回及进入前 25 后可改善的名次。FLARE、MIST 和 CE 比较将使用同一份候选；候选仓结构匹配另由 Kaggle CPU 计算。[Kaggle 实际候选 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pipeline) · [候选仓匹配 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-external-membership)
- 候选仓匹配 v2 在断网 Kaggle CPU 用时约 31 分钟。110 个结构中，**4** 个在 068a 主池，另有 **99** 个在基础 PubChem 仓。合计 103 个 PubChem 命中都在档案前体字段推算的 ±10 ppm 内，两个增量仓未新增命中。余下 7 个在所查质量窗内未找到，E2 的额外结构库未纳入此计算。跨 110 题的前 25 排名仍在生成。[Kaggle 候选仓匹配 v2](https://www.kaggle.com/code/jiyujia/enveda-075b-external-membership)
- MassBank 实际候选 v1 已产出排名。110 题中最终首位 **33** 个、前 25 **58** 个、各路并集 **65** 个，MRR@25 **0.33685**。PubChem 分支经过 ICEBERG 和 GLACIER 重排后，前 25 命中 **59** 个，MRR@25 **0.36715**，其中 6 个真值在最终前 25 消失；其他来源和后处理也补进了 PubChem 未召回的正确结构。39 个通用骨架未见的结构中，前 25 命中 15 个，MRR@25 **0.21781**。参考库实际移除 124,682 条比赛参考谱和 1,099 条外库查询窗内参考谱，APEX 未改排序。[Kaggle 实际候选 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pipeline)
- 在同一批候选上已并行启动 FLARE/MIST 逐题比较和 PubChem 独有结构的分步召回统计。另将 PubChem 的输出深度扩至 100，保持同一质量窗、第一阶段候选和 FPNet 权重，对比第 25、50、100 名的召回以及文档先验的影响。三项均在 Kaggle 断网 CPU 执行。[FLARE/MIST](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-flare-mist) · [分步召回](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-retrieval-audit) · [PubChem 深度](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pubchem-depth)
- 99 个主池外、已在 PubChem 质量窗中确认的结构中，PubChem 原始前 25 命中 **59** 个，各路可见候选共命中 **61** 个，最终前 25 保留 **54** 个。最终未命中的 45 个中，**38** 个从未进入可见候选，**7** 个在融合后丢失。38 个结构未召回的题中，有 **34** 题已在至少一路候选中找到正确分子式，其中 28 题的 Engine 2 候选已含正确分子式。首要缺口是同分子式候选的召回，当前输出只能定位到 PubChem 前 25 之前。[分步召回 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-retrieval-audit)
- MassBank 固定前 25 上，原排序 **0.33685**，普通 FLARE **0.24740**（5 升首、18 丢首），MIST **0.10255**（0 升首、29 丢首）。39 个通用骨架未见的结构上分别为 **0.21781、0.15918、0.09930**。不将两种重排直接接入比赛。20 个同来源、同仪器多实测 eV 的结构中，按同式组重排有 5 题可把真值升到首位，14 题可能改坏已有正确首位，这一子组对全 110 题的最大 MRR 增量为 **0.03955**。已启动仅使用明确单值 eV 的固定能量与逐谱能量余弦对照。[Kaggle FLARE/MIST v1](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-flare-mist) · [Kaggle CE 对照 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-ce)
- MassBank ICEBERG v1 在断网 T4×2 完成 2,430 个条件，预测 52.5 秒，0 失败。110 题原排序 **0.33685**，固定 CE 余弦 **0.30109**（0 升首、7 丢首），逐谱明确实测 CE **0.29456**（0 升首、8 丢首）。20 题覆盖组原排序 **0.73249**，固定及实测 CE 分别 **0.53580、0.49985**。相同余弦下逐谱 CE 对固定 CE 的全组差为 **−0.00653**，覆盖组 **−0.03594**。停止直接 CE 重排，优先扩大可供评分的同式候选。[Kaggle CE 对照 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-ce)
- PubChem 深度 v1 直接输出文档先验和 FPNet 排序，前 25、50、100 中分别命中 **59、68、75** 个。首轮误将它与经过 ICEBERG、GLACIER 重排的排序比较。两个阶段的 MRR **0.24737** 与 **0.36715** 不能用来判断设备差异。v2 已按原 T4 设备复算 PubChem；实际前 25 的比较改用原运行在两种预测谱打分前写出的 `pc_lists.json`。[Kaggle 深度对照](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pubchem-depth)
- 同阶段比较确认，CPU 扩展版与原 T4 的前 25 首位 **110/110** 相同，集合 **109/110** 相同，完整顺序 **101/110** 相同，原始 MRR 均 **0.24737**。ICEBERG 与 GLACIER 在相同前 25 上把 MRR 提到 **0.36715**，首位命中从 19 增至 32。保留文档先验的前 25、50、100 命中 **59、68、75**，仅用指纹分数为 **39、52、64**。下一步把扩大名单中属于原前 25 分子式的候选交给同一套预测谱排序，测能进入前 25 的实际增量。[Kaggle 同阶段比较](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pubchem-depth-audit)
- PubChem 深度 v2 在同一 T4 设备上复得 **59、68、75** 个命中，110 题前 25 候选集合与实际管线全部一致。已用该输出启动原预测谱排序的单独对照，将同式候选范围从 25 扩到 50 和 100，保留原 25 个公式占位，比较最终进入前 25 的实际首位和 MRR。[Kaggle 深度 v2](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pubchem-depth) · [扩大同式候选 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pubchem-forward)
- 扩大同式候选 v1 在断网 T4×2 完成，精确复现原 PubChem 排序 110/110。原深度 25 的前 25 命中 **59/110**、MRR **0.36715**、正确首位 32。原始深度扩到 50 后为 **65/110、0.38216**（+0.01501，升首 2、丢首 1，新增命中 6、丢失 0）；扩到 100 后为 **68/110、0.38640**（+0.01926，升首 3、丢首 2，新增命中 10、丢失 1）。两模型分别补算 7,032 个结构，各用 745.9 秒和 285.0 秒；实际总计约 19 分 28 秒。优先将 50 深度接入完整外部融合，判断分支收益能保留多少。[Kaggle 扩大同式候选 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pubchem-forward)
- 深度 50 的完整融合 v1 已在断网 T4×2 启动。只给原谱库分低于现有阈值的题扩大同式候选，保留原 25 个公式占位和原相对分数。实际输出与同一批输入的原完整排序 **0.33685** 对比。[Kaggle 完整融合 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pubchem50)
- 已从 Kaggle 拉取 Taeyang v2 确认，其内嵌候选生成程序与 068a 逐字节相同，包括 PubChem 和 ICEBERG/GLACIER 所在单元。v2 外层的 FR 和 P5 保留各自输入的前两名。若深度 50 通过整线验证，可复用同一处改动并检验后续融合是否保留收益。[Taeyang 原版 v2](https://www.kaggle.com/code/taeyangg4/casmi26-lb-0-495-network-fr-p5-tail?scriptVersionId=371003154)
- Taeyang v2 的深度 50 变体已打包，复用 MassBank 整线验证中的 PubChem 与预测谱改动，可从同一原版源码重新生成。外层 FR/P5 各自保留前两名，P5 仍可能挤出尾部候选。该变体尚未在 Kaggle 启动。[打包入口](pipeline_candidates/build_taeyang_pubchem50.py)
- 同一深度 50 候选上的公开 FR 对照已提交为断网 CPU 实验。用 Taeyang 原权重比较同式 FR 单项及 FPNet/文档先验与 FR 等权分数，保持原前 25 的分子式占位，并与已完成的 ICEBERG/GLACIER 排序对比。[Kaggle FR 对照](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pubchem-fr)
- 公开 FR 对照 v1 完成，共 5,155 个候选。59.3 秒推理给 108/110 题、5,061 个候选评分，其余两题保留原位。相同深度 50 下，ICEBERG/GLACIER 为 **MRR 0.38216、前 25 命中 65、首位 33**，仅 FR 为 **0.28645、60、24**，等权加入 FPNet/文档先验为 **0.32435、64、27**。等权版较两种预测谱重排升首 3、丢首 9。深度扩展沿用 ICEBERG/GLACIER。[Kaggle FR 结果](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pubchem-fr)
- 在原整线输出上比较一条固定换首规则。17 题的最终首位与 PubChem 首选是同分子式的不同结构，且两种预测谱均有可比较分数。要求 PubChem 在 ICEBERG 和 GLACIER 都严格胜出后才换首，共触发 6 题，全部已在最终前 25。升首 0、丢首 1，前 25 命中仍为 58/110，MRR **0.33685 → 0.33231**。99 题 PubChem 独有组触发 4 题，正确结构名次未变。这条规则不进入候选融合。[Kaggle 换首比较](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pubchem-dual-support)
- 保留原 25 个候选和逐题真实深度 50 补位数，用带先验:仅指纹的 **3∶1** 顺序补入同式候选。两组均给预测谱排序提供 2,405 个额外结构。排序前原先验名单包含 **65/110** 个正确结构，混合名单 **64/110**，新增 0、丢失 1。继续沿用带先验的深度 50 名单。[Kaggle 等候选数比较](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pubchem-mix)
- Taeyang 变体的预测顺序已调整。作者 ICEBERG/GLACIER 会在约 3,000 个任务的块内重排，并在块预测完成后写分；将新增候选混在块中会占用原候选的时间。新入口让两个模型各自先跑完原队列并写出分数，再用同一进程、缓存和剩余时间处理 PubChem 尾部。保持 ICEBERG **5,400 秒**、GLACIER **4,000 秒**。作者两份运行脚本的改写版本已通过编译，仍待 Kaggle 实跑。[Taeyang 打包入口](pipeline_candidates/build_taeyang_pubchem50.py)

## 已完成与实测结果

### 1. 前沿与管线调研（2026-11-18）
- 结合 `071` 深度审计，收敛为两份精简文档：[模型与资产清单](research/models-and-assets.md)、[管线信号与验证口径](research/pipeline-signals-and-eval.md)。
- 公榜复核（2026-11-19 10:42 北京）：第 1 名 `0.550`（Shehab Alshehabi）、第 20 名金牌线 `0.506`、第 30 名 `0.502`；前 30/40 名无任何公开 notebook。公开可复现前沿仍为 `0.492–0.494`（我方 `068a` 稳定 **0.492**）。

### 2. 本地制备核查（2026-11-18）
- **RDKit Murcko 兼容问题**：本机 `rdkit 2026.09.2` 调用 `MurckoScaffold.GetScaffoldForMol` / `MurckoScaffoldSmiles` 6/6 分子均报 `RuntimeError: bad any cast`。化学型/骨架分组必须在 Kaggle 远端（`RDKit 2026.03.3`）计算。
- **结构聚类稀疏度**：对 20,167 个 ≥2 谱结构抽样 5,000（Morgan2 2048-bit，Tanimoto ≥0.65 leader 聚类）得 **4,921 簇**（单结构 4,844、2–4 簇 77、≥5 簇 0，最大簇 3），结构库整体高度分散。

### 3. NX 同分异构判别器验证 → **公榜崩溃，已终止**
- `075c`（`jiyujia/enveda-075c-nx-instrument` v2 `COMPLETE`）：在 `072d` 150 分子面板上用 `ahmedberatozer/casmi26-nx-model` 做同式组重排，MRR `0.8286 → 0.9394`（`fold0 w=1.0`）。因 150 个结构均来自 `train.parquet`（`fold0` 仅留出 1 折），存在训练泄漏，只证明集成路径可用。
- **`075d`（`jiyujia/enveda-075d-nx-rerank` v3，ref `58519267`，2026-11-18 17:20 UTC 提交）**：将 `NX fold0 w=1.0` + `NX_LIB_GATE=0.9` 接入 `068a`（允许同式组内换 rank-1，commit smoke `active 0` 符合预期）。
  - **2026-11-19 读回：`COMPLETE`，public = `0.381`（相对 068a 0.492 暴跌 `−0.111`）**。
  - 结论：NX 在未见结构上替换 rank-1 严重误排（与 `075d` 原笔记历史注记 `e6 0.472→0.414` 同类崩塌）。**NX 路线正式关闭**，相关本地产物与脚本已归档清理。

## 资源与下一步

- **配额（2026-11-19 MCP 核实）**：当前账号周 GPU 已用 `121,537s / 108,000s`（0h 剩余），刷新时间 **`2026-11-21T00:00:00Z`**（北京 11-21 08:00）；全部历史提交均已 `COMPLETE`，无在途任务。
- **下一步**：GPU 刷新后若继续推进，按 [plan.md](plan.md) 先建 `075b` 化学型整组留出面板与零成本信号审计，再单臂评估 `MIST MSG` 候选打分，不恢复已关闭方向。

## 2026-11-19 晚间 深度研究增量

本次读回最近10项提交，075d=0.381、074a=0.492、072e=0.487、083a=0.488、082d/f3/g=0.492均COMPLETE；配额仍为已用121536.588s/允许108000s、reserved0，刷新11-21 08:00北京。本次未启动GPU、训练、推理或新提交。

### 新证据和判断更新

1. **重建旧面板的选样范围**：仅读train第0个rowgroup的身份/来源/adduct等标量列，未读取峰数组。072d150身份681谱；元数据模拟075c每极性前8谱得到839条原始选择、59身份改变、额外158谱（不等于清峰后有效输入）。原始857条匹配谱中63身份同极性多adduct。072d的244/681谱为二聚体，150/150不能外推池外召回已饱和。未由这些风险认定NX掉分原因。
2. **多CE可用接口**：可见test400分子中279在同adduct有≥2不同单值CE，915谱单值、298多值；只作接口覆盖，不作泛化证明，多值谱不拆观测。
3. **新模型FLARE**：作者HF发布83.2MB checkpoint、MSG训练配置、局部peak↔atom late interaction，README声明MIT。预测公式缺失时回填标签公式、负离子预处理疑点和checkpoint训练manifest未核均明确记录；首轮正离子，真实公式仅作单独诊断。
4. **0.495公开配方新证据**：Taeyang平台公开submission58449454=0.495；具体notebook关联依作者说明，Alex/Arsen复现ref是公开自报。Taeyang v2与Alex v3的19 code cells相同，Zohrab复制文案不另计独立复现；FR三项训练派生输入生成链尚未定位。
5. **前沿排除边界**：MARASON原MSG流程使用NIST模型补CE，不是现成干净MSG资产；AIMe/DeepMS2Reasoner由NIST训练、MSG仅评估。ICEBERG2.1加速不自动等于新信息，需区别msg_all与msg_simulation。

### 产物和人工核对

- 本次研究并入当前075：新增[框架决策](research/next-frameworks.md)，更新计划、两份专题、README/轮次索引及归档中的过度归因；不恢复已清理方向。
- 机器证据：[旧面板和CE复算](../../data/references/075-panel-recheck-2026-11-19.json)、[公开前沿](../../data/references/075-public-frontier-2026-11-19.json)、[作者模型资产](../../data/references/075-model-frontier-2026-11-19.json)。原始小源码哈希/版本在JSON，未下载模型大权重。
- 工具与环境：本机Python3/pyarrow读取元数据；`git show pre-cleanup-2026-11-19:...`只读历史源码；Kaggle MCP公开数据/代码/提交/配额；GitHub、作者HF与论文主源。手动使用validate-data的比较口径核查，未创建测试文件/CI。
- 两个他人复现ref的私有提交详情被接口权限拒绝，已停止该路径；仅使用公共团队提交接口确认Taeyang0.495，不将自报视作平台逐ref确认。
- 交叉审阅修正了raw谱数、N1c并未冻结leader、union包含final25、pilot不能证明大幅收益等限定。已核算0.028净提升的条件尺度：无伤害时rank2→1需5.6%分子，rank5→1需3.5%；这不是本赛预期收益。
- 最后手动执行`git diff --check`、本轮Markdown本地链接存在性检查和3份JSON解析，均无异常；校核上述MRR算术。没有运行模型或把这些文档检查写成测试体系。

### 下一步

按更新计划先固定等输入的075b与训练曝光清单，再单臂测FLARE/MIST及CE条件前向评分，量候选oracle空间与实际逐分子ΔRR。原“零成本”仅指导出改动轻，运行底座仍消耗GPU。研究报告完成、新方案效果未测，当前最佳仍是068a的0.492。
