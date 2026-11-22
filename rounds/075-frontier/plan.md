# 075 前沿计划：候选局部判别与条件化多谱

2026-11-22 更新。本项目最高公榜为复现 goodpjw 原版的 **0.498**（`58695195`）。外部 MassBank 实验使用 `068a`（公榜 **0.492**，`58412029`）。固定异式换首在第二批正离子谱上降分，已取消比赛提交计划。逐题结果与运行链接见 [实验进展](progress.md)。

## 当前实验

- 新 E180 96 题的 C34 前 25 漏题共 **37**，其中 **27** 个真结构在实际质量窗内被粗筛淘汰。用原首式全窗补召回后，两臂各保留原前 25 和 **2,362** 个追加候选，覆盖 **59/96 → 80/96**。用原 ICEBERG 与 GLACIER 截回原 25 个分子式槽位，原 C41 为 **0.46920 MRR / 38 首 / 59 前25**，扩池保首为 **0.56556 / 38 / 80**，扩池允许换首为 **0.63240 / 50 / 80**。在未用的 121 个 E180 结构上复核，接入已实得 **0.498** 的完整流程。[首式等量补召回](https://www.kaggle.com/code/jiyujia/enveda-075b-e180-d3-first-formula-recall/versions/1) · [物理重排 v1](https://www.kaggle.com/code/jiyujia/enveda-075b-e180-d3-first-formula-c41/versions/1)
- FraGNNet D3 用 **2,000** 个结构、**4,485** 张单档实测谱做骨架分组的拟合与内部留出。内部验证谱余弦从第 16 轮 **0.47503** 升到第 30 轮 **0.53135**。固定各批真实名单，原 132 题的原 C41、第 16 轮、第 30 轮 MRR 依次为 **0.57987 / 0.58674 / 0.59376**；另 96 个此前未用结构为 **0.46920 / 0.46226 / 0.46052**，原本正确首位 **38** 降至 **36**。结束当前等权融合。[从零训练](https://www.kaggle.com/code/jiyujia/enveda-075b-fragnnet-e180-d3-fit/versions/1) · [续训](https://www.kaggle.com/code/jiyujia/enveda-075b-fragnnet-e180-d3-resume/versions/1) · [两批联合评分](https://www.kaggle.com/code/jiyujia/enveda-075b-fragnnet-e180-d3-joint/versions/1)
- 独立 timsTOF E180 固定 192 题中，完整 PubChem 质量窗包含 185 个真结构。恢复单谱实测 CE 后完整重做检索，带先验 MRR **0.28275 → 0.42973**；使用 655 张已知 CE 的谱再提高到 **0.45431**。单谱与多谱各命中 **134** 道前 25，交换的三个新增结构均属于原单谱首位的分子式。原流程已读取实测 CE 并支持多谱。[单谱完整检索](https://www.kaggle.com/code/jiyujia/enveda-075b-e180-full-ranked-ce/output) · [多谱完整检索](https://www.kaggle.com/code/jiyujia/enveda-075b-e180-full-ranked-multice/output)
- 在预先选定的 60 题上固定完整多谱名单。早期单张谱加默认能量的物理分配单谱指纹时增 **0.01771 MRR**。用比赛原生逐谱 CE 后，单谱指纹 **0.57147**，多谱指纹 **0.59165**；另 132 题仍按早期口径比较也未复现单谱优势（**0.57161 / 0.57330**）。保留原生多谱指纹。[原生物理重排](https://www.kaggle.com/code/jiyujia/enveda-075b-e180-native-physical-ce-replay/versions/1) · [独立 132 题早期口径](https://www.kaggle.com/code/jiyujia/enveda-075b-e180-confirm132-comparison/versions/1)
- 在开发 60 题上，C34 同式扩深并保留原物理首位，从 **0.59165 / 31 首 / 45 前 25** 到 **0.60428 / 31 / 48**。规则冻结后，在独立 132 题仅从 **0.57987 / 68 / 89** 到 **0.58139 / 68 / 90**，未见骨架持平。每个物理模型新增 3,103 个请求，接近原算量一倍；保留接入代码，暂不提交该版本。[开发 60 题](https://www.kaggle.com/code/jiyujia/enveda-075b-e180-native-depth50-keephead/versions/1) · [独立 132 题](https://www.kaggle.com/code/jiyujia/enveda-075b-e180-confirm132-keephead/versions/1)
- 按加合物分别合峰在比赛 Enveda-180 的两折多加合物结构上改善主库排序，折零 256 题 **+0.00940**、折一 512 题 **+0.01235**。单独接入公开原版 C34 后，正式公榜 **0.498 → 0.488**（`58695253`）。后续使用原版合峰逻辑。[合峰版 v1](https://www.kaggle.com/code/jiyujia/enveda-casmi-0498-adduct-merge/versions/1)
- 110 题中 PubChem 原前 25 命中 59 题，前 100 命中 75 题。原前 25 经两种预测谱打分后，MRR 从 0.24737 升到 0.36715；各来源融合后为 0.33685。
- 在原前 25 的分子式占位内加入同式候选后，深度 50 的 PubChem 前 25 命中 **65** 题，MRR **0.38216**，深度 100 是 **68** 题、**0.38640**。选择深度 50 接入完整融合，用同一批谱测最终前 25 和 MRR。[Kaggle 扩大同式候选](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pubchem-forward)
- 深度 50 整线将前 25 命中从 58 增至 60，MRR 从 0.33685 降至 0.33415。PubChem 新增的 6 个正确结构中，只有 2 个进入初融合，后来保留到最终名单。同式尾位替换最多回收 +0.00085 MRR。[整线分解](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pubchem50-comparison) · [同式尾位替换](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pubchem-tail-fuse)
- 深度 50 的新增首选仅在原首位异式且两模型都支持时插首。110 题触发 5 次，新增 1 个首位、无丢首，MRR **0.34586**（+0.00901）。独立 101 题有 4 对可比较，**0 次插首**，MRR 维持 **0.81023**。[首批结果](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pubchem50-novel-leader/output) · [独立结果](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-202603-pc50-verify/output)
- 固定异式换首在用于设计规则的 110 题上 **+0.02391**（3 升首、0 丢首）。在独立的 101 题上，原 101 题均命中前 25，规则触发 8 次，**0 升首、3 丢首**，MRR **0.81023 → 0.79026**（−0.01997）。停止固定双胜规则。[110 题异式排序](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-pubchem-cross-formula) · [101 题独立对照](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-202603-cross-only/output)
- 第一批原主库覆盖 **4/110**，第二批覆盖 **99/101**。第一批已选完既有 2026.11 增量中的 110 个合格精确骨架。后续要评价主库外结构召回，需要独立采集来源中、已确认在 PubChem 质量窗内的正离子结构。[两批主库覆盖比较](https://www.kaggle.com/code/jiyujia/enveda-075b-positive-pool-composition/output)
- 第一批的 38 道主库外漏检题中，16 道能在原打分前 100 找到，另 22 道仍缺。核对原数据库及粗筛，**17 道已过粗筛，5 道被截**。14 道正确式已有原 PubChem 占位，只有 4 道靠重分配现有名额有机会找回。等预算轮取的宽池命中 **71 → 70**，两个新增经预测谱后均未进前 25。MIST 等权检索 **71 → 61**（新增 1、丢失 11）。停止这两套替换规则。[粗筛归因](https://www.kaggle.com/code/jiyujia/enveda-075b-pc-coarse-audit/output) · [同式轮取](https://www.kaggle.com/code/jiyujia/enveda-075b-pc-formula-round/output) · [MIST 等预算](https://www.kaggle.com/code/jiyujia/enveda-075b-pc-mist-equal-rank/output) · [FR 110 题](https://www.kaggle.com/code/jiyujia/enveda-075b-pc-coarse-fr-110/output)
- 按整线各来源并集计，45 道漏检中 **41** 道已经看到正确分子式，优先改同式结构检索。用 E2 首选式追加时，等量加入 38 结构使候选命中 **65 → 66**，不限式对照仍为 **65**。唯一新增 U110 在两种预测谱分上意见相反，按当前同式插入方式两臂 MRR 均 **0.33685**。DreaMS 官能团粗筛在过粗筛仍漏检的 17 题也无新增。结束这两个具体配置。[并集漏检分解](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-union-misses/versions/4) · [E2 实际名次](https://www.kaggle.com/code/jiyujia/enveda-075b-e2-formula-physical/output) · [官能团检索](https://www.kaggle.com/code/jiyujia/enveda-075b-dreams-class-recall/versions/2)
- FLARE 缩到同式前两名仍有 **0 升首、7 丢首**，MRR **0.30503**（−0.03182）。通用骨架未见组的 MRR 持平。这一权重的直接换首不再投入。[同式前两名](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-flare-second)
- FR 等预算宽池持平 **71 → 71**，但经两种预测谱后的 110 题 PubChem MRR **0.38640 → 0.37047**，首位 **33 → 31**。停止按 FR 固定替换一半同式候选的方案。U019 从 PubChem 第 5 到核心流程第 4，末段 FR 后第 7，P5 后最终 **第 13**，原对照未命中。[110 题配对结果](https://www.kaggle.com/code/jiyujia/enveda-075b-pc-fr-paired-merge/output) · [U019 最终名次](https://www.kaggle.com/code/jiyujia/enveda-075b-pc-fr-final/output)
- 固定前 25 的全局权重从一批外部谱迁移到另一批时明显降分，Engine 1 与 PubChem 的系数随训练批次反号。直接学习单题换首收益后，211 道正离子跨批 **−0.01141**，独立 47 道负离子 **−0.01064**，首位均没有新增。结束这两套重排，保留各来源原顺序。[全局权重对照](https://www.kaggle.com/code/jiyujia/enveda-075b-listwise-stored/output) · [逐题换首对照](https://www.kaggle.com/code/jiyujia/enveda-075b-lead-gain/output)
- 跨能量碎片差异的同式前两名实验覆盖 18 题。等输入的逐谱余弦维持 **0.33685**，差异碎片在全 110 题降至 **0.32776**（0 升首、2 丢首）。结束这一差异分数的直接换首实验。[差异碎片配对](https://www.kaggle.com/code/jiyujia/enveda-075b-massbank-ce-pair/output)

## 原则与禁止项

- **本阶段不重训/微调比赛原生模型，不恢复NX**：这是当前预算策略。既往失败提示过拟合和错配风险，但075d本身没有训练，掉分原因也未分离验证，不能当作所有适配学习已被否证。
- **不将泄漏面板或单次数字当泛化**：可见 400 test 为 `train.parquet` 副本（仅作 smoke）；`072d` 只留出谱未留出化学型。所有新信号须先过 **整组化学型/骨架留出** 面板，再决定提交。
- **不新建测试文件/CI 门控**，评估均手动发起，三层状态分开记录（代码成功 → 远端完成 → 公榜提分）。

## 075b 原计划

1. **075b 化学型整组留出面板（先决条件）**：在Kaggle远端RDKit `2026.03.3`按规范身份/骨架分组；U面板保留候选结构、移除其参考谱，并核查各监督模型训练曝光，P/G按实际离线池成员关系另建。后分组不能消除旧权重已见结构；来源未知或已见的结果单列，不能称全链未见泛化。S面板另测已见结构独立谱检索。
2. **共用query并缓存证据**：所有模型用完全相同的逐谱输入，禁止按极性混合不同adduct后沿用首谱precursor。导出`Engine 1 scs`、物理连续分、`Engine 2 frag_ad`、双神经分歧与真实候选并集；代码导出改动轻，但底座重新运行仍有GPU成本。S/U/P/G及模型训练曝光分别报告，答案缺失题留在分母。
3. **外部预训练候选-谱打分器（不重训）**：
   - 新增 **FLARE**现成checkpoint，与**MIST MSG**分别单臂比较；首轮支持正离子，以推断公式评分，关闭FLARE预测公式缺失时回退真值标签的逻辑。先核checkpoint来源/输入一致性，不直接升位提交。
   - **ICEBERG 2.1/GLACIER MSG**先比对底座checkpoint与CE训练口径；版本加速不等于新信号。CFM-ID4保留负离子备选。
4. **CE与差异碎片评分**：确认所选权重确实使用CE，比较原余弦/entropy、正确逐谱聚合、跨CE差异评分三臂。仅使用实际同adduct单值CE观测，多值谱不拆成伪谱；便宜打分保留top25/来源并集，再按残差选择昂贵候选宽度。
5. **完整MSAlign与融合**：DreaMS×MoLFormer须重训匹配的投影头；只有现成通道显示错误互补才考虑这项新头训练，不能标成原版即插即用。低容量融合使用独立校准折的连续分数，不对单一小面板盲扫。
6. **条件候选扩展**：如果P/G或来源截断存在足够oracle空间，再做宽MIST检索/MS-GPT生成；先量新增正确身份和最终MRR，避免只塞尾槽。

## A线升级与高投入备选

- Taeyang 公开方案原版 v2 已经复现公榜 **0.495**，使用作者公开推理资产。FR 的三项派生训练输入仍待补全生成过程，复现推理分数与自行重建 FR 训练分别记录。[原版 v2](https://www.kaggle.com/code/taeyangg4/casmi26-lb-0-495-network-fr-p5-tail?scriptVersionId=371003154)
- goodpjw 的公开 **0.498** 配方得到 Alex 独立同代码复现，Kaggle 确认两份提交 **58655983、58677356**。`jiyujia` 的完整余量约 24 小时，原 v3 已在断网 T4 启动。[公开原版](https://www.kaggle.com/code/goodpjw2008/enveda-casmi-stack-aprior-uniq-lb-0-498/versions/3) · [Alex 复现](https://www.kaggle.com/code/alexchilton/casmi26-0-498-what-moved-the-board-66-arms/versions/4) · [本队原版 v1](https://www.kaggle.com/code/jiyujia/enveda-casmi-0498-aprior/versions/1)
- MARASON式参考谱增强是不同机制，但原MSG流程含NIST模型CE伪标签，无已核实直接可用的合规权重；仅作以后重建训练链的备选。本次不启动长训练或GPU。
- 首次50–100结构只用于接入/吞吐pilot；策略锁定后扩大独立留出并报告结构簇区间，不能凭pilot正号证明+0.02级收益。先设有限研究预算再按实测外推9h，不承诺未经实测的T4时长。

## 止损标准

- 以逐分子`ΔRR`、收益/损失总量及独立结构上的不确定性为主；`fix/harm`次数仅作补充，不再用固定1.5替代MRR。小样本跨零记未定，锁定面板持续无增益才停止具体配置，不公榜盲调。
- 任何模型/资产若发现含 NIST 及其衍生训练权重或不可复现中间产物，立即剔除。
