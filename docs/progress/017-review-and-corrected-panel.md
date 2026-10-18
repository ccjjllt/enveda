# 017 — 016复核与修正：实际进展

日期：2026-10-18（北京时间）。接手`2383112`，初始工作树干净。对应[017计划](../plan/017-review-and-corrected-panel.md)。交接里的同分子对照、全库重训和相对特征已在[018进展](018-same-molecule-reference.md)做完；本页保留017当时的复核结论。

## 结论与实际完成

**暂不提交新的submission。** 已修复016两项关键输入/基线错误，保留其原产物，用同一400面板及原冻结模型完整重算两种参考配置。跨来源外部参考B的direct增量得到确认：MRR0.936458→0.951625，Top1 180→184；但全库参考缺失A仍明显退化，自然缺库MRR0.765440→0.734150（direct）/0.738397（组合）。当前不是可全域替换v9的成品。

本轮没有训练新模型、GPU运行、push、submit或改变最终提交选择。Kaggle CLI只读确认v9r2/ref57028399与v10/ref57293821仍COMPLETE/public0.323。下一项核心工作是**在实际全库参考条件下重建训练证据，学习区分可信同结构证据与同式错误候选，并验证证据不足时保留原v9的策略**；不在这400例上扫描quality阈值。接手入口见[017交接](../handoff/017-corrected-panel-and-next-step.md)。

## 复核发现与修复

1. **P1，ICE读取键错位。** 原`build_016_panel_features.py`用canonical查以SMILES为键的回包；原两套产物均36228×10，ice_present/ice_z全0。实际400分子/30980候选记录与输入对齐，其中280分子、21869个有限fused_merge可复用。修复为读取record内`canon`并复用015 `frozen_v9`。
2. **P1，FP-only误标为v9。** 原脚本直接使用FP `order_top100`，遗漏FP＋ICE z融合、有ICE先排及缺失续后。正确排序改变277/400完整序、220/400前50集合，合计1544个新前50候选；因此已重新计算全部候选参考特征，未只补两列ICE。
3. **候选身份修正。** 8个分子中的10个MB539候选未按v10规则规范化。新增`repair_017_panel_inputs.py`生成017输入/ICE副本；候选SMILES及位置、truth命中向量、所有非canon预测值不变，无重复，不需要重跑GPU。
4. **缓存两项缺陷。** `np.unicode_`在NumPy2被移除；临时路径不以`.npz`结尾又被`savez_compressed`追加后缀，原子rename找不到源。raw分数赋值发生在缓存异常之前，因此保存分数有效，`status=ok`/成功计数不能代表无异常。未来bundler只修新复制worker为`np.str_`及一致的临时后缀；未改旧016包。仅pin NumPy不能修复两个问题。
5. **统计和追溯修正。** bootstrap点估计改为观测配对均值，CI仍20000次分子重采样；保存逐候选概率、完整排名、输入/模型/代码SHA、参考来源与质量。原`fit_015_direct_full.py`恒真的`or True`移除，实际已存模型参数核对正确，未重训或覆盖权重。

修正后基线（每组200）：跨来源FP .954792→真实v9 .936458，Top1 185→180；自然缺库FP .743427→真实v9 .765440，Top1 135→138。ICE并非在两域都优于FP，但所有新排序比较现在基于真实v9融合口径；这不授权按这两个truth定义面板标签选择生产分支。

## 修正后的冻结模型结果

B允许同结构参考；A在选2000参考前全局移除400目标canonical。`外部版`只对cross_source组剔除enveda来源，自然组参考与全来源版相同；它是分组评价条件，**不是可凭隐藏真值选择的生产路由**。

| 面板 / 参考配置 / 场景 | 原v9 | LR基线 | LR direct | LR组合 |
|---|---:|---:|---:|---:|
| 跨来源 / 全来源 / B | 0.936458 | 0.936208 | 0.947458 | 0.952458 |
| 跨来源 / 全来源 / A | 0.936458 | 0.936208 | 0.911375 | 0.911875 |
| 跨来源 / 仅外部 / B | 0.936458 | 0.936208 | **0.951625** | **0.951625** |
| 跨来源 / 仅外部 / A | 0.936458 | 0.936208 | 0.935375 | 0.933958 |
| 自然缺库 / 全来源 / B | 0.765440 | 0.765817 | 0.764700 | 0.764563 |
| 自然缺库 / 全来源 / A | 0.765440 | 0.765817 | **0.734150** | **0.738397** |

关键配对结果（描述性95%CI，不覆盖长期方法选择偏差）：

| 比较 | ΔMRR | 20000次分子bootstrap 95%CI | Top1 |
|---|---:|---|---|
| 跨来源外部B，direct vs v9 | +0.015167 | [+0.005833,+0.026667] | 180→184 |
| 跨来源外部B，组合 vs direct | 0 | [0,0] | 相同 |
| 跨来源外部A，direct vs v9 | −0.001083 | [−0.003333,+0.000333] | 180→180 |
| 跨来源外部A，组合 vs v9 | −0.002500 | [−0.005417,−0.000417] | 180→180 |
| 跨来源全库A，direct vs v9 | −0.025083 | [−0.040833,−0.011500] | 180→171 |
| 自然缺库A，direct vs v9 | −0.031290 | [−0.048030,−0.016475] | 138→128 |
| 自然缺库A，组合 vs v9 | −0.027044 | [−0.042504,−0.013391] | 138→130 |

不能把外部A的direct与组合都称“中性”：组合下降区间未包含0。全来源跨组B组合相对direct +0.005，CI下端0；外部B二者逐分子RR完全相同，尚未形成有说服力的analog独立增量。

## 缺参考退化：固定归因与分布核查

不调参，把冻结direct模型的direct_quality输入统一置0，仅作事后反事实诊断。其输出几乎回到LR基线；在A中，相对这个zero-direct对照，直接特征额外造成跨来源12个、自然组16个分子退化。38个越过truth的错误候选中37个最佳参考来自enveda-180、1个pluskal_ms2，多为原v9第2–5名。

误支撑并非只有低质量单峰：错误quality中位跨来源约0.391、自然组约0.429，最大0.803；匹配峰数中位20/28，最低8/6；存在cosine约0.997、22峰匹配的错候选。因此没有依据认为简单提高绝对quality阈值就能解决同式候选误判，也不能把全库直接检索原理判为无效。

A场景中，每query错误候选具有正direct_quality的平均比例，015抽样训练NP/G1为**3.14%/4.85%**，017全库跨来源/自然组为**53.55%/47.22%**；query至少一个正质量错误候选比例从54.8%/77.6%变为98.5%/90%。外部版跨来源降为8.43%。两个面板本身领域不同，不能将所有变化因果归于库大小；但5万参考训练与251万参考应用之间存在明显证据分布不匹配。下一轮必须先对齐训练与实际参考条件，不能沿用015权重直接扩库后只调阈值。

## 数据独立性与可复用资产

- 新400 canonical与已记录旧800排除正确，未发现truth注入候选。补读旧NP/G1/heldout/R3等所有谱后得到3991 unique hash，原排除清单漏1157；新400查询仍与更完整旧hash集合零碰撞。不能宣称原清单本身完整。
- reference层实际只排除新400代表谱hash，没有应用全部旧query hash；本次保持016已运行参考语义，不暗改成新实验。hash不同不保证采集/重处理独立。
- 跨来源200中199在完整索引具有同极性外部同结构参考；经过最近2000及top50限制，外部B 198个truth进top50，197有同结构pool参考，193正质量。B并非每例必有参考。
- 自然缺库B仍恰有100个同结构其他谱，只有A满足全部目标结构参考移除；原分组指没有外部来源，不是没有所有可用参考。
- 两版A均无truth参考；新面板此前未用，但016已经参与过决策，本轮是纠错复算，不重新命名为盲集。公开FP/ICE可能预训练见过结构，不能称encoder未见。
- 单query谱为016冻结协议，尚未验证v10真实多谱整链。Colab RDKit2026.09.1与Kaggle v10 2026.03.6不同；CE预处理/merge路径人工核对一致，不能据此保证跨环境神经前向逐值相同。
- 旧Colab入口的前两例干跑均为不支持的负模式，只证明输入可解析，未覆盖实际ICE/cache路径；全量回包另有已核实有效预测。下次GPU短检查应至少含一个受支持正离子，检查有限fused及缓存重载，不把0 jobs当作前向成功。

## 运行、产物与人工核对

实际在M5本地CPU运行，Python3.11.16、numpy2.4.6、RDKit2026.03.6；预测用`.venv-015`的sklearn1.9.1/scipy1.17.1，参数及模型SHA保持015全量fit。无GPU/新模型训练。

```bash
/opt/homebrew/bin/python3.11 scripts/repair_017_panel_inputs.py
/opt/homebrew/bin/python3.11 scripts/build_016_panel_features.py --inputs-path artifacts/017-panel-inputs/inputs.json --ice-path artifacts/017-panel-inputs/ice_scores.json --output-dir artifacts/017-corrected-features
/opt/homebrew/bin/python3.11 scripts/build_016_panel_features.py --inputs-path artifacts/017-panel-inputs/inputs.json --ice-path artifacts/017-panel-inputs/ice_scores.json --cross-external-only --output-dir artifacts/017-corrected-features-ext
.venv-015/bin/python scripts/run_016_panel_predict.py --input-dir artifacts/017-corrected-features --diagnostic-zero-direct
.venv-015/bin/python scripts/run_016_panel_predict.py --input-dir artifacts/017-corrected-features-ext --diagnostic-zero-direct
kaggle competitions submissions -c enveda-CASMI26-molecule-id-mass-spectra --csv
```

两版各400 query、36228×10、有限值完整，280query恢复ICE、原top100共21869个有限fused，top50两场景共25490行ice_present=1。特征75.4/82.7秒，预测含报告4.58/4.52秒；原772.7/814.7秒特征耗时因按极性/场景预过滤有序索引而缩短。

- 两种四例probe共16个query×scenario的最近2000行与原Python循环逐行一致，质量距离边界tie按原train row处理。
- 两版各33130条与原016同canonical候选对应的六个参考特征，最大差0，证明加速未改变这些候选证据。差异来自正确ICE、排序/候选集合及已声明的MB身份修复。
- 独立重算两版全部基础特征、候选概率、完整排序、MRR/Top1/25、配对CI及SHA均一致。详见`artifacts/017-review/corrected-{full,external}-review.json`和`corrected-results-review.md`。
- 新复制worker在NumPy2.4.6手动缓存写/读/覆盖成功，无tmp遗留；旧bundle未改、未执行future bundler的前向/打包。后续bundler默认输出改为`artifacts/017-colab`并支持`--output-dir`，避免覆盖016旧包。
- 原016文档和精简记录加显著纠错说明，未删除历史指标；完整新指标/源码与输入hash另写`data/references/research-results-017.json`。没有创建测试文件、测试框架或CI。
- 两版运行都显式传入017输入/ICE路径；运行结束后将builder的CLI默认值也改为同一017路径，避免默认命令再读旧MB身份。产物保留运行时源码SHA，精简记录另存当前源码SHA及这项默认值变更说明，不冒充重新计算。

## 下一步与提交条件

优先按完整索引、相同过滤和候选范围重建**原015训练分子**的B/A特征，并保留同式错误候选；在训练侧校准可信度（相对竞争候选的优势、不同参考的一致性、信号保留与错误匹配密度），确保没有证据时能保留原v9。先报告原固定模型在同分子稀疏/完整参考条件的差异，再做一个预先固定的新模型对照，避免把“换面板”和“换参考库”混成同一原因。

在查看新方法结果前冻结另一批未参与决策的canonical及来源确认集合；016/017只作诊断开发。提交前须用不依赖truth/分组标签的统一推理策略，证明主要参考缺失场景不再明显退化，随后集成一个清楚的变化到新Kaggle离线入口并实际完整跑通。当前没有新提交入口或可报告的新public分数。

## 交接侧重调整（2026-10-18，按用户要求）

交接已改为研究导向：保留直接检索增益、同式误匹配及参考分布变化的证据，下一轮明确分为“同分子稀疏/全库对照→仅换参考条件重训→相对候选证据与特异碎片研究”。运行、hash、重放等细节留在本进展，交接简短引用，不再以门控或测试清单组织后续工作。该偏好已写入项目协作约定；本次为文档调整，实验数值与模型保持原记录。
