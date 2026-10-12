# 008 — 新研究与提交时机审查：实际结果

日期：2026-10-12（北京时间）。对应[下一步计划](../plan/008-submission-readiness-and-recall.md)。本轮只读查询Kaggle、审代码/日志，并手动复算已有产物；未训练、未GPU推理、未push或submission。

## 决定

**目前建议继续一轮有明确目标的研究，不提交当前产物。** 最新有效提交仍为v9r2，ref57028399、版本356902571、public **0.323**；10月12日Kaggle MCP返回的提交列表没有新有效版本。[只读查询快照](../../data/references/submission-decision-2026-10-12.json)。

实质进展是：D落实了先前指定的5项修正并保存OOF；外部观察集从模板变成50分子/59谱；FRIGID/MIST权重加载阶段已实跑。这三项分别属于实现、数据和加载成果，尚未形成适合提交的新方法。

## D：结果可复现，但训练与推理还有错配

新增手动入口 [analyze_008_r1d_evidence.py](../../scripts/analyze_008_r1d_evidence.py)，从候选特征、已保存模型与OOF结果重建。16组指标、改善/恶化及20,000次分子bootstrap与汇总一致；α=0和全无库时完整排序均为250/250一致。10个模型的训练折四类增强均值/标准差可精确复算，未发现标准化读入验证折的证据。

| 集合/场景 | A MRR | D α=0.25 | D α=0.50 |
| --- | ---: | ---: | ---: |
| NP 有参考 | 0.668997 | 0.729408 | 0.755970 |
| NP 真值参考移除 | 0.668997 | 0.641915 | 0.611737 |
| NP 移除riken来源 | 0.668997 | 0.726075 | 0.753653 |
| G1 原场景 | 0.907022 | 0.912117 | 0.915784 |
| G1 移除enveda-180来源 | 0.907022 | 0.907022 | 0.907022 |

NP压力损失分别−0.027082/−0.057260，11/23个原Top1被改错；当前D不部署的决定成立。来源表这次确实重新构造且不含目标来源，不过此移除情景已参与训练增强，不能称未见来源泛化。G1真值仍全部无参考，其pressure与seen相同。仍为两域分别训练模型。

**剩余评分函数错配：** `run_D.py:193–205`训练所有候选的未门控标准化残差，`:97–100`推理才在`has_lib=0`时强制清零。NP10305、G14252条原场景无库候选的原始残差都非零；G1无库真值为+0.0977～+0.2154，却在部署式评分时被抹掉。缺失增强训练的评分函数与实际评价不相同，不能据本次失败否定一致门控的残差方法。

此外`:117`始终按v9为无forward候选排序，忽略库残差；NP34/G1 21个无forward分子因定义而永远不变，其中NP有33个真值实际上有参考。α=0复现基线，不要求α>0仍锁死这部分排序。下一步只给对齐评分函数和可学习候选对的D2一次固定对照，不继续扫描α。

## R3扩池：0→0.64是oracle插入，不是候选发现成功

新增手动入口 [analyze_008_pool_pilot.py](../../scripts/analyze_008_pool_pilot.py)，只读已存JSON，不运行FP。

- `r3_pool_pilot.py:143–169`读取5个B评价分子的已知SMILES/公式，核对其canonical等于truth，然后仅加这5个结构。没有无标签的外部候选搜索/枚举/生成。
- 5个B排名5/2/2/1/1、MRR **0.64**可复算，含义是“正确结构已进入池时的条件排序”。不代表推理时能发现这些结构。
- 插入后的canonical窗为22/3/14/29/12，4个窗≤25。因此4个Top25在模型打分前已保证命中，原“找回至少2个就CONTINUE”没有有效区分能力。
- 每个新增结构只进入自己的B窗，没有进入其余9个query。A质量约760–1021，B约142–238；“A不退化、零新增干扰”不能证明实际扩库安全。
- 原产物写着下一步“仍只加约束内真值”，该方向需要撤回。可以保留oracle诊断，但不能扩大它并作为提交依据。下一步应固定外部结构库并召回真实干扰候选。

50分子数据确已落地，按定义分成25A/25B；C覆盖A全有/B全无来自选样规则，不是新召回方法的成绩。`coverage_50.json`按质量误差/pool index截100，本批最大窗口84，等同整个质量窗；不是FPNet top100，更不是ICE融合后的最终Top25。

首批10例MRR0.900也只有FP原head＋检索诊断，没有完整v9融合；两谱分子采用logits平均而非v9代表谱，6/11谱CE回退25eV。这些限制必须带入方法对比。50集中5张谱的adduct未知/未识别，把precursor当QM的窗口不能当准确中性质量窗；应先修正或单列。272条discards是历史筛选记录，其中35条后来已选入，不应称当前最终丢弃272条。

已记录逐条来源、许可字符串、峰SHA与unknown暴露信息是实际产物；本轮未重新作许可法律判断，也未从发布时间推断保证未见结构。本地B仅表示当前712K候选库外。

## MIST：SAFE调用错误，完整谱条件仍未运行

读取仓库脚本和用户下载的实跑notebook日志，确认DLM Sampler已初始化；MIST两套权重均记录missing=0/unexpected=0；脚本约38.2秒停于SAFE检查。没有MIST encoder forward、de novo或公式生成结果。

`run_mist_single.py:444`使用`SAFEConverter(slicer=None).encoder(truth_smiles)`。[SAFE 0.2.1官方源码](https://raw.githubusercontent.com/datamol-io/safe/0.2.1/safe/converter.py)明确`slicer=None`产生空切分，而`allow_empty`默认False，因此报错不支持“B04太小”的解释。保持同一调用而换大分子也不能解决。官方FRIGID的无切分编码使用`allow_empty=True`，条件生成不要求先编码真值SMILES。[同pin sampler](https://raw.githubusercontent.com/coleygroup/FRIGID/d242a2c06d671e0fb6fbc1a5b31f0e19e24b456d/src/dlm/sampler.py)

`run_arm:255`仍硬编码`fingerprint=None`，峰数组没有进入encoder。因此即使越过SAFE并显示GEN_OK，也仅验证公式条件生成，不能称完整MIST/FRIGID谱条件成功。首批5个B全为负谱，新增50例中已有支持正模式的候选；选样应按模型输入契约，不能为躲SAFE错误按分子大小换样。

其他待修口径：日志缺`data/len.pk`而回退随机长度；按过滤后返回数循环凑B不能反映实际请求槽位；脚本以“检索miss”为由令union MRR等于生成MRR，没有真实联合排序依据。均尚未产生成绩，不能补造效果。

## 下一次提交的具体时机

最快路径是用现有缓存完成一次训练/推理一致的D2；并行把R3变成无真值输入的候选召回实验。新方案与完整v9在同口径下有可解释收益、重要已知退化得到处理、并完成离线notebook后，就可做一次验证性submission，不必等待全部研究完成。public提升无法保证；保留v9r2。

目前不提交D、不提交oracle插入池、不以SAFE_BLOCKED改生成管线。具体有限实验和文件职责见[008计划](../plan/008-submission-readiness-and-recall.md)。

## 产物与人工核对

本地M5、Python3.11/NumPy，手动执行两个新增脚本；D审计seed=20261004、bootstrap20,000次，沿用已存模型而不重新拟合。输出`artifacts/008/r1d-audit.json`、`pool-pilot-audit.json`、`mist-single-audit.json`及日志，原始实验产物不覆盖。MIST审查含官方源码URL、输入SHA和已执行阶段。

```bash
/opt/homebrew/bin/python3.11 scripts/analyze_008_r1d_evidence.py
/opt/homebrew/bin/python3.11 scripts/analyze_008_pool_pilot.py
```

精简记录见[审查证据汇总](../../data/references/research-review-2026-10-12.json)。脚本语法、输出JSON、配对文档链接和`git diff --check`均已手动核对。实跑notebook另确认SAFE版本为0.2.1，与本次核对源码一致。

本轮未查询配额、未启动远端作业、未使用或评价保留250谱，也未引入测试文件/测试框架/CI。README与007的最新解释已同步更正。
