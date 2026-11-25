# MSAlign 谱分箱 × Morgan 固定候选对照

使用作者发布的 MassSpecGym `mces_1` 投影头，对 Pablo v29 的封存候选逐谱打分。同一分子的最终列表保持原有候选、原始槽位和 25 槽上限。多个实测谱先分别编码，再平均同一候选的分数。[MSAlign 作者代码](https://github.com/KrzakalaPaul/MSAlign-NeurIPS2026/tree/97e201b1dd61e792d9086815192f7518a0b62ed8) · [官方权重与数据](https://zenodo.org/records/23372807)

作者在 MCES 整体融合中给分箱 × Morgan 的权重约为零。本试验测它在不同候选池中的独立贡献。SpecBridge 作者在第三版撤稿，说明预处理或评估管线有问题，论文给出的代码链接也返回 404。[SpecBridge 撤稿记录](https://arxiv.org/abs/2601.17204v3)

## 运行

Kaggle CPU 准备资产。代码使用 MIT 许可，权重使用 CC BY 4.0。准备程序保留作者代码许可，校验 Zenodo 原始 ZIP 的 MD5，再校验选定投影头与配置的 SHA-256。运行只需要 Kaggle 自带的 Torch。可选的 DreaMS 通道使用作者发布的谱相似性 TorchScript，当前对照只使用分箱通道。

```sh
python rounds/075-frontier/msalign/build.py \
  --prepared-assets --panel /private/tmp/enveda-075-retrieval-pilot-panel \
  --out /private/tmp/enveda-075-msalign-morgan-prepared
```

生成的私有断网 notebook 挂载 `jiyujia/enveda-075-msalign-bins-assets/1` 和 `jiyujia/enveda-075-np-recall-reseal/1`。它校验封存底座与盲谱的摘要，以及 NPX、E180X 阶段回执，并输出逐谱分数、按分子聚合的分数和封存回执。

配对分析入口为 `scoring/compare.py`。读入原始排序、固定权重的倒数排名融合（0.60）、原前五中与原首位同式组内的融合、单模型排序诊断。对同式组做 32 次确定种子的组内分数置换，保留各题候选与分数分布。汇报 MRR、逐题升首与丢首、进出前五，以及来源与离子类型分层。评价在盲分数回执与两个输出文件的摘要通过校验后读取标签。私有标签可以留在本机，由此入口在下载盲分数后做轻量计数。

## 验证范围

60 个天然产物（98 张实测谱）已按来源和骨架从 Pablo 可用的训练与参考谱掩蔽。对 MSAlign 发布的 MCES 的 28,936 个结构，用比赛的去立体和互变异构规范化筛查，面板在 Train、Val、Test 的结构身份重叠均为 0。用发布的原始 SMILES 比较泛化 Murcko 骨架，Train 中检测到 27 道共享骨架题，Val 中 10 道，Test 中 14 道，各分折的计数可以重叠。负离子结果单独看，发布的 MassSpecGym 通道用正离子训练。

结构索引 `unique_smiles.csv` 的 SHA-256 为 `13ce7bb2e1b4b78bf221fb9ec4150007ef635e56c8f8f495eb96e7866f082b72`，分折 `splits/mces_1.csv` 的 SHA-256 为 `5b485c54775af02e65fb07c91ca1a7bf535561dbe1de0dd20ca496522a5d52d5`。两文件来自发布方同一 ZIP 的索引，通过各成员 CRC 检查。[官方数据发布](https://zenodo.org/records/23372807)
