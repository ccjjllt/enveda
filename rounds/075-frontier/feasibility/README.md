# 同分子式候选的相对碎片覆盖

从 Pablo 0.506 密封候选和同一批 60 道盲谱起跑。原版 Parallax 已做碎片覆盖。这里复用原版的碎片质量枚举，固定每个候选最多 128 个质量、每谱最多 80 个峰，并比较同一批质量提案上的直接覆盖和候选相对覆盖。后者降低同分子式候选共同解释的峰权重。重新计算的原版覆盖使用本实验准备的谱，便于比较每题候选的相关性。原版密封包只提供 Parallax 汇总。

固定每题的原候选、25 槽和空位。只考虑同分子式、质量窗内、单组分中性结构和质子化或去质子化谱。每组至少三候选，最远交换两个位置，只比较第 1 和第 5 名附近。交换要求至少两个可区分的峰、最低绝对覆盖和预设的证据边际。主引擎、Engine 2 和 PubChem 均把同一结构列首时保护原首位。这是碎片质量近似，允许有限的氢迁移。

同一仪器、加合物及相近已知碰撞能量的不同分子谱在两个质量窗中配对。保留原谱的前体、候选和原排名。真实谱与借用谱使用相同的可比较谱子集，分别报告近质量 5 Da 和紧质量 0.01 Da；保留配对距离和可用样本数。近质量对照是探索性的。若紧质量可用样本太少，或相对覆盖在全量验证上与绝对覆盖相当、在匹配子集上与借用谱相当，则现有证据不足以扩训。[MassSpecGym v1.5 研究](https://arxiv.org/html/2611.28096v1)

```sh
python3 rounds/075-frontier/feasibility/build.py \
  --panel /private/tmp/enveda-075-retrieval-pilot-panel \
  --source /private/tmp/enveda-public-0506-pablo-v29/source.ipynb \
  --out /private/tmp/enveda-075-np-relative-feasibility
```

盲推理使用私有、断网的 CPU notebook，挂载固定版本 `jiyujia/enveda-075-np-recall-reseal/2`。输出 `receipt.json`、`audit.json`、`candidate_evidence.csv` 和各组排名 CSV。题目答案仅用于独立的私有比较 notebook。

```sh
python3 rounds/075-frontier/feasibility/build_compare.py \
  --panel /private/tmp/enveda-075-retrieval-pilot-panel \
  --out /private/tmp/enveda-075-np-relative-feasibility-compare
```

比较步骤校验密封文件和逐题候选集合，输出 `metrics.json` 和 `per_query.csv`，记录升首、丢首、进出前五、MRR 净变化及按骨架重采样的区间。

60 题完整结果中，绝对和相对覆盖均从 **0.445208** 升到 **0.472986**，首位 **19 → 22**，前五 **37**、前 25 **48**，两组的正确结构名次逐题相同。三个升首来自三个骨架组。MycoMSBase 两题分别由第 2 和第 3 升首，GNPS Tübingen 一题由第 2 升首。按骨架重采样的 95% 区间为 **[0, +0.06215]**。用相差 5 Da 以内的错误谱作相对覆盖对照时，28 题中只有一个上述升首有可比对照，真实谱升首而错配谱没换首。0.01 Da 内只有 3 题可比，两组正确名次持平。接下来冻结绝对覆盖规则，用独立谱复核。[逐题结果](../../../data/references/075-relative-fragment-feasibility-2026-11-25.json) · [Kaggle 私有计分](https://www.kaggle.com/code/jiyujia/enveda-075-np-relative-feasibility-compare/versions/1)

独立 CASMI 2017 确认先逐字节复现原 60 题的绝对覆盖排序。历史 0.505 底座的 **17 题、19 张谱、425 个候选**均完成复核，整份排序与原序逐字节一致。首位 6、前五 10、前 25 为 12，MRR 保持 **0.454927**。这批独立谱未触发交换，因此 60 题上的升首收益仍待在其他有触发的样本复现。[Kaggle 独立确认](https://www.kaggle.com/code/jiyujia/enveda-075-absolute-confirm-casmi17/versions/1) · [逐题确认回执](../../../data/references/075-absolute-fragment-confirm-casmi17-2026-11-25.json)

Enveda-180 的 121 道留出题用历史 0.498 原版的 299 张 timsTOF 谱、3,025 个候选和三路原始首选复核。规则改动 21 份排序，含 8 次首位交换、14 次第五位交换。121 题无三路首位一致的保护触发，0 升首、1 丢首，前五保持 91，前 25 保持 103，MRR **0.576114 → 0.571982**（−0.004132）。暂缓在整线上默认启用；采用独立实谱检验其适用范围。[Enveda-180 原位次确认](../../../data/references/075-absolute-fragment-confirm-e180-2026-11-25.json)

110 道 MassBank 实谱的 068a 对照，用原 2,750 个候选及三路冻结首选，结果 **0.336852 → 0.332306**。前五 41、前 25 为 58，0 升首、1 丢首，只有非 Eawag 的 BAFG U109 从首位降到第 2。预留的 23 道非 Eawag 一组净差 **−0.021739**。三批确认中没有新增升首，停止以该规则直接换首和送正式提交。[MassBank 逐题确认](../../../data/references/075-absolute-fragment-confirm-massbank-2026-11-25.json)
