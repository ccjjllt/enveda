# 实测参比谱定位预检

用 Pablo v29 的封存原版和 60 题盲谱测试可达性。每题取原前八，再从已保存的溢出、主库、PubChem、Engine 2 和初融合名单轮取未重复结构，最多十六个。参比从比赛的实测训练谱中找，同加合物，沿用面板的来源、身份、骨架和重复谱屏蔽。

对四类单点 O 修饰，先检查候选能否由参比结构增加相应原子组成，再检查实测前体质量差和谱峰匹配。每张谱保留最多三个不同的参比结构。与原谱直接匹配或按前体质量差平移匹配的峰各只能使用一次。仅在同时存在可归属的平移峰和原位峰时，对候选位点计算一致性。每个参比内对同类羟基位点做八次固定置换。只枚举单个非环键的碎裂，因此有些实测峰无法归属。[位点定位方法](https://pubs.acs.org/doi/10.1021/jasms.4c00061) · [参比碎片图匹配](https://arxiv.org/html/2502.17874v2)

```sh
python rounds/075-frontier/anchor/build.py \
  --panel /private/tmp/enveda-075-retrieval-pilot-panel \
  --out /private/tmp/enveda-075-np-anchor-preflight
```

生成私有断网 Kaggle CPU notebook，挂载封存原版 `jiyujia/enveda-075-np-recall-reseal/2`，按摘要校验训练结构表。输出 `anchor_probe_receipt.json`、`anchor_probe_pairs.json` 和 `anchor_probe_candidates.json`。回执分别给出结构可达、同加合物实测可达、有位点分，以及同一参比对多个候选给出不同位点分的题数。逐谱证据保留原位余弦、改良余弦、位点分和置换分。本次只测盲覆盖和证据区分度。

60 题、98 张谱实测中，16 题有结构可达的参比，11 题有同加合物实测谱，6 题共取得 7 对筛选后的谱匹配。7 对全部为正质子谱，每对只覆盖一个候选的一个位点；没有一对同时提供可归属的平移峰和原位峰，最终 **0** 题有位点分。停止用这组实测参比和单键碎裂配置训练换位模型。[Kaggle v1](https://www.kaggle.com/code/jiyujia/enveda-075-np-anchor-preflight/versions/1) · [盲回执](../../../data/references/075-anchor-preflight-2026-11-25.json)
