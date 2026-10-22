# 022研究交接：E006 基线、库外召回与下一步

日期：2026-10-20。目标仍为隐藏集结构识别能力，竞争金牌（公开榜约 0.463）与前五奖金。当前最佳为 **v11 public 0.446**（ref 57405363，kernel `ccjjllt/enveda-020-e006-repro` v4）；v12 为 0.434，未保留。详见[020进展](../progress/020-e006-reproduction.md)、[021进展](../progress/021-local-validation.md)、[022进展](../progress/022-v12.md)。

## 研究到了哪里

**底座已换成公开 E006＋PubChem Promo 的原样 fork，分数从 0.323 提到 0.446。** 本地 NP250 验证器可在 CPU 上原样跑 E006 去模拟器流程（约 80 分钟/次），支持 A/B/R/C 四种遮蔽及逐分子配对比较。

| 证据 | B（有跨来源参考） | C（真值无参考、不在池） | 隐藏集 |
|---|---:|---:|---:|
| 基线 MRR | 0.9569 | 0.1168 | public 0.446 |
| 严格直接参考规则 | +0.0140（7/0） | −0.0046（0/5） | **−0.012**（0.434） |
| MassBank538 扩池 | +0.0001（1/0） | 未测 | — |
| 关闭 PubChem 提升 | +0.0045（3/0） | 未测 | — |

## 最值得解释的现象

**隐藏集的行为更像 C，而不是 B。** 唯一一次本地 B 正、C 负的改动在 public 上为负，说明隐藏集多为无跨来源参考的分子；以 B 为主的本地收益不能外推。

**C 中的命中全部来自 Engine1 的生成路径。** 89 个 top25 命中（4 个 top1）中，PubChem 通道与 Engine2 列表 top100 都不含真值；真值不在 COCONUT/ChEBI/LIPID MAPS 池与 train 中，v2-pool 中 85 个已被遮蔽过滤；Engine1 `casmi/engine.py` 的 `generate(analogs, …)` 对库内类似物做衍生/位置异构生成（排除法结论，未逐个读 is_gen）。常见失败是真值排在其位置异构体之后。

## 下一轮先做什么

1. **加强 Engine1 生成与库外召回。** 在 C 遮蔽上把「真值进 top25」从 89/250 往上推：读取 is_gen 标记确认来源，分析生成规则覆盖不到的真值，研究位置异构体的区分（同式候选碎片证据、逐 CE 预测谱）。以 C 的 top25 召回和 MRR 为主指标，B 只作不伤害检查。
2. **困难负例训练。** `src/train_fingerprint.py` 的排序损失已修（`47600cb`）；从 E006 完整候选窗挖掘高分错误同式结构做困难负例，微调指纹模型，在 C 与 B 上同时评估。
3. **v13 候选：关闭 PubChem 提升（或阈值 250）。** B 上 +0.0045（3/0），但 B 测不到库外收益；须先在 C（保留 PubChem 池中真值的变体遮蔽）上确认不伤害，再决定是否提交。
4. **许可与来源核实。** 16 个上游数据集多数未写许可证；领奖前需逐一核实来源与许可，并准备可复现的训练与推理代码。

## 接手时要记住的数据含义

- 本地数字全部不含 ICEBERG/GLACIER；面板 250 分子且与 018/019 同批，存在乐观偏差。
- C 遮蔽同时把真值移出 PubChem 池，因此 PubChem 通道在 C 中贡献 0 不能当作「PubChem 无用」的反证。
- 可见占位 test 与 train 重叠，Kaggle 交互运行的统计（如 v12 提升 1 个）不代表隐藏集。

## 研究资产与入口

- 验证器：`src/eval_e006_local.py`、`src/casmi_metric.py`；运行根 `/workspace/e021/`（box，本地，不入库）。
- v11 notebook：kernel v4；v12 cell：`notebooks/submissions/submission-v12/v12_direct_ref_cell.py`。
- 训练：`src/train_fingerprint.py`、`src/fp_train_data.py`。
