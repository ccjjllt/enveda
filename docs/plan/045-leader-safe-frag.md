# 045 leader-safe 融合 + 碎片校验 rank-1 规则（计划）

依据：docs/plan/045-public-sota-analysis.md。全部基于我们自己的 model3，不加新数据集。

## 045a：044 + leader-safe 融合
- rank1 仍由原公式 `1/(3+b) + 0.25/(3+n)` 选出（保持 0.466 的首位）；
- rank2–25 去掉 leader 后按 `1/(3+b) + W_MID/(3+n)` 重排，W_MID=0.75（离线扫描选定）。
- 来源致谢：alex chilton、Roman Rozen（公开 0.472 leader-safe），busyaprime（0.477 合并方法）。

## 045b：045a + 碎片校验 rank-1 规则
- 移植 Roman Tamrazov「Spectral Parallax」的 MetFrag-lite（无训练，≤2 键断裂碎片解释强度），Apache-2.0 公开代码。
- 在 W_MID=0.5 的 leader-safe 序上运行，只保留 leader 变更：新 leader 置首，旧 leader 第 2，其余按 0.75 序；关闭交换与末位替换。
- 条件：分子级 lib_max<0.87、可用谱≥3、rank2–6 同分子式、frag≥0.27、margin≥0.095、baseline cfg margin≥0.028、分谱投票≥0.70、model3 z 分差≥0.23、engine2/pubchem/base_tail 任一 top5 支持。
- 与公开实现差异：无 ICEBERG/GLACIER 投票与 neural_cos/fp 检查；lib 门槛用分子级 lib_max（更保守）。

## 提交队列
- 配额重置（10-31 08:00 北京时间）后先推 045a，k_watch 门槛 "NEURAL OK" + "045 LS"；次日 08:05 后推 045b，门槛 "045 FRAG"。每天≤2 次。
