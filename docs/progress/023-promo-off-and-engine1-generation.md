# 023 — v13 关闭 PubChem 提升；Engine1 生成路径研究：实际进展

日期：2026-10-21（北京时间）。接手`158de08`。对应[023计划](../plan/023-promo-off-and-engine1-generation.md)。

## 结论（进行中）

**is_gen 标记确认：C 遮蔽的 89 个 top25 命中全部是 Engine1 生成的候选；而且 Engine1 自己的排序远好于最终输出——Engine1-only MRR 0.2938（top1 66），最终流水线只有 0.1168（top1 4）。** 逐分子 84 个 Engine1 更好、0 个最终更好。损失发生在 Engine1 之后的合并阶段（PubChem 插槽/提升、Engine2 融合）：真值从第 1 名被挤到第 2–5 名。注意 C 同时把真值移出了 PubChem 池，会放大 PubChem 插入的伤害；需在 R 遮蔽（无参考、结构仍在池和 PubChem）上复核。v13 结果待出。

## v13

- 改动：v11 notebook 只把 `or (rel > 150.0)` 改为 `or False`。kernel version 6 因改写时注释吞掉了行尾冒号（语法错误）被 version 7 取代；version 7 编译通过后 push（00:15 左右）。
- 状态：截至 01:00 仍 QUEUED。提交与分数：待出。

## Engine1 生成：C 遮蔽 base 配置（6 类似物 / 每种子 60 / 总 150）

`src/study_e1_gen.py`，执行 E006 前 16 个 cell 后逐分子重跑 `V.run`，按 v4n ranker 排名（不含人气重排与后续融合），结果 `/workspace/e023/C_base.jsonl`。

| 指标 | 值 |
|---|---|
| 有生成候选的分子 | 163 / 250 |
| 平均候选数 / 生成数 | 111.9 / 25.3 |
| top25 中平均生成候选数 | 5.7 |
| 真值被生成（任意名次） | 89 |
| 真值进 top25 / top1 | 89 / 66 |
| Engine1-only MRR | **0.2938** |
| 最终流水线 MRR（021 C_base） | 0.1168（top1 4） |

真值名次分布：第 1 名 66，第 2 名 10，第 3–9 名 13。也就是说一旦生成到真值，ranker 多数能排第一；瓶颈是（1）87 个有生成但没生成到真值、87 个根本无生成候选的分子，（2）合并阶段把生成的第一名挤下去。

## 生成参数变体

big（12/120/400）、loose（big＋min_sim 0.2）运行中。
