# 046 实列表残差重排（计划，045 出分后启动）

依据：2026-10-29 巡检拉到的公开 notebook `ahmedberatozer/casmi26-v33-real-list-residual`（在 busyaprime 0.477 m1 之上学残差；未宣称公榜增益；依赖 sanpier unknown 模型，不可直接用）。

## 时机
- **先等 045a/045b 公榜**（配额 10-31 08:00 北京时间重置后由 k045_queue 自动推送）。
- 若 045 已到 ≥0.472，本轮作冲金余量；若仍平台期，本轮作主探索。

## 假设
在 **我们自己的 045 答案列表**（model3 + leader-safe ± frag，无新数据集）上，用公开目录回放的 query 级特征学一个 **有界残差**，只调整非 leader 名次或受控的 rank1 替换，保留 m1/045 先验。

## 做法（许可面不变）
1. CPU 回放：对 panel/外部目录生成「实列表」行（baseline_rr、neural_rr/score_z、engine 一致、lib_max、ICE/GL z、谱数等——只用我们管线已有字段；去掉 sanpier 专有 neural_cos/fp）。
2. 固定配方：scaffold/query 分组 OOF；β≈0.2 的 log-rank 残差；正则化线性或浅层模型；**所有原 leader 默认锁死**（与 v33 一致，避免重蹈 v22 保护翻车）。
3. 门控：OOF MRR 相对 045 列表有稳定增益再进提交 notebook；否则丢弃。
4. 不引入 sanpier / catalogue-experiment-saner；训练数据仅公开竞赛允许的回放目录。

## 非目标
- 不等同于再训 spectrum model（那是 GPU 配额重置后的并行选项）。
- 不在 045 出分前占用每日提交名额。
