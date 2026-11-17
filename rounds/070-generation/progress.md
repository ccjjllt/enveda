# 070 进展

## 2026-11-17 070a commit

- `jiyujia/enveda-070a-full25` v1 COMPLETE 但 overlay 未生效：`neural_scores.py:32 assert mid in query`——evidence 含 400 分子而 commit run 的 query 谱只覆盖 12 分子，首个缺谱分子即断言失败，无输出、submission 原样。根因是 scorer 要求 evidence 分子必须在 query 中（083a 因 evidence 源自 query 未触发）。
- 修复：evidence 只含 query 覆盖的分子（commit 12 个，hidden rerun 全 400），其余分子保持原样；v2 已推送。审计待拉取：若可见改动只有 ~10 列表或 top-1 零变化，按 kill 规则判死不提交。
- v2 COMPLETE：`neural_full25_audit.json`：**reranked 12，top1_changed 0**（8 个因 `lib_max>=0.90` 冻结 leader）；diff 确认 12 列表变化、top-1 零变化。**070a 按 kill 规则判死，不提交。** 神经分与底座顺序在头部几乎共线，换帅门控全军覆没。
- 070b（ICE+对齐联合）暂缓：070a 已死 + 073/076 已死，联合排序大概率同命；等 082c 出分后再决定是否花 GPU 跑 070b 或转 MIST P0。
