# 070 进展

## 2026-11-17 070a commit

- `jiyujia/enveda-070a-full25` v1 COMPLETE 但 overlay 未生效：`neural_scores.py:32 assert mid in query`——evidence 含 400 分子而 commit run 的 query 谱只覆盖 12 分子，首个缺谱分子即断言失败，无输出、submission 原样。根因是 scorer 要求 evidence 分子必须在 query 中（083a 因 evidence 源自 query 未触发）。
- 修复：evidence 只含 query 覆盖的分子（commit 12 个，hidden rerun 全 400），其余分子保持原样；v2 已推送。审计待拉取：若可见改动只有 ~10 列表或 top-1 零变化，按 kill 规则判死不提交。
