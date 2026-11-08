# 060 进展

- 2026-11-09 06:10 北京：记录 059 公榜 0.481（打平）、058 公榜 0.473。公开扫描见 plan：ozertuu "0.485" = 056 + ranks 2–25 文档先验 0.12 + extlib rank-1 提升（无 v80）；只取前者。
- 构建 `notebooks/060/enveda-060-docprior-012.ipynb` = 056 原文 + 首部说明 + config 改 `POP_W/SPEC_W 0.12` + cell 12 同轮反事实审计（`g60_audit.json`，top1 有差异即回退 056）。
- 2026-11-09 06:12 北京：已推 `ccjjllt/enveda-020-e006-repro` **v40**（RUNNING）。watcher PID 842597（`/workspace/k060/watch.pid`），日志 `/workspace/k060/watch.log`，对比底座 056 `/workspace/kout_v36/submission.csv`。
- 许可：含 sanpier → A 线冲牌。
- 2026-11-09 06:44 北京：v40 COMPLETE（约 30 分钟）。`g60_audit.json` status=ok，先验表 710,701 行均加载；可见 smoke 12 个已处理分子 ranks 2–25 顺序全部改变（diff_full=12），diff_top1=0、diff_top3=1、diff_top5=3；与 056 提交文件对比 N_DIFF_TOP1=0，候选集合完全相同（纯重排）。watcher 用 `-k/-v 40` 提交 ref **58142650**（Kaggle 回报今日剩余 2 次），PENDING（历史出分约 5 小时）。输出 `/workspace/kout_v40/`。
