# 059 进展

- 2026-11-08 18:10 北京：启动。今日标题 0.482/0.481 公开本均非增量（见 plan 扫描表）；stardjyeah v9-pubchem-pool 为整表重写的朴素 PubChem 替换，弃用。按常设兜底走门控池外/Engine1 召回。
- 本地诊断：LOTUS 87.8% / COCONUT 89.6% 已在 Engine1 v2 池；可见 12 分子仅 3 个有同分子式 + 质量带 LOTUS 候选。
- 构建 `notebooks/059/enveda-059-lotus-gated-recall.ipynb` = 056 原文 + 首部说明 + 末尾追加门控 LOTUS cell（`/workspace/k059/cell059.py`）。本地桩测试（随机神经分）：smoke 模式 status=ok 无插入；强制插入路径在第 6 名插入 2 个、rank 1 不变、去重正确。
- 2026-11-08 18:19 北京：已推 `ccjjllt/enveda-020-e006-repro` **v39**（RUNNING）。watcher PID 见 `/workspace/k059/watch.pid`，日志 `/workspace/k059/watch.log`；对比底座 056 `/workspace/kout_v36/submission.csv`。
- 许可：含 sanpier → A 线冲牌。
- 2026-11-08 18:49 北京：v39 COMPLETE，watcher 判定 `status=ok:_no_insert_(056_kept)`、proxy 3 个可测 / 误触发 0.0、N_DIFF_TOP1=0 / N_DIFF_FULL=0（可见集），提交 ref **58124700**。
- 2026-11-09 06:10 北京复核：ref 58124700 **COMPLETE 公榜 0.481**，与 056 打平。隐藏集重跑中第 6 名插入（若有）对 MRR 中性；rank 1 不变的设计验证了"不动 top-1 就不退步"。059 不作新底座（无增益），056 仍为干净底座。
- 结论 / 交接 060：top-1 改动（057 −0.005、058 −0.008）有害，第 6 名插入中性；下一步只动 rank 2–5 或补全候选列表。
