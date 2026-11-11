# 063 进度

- 2026-11-11 12:06 北京：复核 062a/062b 输出——md5 相同属实，但"外部库在隐藏集惰性"推断错误（代码赛评分为隐藏测试全量重跑，commit 只跑 12 个 smoke 分子）；已在 docs/progress/062 追加更正。
- 12:07 北京：拉取公开 notebook（gloriachen0816 0.487、chyibo/xiangqizhang fork、ozertuu 最新、zz3t3rn41 v2/V48）。0.487 版本与 062a 代码 cell 相同，仅输入不同（+popwenman scorer v80、−LOTUS、guhongbin docker）。
- 12:09 北京：推 `ccjjllt/enveda-063a-p2s-scorer-v80` **v1**，立即 RUNNING；watcher `scripts/063/chain.sh`（D=/workspace/k063a，O=/workspace/kout_063a）。
- 12:10 北京：推 `ccjjllt/enveda-063b-p2s-scorer-v80-lotus` **v1**；watcher 同上（D=/workspace/k063b，O=/workspace/kout_063b）。
- watcher 修正：原 `k062a/run3.sh` 审计调用的 `/workspace/k062/venv/bin/python` 已不存在（会导致审计输出为空→SKIP），063 改用系统 python3（pandas 3.0.6），已用 kout_062a 验证审计 `DECISION SUBMIT`。
- 预计完成：约 6–9 小时后（时间守卫 8.55h），出分再约 4–6 小时。

## 提交记录（2026-11-11 18:16 北京时间）
- 063a 内核 v1 于 14:41 完成，watcher 审计通过（smoke 输出与 056 top-1 差 0、全列表差 12，extlib hits 0），18:09 以 kernel 提交，ref 58233824，PENDING。
- 063b（+LOTUS）内核 v1 于 14:42 完成，审计同上，18:15 提交，ref 58233993，PENDING。
- 预计公开分 ~23:00–00:00 出炉；≥0.485 则 A 轨切换到 063 基线，否则保留 062a（0.484）。

## 出分（2026-11-11 23:58 北京时间核对）
- 063a v1 ref **58233824** 公榜 **0.487**（新最佳，相对 062a 的 0.484 为 +0.003）；与 gloriachen0816 / alexchilton 公开 0.487 一致，复现成功。
- 063b v1（+LOTUS）ref **58233993** 公榜同样 **0.487**。LOTUS 候选数据集在 3 位小数下中性（不增益也不退步），与 059 门控 LOTUS 打平 056 的结论同向。
- **A 轨切换到 063a**（无 LOTUS，输入更干净；代码与 062a 相同，增益来自挂上 popwenman `casmi26-scorer-train-v80` 内核输出，以及/或 guhongbin docker；LOTUS 已由 063b 排除）。docker 未单独消融。
- 榜单快照（2026-11-11 23:59 北京，5136 队）：团队 ccjjllt 第 **503** 名。0.487 是大平台（第 70–547 名共 478 队同分）；0.488 仅 16 队。要明显跳出该平台需要 ≥0.488。公开标题为 0.488 的 `nugrahakc/casmi26-0-488-haloramarama` 正文仍是旧 0.477 管线，作者团队 Rama Sanjaya 公榜也是 0.487，不可当作 0.488 配方。0.488 档暂无已核实的公开 notebook。
- 本周 Kaggle GPU 配额仅剩 2.65 小时（2026-11-14 08:00 北京刷新）；完整推理约 6–9 小时，且账号上另有 4 个 pump-fun 内核 RUNNING 会继续耗配额。064 已写计划，等配额刷新后再推，不在此刻强行排队。
