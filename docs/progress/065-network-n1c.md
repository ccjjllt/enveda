# 065 进度

- 2026-11-14 06:10 北京（例行检查）：无新自有提交，最佳仍是 063a/063b 公榜 **0.487**（ref 58233824 / 58233993）。本日提交额度 0/5（剩余 5，08:00 北京重置）。每周 GPU 仍透支耗尽（37.13h / 30h，剩余 0），刷新 2026-11-14 08:00 北京。自有与他人项目内核均为 COMPLETE，工作区无 enveda 后台任务。
- 公榜快照（2026-11-13T22:05:46Z，5248 队）：ccjjllt 第 **559**；0.487 平台第 178–825 名共 648 队；≥0.488 共 **177** 队。相对 2026-11-13T16:00:03Z 快照（5228 队、第 536 名、≥0.488 共 112 队）约 6 小时内 ≥0.488 新增 65 队。
- **已核实公开 0.488**：拉取 `alexchilton/casmi26-lb-0-488-peaks-to-structures-network`，与 063a 逐格比对后，唯一代码增量是外部库 overlay 之前的 N-1c 网络重排格；配置仅 `COMMIT_MODE` 为 `'fast'`（我们保留 `'smoke'`）。作者说明该格来自 analyticaobscura/casmi-2026，其账户提交 58314720 得 0.488。alex chilton 与 ozertuu 本人公榜均已到 0.488。
- 已写 `notebooks/065a`（slug `ccjjllt/enveda-065a-network-n1c`）及本轮计划。等 GPU 刷新后再推。064a（`LIB_HIT_GATE` 0.75）延后，不与 065a 抢第一轮配额。
- 2026-11-14 08:17 北京：每周 GPU 已刷新（0.00h / 30.00h，剩余 30.00h，下次刷新 2026-11-21 08:00 北京）。今日提交额度 0/5。已推 `ccjjllt/enveda-065a-network-n1c` **v1**，状态 RUNNING。watcher：`scripts/063/watch.sh`（D=/workspace/k065a，O=/workspace/kout_065a，HOURS=30），完成后单文件拉取 `submission.csv`、审计通过则 `-k -v` 提交。未推 064a。预计 6–9 小时跑完。
- 2026-11-14 12:10 北京（例行检查）：065a kernel v1 于 10:57 北京 COMPLETE（墙钟约 2.7h，短于原估 6–9h）。watcher 审计通过（TOP1_VS_056=0，FULL_VS_056=12，ExtLib 0 hits / 0 promoted），11:13 北京自动提交 ref **58329677**，状态 PENDING。今日提交额度 1/5（剩余 4）。
- N-1c 运行日志：molecules 400, with spectra 391, with neighbors 114, edges 109, changed lists 2, top1 changed 0, 77s。与 063a submission.csv 逐列表比对：仅 m_01e922（自第 2 名起重排）与 m_068354（自第 3 名起重排）不同，400 条 top-1 全部相同；merge 后、N-1c 前 sha256 与 063a 一致（6023b695b3f84b0c59e2c94ad9d0834273a2636d2755df61070a7139f829f169），底座对齐。指标为 MRR@25，仅改 mid-rank 仍可能兑现公开配方声称的 +0.001。预计出分约 16:13 北京（按既往约 5h 评分）。
- 每周 GPU：0.36h 已用 / 29.64h 剩余 / 30.00h，下次刷新 2026-11-21 08:00 北京。无运行中内核，工作区无 enveda 后台任务。
- 公榜快照（2026-11-14T04:06:37Z，5254 队）：ccjjllt 第 **571**（仍 0.487）；0.487 平台第 205–843 名共 639 队；>=0.488 共 **204** 队（约 6 小时前 177 队）；>=0.489 共 99 队。alex chilton 与 ozertuu 本人仍为 0.488；heliosli 本人 0.499。
- 抽查新公开 notebook：heliosli/casmi26-0-488（66 票，最近跑 2026-11-13 23:44 UTC）dataset_sources 仍有两个空槽（私有），不可复现；heliosli/the-dataset-is-public-now-enjoy 仍挂 heliosli/qcc-circuits-live，属 QCC 而非 CASMI。未见已核实的公开 0.489 配方。064a 继续等 065a 出分后再考虑。
