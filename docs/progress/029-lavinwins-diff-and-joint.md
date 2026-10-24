# 029 进展

> **030复核更正（2026-10-22晚）：** 029a漏移植 `N1_COMPLEMENT=False`；旧开关会删除有有限ICE分数的N1代理GL结果，最终joint分支因此缺少双模型证据。原文件保留为历史产物，不直接push。当前优先运行从v465单独构建的完整N1 challenger，见[030进展](030-n1-challenger-and-score-strategy.md)。用户选择配额恢复后手动继续，先前“028a→029a”顺序已被030方案取代。

## 2026-10-22 12:00–12:30（北京时间）离线准备

- 完成 lavinwins vs v465/028a 代码对照；**纠正**此前 progress/028 中「POP_LAM=0.25 / POOLPOP_MU=0.25」的误读（公开 notebook 赋值实为 0.15；0.464 来自 joint ICE+GL）。
- 备好 029a notebook 与 `/workspace/k029`（含 `hengck23-dreams-enveda-casmi26` 数据源）。**未** push、**未**占用 GPU。
- 028a 本地池扩展复验：`Pool` 724,938（+14,237 src=4），质量单调，与 028 记录一致。
- 排行榜（CLI，2026-10-22 ~12:05 北京时间）：#1 Oleksandr Parfionov **0.504**；约 #20 **0.466**（金牌线附近）。我们仍 **0.456**。

## 待配额重置后

见 [029 计划](../plan/029-lavinwins-diff-and-joint.md)：先 028a，后 029a。

## 2026-10-22 18:05（北京时间）定时巡检

- GPU 周配额仍未重置（2026-10-24 08:00），无运行中 kernel，未 push。
- 排行榜：#1 仍 0.504；#2 0.469、#3 0.468（金牌线随之上移）。我们 0.456。
- 新公开 notebook 留意：`likedge/casmi26-n1-challenger-pair-lambdarank`（N1 challenger + pair LambdaRank）、`ahmedberatozer/casmi26-v25-unq`，配额重置后可对照。

## 2026-10-23 00:06（北京时间）定时巡检

- GPU 周配额仍未重置（预计 2026-10-24 08:00），无运行中 enveda kernel，未 push。
- 排行榜（CLI）：#1 Oleksandr Parfionov 0.504，#2 pikachu 0.499，#3 CHU1PC 0.495；#20 约 0.469（金牌线附近）。我们仍 0.456。
- 下一步不变：配额重置后先 028a，后 029a。

## 2026-10-23 12:08（北京时间）定时巡检

- GPU 周配额仍未重置（已用 30.10/30h，剩余 0；预计 2026-10-24 08:00 北京时间恢复），无运行中 enveda kernel，按用户选择**未** auto-push 029a/030。
- 提交仍最佳 v14 public **0.456**（ref `57462581`）；名次约 **583 / 3837**（队伍增多，分数未变）。
- 排行榜（CLI）：#1 Oleksandr Parfionov 0.504，#2 pikachu 0.499，#3 CHU1PC 0.495；#5 0.489；#20 约 0.469。
- 新公开 notebook 留意（配额恢复后对照，不混入本次 030）：`likedge/casmi26-a-gated-top1-swap`（pair 边际门控 top1 交换，`SWAP_TAU=0.95`）、`likedge/casmi26-det-gate-{a,b,c,d}`、`likedge/casmi26-funnel-r{0,1}-pilot8`（结构留出小样本）。
- 下一步不变：配额恢复后由用户手动继续 **030** N1 challenger；例行检查不自动 push。

## 2026-10-23 18:08（北京时间）定时巡检

- GPU 周配额仍耗尽（已用 30.10/30h，剩余 0；预计 **2026-10-24 08:00** 北京时间恢复），无运行中 enveda GPU kernel，按用户选择**未** auto-push 029a/030。
- 提交仍最佳 v14 public **0.456**（ref `57462581`）；名次约 **613 / 3858**（队伍增加，分数未变；上次巡检约 583/3837）。
- 排行榜（CLI）：#1 Oleksandr Parfionov 0.504，#2 pikachu 0.499，#3 CHU1PC 0.495；#4 Savko 0.490；#5 JJ 0.489；#20 Faraz92 **0.470**（上次约 0.469）。顶部分数未实质移动。
- 监视 notebook：原列表 lastRun 时间相对 12:08 巡检无新版本。新增留意（配额恢复后对照，不混入本次 030）：`likedge/casmi26-funnel-r1-full`（COMPLETE，R1 结构留出全量，`PILOT_N=0`）、`likedge/casmi26-funnel-r2-full`（ERROR）。`a-gated-top1-swap` 与 `det-gate-a` 当前 status=ERROR；`det-gate-d` / `funnel-r1-pilot8` / `n1-challenger-pair-lambdarank` / `v25-unq` 仍为 COMPLETE。Edge（likedge）公榜约 **0.463**（#356），高于我方 0.007，方向仍与已备 030 N1 一致，无新可立即采纳的公榜分数跃迁。
- 无安全可立即执行的 CPU/Colab 增产项；不起草 032。下一步不变：配额恢复后由用户手动继续 **030**。

## 2026-10-24 06:10（北京时间）定时巡检

- GPU 周配额仍耗尽（已用 30.10/30h，剩余 0；预计 **2026-10-24 08:00** 北京时间恢复，约 2 小时后），无运行中 enveda GPU kernel，按用户选择**未** auto-push 029a/030。
- 提交仍最佳 v14 public **0.456**（ref `57462581`）；名次约 **666 / 3902**（队伍继续增加，分数未变；上次巡检约 613/3858）。
- 排行榜（CLI 全量）：#1 Oleksandr Parfionov 0.504，#2 pikachu 0.499，#3 CHU1PC 0.495；#4 Savko 0.490；#5 JJ 0.489；#20 Shehab Alshehabi **0.470**。顶部分数相对 18:08 巡检无实质移动。
- 监视 notebook：相对 `watch_nb_hashes/2026-10-23-1808.json` **SHA256 全无变化**；status 仍为 a-gated-top1-swap / det-gate-a / funnel-r2-full = ERROR，其余 COMPLETE。Edge（likedge）公榜约 **0.463**（#419），lavinwins / Hongbin Gu 约 **0.464**。
- 新近公开 notebook（仅留意、不混入本次 030）：`senanuretin/casmi26-exact-public-example-overlap-audit`、`bhuvneshrichhariya/casmi26-*`、`d1m4official/enveda-casmi-2026-solution-v29`、`hailonglv/casmi26-v16-budget` 等；未见明显高于我方且可立即采纳的公开跃迁。
- 下一步不变：配额恢复后由用户手动继续 **030** N1 challenger；例行检查不自动 push。

## 2026-10-24 12:05（北京时间）定时巡检

- **GPU 周配额已重置**：ccjjllt 现为已用 **0.00h / 30h**，剩余 **30.00h**；下次刷新 UTC 10-31 00:00（北京时间 10-31 08:00）。TPU 剩余 20h。
- 无运行中 enveda GPU kernel；`ccjjllt/enveda-020-e006-repro` 仍为 COMPLETE（旧 v8 / v14 得分版本）。**按用户选择未 auto-push 029a/030。**
- 提交仍最佳 v14 public **0.456**（ref `57462581`）；名次约 **683 / 3914**（队伍继续增加，分数未变；上次 06:10 巡检约 666/3902）。
- 排行榜（CLI 全量）：#1 Oleksandr Parfionov **0.504**，#2 pikachu 0.499，#3 CHU1PC 0.495；#4 Savko 0.490；#5 JJ 0.489；#20 Shehab Alshehabi **0.470**。顶部分数相对 06:10 无实质移动。
- 监视 notebook 状态：`n1-challenger-pair-lambdarank` / `funnel-r1-full` / `det-gate-d` / `funnel-r1-pilot8` / `v25-unq` / `guhongbin/v465` = COMPLETE；`a-gated-top1-swap` / `det-gate-a` / `funnel-r2-full` / `lavinwins/casmi26-top-1-gold` = ERROR。未见新的明显高于我方且可立即采纳的公开跃迁。
- 快照：`artifacts/runs/029/patrol_2026-10-24-1203/`（quota / submissions / leaderboard_summary / watch_status）。
- **阻塞**：030 N1 challenger 代码已备好，配额已恢复，但仍待用户**手动** `kaggle kernels push -p notebooks/030` 后观察完成再提交。例行检查不代为 push。

