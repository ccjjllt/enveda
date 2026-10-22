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
