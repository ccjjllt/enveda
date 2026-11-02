# 046-sanpier 进度

- 2026-10-31 16:20 拉取 busyaprime 0.477 公开源码（只拉源码），核对 leader-safe、碎片规则门槛与 merge 规则。
- 16:21 / 16:24 独立 kernel ccjjllt/enveda-046-sanpier v1、v2：ERROR（Python 3.13 镜像，RDKit 2025.3.6 cp312 wheel 不兼容）。
- 16:26 改推 ccjjllt/enveda-020-e006-repro **v22**（046A：sanpier + LS W_MID0.75 + 碎片 rank-1 on V2），RUNNING。
- 监控：/workspace/k046/run.sh（setsid nohup，输出 /workspace/k046_watch.out），门槛 = submission ≥100 行 + 日志含 "NEURAL OK" 与 "046 SANPIER LS FRAG" → 自动提交（今天第 2 次，符合每日 ≤2）。
- 045b 队列（/workspace/k045_queue.sh）未改动，11-01 08:05 推送。
- 公榜：待出分。
- 2026-11-02 00:06 巡检发现：10-31 16:46 v22 已 COMPLETE，但 k_watch 下载只拿到 baseline_submission.csv（submission.csv 未下载，疑为下载超时），门槛判定 NOT_SUBMITTED，046A 漏提交约 31h。按版本直接提交 `-v 22`：ref **57877235**（PENDING）。
- 教训：k_watch 下载缺文件时应重试下载，而不是直接判 NOT_SUBMITTED；`kaggle kernels output owner/slug/<ver>` 对私有 kernel 报权限拒绝，只能拿最新版本输出。
- 2026-11-02 06:05 巡检：046A（ref 57877235，020 v22）公榜 **0.468**，新最佳（> v19 0.465，+0.003），但低于预期 0.472–0.477。
  - 判断：差距主要来自与公开 0.477 版的门槛差异（未接 ICE/GL simulator_vote，碎片 leader 改动召回偏少），以及底座 v19 与公开底座的细微差别。
  - 下一步：按计划优先级 1 补齐 simulator_vote（ICE/GL 分数传入碎片规则，supports<1 但 sim≥1 放行、sim≤−1 拒绝），再推 A 线一次。
- 2026-11-02 12:05 开始 046B：从 /tmp/pubk 公开 0.477 源码抽出 `_post` 中 decision()，确认 simulator_vote 门槛（a≥b +1，a<b−.35 −1；supports<1&sim<1 拒，sim≤−1 拒；+.025·sim 计分）及 supports 用 metric identity 排名（export_evidence: main=BASE[0]、engine2=ENG、pubchem=PC）。
- 实现 `notebooks/046/build_b.py`；本地 mock 校验：sim/supports 计算与放行/否决逻辑符合预期（sup=0 但 sim=1 的候选被放行并夺 rank-1；ICE −1 + GL +1 = 0 且 sup=0 被拒）。
- 当日（UTC 11-02）提交数 0。12:09 推送 ccjjllt/enveda-020-e006-repro **v24**（046B），RUNNING。
- 监控：/workspace/k046b/run.sh（setsid nohup，日志 /workspace/k046b_watch.out）：墙钟循环（30s 短 sleep、每 2 分钟查状态，14h 截止）；COMPLETE 后下载输出最多重试 6 次直到拿到 submission.csv + log；门槛 = ≥100 行 + "NEURAL OK" + "046B SANPIER LS FRAG SIM"；提交前查 UTC 当日提交数 <2；按版本 `-v 24` 提交，成功后写 /workspace/k046b/SUBMITTED。
- 公榜：待出分。
