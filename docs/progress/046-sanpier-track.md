# 046-sanpier 进度

- 2026-10-31 16:20 拉取 busyaprime 0.477 公开源码（只拉源码），核对 leader-safe、碎片规则门槛与 merge 规则。
- 16:21 / 16:24 独立 kernel ccjjllt/enveda-046-sanpier v1、v2：ERROR（Python 3.13 镜像，RDKit 2025.3.6 cp312 wheel 不兼容）。
- 16:26 改推 ccjjllt/enveda-020-e006-repro **v22**（046A：sanpier + LS W_MID0.75 + 碎片 rank-1 on V2），RUNNING。
- 监控：/workspace/k046/run.sh（setsid nohup，输出 /workspace/k046_watch.out），门槛 = submission ≥100 行 + 日志含 "NEURAL OK" 与 "046 SANPIER LS FRAG" → 自动提交（今天第 2 次，符合每日 ≤2）。
- 045b 队列（/workspace/k045_queue.sh）未改动，11-01 08:05 推送。
- 公榜：待出分。
- 2026-11-02 00:06 巡检发现：10-31 16:46 v22 已 COMPLETE，但 k_watch 下载只拿到 baseline_submission.csv（submission.csv 未下载，疑为下载超时），门槛判定 NOT_SUBMITTED，046A 漏提交约 31h。按版本直接提交 `-v 22`：ref **57877235**（PENDING）。
- 教训：k_watch 下载缺文件时应重试下载，而不是直接判 NOT_SUBMITTED；`kaggle kernels output owner/slug/<ver>` 对私有 kernel 报权限拒绝，只能拿最新版本输出。
