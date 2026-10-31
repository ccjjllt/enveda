# 046-sanpier 进度

- 2026-10-31 16:20 拉取 busyaprime 0.477 公开源码（只拉源码），核对 leader-safe、碎片规则门槛与 merge 规则。
- 16:21 / 16:24 独立 kernel ccjjllt/enveda-046-sanpier v1、v2：ERROR（Python 3.13 镜像，RDKit 2025.3.6 cp312 wheel 不兼容）。
- 16:26 改推 ccjjllt/enveda-020-e006-repro **v22**（046A：sanpier + LS W_MID0.75 + 碎片 rank-1 on V2），RUNNING。
- 监控：/workspace/k046/run.sh（setsid nohup，输出 /workspace/k046_watch.out），门槛 = submission ≥100 行 + 日志含 "NEURAL OK" 与 "046 SANPIER LS FRAG" → 自动提交（今天第 2 次，符合每日 ≤2）。
- 045b 队列（/workspace/k045_queue.sh）未改动，11-01 08:05 推送。
- 公榜：待出分。
