# 053 进展

- 背景：对 051 本地下载 submission 做 FREEZE 时发现 388/400 为占位 `CCO`（可见 test 特性），离线 top1 指标无意义；改为纯 Kaggle 端 overlay。
- 数据集 `ccjjllt/casmi26-053-fragfreeze` 已创建（train_ik14 + frag_lin + frag_feats + ff_config）。
- notebook = 051 + FREEZE cell；020 kernel **v33** 已推送 RUNNING（北京 2026-11-06 ~06:18）。
- watcher：`/workspace/k053/run.sh`（2 分钟轮询 + 13h 墙钟；下载 audit/log 重试；仅当 `reranked>0` 且 `changed/top1_changed>0` 时 `-k/-v 33` 提交）。
