# 052 进展

- 自建 refs：200168 谱 / 145486 结构（takumuhata 192004 + novel 6800 + MassBank permissive 865 + MSnLib 471 + GNPS 28）；排除 train ik14=275810。
- 校准：Enveda-180 timsTOF 2500 分子，TAU=0.55 K=3，假阳 8/2500=0.32%；LIB_TAU=0.9。
- 离线叠加 051：hits=6 changed=6 inserted=6 moved_up=0 skipped_library=12；自测 3/3 通过。
- 数据集 `ccjjllt/casmi26-extlib-052`；notebook = 051 + xl 单元；推送 020 新版本并由 watcher 提交。

- 2026-11-05 18:24 北京时间：数据集 `ccjjllt/casmi26-extlib-052` 已上传（ready）；020 kernel **v32** 已推送 RUNNING；watcher `/workspace/k052/run.sh`（2 分钟轮询 + 13h 墙钟，精确下载 submission.csv，`-v 32` 提交）。
- 2026-11-06 00:12 北京时间：kernel v32 **COMPLETE**。日志 xl：hits=6 changed=6 inserted=6 moved_up=0 skipped_library=12（与离线一致）；自测通过。
- 2026-11-06 06:09–06:12 北京时间：watcher 卡在 `kaggle kernels output`（经典问题，kout_v32 仅 stale bootstrap.log）。杀掉卡死下载 PID 后改用 `kaggle competitions submit -k ccjjllt/enveda-020-e006-repro -v 32 -f submission.csv`（直接提交 kernel 产出，无需完整下载）。离线产物 `artifacts/runs/052/submission_051_plus_xl.csv` 已校验：400 行、相对 051 top1 改 6、审计与 kernel 日志一致，并复制到 `/workspace/kout_v32/submission.csv` 作本地备份。
- **已提交** ref **58036512**（2026-11-05 22:12:03 UTC / 北京 06:12，PENDING 评分，约 5h）。当日剩余提交额度 3。
- xl 审计摘要：hit_cosines≈0.55–0.57（偏弱，有误插风险，按计划仍提交看公榜）；命中 mid：m_30024e / m_6ebcf9 / m_82f2e2 / m_930a3b / m_ab91a8 / m_e8359d。
