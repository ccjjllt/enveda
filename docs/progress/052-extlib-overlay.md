# 052 进展

- 自建 refs：200168 谱 / 145486 结构（takumuhata 192004 + novel 6800 + MassBank permissive 865 + MSnLib 471 + GNPS 28）；排除 train ik14=275810。
- 校准：Enveda-180 timsTOF 2500 分子，TAU=0.55 K=3，假阳 8/2500=0.32%；LIB_TAU=0.9。
- 离线叠加 051：hits=6 changed=6 inserted=6 moved_up=0 skipped_library=12；自测 3/3 通过。
- 数据集 `ccjjllt/casmi26-extlib-052`；notebook = 051 + xl 单元；推送 020 新版本并由 watcher 提交。

- 2026-11-05 18:24 北京时间：数据集 `ccjjllt/casmi26-extlib-052` 已上传（ready）；020 kernel **v32** 已推送 RUNNING；watcher `/workspace/k052/run.sh`（2 分钟轮询 + 13h 墙钟，精确下载 submission.csv，`-v 32` 提交）。
