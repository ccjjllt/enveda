# 044 进度
- 新增 src/own_spec/train_hn.py + build_044.py → notebooks/044/own_spec_hn.py（训练脚本）；notebooks/044/own_hn_cell.py + enveda-044-own-hn.ipynb（提交 notebook，kernel-metadata-sub.json）。
- 本地 CPU 冒烟（2 个 row group 各 4000 行、外部池各 3000）：全流程通过，随机初始化下 pool val 明显低于 isomer val（0.39 vs 0.68），说明外部候选确实更难、旧 val 偏乐观；推理 cell 冒烟 NEURAL OK（scored 299）。
- 北京时间 10-27 18:09 推送训练 kernel ccjjllt/enveda-039-own-spec **version 6**（T4×2，热启动 042 model2.pt），预计 10-28 约 04:30 完成（+输出上传）。
- watcher：`nohup bash /workspace/k044_watch.sh > /workspace/k044_watch.log`，过门槛后自动建 dataset 版本、推送提交 kernel 并提交（预计 v24）。
- 待办：完成后核对日志 `warm start from .../model2.pt`、`VAL init` isomer≈0.76、外部池保留数量，再记录 FINAL isomer/pool val 与公榜结果。
