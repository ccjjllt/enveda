# 054 进展

- 2026-11-06 18:20 北京：巡检确认 054 仅下载了公开 notebook（`/workspace/pub051/roman482`），**未建 notebooks/054、未推 kernel、无 watcher**。020 当前 COMPLETE（v33=053）。
- 原 `romanrozen/0-482-casmi-from-peaks-to-structures` 的 `kernel_sources: deleted/142141665`；与 jtheyogi fork 的 notebook 字节级一致，改用 `bgunshui/casmi26-scorer-v5-model` + `qwertykeypadapple/casmi26-lotus-candidates`。
- 已创建 `notebooks/054/enveda-054-romanrozen-0482.ipynb` + kernel-metadata（挂到 020），准备推送 v34。

- 2026-11-06 18:25 北京：准备 `kaggle kernels push` → 020 v34；watcher 用 `-k/-v` 提交（避开 output 下载挂死）。

- 2026-11-06 18:30 北京：v34 推送后 watcher 误把 **fast-commit 占位** submission（全分子同一串 `CO;CCO;…`，日志 `skipped_fast_commit`）用 `-k/-v 34` 提交为 ref **58054713** PENDING。原 notebook `COMMIT_MODE='fast'`，完整管线只在 `KAGGLE_IS_COMPETITION_RERUN` 时跑；CLI `-k/-v` 若只取 commit 产物则会交占位（预期公榜≈0）。
- 2026-11-06 18:40 北京：已改 `COMMIT_MODE='smoke'`（与 051 一致，commit 即跑全管线），推送 **020 v35**；watcher `/workspace/k054/run_v35.sh`（先等到 RUNNING 再等 COMPLETE；拒绝 top1 全相同占位）再用 `-k/-v 35` 提交。
