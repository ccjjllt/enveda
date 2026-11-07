# 057 进展

- 2026-11-08 00:10 北京：056 出分 0.481 后启动。源本已拉到 `/tmp/sota057/kozy/`。目标：smoke + MASS + domain-evidence 推 020。
- 2026-11-08 06:05 北京：落地完成并推送。
  - notebook：`notebooks/057/enveda-057-domain-evidence.ipynb`（源自 kozykappa v23；强制 `COMMIT_MODE='smoke'`，保留 `V8='MASS'`、`V23=True`；禁用 BOTH）。
  - metadata：同 056（`kernel_sources` 含 `popwenman/casmi26-scorer-train-v80`，GPU T4，internet off）。
  - 推送：`ccjjllt/enveda-020-e006-repro` **v37**，状态 RUNNING。
  - watcher：`/workspace/k057/run.sh`（对比 `/workspace/kout_v36/submission.csv`；identical top1 → `SKIP_IDENTICAL_056` 不交；拒 placeholder / skipped_fast_commit；COMPLETE 后 `-k/-v` 提交）。
  - 许可：含 sanpier → A 线冲牌。
  - 致谢：kozykappa/Kozykorpesh；上游 alex chilton、popwenman、hoowoo、sanpier、Dmitrii Gluzdov、romanrozen、zz3t3rn41、qwertykeypadapple。
