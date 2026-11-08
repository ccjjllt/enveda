# 058 进展

- 2026-11-08 12:20 北京：057 出分 0.476（退步）后启动。公开扫描结论见 plan：heliosli 0.482 数据集仍私有；popwenman 标题 0.482 = 056 同方；claw-maxzero 是 0.480 线。选定 eason V50 多碰撞能 rank-1 救援叠在 056 上。
- 构建：`notebooks/058/enveda-058-multice-rank1-rescue.ipynb`（056 原文 + 原样 V50 cell），`COMMIT_MODE='smoke'`，`V8='MASS'`，metadata 增挂 `sergiomalv/casmi-open-msms-spectra`，保留 popwenman v80。
- 许可：含 sanpier → A 线冲牌。
- 致谢：easonyanyan、sergiomalv；上游 alex chilton、popwenman、hoowoo、sanpier、Dmitrii Gluzdov、romanrozen、zz3t3rn41、qwertykeypadapple。

- 2026-11-08 12:18 北京：已推 `ccjjllt/enveda-020-e006-repro` **v38**，状态 RUNNING。watcher `/workspace/k058/run.sh`：先观察到 RUNNING 再等 COMPLETE；拒 placeholder / skipped_fast_commit；`v50_xl_audit.json` 为 ok 时即使 commit top1 等于 056 也用 `-k/-v 38` 提交（可见 smoke 的 LIB_MAX=1.0，V50 按设计会 skipped_library）；库未挂载或自检失败且 top1 相同则不交。对比底座为 056 `/workspace/kout_v36/submission.csv`（057 已退步）。

- 2026-11-08 14:57 北京：v38 COMPLETE。`v50_xl_audit.json` status=ok，hits=5、changed=5、inserted=5、skipped_library=12；与 056 对比 N_DIFF=5（m_3f8028、m_60e7b1、m_7b993b、m_80b537、m_bbc0aa）。watcher 用 `-k/-v 38` 提交 ref **58118277**，18:20 北京仍 PENDING（历史出分约 5 小时）。输出 `/workspace/kout_v38/`。
- 2026-11-09 06:10 北京复核：ref 58118277 COMPLETE 公榜 **0.473**（相对 056 −0.008），V50 rank-1 救援有害，不作底座。
