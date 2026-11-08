# 057 进展

- 2026-11-08 00:10 北京：056 出分 0.481 后启动。源本已拉到 `/tmp/sota057/kozy/`。目标：smoke + MASS + domain-evidence 推 020。
- 2026-11-08 06:05 北京：落地完成并推送。
  - notebook：`notebooks/057/enveda-057-domain-evidence.ipynb`（源自 kozykappa v23；强制 `COMMIT_MODE='smoke'`，保留 `V8='MASS'`、`V23=True`；禁用 BOTH）。
  - metadata：同 056（`kernel_sources` 含 `popwenman/casmi26-scorer-train-v80`，GPU T4，internet off）。
  - 推送：`ccjjllt/enveda-020-e006-repro` **v37**，状态 RUNNING。
  - watcher：`/workspace/k057/run.sh`（对比 `/workspace/kout_v36/submission.csv`；identical top1 → `SKIP_IDENTICAL_056` 不交；拒 placeholder / skipped_fast_commit；COMPLETE 后 `-k/-v` 提交）。
  - 许可：含 sanpier → A 线冲牌。
  - 致谢：kozykappa/Kozykorpesh；上游 alex chilton、popwenman、hoowoo、sanpier、Dmitrii Gluzdov、romanrozen、zz3t3rn41、qwertykeypadapple。

- 2026-11-08 12:16 北京：020 **v37** COMPLETE（约 07:39 北京），watcher 已于 07:41:36 北京提交 ref **58107366**。
  - vs 056：`SAME_TOP1 0 N_DIFF 1 N 400`，仅 1/400 个 top-1 改变。
  - V23 arbitration：`rank1_changed=1`，`promote_leader_without_reference=1`，分子 `m_050bf1` 从 rank 2 → 1（`new_leader_sim=1.0`，`old_leader_sim=null`，`old_rank_of_new_leader=2`）。
  - 其他：`ranks_reordered=12`，`lib_hit_molecules=12`，`molecules_with_outside_hit=0`，merge `leader_moves=0`；evidence module `f1f1c2c44e35`。
  - 产出：`/workspace/kout_v37/`（submission.csv + 日志）。管道日志确认 V23 跑通。
  - **公榜 0.476**（相对 056 的 0.481 退步 −0.005）。单次错误的 rank-1 提升足以解释该幅度（公榜约 125 题，rank1 对错互换约 0.004–0.008）。V23「无参考也提升 leader」这条规则有害，057 **不作新底座**。
  - 最佳仍为 **056 ref 58089862 = 0.481**。下一轮 058 回到 056 底座，叠加有文档的 eason V50 多碰撞能 rank-1 救援（heliosli 0.482 的两个数据集仍私有；「dataset is public now」是 QCC 而非 CASMI）。
