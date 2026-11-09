# 062 进展

- 2026-11-10 00:05 北京：061b ref 58160252 仍 PENDING。公开 notebook 扫描：alexchilton 0.484（ref 58141942，ozertuu + ExtLib TAU0.80，全部公开数据集）、heliosli 0.486（仍挂私有数据集，不可复现）、ozertuu/zz3t3rn41/rajrajak99/chyibo 等 0.485–0.487 标题未见实测记录（多为 alex 版本小改：LIB_HIT_GATE 0.75、NP 提升、LOTUS np_map 增强）。LB #1 0.537。
- 00:0x 北京：构建 `notebooks/062a`（alexchilton 原样，仅 COMMIT_MODE=smoke），推 `ccjjllt/enveda-062a-extlib-tau080` **v1**（QUEUED；GPU 会话被其他项目占用）。watcher `/workspace/k062a/run.sh`（单文件拉取 submission.csv / submission_pre_apex_extlib.csv / apex_extlib_audit.json → 审计 → `-k/-v` 提交）。
- 离线标定（`scripts/062/extlib_tau_calib.py`，匹配逻辑与 overlay 一致；外部库 A 200,724 谱 + B 124,160 谱）：
  - enveda-180 timsTOF 20,000 分子（结构在外部库仅 0.015%）：最佳命中 ≥TAU 比例 0.60:0.60% / 0.65:0.22% / 0.70:0.08% / 0.75:0.04% / 0.80:0.02% / 0.85:0.01%，命中全部错误（即每 400 分子 0.70 阈值约 0.3 个误触发）。
  - pluskal_ms2 等结构在库内 3,000 分子（去 cos≥0.995 同源谱）：TAU 0.70/0.75/0.80/0.85/0.90 触发 45%/43%/40%/35%/27%，精度 0.880/0.889/0.894/0.903/0.920；分段精度 0.70–0.75 = 0.73、0.75–0.80 = 0.82、0.80–0.85 = 0.83；推算 0.65–0.70 ≈ 0.58、0.60–0.65 ≈ 0.47。
  - 结论：TAU 降到 0.70 的增量触发精度 ≥0.73、误触发极低，期望为正；0.65 以下边际精度快速下滑。
- 构建 `notebooks/062b`（TAU 0.70）；推送遇 "Maximum batch GPU session count of 2" → `/workspace/k062b/chain.sh` 每 5 分钟重试推送，成功后接 watcher 自动审计提交。
