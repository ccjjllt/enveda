# 062 进展

- 2026-11-10 00:05 北京：061b ref 58160252 仍 PENDING。公开 notebook 扫描：alexchilton 0.484（ref 58141942，ozertuu + ExtLib TAU0.80，全部公开数据集）、heliosli 0.486（仍挂私有数据集，不可复现）、ozertuu/zz3t3rn41/rajrajak99/chyibo 等 0.485–0.487 标题未见实测记录（多为 alex 版本小改：LIB_HIT_GATE 0.75、NP 提升、LOTUS np_map 增强）。LB #1 0.537。
- 00:0x 北京：构建 `notebooks/062a`（alexchilton 原样，仅 COMMIT_MODE=smoke），推 `ccjjllt/enveda-062a-extlib-tau080` **v1**（QUEUED；GPU 会话被其他项目占用）。watcher `/workspace/k062a/run.sh`（单文件拉取 submission.csv / submission_pre_apex_extlib.csv / apex_extlib_audit.json → 审计 → `-k/-v` 提交）。
- 离线标定（`scripts/062/extlib_tau_calib.py`，匹配逻辑与 overlay 一致；外部库 A 200,724 谱 + B 124,160 谱）：
  - enveda-180 timsTOF 20,000 分子（结构在外部库仅 0.015%）：最佳命中 ≥TAU 比例 0.60:0.60% / 0.65:0.22% / 0.70:0.08% / 0.75:0.04% / 0.80:0.02% / 0.85:0.01%，命中全部错误（即每 400 分子 0.70 阈值约 0.3 个误触发）。
  - pluskal_ms2 等结构在库内 3,000 分子（去 cos≥0.995 同源谱）：TAU 0.70/0.75/0.80/0.85/0.90 触发 45%/43%/40%/35%/27%，精度 0.880/0.889/0.894/0.903/0.920；分段精度 0.70–0.75 = 0.73、0.75–0.80 = 0.82、0.80–0.85 = 0.83；推算 0.65–0.70 ≈ 0.58、0.60–0.65 ≈ 0.47。
  - 结论：TAU 降到 0.70 的增量触发精度 ≥0.73、误触发极低，期望为正；0.65 以下边际精度快速下滑。
- 构建 `notebooks/062b`（TAU 0.70）；推送遇 "Maximum batch GPU session count of 2" → `/workspace/k062b/chain.sh` 每 5 分钟重试推送，成功后接 watcher 自动审计提交。
- 2026-11-10 12:10 北京检查：
  - **061b ERROR 原因**：API `error_description` = "A system error. Please try resubmitting to resolve the error and contact Kaggle Support if it persists."——Kaggle 侧系统错误，与代码/格式无关。对照：`/workspace/kout_061b/submission.csv` 与成功的 `kout_v36/submission.csv` 字节数相同（17303）、400 行、ID 顺序一致、LF、无 BOM，smoke 占位格式一致（仅 12 行第 2–25 名不同，top-1 差 0）；061b 与 061a（成功 0.478）笔记本只差权重常量。→ 12:12 原样以 `-k ccjjllt/enveda-061b-docprior020 -v 1` **重新提交 ref 58185737**（PENDING，今日剩 4 次）。
  - 062a v1 自 00:10 起 QUEUED 12 小时；12:10 时 GPU 会话 = pump-fun-139（另一个项目，10:01 北京起 RUNNING）+ 062b v1（chain 于 12:10 推送成功，立即 RUNNING），062a 仍 QUEUED（疑似队列卡死）。
  - 原 062a watcher 12h 窗口 12:10 到期 → 已停掉并用 `/workspace/k062a/run3.sh`（`run.sh` 副本：`HOURS` 可配，30h；额外拉 `submission_neural.log`；`SKIP_IF` 兄弟已提交则不重复提交；`NEURAL_CHECK` 神经打分日志校验）重启。
  - **GPU 队列绕行**：评估 062a 笔记本只有神经打分器子进程需要 GPU（`device='cuda'` + assert），其余阶段本就是 CPU；时间守卫为整场 8.55h/固定阶段预算，CPU 机器同为 4 vCPU。→ 新建 `notebooks/062a_cpu`（与 062a 唯一差别：两处神经打分调用 `'cuda'→'cpu'`，`enable_gpu=false`），推 `ccjjllt/enveda-062a-extlib-tau080-cpu` **v1**（12:15 RUNNING，CPU kernel 不占 GPU 批会话额度）。watcher `/workspace/k062a_cpu`（run3.sh，NEURAL_CHECK=1，SKIP_IF=/workspace/k062a/SUBMITTED）；GPU 版 watcher 设 SKIP_IF=/workspace/k062a_cpu/SUBMITTED——谁先完成谁提交，避免重复占名额。CPU 浮点与 GPU 可能有极小差异，属于有效复现而非逐字节复现。
  - 062b v1 RUNNING（watcher `/workspace/k062b/watch.out`）；062c 决策器改为 `/workspace/k062c/decide2.sh`（同时识别 GPU 版 "062a alexchilton" 与 CPU 版 "062a-cpu = alexchilton" 提交描述），12:17 重启。
- 12:18 北京：**062a-cpu v1 ERROR**——内层冻结管线 `frozen_0465.ipynb`（SHA256 锁定，cell 28 `_e004_preflight`）强制 `nvidia-smi` 并断言有 GPU；且 Engine 2 两个排序器在 GPU 上**现训**（smoke 下 618 s，训练量与测试规模无关），cell 41 等 12 处 cuda。CPU 上需改冻结管线并重算哈希，训练预计慢 10–50 倍，会触发 8.55h 时间守卫跳阶段 → 不再是有效复现。**结论：CPU 路线不可行**；Colab/本机离线也不可行（代码赛，分数只来自 Kaggle 对 kernel 的隐藏测试重跑，本地无隐藏 test）。CPU watcher 已 FAIL 退出，`notebooks/062a_cpu` 保留作记录，不再使用。
- 维持 GPU 排队：新增 `/workspace/k062a/requeue.sh`——等 062b 结束让出 GPU 会话后再给 062a v1 30 分钟；若仍 QUEUED 则重推 `notebooks/062a` 为新版本，并把 watcher（run3.sh，30h）切到新版本号；不取消 v1。
