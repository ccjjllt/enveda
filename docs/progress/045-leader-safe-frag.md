# 045 进度

## 离线 W_MID 扫描（CPU，notebooks/045/sweep_wmid.py）
- 集合：044 同一 holdout（seed 20261025, 2%），真分子 + ext pool 同分子式候选（取 300，按 baseline 取前 24），600 个分子。
- baseline 代理：prvsiyan FPNet（Engine2）对数似然；neural：model3（fp_w 同 044）。
- 注意：FPNet 在全量 train 上训练（泄漏），baseline 偏强，绝对值仅作相对比较。

| 方案 | MRR | top1 |
|---|---|---|
| baseline | 0.9806 | 0.9733 |
| model3 | 0.9580 | 0.9367 |
| V1 W0.25 | 0.9823 | 0.9750 |
| LS 0.5 | 0.9834 | 0.9750 |
| **LS 0.75** | **0.9841** | 0.9750 |
| LS 1.0 | 0.9838 | 0.9750 |

结论：0.75 最优，与公榜趋势一致（0.5→0.468，0.75→0.472，1.0→0.470），选定 W_MID=0.75。

## Notebook
- notebooks/045/build.py 从 044 生成 enveda-045a-leader-safe.ipynb 与 enveda-045b-frag.ipynb；frag_physical.py 为移植的 MetFrag-lite。
- 本地单元测试：leader-safe 排序正确；碎片规则在构造样例上触发、门槛不满足时返回 None；规则异常只记 frag_err，不影响提交。

## 队列
- 旧 k044_retry（PID 10881）已停，替换为 /workspace/k045_queue.sh（setsid nohup），当前每 30 分钟重试，GPU 周配额已满。

## 巡检 2026-10-29 12:00 北京
- 仍无新提交；队列存活（sleep 1800 轮询），最近失败仍为周 GPU 配额满。
- 公开侧未见稳定突破 0.477 的可复现新配方；ahmed v33 残差有启发但依赖 sanpier，记入 046 草稿。

## 巡检 2026-10-29 18:05 北京
- 仍无新提交（最佳仍 v19 0.465 / 最近 v23 0.464）；k045_queue.sh 存活（PID 轮询 sleep 1800），最近失败仍为周 GPU 配额满；距重置约 38h（10-31 08:00 北京）。
- LB 4307 队；team ccjjllt 0.465 约第 883；#1 0.516；约第 18≈0.481；≥0.477 约 299 队、≥0.472 约 488 队。
- 公开侧同日仍大量 busyaprime/0.477 fork；guhongbin `casmi26-v537-frag-wmid075` 等同向（frag + W_MID=0.75），与 045 方向一致，无新可复现突破需改队列。
- 046 仍等 045 公榜，未开工。

## 巡检 2026-10-30 06:07 北京
- 仍无新提交（最佳仍 v19 0.465 / 最近 v23 0.464）；k045_queue.sh 存活（setsid 于 00:31 北京重启后仍在，PID 轮询 sleep 1800），最近失败仍为周 GPU 配额满（日志至 06:02）；距重置约 26h（10-31 08:00 北京）。
- LB 下载快照约 4345 队；team ccjjllt 0.465 约第 929；#1 0.517；约第 18≈0.482；银 5%≈0.477（~217）、铜 10%≈0.472（~434）。
- 045a/b notebook 与 model3.pt（ccjjllt/enveda-own-spec）仍就绪；046 仍等 045 公榜，未开工。无需叫醒用户。

## 巡检 2026-10-31 12:09 北京
- **045a 已推送并提交**：10:17 北京 `kaggle kernels push` 成功（kernel slug `ccjjllt/enveda-020-e006-repro` v21）；10:19–10:48 RUNNING→COMPLETE；`k_watch` 已提交 ref `57824441`（描述：045a leader-safe W_MID0.75 on model3），当前 **SubmissionStatus.PENDING**，尚无 publicScore。
- `k045_queue.sh` 仍存活（PID 106020，自 00:20 北京起；`do_wait`），按脚本在 045a 提交日后一天 08:05 北京唤醒后推送 045b。
- 最佳公榜仍 v19 **0.465**（最近完整分 v23 0.464）；LB 快照约 **4423** 队；team ccjjllt 0.465 约第 **1032**；#1 0.517；约第 18≈0.482；银/铜 5%/10% 分位均≈0.477（公开 SOTA 平台堆积）。
- 046 仍等 045 公榜，未开工。无新分、无 blocker，不叫醒用户。

## 045a 结果 2026-10-31（北京）
- v24（ref `57824441`，045a = model3 + leader-safe W_MID 0.75）公榜 **0.462**，低于 v21（自研 W0.25，0.464）和最佳 v19（0.465）。
- 结论：**W_MID 0.75 对 model3 有害**。公开方案的 0.75 建立在 sanpier 更强的模型上，换成我们较弱的自研模型后不成立。
- 045b 已改：不再叠加 045a。底座改为 model3 全名次 W0.25（即 044 融合），只加碎片 rank-1 规则：规则在 W0.25 序上运行，原第 1 名降到第 2，其余保持 W0.25 序。日志标签仍为 "045 FRAG"。k045_queue.sh 推送时读取重建后的 notebook（11-01 08:05 北京）。
