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
