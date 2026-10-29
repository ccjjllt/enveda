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
