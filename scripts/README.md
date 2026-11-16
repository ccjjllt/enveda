# 研究脚本导航

当前只保留 A 线 063 的推送/看守脚本；001–062 的脚本已随放弃方向清理，结论见[已放弃研究总览](../docs/research-archive.md)。

| 目录 | 用途 |
| --- | --- |
| [063](063/) | `chain.sh`：循环推送 kernel（GPU 会话上限时每 5 分钟重试）；`watch.sh`：等完成、单文件拉取、审计、`-k/-v` 提交 |

## 用法

脚本按远端工作区的路径编写（`/workspace/enveda`、`/workspace/kenv.sh`），在远端 shell 手动执行，不从本机直接运行。当前 068a 的推送顺序：

```bash
NB=068a K=ccjjllt/enveda-068a-n1c-lam035 D=/workspace/k068a O=/workspace/kout_068a \
  MSG="068a N-1c LAM=0.35" bash scripts/063/chain.sh
```

`watch.sh` 会检查 400 行、ID 集合、候选列表合法性，并对比 056 底座统计 top-1 与完整列表差异；只有 `DECISION SUBMIT` 才提交。审计失败写 `NOT_SUBMITTED`，不重试提交。

脚本存在不代表已经运行；实际版本、日志、分数与配额见[068 进展](../docs/progress/068-n1c-lam035.md)。
