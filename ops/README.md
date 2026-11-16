# 远端操作

比赛规则、数据、Kaggle CLI、提交与配额：[remote-workflow.md](remote-workflow.md)。

## 推送/看守脚本（`063/`）

`chain.sh`：循环推送 kernel（GPU 会话上限时每 5 分钟重试）；`watch.sh`：等完成、单文件拉取、审计、`-k/-v` 提交。按远端工作区路径编写（`/workspace/enveda`），在远端 shell 手动执行，不从本机直接运行。

```bash
R=068-n1c-lam035 NB=068a K=ccjjllt/enveda-068a-n1c-lam035 D=/workspace/k068a O=/workspace/kout_068a \
  MSG="068a N-1c LAM=0.35" bash ops/063/chain.sh
```

`watch.sh` 会检查 400 行、ID 集合、候选列表合法性，并对比 056 底座统计 top-1 与完整列表差异；只有 `DECISION SUBMIT` 才提交。脚本存在不代表已经运行；实际版本、日志、分数与配额见各轮次的 `progress.md`。
