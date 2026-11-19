# 远端操作

比赛规则、数据、Kaggle CLI、提交与配额：[remote-workflow.md](remote-workflow.md)。

## 推送/看守脚本（`063/`）

`chain.sh`：循环推送 kernel（GPU 会话上限时每 5 分钟重试）；`watch.sh`：等完成、单文件拉取、审计、`-k/-v` 提交。按远端路径（`/workspace/enveda`）编写，在远端 shell 手动执行：

```bash
R=068-n1c-lam035 NB=068a K=ccjjllt/enveda-068a-n1c-lam035 D=/workspace/k068a O=/workspace/kout_068a \
  MSG="068a N-1c LAM=0.35" bash ops/063/chain.sh
```
