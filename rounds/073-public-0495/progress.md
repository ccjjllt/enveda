# 073 进展

## 2026-11-18 07:24 UTC 推送 v1

- 复查公榜：top1 0.550；0.500 → 第 30 名；0.495 → 第 83–120 名（17 队同分）；**前 600 名全部 ≥0.492**。0.492 已是前 600 的地板，先前「0.492 在第 100–230 名」的记录过期。
- 公开 notebook 逐个核对作者公榜：heliosli（0.495 demo）不在前 600；flexonafft/Igor Zharov、heetp0/Heet Patel、zz3t3rn41、ozertuu 均为 **0.492**；`ccuuuuy`（Dante）的 network-rerank fork 为 **0.494**。⇒ 公开复现上限应从 0.492 修正为 **0.494**，没有任何公开 notebook 的作者持有 >0.494。
- 读到 0.495 源码：`CUSTOM(w_top=0.25, w_mid=0.75, rescue=0)` + `PHYSICAL=True`（未训练 MetFrag-lite 碎片检查）+ `FRAG_ON_V2=True`（只采纳碎片检查的榜首移动）。依赖 lightgbm / rdkit / torch，与我们的镜像能力一致。
- 复制为 `jiyujia/enveda-073a-0495-repro`，保持原 docker 镜像与 14 个数据集，`is_private: true`，07:24 UTC 推送 → `RUNNING`（镜像与数据集均可挂载，未在元数据阶段失败）。

### 待办

- [ ] kernel `COMPLETE` 后核对日志各阶段与 `submission.csv` 行数/格式
- [ ] 用 1 次提交验证公榜
