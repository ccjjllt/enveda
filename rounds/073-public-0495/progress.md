# 073 进展

## 2026-11-18 07:24 UTC 推送 v1

- 复查公榜：top1 0.550；0.500 → 第 30 名；0.495 → 第 83–120 名（17 队同分）；**前 600 名全部 ≥0.492**。0.492 已是前 600 的地板，先前「0.492 在第 100–230 名」的记录过期。
- 公开 notebook 逐个核对作者公榜：heliosli（0.495 demo）不在前 600；flexonafft/Igor Zharov、heetp0/Heet Patel、zz3t3rn41、ozertuu 均为 **0.492**；`ccuuuuy`（Dante）的 network-rerank fork 为 **0.494**。⇒ 公开复现上限应从 0.492 修正为 **0.494**，没有任何公开 notebook 的作者持有 >0.494。
- 读到 0.495 源码：`CUSTOM(w_top=0.25, w_mid=0.75, rescue=0)` + `PHYSICAL=True`（未训练 MetFrag-lite 碎片检查）+ `FRAG_ON_V2=True`（只采纳碎片检查的榜首移动）。依赖 lightgbm / rdkit / torch，与我们的镜像能力一致。
- 复制为 `jiyujia/enveda-073a-0495-repro`，保持原 docker 镜像与 14 个数据集，`is_private: true`，07:24 UTC 推送 → `RUNNING`（镜像与数据集均可挂载，未在元数据阶段失败）。

### 待办

- [ ] kernel `COMPLETE` 后核对日志各阶段与 `submission.csv` 行数/格式
- [ ] 用 1 次提交验证公榜

## 2026-11-18 07:55 UTC commit run 完成（smoke）

`jiyujia/enveda-073a-0495-repro` v1 `COMPLETE`，用时约 30 分钟——**符合 smoke 预期**（commit run 只算 12 个可见分子，隐藏重跑才跑 400）。

内核自报的关键回执：

```
candidate_audit.json: hidden_rerun false | active_molecules 12 | output_rows 400 | changed_molecules 0
execution_receipt.json: mode submission | is_full_run false | processed_molecules 12
cvjoint_overlay_audit.json: mode swap | molecules 400 | candidates 76894 | agree 365 | swapped 0 | below 35
stages: ICE ok (11/11 覆盖, 77.9s) | GLACIER ok (11/11, 43.2s) | runtime 1172.4s
```

- ICE/GLACIER 两个神经打分阶段在我们镜像下都跑通（cuda，RDKit 2026.03.3），没有 error。
- `swapped 0` / `changed_molecules 0` 是可见 12 分子的正常结果（它们本来就是库命中），真正差异只会在隐藏重跑出现。
- `output_rows 400`：提交文件已按 400 行写出，格式与比赛要求一致。

下一步：确认 `submission.csv` 落到本地、校验行数/候选数/无重复，然后用 1 次提交触发隐藏重跑。

## 2026-11-18 08:15 UTC 判决：demo 空壳，停止复现

提交被拒，错误码 400：

```
Submission not allowed: Your Notebook uses non-versioned datasets
[/datasets/heliosli/casmi26-joint-ranker, ...rr-v3, ...m5-v2d, ...frigid-e1,
 ...pxv2-deploy, ...rt-np, ...analyticon-megx]
```

查这 7 个数据集的版本说明后，实验到此结束——作者把真实权重删掉了：

```
PUBLIC-DEMO-1116-R1: Runnable demonstration artifacts; original competition weights excluded.
Not a reproduction of 0.494 ... has no claimed leaderboard score.
Private learned parameters are replaced with initialized demo weights
(the joint ranker uses a synthetic fit).

analyticon-megx: Synthetic molecular catalogue ... 3,683 generated examples.
```

三条独立证据互相吻合：

1. 数据集描述明说权重是 demo、catalogue 是合成、**不声称任何公榜分数**；
2. 作者不在公榜前 600（前 600 全部 ≥0.492）；
3. 我们的运行里 `rr3 reranked 0 / p1gate scored 0 / m5a actionable 0 / cvjoint swapped 0`，符合「随机初始化权重没有判别力」。

### 结论

- **0.495 不可复现，也不该再花 GPU/提交。** 标题里的 0.495 是演示标签，作者自己的数据集写的是 0.494。
- 公开可复现前沿 = **0.494**（Dante 的 network-rerank fork，真实公榜 0.494），比 0.492 只高 +0.002。
- 0.500+ 的 30 支队没有任何公开 notebook。

### 成本

GPU 已用 21.89h → **22.33h**（这次 smoke 用了 0.44h，剩 7.67h），提交 **0 次**（今日仍 1/5）。`073a` kernel 保留为只读记录，不再推送新版本。
