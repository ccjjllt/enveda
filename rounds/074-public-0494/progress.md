# 074 进展

## 2026-11-18 09:20 UTC 推送 v1

- 选定方案：`ccuuuuy/casmi26-0-492-test-set-network-rerank`（作者 Dante，公榜 **0.494**）。这是唯一「作者本人真实持有 >0.492」的公开 notebook；073 的 0.495 demo 已证明是空壳。
- 读到源码：主线是 alexchilton 0.488 网络 + `LAM 0.35`（=0.492，与我方 065a/068a 同源），再叠 `CUSTOM(w_top=0.25, w_mid=0.75)` 共识、`PHYSICAL` 碎片检查、`FRAG_ON_V2` 只采纳榜首移动、`ENABLE_K14_DEDUP`、以及 baidalinadilzhan 的文档先验层。作者的 0.494 应来自这些叠加。
- 唯一改动：`COMMIT_MODE 'fast' → 'smoke'`，让 commit run 真正跑各阶段并写出合法 400 行文件（约 0.5h GPU）；隐藏重跑原本就是全量。
- 数据集与镜像与原作者完全一致（10 个真实数据集 + `popwenman/casmi26-scorer-train-v80`），非 demo 资产。
- 推送 `jiyujia/enveda-074a-dante-0494-repro` v1（私有，T4）。

### 待办

- [ ] commit run `COMPLETE` 后核对阶段日志与 `submission.csv` 400 行
- [ ] 提交 1 次，等公榜：≥0.494 记提分；=0.492 记中性；<0.492 记有害

## 2026-11-18 10:21 UTC commit run COMPLETE + 已提交

commit run 用时约 20.5 分钟（smoke），关键日志：

```
done in 19.6 min              (冻结 0.465 管线)
PARALLAX REPORT rows 400 changed_rows 2 top1_changed ...
SUBMISSION READY: rows 400 control CUSTOM
MERGE applied rows 400 ranks_reordered 12 dedup_swaps 0 leader_moves 0 output_equals_custom false
SUBMISSION READY: molecules 400 | V8 MASS | mass lead ran | RR off in this preset
```

两点值得注意：

1. 这份 notebook 的底座同样是 `V8='MASS'`（`RR off in this preset`），所以它的增益完全来自共识/物理/去重/先验那几层，与 RR8 无关。
2. `dedup_swaps 0`、`leader_moves 0`：K14 去重和碎片检查在 12 个可见分子上没有动作，与 072e/073a 的现象一致（可见分子都是库命中、已是满分榜首）。

提交：**ref `58505232`**，10:21 UTC，今日第 2 次（剩 3 次）。隐藏重跑约 2.7h，出分后记录。

## 2026-11-18 15:08 UTC 公榜：0.492（与现有最佳打平，未复现 0.494）

`58505232` `COMPLETE`，**public 0.492**，提交到出分 4h47m（10:21 → 15:08）。

| 对比 | 分数 |
| --- | --- |
| 我们 068a | 0.492 |
| 074a（Dante 叠加方案） | **0.492** |
| 作者公榜 | 0.494 |

结论：

1. **这份 notebook 实际只到 0.492。** 它叠的四层（共识 `w_mid=0.75`、物理碎片检查、K14 去重、文档先验）在隐藏集上没有净收益——与我们 065a/068a 同源主线打平。
2. **作者的 0.494 不是从这份公开代码得到的**，只能来自他没有公开的私有变体。这与「top-30 无公开物」的结论一致。
3. 公开可复现前沿就此确认在 **0.492–0.494** 之间，且 0.494 无可复现路径；073 的 0.495 是 demo 空壳。
4. `ranks_reordered 12 / dedup_swaps 0 / leader_moves 0`：这些层在 12 个可见分子上也没动作，与最终打平一致。

不提交新版本，本方向关闭。资源：今日提交 3/5 用掉（072e、074a、? 按实际），GPU 见配额。
