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
