# 060 计划：056 底座 + 第 2–25 名文档先验权重 0.10/0.10 → 0.12/0.12（rank 1 不动）

## 背景
- A 线最佳：056（ref 58089862）= 059（ref 58124700）= 公榜 **0.481**。
- 057（V23 无参照 leader 提升）0.476、058（V50 多碰撞能 rank-1 救援）0.473：**改 top-1 均退步**；059 第 6 名门控插入中性。
- 指标是 MRR@25：真值在第 2 名得 0.5、第 3 名 0.33，重排 2–5 名仍有真实收益，且不碰 rank 1。历史上能让公榜涨分的非 top-1 旋钮只有两个：ranks 2–25 的 w_mid（0.468→0.472）与 baidalinadilzhan 文档先验（0.477→0.480）。

## 公开扫描（2026-11-09 06:00 北京）
| 候选 | 结论 |
|---|---|
| ozertuu/0-485-sota-casmi26-from-peaks-to-structures、ozertuu/casmi26-apex-gold-486（代码 md5 相同） | = 056 + `POP_W, SPEC_W` 0.10→0.12 + 放宽两个 sha 断言 + 末尾 extlib（takumuhata + MassBank harmonized）cos≥0.85 / K≥4 / lib_max<0.90 **直接 rank-1 提升**；`kernel_sources=[]`（无 popwenman v80）、`COMMIT_MODE='fast'`。CLI 按分数排序位列第 2（heliosli 0.482 之后、alexchilton 0.481 之前），具体分数不可见；标题 0.485/0.486 无凭据。rank-1 提升部分与 052/058 同机制（均退步），不采纳；只取 ranks 2–25 旋钮。 |
| roshanrobinc sovereignzenith v32 "0.486+" | 同一 ozertuu 谱系 + v80，换皮，无新机制证据。 |
| zz3t3rn41/casmi26-peaks-to-structures-v2、ndaedzomukhuba/enveda-apex-0500 | 无 V8 的旧线 + bgunshui v2/v7 三神经集成 + extlib rank-1 overlay；非 056 增量，分数未证实。 |
| ezberch v9 rr8 | V8='BOTH' + fast（054 失败配方）。 |
| duongynhi gold-apex | popwenman 0.482 V22 复刻 + 额外 seeds/specialist 数据集，fast。 |

## 方法
- 056 原文只改 config：`POP_W, SPEC_W, NP_W = 0.12, 0.12, 0.12`（056 为 0.10/0.10/0.12）。这些权重只进入 `pxm_merge` 中 ranks 2–25 的 `rest.sort`，leader 在排序前已定。
- cell 12 追加同轮反事实审计：用 056 权重再跑一次 `pxm_merge`，记录 diff_full / top1 / top3 / top5 到 `g60_audit.json`；若 top1 有任何差异则整体回退到 056 合并结果。
- 后续 v8 质量 rank-1 规则（cell 13）可能因 2/3 名顺序变化而选不同候选，概率极低（leader 质量偏离 >5ppm 且库 <0.9 且 2/3 名同时在带内）。
- 不叠 057/058/059；`COMMIT_MODE='smoke'`、`V8='MASS'`、popwenman v80，元数据与 056 相同。

## 提交规则
- 020 v40；watcher `/workspace/k060/run.sh`：RUNNING→COMPLETE；需 submission.csv + g60_audit.json；status=ok、可见集 top1 与 056 一致、审计 diff_top1=0 才 `-k/-v 40` 提交。
- Kaggle 返回 "3 submissions remaining today"（11-08 UTC 日已用 2 次）→ 实际日上限为 5 次/UTC 日（北京 08:00 重置），非 2 次。

## 预期
- 小幅（±0.001–0.002）；若 ≥0.482 则继续扫 0.14/0.16 与 NP_W；若 ≤0.480 则反向扫 0.08。

## 许可
含 sanpier → A 线冲牌。

## 致谢
ozertuu（权重旋钮）、baidalinadilzhan（文档先验层）、alex chilton、popwenman、hoowoo、sanpier、Dmitrii Gluzdov、romanrozen、zz3t3rn41、qwertykeypadapple。
