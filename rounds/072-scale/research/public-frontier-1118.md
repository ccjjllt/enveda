# 公开前沿复查（2026-11-18 07:00 UTC）

目的：回答「现在有没有分数更高的公开方案」。数据全部来自 Kaggle 实时读取（`get_competition_leaderboard` 600 行、`search_notebooks` 按分数/日期排序、`kaggle kernels pull` 实际源码）。

## 1. 公榜已经整体上移，0.492 不再是前 100

```
top1 0.550 (Shehab Alshehabi)   top2 0.540   top3 0.536   top4 0.532   top5 0.523
0.500 -> 第 30 名
0.496 -> 第 55 名
0.495 -> 第 83-120 名（17 支队同分）
0.494 -> 第 83 名起（51 支队在 0.494-0.496）
>=0.492 -> 前 600 名全部 >=0.492（600/600 行）
```

我们 068a 的 0.492 已落在**前 600 名的地板上**。11-16 记录里「0.492 在第 100–230 名」已经过期。

## 2. 公开 notebook 的高分标题基本不可验证

按「分数」排序的前 8 个 notebook，逐个核对其作者在公榜的位置：

| notebook | 标题声称 | 作者 | 作者公榜 |
| --- | --- | --- | --- |
| `heliosli/casmi26-0-495-public-demo` | 0.495 | Heliosli | **不在前 600** |
| `flexonafft/casmi26-anchored-spectral-network` | —（49 票） | Igor Zharov | 0.492 |
| `heetp0/notebook0b6a5a21c6` | — | Heet Patel | 0.492 |
| `zz3t3rn41/casmi26-peaks-to-structures-v2` | —（131 票） | zz3t3rn41 | 0.492 |
| `ozertuu/0-492-casmi26-dual-neural-physical-isomer` | 0.492 | Ertuğrul Özer | 0.492 |
| `mathleteutd/casmi26-0-492-test-set-network-rerank` | 0.492 | Rohan | — |
| `ccuuuuy/...network-rerank` | 0.492 | **Dante** | **0.494**（其 fork 拿到 0.494） |
| `gloriachen0816/casmi26-0510-trainret` | 0.510 | Siyuan Chen | 不在前 600 |

**结论：没有任何一个公开 notebook 的作者在公榜上持有 >0.494 的分数。** 0.495 是 17 支队真实达到的水平，但公开可读的那份 0.495 来自一个不在前 600 的账号，属于**未验证声明**。0.500+ 的 30 支队没有公开 notebook。

注意先前判断需修正：0.494 的 fork 是真实存在的（Dante 0.494），所以「公开复现上限 0.492」应改为 **0.494**。

## 3. 0.495 那份 notebook 到底改了什么

源码已拉到本地（30 cells，735k 字符，主内容是冻结的 0.465 管线 + 嵌入模块）：

```
V0 冻结 0.465 基线
V1 神经 RRF，全局权重 0.25                       -> 0.466
V2 rank1 权重 0.25，rank2-25 权重 0.50（上游默认）
V3 V2 + 尾部 3 个位置允许外来候选
V4 rank2-25 权重 1.00
本 notebook: CUSTOM(w_top=0.25, w_mid=0.75, rescue=0) + PHYSICAL 碎片检查 + FRAG_ON_V2 合并
```

- 融合式：`1/(3 + rank)` 的 reciprocal-rank fusion，**榜首用 0.25 的神经权重**（leader-safe），**2–25 名用 0.75**。
- `PHYSICAL=True`：跑一个**未训练的 MetFrag-lite 碎片解释度重打分**（≤2 键断裂），在 V2 顺序上执行，之后的合并只采纳它的**榜首移动**。
- 依赖的 14 个数据集（`sanpier/casmi-spectrum-model-v1`、`dmitriigluzdov/casmi26-*`、`heliosli/casmi26-*` 等）用我们账号逐个查过，**全部可读**，不是作者私有。

与我们的 068a 家族对比：068a 是「mass prior + N-1c + parallax/merge」，这条线是「神经 RRF 共识 + 碎片检查」。065a/066/067 曾做过同族的 mid-rank 叠加（0.488 中性），但**从未测过 `w_mid=0.75` + 物理碎片检查 + 只保留榜首移动**这一组合。

## 4. 可行动作与成本

1. 把 `heliosli/casmi26-0-495-public-demo` 原样复制到我们账号推送，跑一次（T4，约 2.6–2.8h GPU），确认数据集能挂载、产出合法 400 行 submission。
2. 用 1 次提交验证公榜是否真的 0.495。今日剩 4 次。
3. 若复现为 0.495：把它作为新底座，再考虑能否在其上叠加我们已验证有效的部件；若失败（数据集版本/路径），成本仅一次失败的 kernel 运行。

风险：0.495 相对 0.492 只差 0.003（约 0.4 个分子），而复现差异可能吞掉它；但 17 支队同分说明这个平台是真实的，不是单次噪声。
