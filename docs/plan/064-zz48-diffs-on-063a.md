# 064 计划：在 063a（0.487）底座上移植 zz3 仍可能生效的差异（A 线）

## 背景
- 063a ref 58233824 与 063b ref 58233993 公榜均为 **0.487**（当前最佳）。A 轨底座为 063a：062a 代码 + popwenman scorer v80、不挂 LOTUS、guhongbin docker。LOTUS 已由 063b 证明在 3 位小数下中性。
- 063a 配置是 `V8='MASS'`（只开 timsTOF 质量先验），因此 `RR8 = False`，异构体重排关闭，`RR8_ALPHA` 不会被调用。
- 榜单快照（2026-11-11 22:04 UTC / 11-12 06:04 北京，5150 队）：团队 ccjjllt 第 **509** 名；恰好 0.487 的平台为第 79–596 名共 518 队；≥0.488 共 78 队。要跳出平台需 ≥0.488。公开标题「0.488」或「0.500+」仍未核实为可复现配方。

## 已核对、且必须修正的原 064a 设想
- 原计划优先做「063a 只改 `RR8_ALPHA` 1.0→0.25」。**这是空操作，已取消，不要推。** zz3 线把 alpha 改成 0.25 是因为它同时把 `V8` 设为 `'BOTH'`（重排开启）；063a 仍是 `'MASS'`，只改 alpha 不会改变任何列表。
- 公开证据也不支持先开重排：063a 笔记里的已有消融写明，在 Gold Apex V22 上把 `V8` 从 `'MASS'` 改成 `'RR'`/`'BOTH'` 会从 0.481 掉到 0.460/0.461。ozertuu 标题为「0.500+」的 `ozertuu/0-500-casmi26-dual-neural-physical-isomer`（2026-11-11 16:36 UTC 跑完后提交）用的是 `V8='BOTH'`、`RR8_ALPHA=0.20`、`COMMIT_MODE='fast'`，作者 Ertuğrul Özer 公榜仍是 **0.487**。rogersjohnson `casmi26-s3-alpha10`（`V8='BOTH'`、`RR8_ALPHA=1.0`）之后作者也仍是 0.487。
- 当前官方 `zz3t3rn41/casmi26-peaks-to-structures-v2` 与 rogersjohnson `zz3v49-1111` 均为 `V8='BOTH'`、`RR8_ALPHA=0.25`、`LIB_HIT_GATE=0.75`，不挂 v80、另挂 bgunshui scorer v2 与 LOTUS，作者公榜仍是 0.487。zz3 现已挂上 `youyouwou/casmi26-v9np-bundle`，但作者分数未因此跳出 0.487，不把 v9np 当作本轮主实验。
- 063a 的外部库覆盖已是 `TAU=0.80`、`K_PEAKS=4`，与 zz48 同名常量一致，不是差异。

## 实验（等 GPU 配额刷新后再推）
单旋钮、一次只改一项，元数据与 063a 对齐（v80、无 LOTUS、同一 docker），`COMMIT_MODE='smoke'`。
1. **064a**（已在 `notebooks/064a` 备好，尚未推送）：063a 底座只改活旋钮 `LIB_HIT_GATE` 0.78→0.75（最终层 ranks 2–25 的谱库加分门控，`V8='MASS'` 下会实际触发）。内核 slug：`ccjjllt/enveda-064a-libhit-gate075`。
2. 若 064a 不退步，再考虑加合物感知谱库对齐（代码级差异，不是单常量）。
3. 不要把「只改 `RR8_ALPHA`」或「先把 `V8` 改成 `'BOTH'`」当作下一刀。

审计/提交沿用 `scripts/063/watch.sh` 逻辑（单文件拉取、退避、smoke 门槛、通过则 `-k -v`）。

## 配额
完整隐藏重跑约 6–9 小时。本周 GPU 配额（`quota_view`，2026-11-12 06:04 北京）`timeUsed` 按 12044.734 s 读、总量 21600 s，约剩 **2.65 小时**，刷新时间 2026-11-14 08:00 北京。pump-fun 内核已全部 COMPLETE、当前无在跑 GPU 内核，但剩余仍不够一次完整推理，**刷新前不推送**。

## 许可
A 线（公开 notebook 差异移植，sanpier 等模型许可 unknown），仅冲奖牌/名次。


## 2026-11-14 更新
公开 0.488 已核实为 N-1c 网络重排（见 [065 计划](065-network-n1c.md)）。064a 仍保留，但 **2026-11-14 08:00 北京 GPU 刷新后的第一刀改为 065a**，不先推 `LIB_HIT_GATE`。
