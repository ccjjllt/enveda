# 072e：启用重排 + 保证金式榜首晋级（V8='BOTH'）

日期：2026-11-18。072d 仪器指出唯一还有空间的位置：150 个留出分子里有 **42 个真值停在名次 2**，而 0.492 配方把重排整条关掉了（`V8='MASS'` ⇒ `RR8=False`）。本实验就是那条从未尝试过的配置。

## 改动（相对 068a，仅两处）

1. `V8 = 'MASS'` → `V8 = 'BOTH'`，启用 RR8 重排（mass prior 仍然开）。
2. `rerank_list` 允许替换榜首，但只在融合分**领先现任 >1.0 个 z 单位**时；否则榜首不动、只重排 2–25 名（072b 的 leader-safe 行为）。

`_RR8_OVERLAY` 的 SHA-256 锁已同步为 `9ae410ce…`。

## 保证金阈值从哪来

用已发布 booster 原样（不重训）在 120 个带标签面板上扫阈值：

| margin（z） | ΔMRR | top-1 改动数 |
| --- | --- | --- |
| 0.00 | −0.0013 | 5 |
| 0.25 | −0.0013 | 3 |
| 0.50 | +0.0042 | 2 |
| **0.75 / 1.00 / 1.25 / 1.50** | **+0.0084** | **1** |
| 2.00 / 3.00 | +0.0042 | 0 |

取 **1.0**（最优平台中央）。不加门是 −0.0013，纯 leader-safe 是 +0.0042，加门后 +0.0084。

## 门控自检

```
z(C1)-z(C0) = 0.179  -> small z-gap keeps leader C0: True | multiset ok: True
z(C1)-z(C0) = 3.240  -> large z-gap promotes C1:    True | multiset ok: True
frozen leader kept: True | multiset ok: True
```

候选多重集在每种情况下都保持不变。

## 运行结果（commit run）

`jiyujia/enveda-072e-margin-gate` v1，`KernelWorkerStatus.COMPLETE`，约 0.5h GPU。

```
V8 BOTH | mass prior {"status":"on","molecules":400,"candidates":688,"penalised":447,"lists_touched":394}
SUBMISSION READY: ... | V8 BOTH | mass lead {"status":"ran","leader_moves":0}
                        | RR {"status":"ran","molecules":400,"reranked":0,"changed":0,
                              "top1_changed":0,"skipped_library":12,...}
```

`reranked: 0` 是可见 12 分子全部 `library_max=1.0`（被 `lib_tau≥0.9` 跳过）导致的，符合预期；**真实效果只在隐藏重跑里出现**。提交文件合法：400 行、id 顺序与 sample 一致、列表 1–25、无重复。

## 提交

当日（11-17 UTC）5/5 已用尽，首次提交返回 `400 Bad Request`；配额在 00:00 UTC 重置后重试。

## 验收

公榜 **>0.496** 为有效。若回 0.492，则连同 072c（mass prior +0.00000）、072d（召回 100%）、59 臂表（最大 +0.006）一起，构成「0.492 是本配方族上限」的完整证据链。
