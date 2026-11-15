# 068 计划：在 065a（0.488）上复现已核实的公开 0.492（N-1c LAM 0.25→0.35，A 线）

## 背景
- 067a（065a + 领头保护的自研 044 model3 mid-rank，λ=0.20，钉死 top-1）ref **58380689** 公榜 **0.488**，与 065a/066a 打平，无增益。不作为新底座，也不再加大 model3 权重。
- 当前自有最佳仍是 065a 公榜 **0.488**（ref 58329677）：063a + N-1c（τ=0.6，λ=0.25）。
- 官方快照 2026-11-15T16:19:51Z（5331 队，约 2026-11-16 00:19 北京）：ccjjllt 第 **211** 名，仍 0.488。恰好 0.488 的平台为第 184–510 名共 327 队；**≥0.489 共 183 队**（18:26 快照为 122 队）；恰好 **0.492 共 65 队**，说明有新的公开配方落地。
- 周 GPU 已用尽：31.15h / 30.00h，剩余 **0.00h**，下次刷新 **2026-11-21 08:00 北京**。CPU 路线在 062a 已证明不可行（冻结管线强制 GPU，Engine 2 现训，时间守卫会跳阶段）。本轮只备好 notebook，刷新后再推。

## 已核实的公开 0.492 配方
- 唯一必要改动：把 N-1c 网络重排权重 **`LAM` 0.25→0.35**（τ=0.6 不动）。
- 独立核实：
  - Rogers Johnson [casmi26-l4-lam035](https://www.kaggle.com/code/rogersjohnson/casmi26-l4-lam035)；作者本人现为 0.492（提交 2026-11-15 01:36 UTC）。
  - alex chilton [casmi26-0-492-what-moved-the-board-59-arms](https://www.kaggle.com/code/alexchilton/casmi26-0-492-what-moved-the-board-59-arms) 声明原样跑该笔记，提交 **58380172** 公榜 **0.492**（相对其 0.488 的 +0.004）。
  - mathleteutd 公开消融表：LAM 0 / 0.15 / 0.25 / **0.35** / 0.50 → 公榜 0.487 / 0.487 / 0.488 / **0.492** / 0.484。LAM=0.50 仍会退到 0.484，不跟。
- 先前（067 计划）把 Rogers 的 L 系列视为未核实 λ 扫描而没有跟；现已被第二账户独立跑出 0.492，且榜上出现 65 队的 0.492 平台，作为已核实配方跟。
- 不做：mathleteutd 的 learned reranker（只报了 rehearsal MRR，未在公榜单独核实）；不叠已中性的 066a libcos / 067a model3 mid-rank；不改 `LIB_HIT_GATE` / `V8` / ExtLib TAU。
- heliosli/casmi26-0-489 标题仍是 0.489，但拉取后正文仍是旧的 0.477 笔记，不作为本轮配方。

## 实验
1. **068a**（`notebooks/068a`，本轮主实验）：065a 底座只改 N-1c 函数默认 `LAM=0.35`（调用仍是 `_n1c()`，无显式传参）。其余常量不动（`V8='MASS'`、`LIB_HIT_GATE=0.78`、ExtLib `TAU=0.80`、N-1c τ=0.6、`COMMIT_MODE='smoke'`），输入与 docker 与 065a 相同。
   - 内核 slug：`ccjjllt/enveda-068a-n1c-lam035`。
   - 周 GPU 刷新前不推送；刷新后用 `scripts/063/chain.sh` + `watch.sh` 推送，审计通过则 `-k -v` 提交。
2. 若 068a ≥0.492，再考虑正交方向（门控召回/库外生成），不继续扫 LAM。若 ≤0.487，回退 065a。

## 许可
A 线（公开差异底座，含 sanpier 等许可 unknown 的模型），仅冲奖牌/名次。B 线许可干净的自研整管线本轮不推，最终选交仍需各一。

## 署名
N-1c LAM 0.35 / 公开 0.492：Rogers Johnson、alex chilton、mathleteutd。N-1c / 0.488 底座：analyticaobscura、alex chilton，以及 065a/063a 已列出处。
