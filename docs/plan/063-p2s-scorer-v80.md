# 063 计划：公开 0.487（Peaks to Structures + popwenman scorer v80）复现 + 输入消融（A 线）

## 背景
- 当前最佳 062a 0.484（ozertuu P2S + ExtLib TAU0.80）。062b TAU0.70 = 0.479，TAU 方向已被 alexchilton 扫完（0.75/0.80 = 0.484，0.85/0.90 = 0.478，K_PEAKS 3、LIB_TAU 0.95 无变化）→ 不再扫 TAU。
- 2026-11-11 12:07 北京扫描公开 notebook：gloriachen0816（alexchilton 团队账号）《CASMI26 LB 0.487 | Peaks to Structures + Scorer》= guhongbin *casmi26-v589-alex0484-extlib*，**代码 cell 与 062a 逐字节相同**，只改输入：
  1. 挂 popwenman kernel `casmi26-scorer-train-v80`（管线在 /kaggle/input 搜 `final_model.pt`+`config_used.json` 用作神经打分器）；
  2. 不挂 qwertykeypadapple LOTUS 候选数据集（LOTUS 空位回填跳过）；
  3. guhongbin 的 docker 镜像（sha256:37c64f7d…）。
  作者账号实测 ref 58185447 = **0.487**，未拆分哪项输入带来增益。chyibo、xiangqizhang 的 fork 与之相同。另 rogersjohnson *zz3v2-1109* 原样也 0.487（不同管线）。

## 实验
- **063a**：0.487 原样（仅 `COMMIT_MODE='fast'→'smoke'` 以便审计），kernel `ccjjllt/enveda-063a-p2s-scorer-v80`，元数据（数据集、v80 kernel source、docker）与原作相同。预期 0.485–0.489。
- **063b**：063a + 重新挂 LOTUS 数据集（代码相同），kernel `ccjjllt/enveda-063b-p2s-scorer-v80-lotus`。输入消融：若 ≥063a，说明增益来自 v80，LOTUS 回填可叠加；若 <063a，说明去掉 LOTUS 是增益之一。
- 后续（视出分）：在 063a/063b 胜者上试 rogersjohnson zz3v2-1109 的差异项，或 zz3t3rn41 v48 的 adduct-aware extlib（需先在作者账号外核实分数）。

## 审计与提交
watcher `scripts/063/chain.sh` → `scripts/063/watch.sh`：推送失败（GPU 会话上限 2）每 5 分钟重试；COMPLETE 后等 10 分钟，单文件拉取（每文件 10 次退避），审计 400 行 / top-1 去重 ≥12 / 每行 1–25 无重复无空 / ID 集合同 056，通过则 `-k -v` 提交；提交失败等到次日 08:05 北京重试。

## 许可
A 线（公开 notebook 复现，sanpier 模型许可 unknown），仅冲奖牌/名次。
