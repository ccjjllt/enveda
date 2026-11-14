# 066 计划：在 065a（0.488）上做领头保护的训练库连续余弦 mid-rank 重排（A 线）

## 背景
- 当前自有最佳是 065a 公榜 **0.488**（ref 58329677）：063a + N-1c（测试集内网络重排，τ=0.6，λ=0.25）。官方快照 2026-11-14T17:03:13Z（5272 队，约 2026-11-15 01:03 北京）：团队第 **255** 名；恰好 0.488 的平台为第 109–340 名共 232 队；**≥0.489 共 108 队**。要跳出平台需 ≥0.489。
- 2026-11-15 01:00–01:10 北京再次审计公开 Code 页与作者公榜：仍 **未见已核实的公开 0.489+ 配方**。标题写 0.489/0.499/0.500+ 的 notebook 要么输入有空槽（heliosli），要么作者本人仍是 0.488/0.487（ozertuu、zz3t3rn41），要么正文仍是旧流程。analyticaobscura 本人 0.493，公开 casmi-2026 最近运行仍是 2026-11-11，增量不可复现。

## 公开审计要点（相对 065a，不是相对 056）
- 已核实的 0.488 配方仍是 N-1c；alex chilton、ozertuu、gloriachen0816、zz3t3rn41、rogersjohnson、guhongbin、sanpier、romanrozen 本人均为 0.488。
- rogersjohnson 消融：x1 `LAM=0.10` 作者提交后仍为 0.488；x2 `TAU=0.8` 已跑完但作者最后提交仍是 10:07 UTC 的 0.488，未见跳出；x3 是对照（LAM/TAU 与 065a 相同）。公开表此前已显示 `LAM=0.50` 退至 0.484，`LIB_HIT_GATE` 0.75↔0.78 中性。**不把这些旋钮当下一刀。**
- bill123987 的 n1ctier2 / n1ccons / n1crr8 把 N-1c 参数化（PINTOP、SYM、ROUNDS、MASS_PPM），默认值与 065a 等价或缺少 PINTOP 可能直接失败回退；作者 0.488，不跟。
- zz3t3rn41 v2 的 LAM=0.28 + extlib2 组合、youyouwou / riochiang 的 v9np / triple，作者分别为 0.488 或 0.487，不跟整份厨房水槽。
- liuzewen123/casmi26-0488-libcos：在 0.488 栈上增加「训练库连续余弦」，用 enveda-180（与测试同一仪器家族）按 InChIKey-14 建谱索引，给候选打连续余弦。**公开原文是 rank-1 提升**（同分子式、margin 0.10、min cosine 0.30）。作者最后提交早于这些 0488 内核，公榜仍 0.481，**没有已核实的公榜分数**。改 top-1 在我方 057/058 上有害，故不原样移植。

## 实验假设
管线里「库支持」主要是分子级二元 `library_max`（可见集约 12/400=3% 过门），而讨论中 Class 1（库内真谱）估计约 28%，普通库检索在有参考谱的天然产物上 MRR@25 很高。大量候选其实有训练库谱，只是不够强、没被二元门用上。N-1c 证明 **只改 mid-rank、不动 top-1** 也能 +0.001。把连续库余弦用在第 2–25 名、钉死第 1 名，与 N-1c（测试集内邻居）正交，有机会再拉 0.001。

## 实验
1. **066a**（`notebooks/066a`，本轮主实验）：065a 底座只在 ExtLib overlay **之后**插入领头保护的 libcos mid-rank 格。
   - 索引/打分移植 liuzewen123 libcos（enveda-180、IK14、分箱 sqrt 余弦）。
   - **钉死原第 1 名**；仅当至少 2 个第 2–25 名候选在索引中时，用 `z(-log rank) + λ 0.25 × z(libcos)` 重排中间名次。
   - `library_max >= 0.90` 的分子跳过；异常则 submission 原样保留；候选集合必须是原列表的置换。
   - 其余常量不动（`V8='MASS'`、`LIB_HIT_GATE=0.78`、ExtLib `TAU=0.80`、N-1c λ=0.25/τ=0.6、`COMMIT_MODE='smoke'`）。
   - 内核 slug：`ccjjllt/enveda-066a-libcos-mid`，docker / 输入与 065a 相同（guhongbin 镜像 + popwenman v80，无 LOTUS）。
2. **不做**：064a（LIB_HIT_GATE 0.75）、把 λ 提到 0.50、把 N-1c τ 改到 0.80、把 `V8` 改成 `'BOTH'`、原样移植 libcos 的 rank-1 提升、整份 zz3 v2 / v9np。
3. 若 066a ≥0.489，再考虑把同样的 mid-rank 思路扩到 ExtLib 连续余弦（仍钉死 top-1）。若 066a ≤0.487，回退 065a，下一刀改走召回/库外生成或难负例，而不是再调 N-1c 权重。

审计/提交沿用 `scripts/063/chain.sh` + `watch.sh`（会话上限重试、单文件拉取、退避、smoke 门槛、通过则 `-k -v`）。

## 许可
A 线（公开差异移植 + sanpier 等模型许可 unknown），仅冲奖牌/名次。B 线许可干净的自研模型本轮不推，规划中保留，最终选交仍需各一。

## 署名
libcos 索引/打分：liuzewen123。N-1c / 0.488：analyticaobscura、alex chilton。底座沿用 065a/063a 已列出处。
