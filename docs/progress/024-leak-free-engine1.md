# 024 — 留出 NP 的 Engine1 指纹模型复核：实际进展

日期：2026-10-21（北京时间）。接手`5673ecb`。对应[024计划](../plan/024-leak-free-engine1.md)。

## 模型来源核对

- `fpnet_0.pt`（v4b 自带，训练代码默认留出 np/nplib/syn/plusk/twin）：checkpoint 记录 val np **0.578**。
- `fpnet_full1.pt`（v11 Engine1 实际使用）：val np **0.918**，基本可以判定训练时见过 NP。
- 验证器开关 `E024_FP0=1` 让 Engine1 回退到 fpnet_0（日志 `engine FP bank: fpnet_0 (A) FALLBACK`）；Engine2、PubChem 通道复用 021 输出。

## C 遮蔽（fpnet_0）

```
C_fp0 base MRR 0.1281 top1 3
  protect tau 0.3: applied 11 MRR 0.1553 d +0.0272 W/L 10/0 CI [+0.0126,+0.0445]
  protect tau 0.5: applied 26 MRR 0.1874 d +0.0593 W/L 22/0 CI [+0.0349,+0.0823]
  protect tau 9: applied 76 MRR 0.2879 d +0.1598 W/L 65/0 CI [+0.1245,+0.1925]
C_fp0 promo_off MRR 0.1569 top1 11
  protect tau 0.3: applied 11 MRR 0.1799 d +0.0230 W/L 9/0 CI [+0.0093,+0.0383]
  protect tau 0.5: applied 26 MRR 0.1986 d +0.0417 W/L 17/0 CI [+0.0233,+0.0630]
  protect tau 9: applied 76 MRR 0.2886 d +0.1317 W/L 57/0 CI [+0.1023,+0.1617]
  promo_off vs base: d +0.0288 W/L 20/0 CI [+0.0156,+0.0437]
```

换成留出模型后，C 基线 0.1281（与 full1 的 0.1168 相近），保护规则仍全部为正且 0 loss，关闭 PubChem 提升 +0.0288（20/0）。R、B 运行中。

## R 遮蔽（fpnet_0）

```
R_fp0 base MRR 0.4425 top1 26
  protect tau 0.3: applied 0 MRR 0.4425 d +0.0000 W/L 0/0 CI [+0.0000,+0.0000]
  protect tau 0.5: applied 1 MRR 0.4405 d -0.0020 W/L 0/1 CI [-0.0060,+0.0000]
  protect tau 9: applied 1 MRR 0.4405 d -0.0020 W/L 0/1 CI [-0.0060,+0.0000]
R_fp0 promo_off MRR 0.4974 top1 37
  protect tau 0.3: applied 0 MRR 0.4974 d +0.0000 W/L 0/0 CI [+0.0000,+0.0000]
  protect tau 0.5: applied 1 MRR 0.4954 d -0.0020 W/L 0/1 CI [-0.0060,+0.0000]
  protect tau 9: applied 1 MRR 0.4954 d -0.0020 W/L 0/1 CI [-0.0060,+0.0000]
  promo_off vs base: d +0.0548 W/L 45/0 CI [+0.0382,+0.0736]
```

R 基线 0.4425（full1 为 0.4547）。保护规则 τ=0.3 在 R 上 0 生效；τ≥0.5 有 1 例 −0.0020。关闭 PubChem 提升在 R 上 +0.055（45/0），但 v13 在隐藏集为 −0.014，说明本地遮蔽对 PubChem 通道系统性误判。

## 转向：验证缺口诊断（v13 公榜 0.432 < v11 0.446）

- v12（0.434）与 v13（0.432）的本地增益在公榜上都反转了，说明 NP250 遮蔽面板跟隐藏测试的分布不一致。
- **NP250 与可见测试的差异：**

  | | NP250 | 可见测试 |
  |---|---|---|
  | 仪器 | 混合 | 全部 timsTOF |
  | 正离子 | 67% | 81% |
  | [M+H]+ | 48% | 79% |
  | 母离子 m/z（10–90%） | 223–677 | 280–379 |
  | 每分子谱图数 | 4.74 | 3.03 |

  NP250 以天然产物为主；可见测试是 enveda-180 的合成化合物。
- 可见测试的 1213 张谱图全部与 enveda-180 训练谱图逐字相同，由此恢复出 400 个分子的标签（`/workspace/e024/visible_solution.csv`）。
- 之前在可见测试上得到的 v11/v12/v13 分数（0.0288/0.03/0.0288）是 smoke 伪影：notebook 检测到可见测试的签名后只处理 12/400 个分子，所以这组分数没有任何信息量。
- **新面板 E180 / 遮蔽 H（与隐藏集同分布）：**
  - 以可见测试的 400 个分子作为查询。
  - 从训练集删除这些结构的全部谱图，共 2536 张。
  - Engine1 v2-pool 只删除训练来源（src==1）的条目，共 400 条，全部来自 src==1。
  - COCONUT、ChEBI、PubChem 保留。
  - 运行时设置 `KAGGLE_IS_COMPETITION_RERUN=1`，强制处理全部分子。
- 脚本：`src/panel_e180.py`；`eval_e006_local.py --mask H`。
- B_fp0 在 cell 23 被 OOM 杀掉（与面板构建并发），已放弃；B 遮蔽本身已证明不可信。
