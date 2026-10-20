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
