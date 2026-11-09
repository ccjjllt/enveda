# 062 计划：公开 0.484 复现 + 外部谱库 rank-1 覆盖阈值离线标定（A 线）

## 背景
- 056/059/060 公榜 0.481 打平；061a（NW_MID 1.00）0.478；第 2–25 名小旋钮已到头，top-1 盲目提升（057/058）有害。
- 2026-11-10 00:05 北京检查公开 notebook：alexchilton《CASMI26 LB 0.484 | Peaks to Structures + ExtLib》（作者账号实测 ref 58141942 公榜 0.484）= ozertuu《From Peaks to Structures》仅改外部谱库覆盖 `TAU 0.85→0.80`。所有数据集公开、可复现。
  - 底座：popwenman Gold Apex V22（V8=MASS，POP_W/SPEC_W 0.12），**不挂** popwenman v80 kernel（作者说明 v80 对 0.480 系列中性）。
  - 新层：外部实测谱库（Takumu Hata casmi26-extlib 20 万谱 + samartalwar MassBank harmonized 去 is_casmi_subset）rank-1 覆盖：前体 10 ppm+0.002 Da、同离子模式、sqrt 强度余弦 ≥TAU、匹配峰 ≥4，且管线自身库匹配 <0.90 时把命中结构提到第 1。
  - 作者账号：ozertuu 原版 TAU0.85 = 0.478，TAU0.90 = 0.478，TAU0.80 = 0.484 —— 阈值单调放宽有收益。
- 这正是"只在强独立证据下改 top-1"的门控方向：证据来自外部实测参考谱，而非 057/058 的同管线内部证据。

## 离线标定（scripts/062/extlib_tau_calib.py）
- 误触发率：train 中 enveda-180 timsTOF 分子（与隐藏测试同仪器/同来源；其结构几乎都不在外部库，0.015%）抽 20,000 个，逐谱按 overlay 原逻辑查库，统计最佳命中余弦 ≥TAU 的比例（这些触发几乎全错）。
- 真命中精度：train 中 pluskal_ms2/enveda-np-examples 且结构在外部库的分子 3,000 个（剔除 cos≥0.995 的疑似同源重复谱），统计各 TAU 下命中结构=真值的比例（上界式代理，非 timsTOF）。
- 期望值：在被触发分子上，命中正确收益≈1−1/r_old（约 0.5），错误损失≈P(原 top1 对)×0.5（约 0.2）→ 增量精度只要高于约 0.3 就为正。

## 实验
- **062a**：alexchilton 0.484 原样（仅 `COMMIT_MODE='smoke'` 以便审计），新 kernel `ccjjllt/enveda-062a-extlib-tau080`。预期 0.482–0.486（作者称同代码因时间守卫 ±0.002）。
- **062b**：062a + `TAU 0.80→0.70`（kernel `ccjjllt/enveda-062b-extlib-tau070`），依据离线标定。
- 视 062a/062b 出分决定 062c（TAU 0.65 或 K_PEAKS 放宽 / 加 v80 第二打分器），本轮最多 3 次新提交。

## 审计与提交门槛
400 行、top-1 去重数 >20（非占位）、每行 1–25 个且无重复/空 SMILES、ID 集合与 056 一致；记录与 056 的 top-1/全列表差异与 `apex_extlib_audit.json`。smoke 下可见 test 为训练集复本，外部库几乎不触发，真实效果只能看公榜。

## 许可
A 线（公开 notebook 复现，sanpier 模型许可 unknown），仅冲奖牌/名次；B 线自研许可干净路线不受影响。
