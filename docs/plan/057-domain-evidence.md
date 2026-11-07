# 057 计划：056 底座 + kozykappa Domain-Evidence Layer（冲 >0.481）

## 背景
- 056 = alexchilton Gold Apex V22 + `V8=MASS` + popwenman v80，公榜 **0.481**（ref 58089862，新最佳）。
- 公榜 #1 ≈0.537；公开可复现增量优先于盲扫。
- kozykappa `gold-apex-v22-mass-prior-domain-evidence-layer`（2026-11-07）：自称 v23，在 0.481 MASS 路径上加 ranks 2–25 的 domain-evidence（library entropy / ionization / NP-likeness / heteroatoms），开关关闭时回退 0.481。

## 假设
在已验证的 056 路径上启用同一 domain-evidence 层，公榜有望 >0.481；若无增益则关闭该层，改试 heliosli 0.482 路径或 aiokajima/claw 变体。

## 实验设计
1. 以 056 notebook 为底座，移植 kozy v23 domain-evidence 模块（保留 SHA pin / 失败回退）。
2. **强制 `COMMIT_MODE='smoke'`**（kozy 默认 fast，易交占位）；**`V8='MASS'`**；`kernel_sources` 含 `popwenman/casmi26-scorer-train-v80`；禁用 bgunshui BOTH 路径。
3. 推 `ccjjllt/enveda-020-e006-repro` 下一版本；COMPLETE 后审计 top1 多样性与 vs 056 差异，再提交。
4. 许可：仍含 sanpier → **A 线冲牌**。

## 致谢
kozykappa / Kozykorpesh；上游 alex chilton、popwenman、hoowoo、sanpier 等。
