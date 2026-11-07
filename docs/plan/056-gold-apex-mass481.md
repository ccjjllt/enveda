# 056 计划：复现 alexchilton 公榜 0.481（Gold Apex V22 + MASS + popwenman v80）

## 背景
- 最佳仍 **051 = 0.480**（alexchilton Neural Ensemble，020 v30）。
- 052=0.470；053 未交；054/054b=0.459（退步）。
- 055 严格 xl（TAU0.85 K4）因 052 max cosine≈0.567 → TAU≥0.60 零命中，**已关闭（no-op）**。

## 054 根因（关键）
054 按 romanrozen 页面自称 0.482，用了：
- `V8='BOTH'`（质量先验 + 同式异构体重排）
- 数据集 `bgunshui/casmi26-scorer-v5-model`，**无** `popwenman/casmi26-scorer-train-v80` kernel_source

alexchilton 自己的对照表写明：romanrozen 代码 + `V8='BOTH'` + bgunshui scorer v2 → 其账号公榜 **0.461**（页面声称 0.481）。真正拿到 **0.481** 的是：
- `V8='MASS'`（**不是 BOTH**；异构体重排关闭）
- kernel_sources：`popwenman/casmi26-scorer-train-v80`
- 与 `hoowoo/casmi26-gold-apex-mass-prior-repro` 字节级一致（md5 `8a2ca55e4d9d`）

## 假设
原样复现 alexchilton `casmi26-lb-0-481-gold-apex-v22-mass-prior`（smoke 提交路径）可到约 **0.481**，超过 051 的 0.480。

## 实验设计
1. 复制 alex481 notebook → `notebooks/056/enveda-056-gold-apex-mass481.ipynb`。
2. **`COMMIT_MODE='smoke'`**（禁止 `fast`；避免 054 v34 占位误交）。
3. 保持 **`V8='MASS'`**；数据集与 alex 一致；`kernel_sources=['popwenman/casmi26-scorer-train-v80']`；**不用** bgunshui scorer dataset。
4. 挂到 `ccjjllt/enveda-020-e006-repro` 下一版本；COMPLETE 后优先下载审计 top1 多样性，失败则 `-k/-v` 提交（smoke 保证完整管线）。
5. 许可：含 sanpier 等 → **A 线冲牌**，不参与奖金。

## 若 ≤0.480 的下一候选（本轮不启动）
- (a) aiokajima 的 popwenman 轻微变体
- (b) gloriachen0816 claw-maxzero
- (c) 硬负例 / 自研 B 线

## 致谢
alex chilton、popwenman、hoowoo/CavaradØssi、sanpier、Dmitrii Gluzdov、qwertykeypadapple；上游 romanrozen / zz3t3rn41 / bgunshui 等。
