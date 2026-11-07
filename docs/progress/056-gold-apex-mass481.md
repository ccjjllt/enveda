# 056 进展

- 2026-11-07 18:10 北京：关闭 055（TAU≥0.60 → 0 hits，无 notebook）。启动 056：原样复现 alexchilton 公榜 **0.481** 配方（`V8='MASS'` + `popwenman/casmi26-scorer-train-v80`）；源本已落盘 `/tmp/sota056/alex481/`（与 hoowoo 同 md5）。
- 根因备忘：054 用 `V8='BOTH'` + bgunshui scorer → 公榜 0.459；alex 表内同配置实测 0.461。正确路径是 MASS-only + popwenman v80。
- 构建：`notebooks/056/enveda-056-gold-apex-mass481.ipynb`，`COMMIT_MODE='smoke'`，`V8='MASS'`；metadata 挂 020 + popwenman v80 + LOTUS 等 alex 同源资产。
- 若 056 公榜 ≤0.480：下一候选 (a) aiokajima 变体 (b) gloriachen0816 claw-maxzero (c) 硬负例/自研 B 线——本轮不启动。

- 2026-11-07 18:12 北京：020 **v36** 已推送（`enveda-056-gold-apex-mass481.ipynb`，smoke + V8=MASS + popwenman v80；无 bgunshui）。watcher `/workspace/k056/run.sh` 待 RUNNING→COMPLETE 后审计/提交。git `1a56036`。
