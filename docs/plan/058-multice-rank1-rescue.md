# 058 计划：056 底座 + eason V50 多碰撞能 rank-1 救援

## 背景
- A 线最佳：**056** = Gold Apex V22 + `V8='MASS'` + popwenman v80，ref **58089862** 公榜 **0.481**。
- 057 Domain-Evidence（kozy v23）ref **58107366** 公榜 **0.476**（−0.005）：只翻转 `m_050bf1` rank2→1（`promote_leader_without_reference`），该规则有害，057 不作底座。
- 公榜 #1 = 0.537。公开可复现、且明确写了增量方法的 notebook 优先于盲扫。

## 公开扫描（2026-11-08 12:15 北京）
| 候选 | 结论 |
|---|---|
| heliosli/casmi26-0-482（057 书面备选） | 两个 `dataset_sources` 仍为空（私有）。`heliosli/the-dataset-is-public-now-enjoy`（当天 11:58 北京）挂的是 `heliosli/qcc-circuits-live`，属于 QCC 而非 CASMI。 |
| popwenman/casmi26-0482-gold-apex-v22 | 与 056 同一条 MASS+v80 配方，只是 `COMMIT_MODE='fast'`。原样复现是空操作。 |
| gloriachen0816/casmi26-r10-claw-maxzero | 0.480 bgunshui 神经集成线 + promotion margin 微调，低于 0.481 地板。 |
| ozertuu 0.500 标题本 | 仍是 0.480 管道 + TAU=0.85 薄库叠加的标题党（052/055 已证伪薄库/高 TAU 无命中或误插）。 |
| easonyanyan/casmi26-v47-extlib-rank1-rescue | 作者记录：takumuhata 库（约 1.3 谱/结构）隐藏重跑 **0 次有效提升**，停在 0.4810。 |
| **easonyanyan/casmi26-v50-multice-rank1-rescue** | 在其 V44（自称 0.4810）上只追加一个失败回退 cell；库为公开的 `sergiomalv/casmi-open-msms-spectra`（421,671 谱 / 60,377 结构，约 7 个碰撞能）。离线：Enveda timsTOF 有标签训练上 cos≥0.78 / K≥10 命中真结构 37.45%、误触发 0.22%；自然流行率下期望 +0.00115 MRR。V47 已证明薄库不行，多碰撞能覆盖是文档里的下一变量。 |

## 假设
把 V50 救援 cell 原样接到已验证的 056 底座（不要改写底座树；eason 记录手搬底座曾 −0.024/−0.008），隐藏测试上若有高置信库命中，公榜有望 ≥0.481；若 0 次提升则 byte-for-byte 回退 056，期望持平 0.481。

## 实验设计
1. 以 `notebooks/056/enveda-056-gold-apex-mass481.ipynb` 为底座，在末尾追加 eason V50 cell（及一段说明 markdown）。
2. **强制 `COMMIT_MODE='smoke'`**，**`V8='MASS'`**；`kernel_sources` 保留 `popwenman/casmi26-scorer-train-v80`；`dataset_sources` 增加 `sergiomalv/casmi-open-msms-spectra`。
3. 可见 smoke 的 12 个分子 `LIB_MAX=1.0`，V50 按设计会 `skipped_library`（commit CSV 顶 1 应等于 056）。隐藏重跑才是真正实验。因此 watcher 在 `v50_xl_audit.json` 为 `status=ok`（库已加载且自检通过）时，即使 commit top1 与 056 相同也通过 `-k/-v` 提交；占位 / 库未挂载 / 自检失败则不交。
4. 对比底座：056 `/workspace/kout_v36/submission.csv`（057 已退步，不用 v37）。
5. 许可：含 sanpier → **A 线冲牌**。

## 致谢
easonyanyan（V50 方法）；sergiomalv（公开 MS/MS 库）；上游 alex chilton、popwenman、hoowoo、sanpier、Dmitrii Gluzdov、romanrozen、zz3t3rn41、qwertykeypadapple。
