# 054 计划：复现 romanrozen 公榜 0.482（Gold Apex + MASS + RR）

## 背景
- 当前 A 线最佳：051 = alexchilton Neural Ensemble 公榜 **0.480**（020 v30，ref 58008856）。
- 052 TAU0.55 xl 叠加公榜 0.470（退步）；053 FREEZE 因可见 test 占位导致 skipped_short=388，SKIP_NO_RERANK 未提交。
- 公开 notebook `romanrozen/0-482-casmi-from-peaks-to-structures` 声称 **0.482**：在 zz3t3rn41 Gold Apex V22（≈0.480 + LOTUS + 更高 NP 先验）之上加 timsTOF 质量先验（MASS）与同式异构体 LightGBM 重排（RR）。`V8='BOTH'`。
- 原 notebook 的 `kernel_sources` 指向已删除内核；jtheyogi fork 用公开数据集 `bgunshui/casmi26-scorer-v5-model` 替代，内容与原 notebook 一致。

## 假设
完整复现 romanrozen v9（V8=BOTH + LOTUS + bgunshui scorer）可到约 0.482，高于 051 的 0.480。

## 实验设计
1. 原样推送 romanrozen notebook 到 `ccjjllt/enveda-020-e006-repro` 下一版本（v34）。
2. 数据集：sanpier 谱模型 + dmitriigluzdov 资产 + bgunshui/casmi26-scorer-v5-model + LOTUS + rdkit wheel；无 kernel_sources（避免 deleted）。
3. COMPLETE 后用 `-k/-v` 直接提交（规避 kernels output 下载挂死）。
4. 许可：含 sanpier 等 unknown → **A 线冲牌**，不参与奖金。

## 不做什么
- 不重做失败的 052 TAU0.55 xl、W_OWN、W_FR 无改动方案。
- 本轮不做 ozertuu 0.500+（054 出分后再评估为 055）。

## 致谢
romanrozen、zz3t3rn41、alex chilton、sanpier、Dmitrii Gluzdov、baidalinadilzhan、bgunshui、Roman Tamrazov、Ahmed Berat Ozer、qwertykeypadapple（LOTUS）。
