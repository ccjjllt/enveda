# 047 融合权重扫描（A 线，sanpier）

## 背景
- 046A（sanpier + leader-safe W_MID0.75 + 碎片 rank-1）公榜 0.468 为当前最佳；046B 补 simulator_vote 后 0.467，诊断显示门槛不是瓶颈。
- W_MID=0.75 来自公开 alex chilton / Roman Rozen 配方，并未在我们的 v19 底座上调过；045a 证明 0.75 对 model3 有害（0.462），说明最优 W_MID 依赖底座。

## 方案
- 只动一个参数：W_MID（rank 2–25 的神经融合权重），其余与 046A 完全相同（W_TOP=0.25、W_FRAG=0.5、碎片规则同 046A）。
- 047a：W_MID=0.60。若 > 0.468 继续往下试 0.5 附近；若 < 0.468 则试 0.90。
- 构建：`notebooks/047/build.py [W_MID]`（复用 046/build.py，仅改标记、输出文件、kernel id），推到 ccjjllt/enveda-020-e006-repro。
- 打印标记 `047 SANPIER WMID SCAN`，watcher 门槛同 046B（≥100 行 + NEURAL OK + 标记，UTC 当日提交 <2）。

## 预期
- ±0.002 量级；主要用于确定 W_MID 方向，之后转 model3+sanpier 集成（计划 046 优先级 3）与同分子式重排器（优先级 4）。
