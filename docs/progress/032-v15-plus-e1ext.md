# 032 进展

- 构建：`notebooks/032/enveda-032-n1-e1ext.ipynb` = 030 notebook + 028 扩池 cell（cell 18）+ base lists 一行修改（cell 21）。37 个 cell，语法人工检查通过。
- 2026-10-24 19:3x（北京时间）push 为 kernel v10，运行中。
- v10 完成（约 18 分钟），但扩池 cell 报 `NameError: np`（该 cell 位于 numpy 导入之前；028 本地单元检查用的是另一套环境，没暴露）。扩池回退原池，等于 v15 复刻，不提交。
- 修复：cell 开头补 `import os, glob, numpy, pandas`（028 的 e1ext_cell.py 同步修复）。push 为 kernel v11。
- v11（20:16 push，20:37 北京时间 ERROR）：扩池本身成功（smoke 12 分子新增 656 个结构，23 s），但 v4b 特征族 `fe_v4/fams/frag2.py` 断言失败：`pool_f1_off/pool_cap` 与池长度不一致。
- 原因：v465 的 Engine1 用 v4b 特征族，它们带有按池下标对齐的预计算资源（`res/frag2/pool_cap|f1_off|f1_mass`、`res/analog_struct/pool_*`、`res/fragnet/rec_*`、chemfeat `pool_rec`），028 只扩了池本身。028 当时的单元检查没有覆盖这些族。
- 结论：028 扩池在当前 v465 引擎下不能直接接入，未提交，公榜仍是 v15 0.464。GPU 用掉约 40 分钟。
- 可选修法（未做）：把扩展结构作为“生成候选”（pid<0）注入，各特征族已有按 SMILES 计算的生成候选路径；或为新条目补齐所有对齐资源。前者改动更小，列为后续选项。
