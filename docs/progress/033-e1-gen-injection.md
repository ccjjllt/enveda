# 033 进展

- 新 cell：`notebooks/033/e1gen_cell.py`，完整 notebook `notebooks/033/enveda-033-e1gen.ipynb` = 030 notebook + cell 19（v1 engine 构建之后），其余 cell 不变。
- 本地校验（`scripts/033_local_check.py`，venv-e006，RERUN=1，前 8 个分子，跳过 Engine2/PubChem/模拟器）：
  - E1GEN ok：候选 14,237，标准化去重后 13,802，用时 84 s（CPU）。
  - 8 个分子共注入 250 个结构，n_err 0，v4b 特征族无报错。
  - 与不注入对照：8 个分子 top1 全部不变；3 个分子的 top5 出现新结构（1/2/1 个）。
- 2026-10-24 21:0x（北京时间）push 为 kernel v12（GPU 全流程）。
- v12 运行 COMPLETE（约 24 分钟）：E1GEN ok（smoke 12 分子窗口 656 候选 → 638 个），base lists errors 0；ICE/GL/N1 全部 ok；输出 400 行。
- 2026-10-24 21:23（北京时间）提交 v12，命名 v16，ref `57595662`，待评分。

- 评分：v16 public 0.450（< v15 0.464，退步 0.014）。028 结构作为生成候选注入不采用，基线仍为 v15。
