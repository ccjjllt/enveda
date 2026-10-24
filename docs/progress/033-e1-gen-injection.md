# 033 进展

- 新 cell：`notebooks/033/e1gen_cell.py`，完整 notebook `notebooks/033/enveda-033-e1gen.ipynb` = 030 notebook + cell 19（v1 engine 构建之后），其余 cell 不变。
- 本地校验（`scripts/033_local_check.py`，venv-e006，RERUN=1，前 8 个分子，跳过 Engine2/PubChem/模拟器）：
  - E1GEN ok：候选 14,237，标准化去重后 13,802，用时 84 s（CPU）。
  - 8 个分子共注入 250 个结构，n_err 0，v4b 特征族无报错。
  - 与不注入对照：8 个分子 top1 全部不变；3 个分子的 top5 出现新结构（1/2/1 个）。
- 2026-10-24 21:0x（北京时间）push 为 kernel v12（GPU 全流程）。
