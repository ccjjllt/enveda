# 033 计划：028 结构作为 Engine-1 生成候选注入

- 背景：032 把 028 结构直接加进 v2 池，v4b 特征族的池对齐资源（frag2/analog_struct/fragnet/chemfeat）随之错位，断言失败。
- 方案 A：不改池，在 `Engine.candidates()` 外包一层，把母离子窗口内、不在池中的 PubChemLite CC0 + ChEBI/LIPID MAPS 结构以生成候选（pid=-1，sim/steps/tan=0）追加。生成候选本来就按 SMILES 计算全部特征，不读池对齐资源。
- 校验：本地 CPU（`KAGGLE_IS_COMPETITION_RERUN=1`，可见测试前 8 个分子，跳过 Engine2/PubChem）确认 v4b 特征路径不崩，再做一次 GPU 全流程运行；干净完成才提交。
- 风险：新结构被 ranker 当作“生成候选”看待（is_gen=1），可能排名偏低；生成候选不享受池内流行度加分。
- 若不可行：转 030 第 3 步，针对同式、缺参考谱、负离子错例做定向修复。
