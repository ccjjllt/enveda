# 039 进度
- src/own_spec 编写完毕；本地 CPU 冒烟（2 个 row group 各 4000 行）全流程通过。
- kernel ccjjllt/enveda-039-own-spec v1 已 push（GPU），训练中。
- 与 038 冲突：一次只跑一个重任务，038 排在训练之后。
- v1 失败：Kaggle 镜像无 rdkit → 指纹全空（已加断言）；v2 失败：cp311 wheel 不适配 py3.13；v3 训练 kernel 开 internet 在线装 rdkit==2026.3.3，RUNNING（约 20:50 UTC+8 起，预计 7.3h）。
- notebooks/040/own_neural_cell.py：自研模型推理 cell（由 036 cell 改写），用 CPU 冒烟模型本地测试通过（scored 12）。待训练完成后：下载 own_spec/ → 建私有 dataset ccjjllt/enveda-own-spec → 040 notebook = 030 + 本 cell（+038 保护可选）。
- v3 COMPLETE：66068 步，7.05h；FINAL VAL isomer MRR（n_val=2535）{fp0: 0.7583, 0.5: 0.7606, 1.0: 0.7589, 2.0: 0.7571, 4.0: 0.7482} → OWN_FP_W=0.5。
- 输出已建私有 dataset ccjjllt/enveda-own-spec（own_spec/ 目录 zip）。
