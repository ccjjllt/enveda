# 039 进度
- src/own_spec 编写完毕；本地 CPU 冒烟（2 个 row group 各 4000 行）全流程通过。
- kernel ccjjllt/enveda-039-own-spec v1 已 push（GPU），训练中。
- 与 038 冲突：一次只跑一个重任务，038 排在训练之后。
- v1 失败：Kaggle 镜像无 rdkit → 指纹全空（已加断言）；v2 失败：cp311 wheel 不适配 py3.13；v3 训练 kernel 开 internet 在线装 rdkit==2026.3.3，RUNNING（约 20:50 UTC+8 起，预计 7.3h）。
