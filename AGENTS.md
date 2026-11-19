# 项目协作约定

本项目研究 Kaggle **Enveda CASMI 2026 — Molecule ID From Mass Spectra**（`enveda-CASMI26-molecule-id-mass-spectra`），交流和研究文档默认使用中文。

## 研究方式

- 专注研究假设、实验、模型和分数，不搭建与研究无关的工程门控。
- **不创建测试文件，不引入自动化测试、测试框架或 CI 测试门控**；所有检查与验证手动发起。
- 当前保留基线为 `068a`（N-1c `LAM=0.35`，public **0.492**，ref `58412029`）；未封闭的前沿研究集中在 `rounds/075-frontier/`（MS2 同分异构体判别：化学型整组留出、MIST MSG / ICEBERG 2.1 / MSAlign 与管线未用信号）。
- 代码完成、运行成功、离线指标提升和排行榜提升分开记录；没有实际结果，不声称有效或提分。
- 本机 M5 / 32 GB 只做代码、文档与轻量审计；完整推理与训练在 Kaggle 断网 GPU 执行，最终输出 `submission.csv`。
- 不把可见占位 `test.parquet`（训练集 1:1 副本）或仅留出谱的命中率当作泛化；未见结构须按规范化结构/骨架整组留出，另设独立谱验证衡量库检索。

## 文档与清理约定

- 当前仅保留 `rounds/068-n1c-lam035/`（0.492 底座）与 `rounds/075-frontier/`（前沿计划与调研）；文档保持精简，总入口为根 `README.md` 与 `rounds/README.md`，远端操作见 `ops/remote-workflow.md`。
- `001–075d` 中已终止方向的研究概况、代表分数、失败机制、放弃原因与删除范围统一记录在 `archive/research-archive.md`。
- 方向终止后先更新 `archive/research-archive.md` 并打 Git 标签记录提交号，再删除明细文档、脚本、notebook 与 `artifacts/` 运行产物（未跟踪产物不可从标签恢复，须在总览写明）。
- 原始比赛数据 `data/raw/`、机器可读证据 `data/references/` 与当前基线入口 `068a/` 不因清理而删除。

## 远程工作

- Kaggle MCP 用于读取比赛、数据、公开代码、运行状态、日志、提交和配额；Kaggle CLI 用于推送与提交（命令见 `ops/remote-workflow.md`）。研究账号为 `ccjjllt`（必要时进程内切换认证，不改全局默认）。
- `kernels push` 不同步整个仓库，入口须自包含或挂载输入；记录版本、输入、设备、时长、指标与输出位置，凭据、原始数据与权重不入 Git。
- 已授权的例行、可逆研究工作直接推进，不重复确认；长期训练或公开发布按实际授权范围处理。
