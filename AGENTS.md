# 项目协作约定

本项目研究 Kaggle **Enveda CASMI 2026 — Molecule ID From Mass Spectra**（`enveda-CASMI26-molecule-id-mass-spectra`），交流和研究文档默认使用中文。

## 研究方式

- 专注研究假设、实验、模型和分数，不搭建与研究无关的工程门控。
- **不创建测试文件，不引入自动化测试、测试框架或 CI 测试门控**；所有检查与验证手动发起。
- 当前保留基线为 `068a`（N-1c `LAM=0.35`，public **0.492**，ref `58412029`）；未封闭的前沿研究集中在 `rounds/075-frontier/`（同式前五重排、等预算补召回、FIDDLE 分子式与结构编辑先验）。
- 代码完成、运行成功、离线指标提升和排行榜提升分开记录；没有实际结果，不声称有效或提分。
- 本机 M5 / 32 GB 做代码、文档与轻量审计。自主训练优先使用 `jiyujia` 的 Kaggle GPU，按实测显存、吞吐和剩余配额决定是否转 Colab。正式比赛推理在 Kaggle 断网环境完成，输出 `submission.csv`。
- 不把可见占位 `test.parquet`（训练集 1:1 副本）或仅留出谱的命中率当作泛化；未见结构须按规范化结构/骨架整组留出，另设独立谱验证衡量库检索。

## 文档与清理约定

- 当前仅保留 `rounds/068-n1c-lam035/`（0.492 底座）与 `rounds/075-frontier/`（前沿计划与调研）；文档保持精简，总入口为根 `README.md` 与 `rounds/README.md`，远端操作见 `ops/remote-workflow.md`。
- 已终止方向的研究概况、代表分数、失败机制、放弃原因与删除范围统一记录在 `archive/research-archive.md`。
- 方向终止后先更新 `archive/research-archive.md` 并打 Git 标签记录提交号，再删除明细文档、脚本、notebook 与 `artifacts/` 运行产物（未跟踪产物不可从标签恢复，须在总览写明）。
- 原始比赛数据 `data/raw/`、机器可读证据 `data/references/` 与当前基线入口 `068a/` 不因清理而删除。

## 远程工作

- Kaggle MCP 用于读取比赛、数据、公开代码、运行状态、日志、提交和配额；Kaggle CLI 用于推送与提交（命令见 `ops/remote-workflow.md`）。研究账号为 `ccjjllt`（必要时进程内切换认证，不改全局默认）。
- `kernels push` 不同步整个仓库，入口须自包含或挂载输入；记录版本、输入、设备、时长、指标与输出位置，凭据、原始数据与权重不入 Git。
- 已授权的例行、可逆研究工作直接推进，不重复确认；长期训练或公开发布按实际授权范围处理。

### Colab 备用算力

- 本机连接前设置 `export COLAB_PROXY=http://127.0.0.1:7897`。包装器保留已有的 HTTP(S) 代理，若环境指向其他端口，在当前 shell 执行 `export HTTPS_PROXY="$COLAB_PROXY" HTTP_PROXY="$COLAB_PROXY"`。项目以 7897 连接。
- `colab-accounts` 查看标签。可用 `export COLAB_ACCOUNT=ccjjllt0` 为本次研究选择普通账号，或用 `colab-use ccjjllt0` 切换默认账号。`rainer` 是用户提供的 Ultra 备用账号，使用前通过 `COLAB_ACCOUNT=rainer colabx usage` 查看实际余额。Kaggle T4 能完成的作业优先留在 Kaggle。
- 创建与挂载示例为 `colabx new -s enveda-train --gpu T4` 和 `colabx drivemount -s enveda-train /content/drive`。将所需输入放入同账号的桌面 Google Drive 专用目录，远端从 `/content/drive/MyDrive/` 下读取。授权交互由账号持有人完成。
- 通过 `colabx exec -s enveda-train -f <训练入口.py> --timeout <秒数>` 运行。把模型、优化器状态、当前步数和分折清单定期保存到 Drive；会话回收后重新挂载并从最近检查点继续。`colabx status -s enveda-train` 查状态，结束后 `colabx stop -s enveda-train` 释放资源。训练结束后记录配置与可重建的输入来源，再将权重用于 Kaggle 断网推理。竞赛提交手动发起。
