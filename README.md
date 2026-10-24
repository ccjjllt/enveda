# Enveda CASMI 2026 研究项目

从串联质谱（MS/MS）预测分子结构，研究候选召回、排序、多谱融合与模型训练。比赛：[Enveda CASMI 2026 — Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)。交流与研究记录默认使用中文。

## 当前状态：030（2026-10-24，v15已提交待评分）

**实际最佳 v14/v465 public 0.456；030的kernel v9已COMPLETE，并于10月24日13:05正式提交为v15，ref `57583178`，最后查询PENDING，尚无新分数。** 最佳提交仍为ref `57462581`，kernel `ccjjllt/enveda-020-e006-repro` v8。10月22日已核实名次515/3802，榜单第1/5/20为0.504/0.484/0.469，均为当时公榜快照。

030以实际已得分的v465源码为基线，保留有ICE分数的N1代理GLACIER结果，并在这些分子上联合重排。029a漏移植`N1_COMPLEMENT=False`，保留作历史研究产物；下一次直接使用030。

- [030计划](docs/plan/030-n1-challenger-and-score-strategy.md)：研究假设、完整N1路径与手动继续顺序。
- [030进展](docs/progress/030-n1-challenger-and-score-strategy.md)：实际提交/配额、产物、已核对内容与未验证边界。
- [030提交notebook](notebooks/030/enveda-030-n1-challenger.ipynb)及[改动与来源](notebooks/030/PATCHES.md)。

本轮可见运行25分33秒，Python3.12.13/T4×2，ICE/GL均成功；smoke中真实N1适用分子为0，合成代理路径跑通，不能当作提分结果。提交后用户表示“跑完告诉你”，现等待通知再查 **ref57583178**，不重复push或提交，也不安排自动任务。本轮使用ccjjllt认证；后续实际分数见030进展。

## 研究优先级与已知边界

1. 等v15隐藏重跑评分，读取ref57583178的实际分数，与0.456比较；可见运行、隐藏重跑和公榜提高分别记录。
2. 建立包含ICEBERG/GLACIER的结构留出验证。旧CPU面板跳过模拟器，多次看好实际掉分的融合改动；NP250/P/Q只能作开发集。
3. 独立验证028的候选召回增量，再研究缺参考、同式候选、负离子与不支持加合物的证据。028本地扩池增加14,237结构，尚无远端运行或提分结论。
4. 保留多CE、完整生成和困难负例训练方向，以实际失败机制驱动实验；不从public总分推断隐藏类别比例。

可见`test.parquet`是训练集占位副本，只检查运行与输出。未见结构按规范化结构整组留出；已见结构的独立谱检索另报。比赛指标为**MRR@25**，最终通过断网Kaggle notebook生成`submission.csv`并提交。

## 目录与入口

```text
enveda/
├── AGENTS.md              # 协作约定与研究记录要求
├── docs/                  # 配对计划/进展、历史、远端工作流
├── src/                   # 可复用研究、训练、推理模块
├── scripts/               # 按三位轮次归类的手动研究脚本
│   ├── 030/               # 当前notebook构建脚本
│   └── legacy/            # 未标轮次的历史脚本
├── notebooks/
│   ├── 028/ 029/ 030/     # 近期研究与当前提交候选
│   ├── submissions/       # 历次提交入口，保留原版本目录名
│   └── archive/           # 早期baseline、Colab、试点与推理草稿
├── data/
│   ├── references/        # 精简来源、人工核对与研究结果
│   ├── raw/               # 原始比赛数据，不入Git
│   └── processed/         # 本地预处理数据，不入Git
├── datasets/              # Kaggle输入资产说明、构建入口与metadata
└── artifacts/             # 本地研究产物、日志、模型，不入Git
    ├── runs/              # 数字轮次产物
    ├── shared/            # 跨轮复用候选池等资产
    ├── submissions/       # 历次提交输出
    └── legacy/            # 早期具名实验与训练bundle
```

- [文档总索引](docs/README.md)：001–031配对计划与进展、交接文档。
- [研究脚本](scripts/README.md)、[notebook入口](notebooks/README.md)、[公共模块](src/README.md)：按用途定位代码。
- [研究历史](docs/research-history.md)：各轮主要结论、失败机制、旧分数与历史判断。
- [比赛与远端工作流](docs/remote-workflow.md)：数据、规则、硬件、Colab连接、Kaggle MCP/CLI与提交示例。
- [031整理记录](docs/progress/031-workspace-organization.md)：目录迁移、缓存清理和人工核对记录。

## 工作约定

接手先读对应计划与进展，结束后更新进展。脚本按研究轮次归类，临时缓存与大文件放忽略目录，原始数据和权重不提交Git。失败实验及其实际结果保留用于研究复盘。

不创建测试文件、测试框架或CI门控；数据读取、语法检查、notebook执行与研究指标核对均手动发起。复现记录包含notebook/资产版本、seed、设备、耗时、指标与输出位置。凭据留在用户认证配置中，不写入仓库、文档或日志。完整约定见[AGENTS.md](AGENTS.md)。
