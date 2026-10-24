# Enveda CASMI 2026 研究项目

- 034 负离子-only 分子：[M-H]- 平移 +2H 作 [M+H]+ 代理喂 ICE/GL（本地 ICE 0.681/GL 0.731 vs 随机 0.227），kernel v13 运行中。见 docs/plan/034-negproxy.md
> 033（2026-10-24）：028 结构改为 Engine1 生成候选注入，GPU 运行干净，已提交 v16（ref 57595662），待评分。当前最佳 v15 0.464。

> 032（2026-10-24）：v15 + 028 扩池两次 GPU 运行均未成功（缺 numpy 导入；v4b 特征族的池对齐资源与扩池不兼容），未提交。当前最佳仍 v15 公榜 0.464。详见 docs/progress/032-v15-plus-e1ext.md。

从串联质谱（MS/MS）预测分子结构，研究候选召回、排序、多谱融合与模型训练。比赛：[Enveda CASMI 2026 — Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)。交流与研究记录默认使用中文。

## 当前状态：030（2026-10-24，v15 public 0.464，新最佳）

**新最佳：v15（030 kernel v9，ref `57583178`）public 0.464，较v14/ref`57462581`（kernel v8）的0.456提升+0.008。** 名次快照：10月24日18:10查询170/3939（0.464同分区间约170–352），同时点榜单#1/#10/#20/#50为0.504/0.475/0.471/0.467；18:16复核171/3940，第1/5/20为0.504/0.489/0.471。10月22日已核实名次515/3802，榜单第1/5/20为0.504/0.484/0.469；均为当时公榜快照，不代表最终获奖分数线。

030以实际已得分的v465源码为基线，保留有ICE分数的N1代理GLACIER结果，并在这些分子上联合重排。029a漏移植`N1_COMPLEMENT=False`，保留作历史研究产物；下一次直接使用030。

- [030计划](docs/plan/030-n1-challenger-and-score-strategy.md)：研究假设、完整N1路径与手动继续顺序。
- [030进展](docs/progress/030-n1-challenger-and-score-strategy.md)：实际提交/配额、产物、已核对内容与未验证边界。
- [030提交notebook](notebooks/030/enveda-030-n1-challenger.ipynb)及[改动与来源](notebooks/030/PATCHES.md)。

本轮可见运行25分33秒，Python3.12.13/T4×2，ICE/GL均成功；smoke中真实N1适用分子为0，合成代理路径跑通。提分依据是正式提交0.464，尚无隐藏N1分层日志或独立离线泛化结果。本轮已完成，不重复push或提交，也未安排自动任务。

## 研究优先级与已知边界

1. 保留v15作为新基线，固定后续资产版本；用P250完整GPU缓存重放“模拟器有/无 × rel-promotion开/关”，定位旧CPU评估方向反转。这个开发诊断不评估N1收益。
2. 建立包含ICEBERG/GLACIER、且模型训练排除和跨库重复可核实的结构留出验证。旧CPU面板跳过模拟器，多次看好实际掉分的融合改动；NP250/P/Q只能作开发集。
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
