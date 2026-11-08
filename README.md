# Enveda CASMI 2026 研究项目

- 056（A 线）：复现 alexchilton 公榜成功，ref 58089862 公榜 **0.481**（新最佳，超 051 0.480）；`V8=MASS` + popwenman v80 + smoke（020 v36）。见 docs/progress/056-gold-apex-mass481.md
- 057（A 线）：kozykappa Domain-Evidence Layer（V23 + V8=MASS + smoke + popwenman v80）020 **v37** COMPLETE，ref **58107366** 公榜 **0.476**（相对 056 的 0.481 退步 −0.005；仅翻 1 个 top-1：m_050bf1 rank2→1）。不作新底座。见 docs/progress/057-domain-evidence.md
- 058（A 线）：056 底座 + easonyanyan V50 多碰撞能 public MS/MS rank-1 救援（sergiomalv/casmi-open-msms-spectra，cos≥0.78 / K≥10，LIB_MAX≥0.87 不动）；smoke + V8=MASS + popwenman v80，020 **v38** COMPLETE，2026-11-08 14:57 北京提交 ref **58118277** 公榜 **0.473**（V50 插入 5 个、vs056 N_DIFF=5；top-1 救援退步 −0.008）。见 docs/progress/058-multice-rank1-rescue.md
- 059（A 线）：056 底座 + 门控 LOTUS 天然产物同分子式异构体召回（仅 V8_LIB<0.60；质量带 + 同分子式；双神经联合前 2 才插到第 6 名；同轮强库命中分子误触发代理>15% 即中止），不叠 057/058；smoke + V8=MASS + popwenman v80，020 **v39** COMPLETE，ref **58124700** 公榜 **0.481**（与 056 打平，无增益；rank 1 不动不退步）。见 docs/progress/059-lotus-gated-recall.md
- 060（A 线）：056 底座只改第 2–25 名文档先验 `POP_W/SPEC_W` 0.10→0.12（取自 ozertuu 快照，不取其 extlib rank-1 提升），rank 1 不动 + 同轮反事实审计；smoke + V8=MASS + popwenman v80，已推 020 **v40**，watcher `/workspace/k060/`。见 docs/progress/060-docprior-ranks2-25.md

- 047b（A 线）：047a W_MID=0.60 公榜 0.466 < 046A 0.468；改试 W_MID=0.90，020 kernel **v26** 已推送，watcher 自动提交。见 docs/plan/047-wmid-scan.md
- 047 融合权重扫描（A 线）：046A 底座只改 W_MID=0.60，020 kernel **v25** 已推送，watcher 自动提交；046B 公榜 0.467（simulator_vote 未改变任何决策），最佳仍 046A 0.468。见 docs/plan/047-wmid-scan.md
- 046B（A 线）：046A + ICE/GL simulator_vote 接入碎片 rank-1 规则（对齐公开 0.477 门槛：supports<1 但 sim≥1 放行、sim≤−1 否决），020 v24 公榜 0.467（无增益）；见 docs/plan/046-sanpier-track.md
- 046-sanpier（A 线，sanpier 冲奖牌）：046A = 036 + leader-safe W_MID0.75 + 碎片 rank-1(V2 序) 复现公开 0.477，020 kernel v22 运行中、跑完自动提交；见 docs/plan/046-sanpier-track.md
- 045a leader-safe（W_MID=0.75 on model3）已提交 ref 57824441（PENDING 出分）；045b 排队明日配额日推送，见 docs/progress/045-leader-safe-frag.md
- 046 实列表残差（草稿）：045 出分后在自研管线上做有界残差，见 docs/plan/046-real-list-residual.md
- 044 硬负例 model3 已训完并入库；提交被周 GPU 配额挡住，由 045 队列承接，见 docs/progress/044-hard-negative.md
- 036 v15 + 公开神经重排（sanpier spectrum model，W=0.25 RRF），v19 公榜 0.465。见 docs/plan/036-neural-rerank.md
- 046A（A 线，sanpier + leader-safe W_MID0.75 + 碎片 rank-1，020 v22）公榜 0.468（已被 051 的 0.480 超越；许可 unknown 不参与奖金）；B 线自研最佳仍 0.464。见 docs/plan/046-sanpier-track.md
- 040 v15 + 自研神经重排（W0.25, fp0.5）kernel v18 运行中，待自动提交 v21。见 docs/plan/040-own-neural.md
- 037 v20 W0.40 公榜 0.453（< W0.25 的 0.465），保留 W0.25；038 top1 保护已构建未提交；sanpier 模型许可证 unknown（获奖风险）
- 039 自研谱模型训练完成：val isomer MRR 0.7606（fp 权重 0.5），私有 dataset ccjjllt/enveda-own-spec
- 040 v21 自研神经重排 0.464（≈v19 0.465，许可证干净）→ 主线；v20 W0.40 0.453。041 = v21 + top1 保护 → v22 0.398 大幅回退，放弃；主线保持 v21。042 自研模型续训中
- 037 神经重排权重扫描（W=0.40，v20 运行中）。见 docs/plan/037-neural-weight.md
- 035 v15 + 仅 034 负离子代理（去掉 033 注入），v18 已提交（ref 57608348，待评分）。见 docs/plan/035-v15-negproxy.md
- 034 负离子-only 分子：[M-H]- 平移 +2H 作 [M+H]+ 代理喂 ICE/GL（本地 ICE 0.681/GL 0.731 vs 随机 0.227），kernel v13 完成，v17 public 0.446（退步，不采用）。见 docs/plan/034-negproxy.md
> 033（2026-10-24）：028 结构改为 Engine1 生成候选注入，GPU 运行干净，v16 public 0.450（退步，不采用）。当前最佳仍为 v15 0.464。

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
- 043：自研双模型集成重排（042 续训 val 0.7636）；v22 保护=0.398 已废弃；v23 公榜 0.464（持平，重排进入平台期）
- 044：自研模型「候选池硬负例」微调（PubChemLite+COCONUT 同分子式在线挖掘 + 推理同款打分 listwise CE），新增贴近测试的 pool val；训练 kernel v6 运行中，k044_watch 过门槛自动建 dataset→推送提交 kernel→提交。见 docs/plan/044-hard-negative.md
- 045：045a = 044 + leader-safe 融合（rank1 W0.25，rank2–25 W0.75，离线扫描选定）；045b = 045a + 碎片校验 rank-1 规则（移植 Roman Tamrazov MetFrag-lite）。致谢 alex chilton、Roman Rozen、busyaprime、Roman Tamrazov。见 docs/plan/045-leader-safe-frag.md
- 045a 结果：v24 公榜 0.462（ref 57824441），W_MID 0.75 对 model3 有害；045b 改为 044 融合（W0.25）+ 碎片 rank-1 规则

- **048 特征工程**：自研断键碎片解释/中性丢失特征 + 线性 listwise 残差，model3 独立 holdout isomer MRR 0.8174→0.8345（+0.017，CI>0）；020 v27 已推送自动提交（B 线）。见 docs/progress/048-feature-engineering.md。
- **049 A×B 集成**：046A sanpier 流水线 + 自研 model3 rank 项 W_OWN=0.20（三种融合序统一加项，失败回退 046A）；020 v28 已推送，watcher 自动提交。见 docs/progress/049-ens-ab.md。
- **050 A 线碎片残差**：046A + 048 自研碎片解释/中性丢失线性残差 rank 项 W_FR=0.15（日志 top1_vs_046A）；020 v29 已推送，watcher 于北京时间 11-05 08:05 后自动提交（top1 无改动则跳过）。050a 结果 fr_used=12、top1_vs_046A=0，未提交；050b W_FR=0.30 已推送 020 v31，watcher 仅在 top1 有改动时提交。见 docs/progress/050-sanpier-fragres.md。


### 054 复现 romanrozen 0.482（A 线，冲牌）
- Gold Apex V22 + MASS + RR（`V8=BOTH`）+ `bgunshui/casmi26-scorer-v5-model` + LOTUS。
- v34 误交 fast-commit 占位 ref **58054713** PENDING（预期≈0）；已改 `COMMIT_MODE=smoke` 推送 **v35**，watcher `k054/run_v35.sh` 门控后 `-k/-v 35` 重交。见 docs/progress/054-romanrozen-0482.md。

### 051 公开榜首方案复现（当前 A 线最佳）
- 拆解 heliosli 0.482(私有异构体重排器+外部新版谱库，当时数据集私有不可复现)与 alexchilton 0.480(可复现)。
- **020 kernel v30 / ref 58008856 公榜 0.480**（新最佳，超过 046A 0.468）。详见 docs/plan/051-public-top-replication.md 与 docs/progress/051-public-top-replication.md。
- 050b（v31，W_FR=0.30）top1_vs_046A=0，未提交。见 docs/progress/050-sanpier-fragres.md。

### 053 碎片 FREEZE 重排（A 线，051 底座）— 未提交
- 020 v33 COMPLETE：`reranked=0`，`skipped_short=388`（可见 test 占位），GATE=SKIP_NO_RERANK，**未提交**。见 docs/progress/053-frag-isomer-freeze.md。

### 052 外部新版谱库叠加（A 线，051 底座）— 退步
- 020 v32 / ref **58036512** 公榜 **0.470**（相对 051 0.480 退步）。TAU0.55 xl 误插，停止同阈值。见 docs/progress/052-extlib-overlay.md。
