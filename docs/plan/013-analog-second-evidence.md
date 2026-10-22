# 013 — 可追溯第二证据引擎：参考谱/analog 支撑审计与排序试点

日期：2026-10-17（北京时间）。承接 `docs/progress/012-score-plateau-and-complementary-evidence.md` 后续实施顺序 1/2。对应进展 `docs/progress/013-analog-second-evidence.md`。

## 背景与继承

- v10 public 0.323 与 v9r2 同分；538/712199 改动太窄、排序未变是待检验解释，非已证事实。
- 012 已完成碎片（55 不支持 MRR +0.032 但 CI 含 0）、ChEBI 三星（+8457）、PubChemLite（窗内 +12092，直接 FP 干扰 NP 0.5146→0.4784）。不直接扩库提交。
- 高分复查：强 recipe 实质为 PubChem 质量窗、直接/adduct-shift 参考谱、analog Tanimoto 传播、GBM 学习排序；Ahmed v11 仅阈值 250→150 + 第二引擎插 rank2；0.451/0.452 为自报/Target。
- 本轮 Kaggle CLI（ccjjllt）已核实：提交 v10/v9r2 均为 0.323；榜首 0.495、第五 0.468；GPU 剩 27.39h、TPU 20h；已拉取 prvsiyan analog、Berat 0.336、Seyit 0.446 源码到 `artifacts/runs/013-notebooks/`，只读源码、未执行其下载/权重逻辑。

## 目标与假设

1. 先审计后排序：对 550 已有开发 query（NP250/G1-250/R3-50），统计真值在 train 253 万谱中的直接参考支撑率与 mass-shift analog 支撑率；canonical 整组口径，不按标签选参考。
2. 假设：缺参考谱场景需 analog 一致性证据（query-近邻谱相似 vs 候选-近邻结构相似），不要求候选自身有谱；已见场景按谱 hash 留出 query 本谱。
3. 不把支撑率当 MRR 增益；排序试点固定 FP+ICE 为对照，同式 top50 槽位内重排，GBM 与 logistic 消融分开报告。

## 实验设计（CPU 有界）

- 参考库：只用 `data/raw/train.parquet` 允许来源（2539608 行），键为 `inchikey14`；adduct→中性质量用 Berat ADDUCTS 表同口径（未知载荷跳过，不猜）。
- 查询谱：`artifacts/legacy/f1-bundle/inputs.json`（NP250 谱）、`artifacts/legacy/g1-validation/inputs.json`（G1-250）、`artifacts/runs/007-r3/observations_50.json` + `peaks/*.npz`（R3-50）。先冻结 query 清单与质量窗，再访问 truth。
- 清洗：floor 0.002、top256、0.01Da 容差、强度归一；pilot v1 用余弦（entropy 加权后续消融），direct 取同组 max，analog 取 ±200Da 内非同组 max + mass-shift（整体平移 Δ=query_neutral - ref_neutral）best-of。
- 两域：A 无真值域建特征前移除该组全部参考；B 已见域按谱 hash/duplicate 留出本谱。GroupKFold 按 canonical 组。
- 成本：先轻量索引（无峰列）全量排序，再按需分批读峰；550 query 目标 CPU <30 分钟，内存 <8GB；先 5 例 probe 计时再全量。

## 待修改代码与交付

- `scripts/013/build_013_analog_pilot.py`（新建，CPU only，不改 v9/v10），产物 `artifacts/runs/013-analog/`（索引、冻结清单、审计 JSON、手动核对）。
- 精简结果入 `data/references/research-results-013.json`；本轮不 push、不提交，远端大规模训练/候选先有离线证据再定。
- 禁止测试文件、自动化测试、CI；运行成功、离线信号、public 提升分开记录。
