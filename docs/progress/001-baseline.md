# 001 — 数据理解与检索排序基线：实际进展

对应计划：[001-baseline](../plan/001-baseline.md)。最后更新：2026-09-30。

**当前状态（2026-10-01 复核）：train、初步 split、B0/B2 本地运行和权重下载检查已完成；评分仍为旧口径，B2 存在 query/library 重叠，不能据旧数值判断融合收益。完整更正与下一轮见文末及 004 配对文档。以下保留各阶段历史记录。**

## 2026-09-30 补充：001-2 最小闭环（train 实测与 split 清单，已人工验证）

## 2026-09-30 补充：数据复核与 B0 骨架（已人工验证）

- 复核 `data/raw/test.parquet`：1213 行 / 400 分子 / 12 列，adduct 分布 `[M+H]+` 959、`[M-H]-` 193、`[M+CH2O2-H]-` 31、`[M+Na]+` 22、`[M+NH4]+` 4、`[M+K]+` 2、`[M+Cl]-` 2，与既有 JSON 一致；`ionization positive 987 / negative 226`；`ms2` 前 1024 行配对 0 mismatch；`sample_submission.csv` 400 行×25 `CCO`，ID 双向差集 0。
- MCP 核对：`train.parquet` 3,033,286,496 B；`column_summary` 仅 2 列，无代表性；讨论 742042 确认 train 18 列、无 `molecule_id/spectrum_id`（本项目未下载 train，未独立验证）；讨论 742278 确认可见 test verbatim 来自 train `enveda-180`，不可当泛化集。
- 新建 `src/data.py`（流式读 parquet、缺列兼容、ms2 配对检查）、`src/retrieval.py`（adduct 占位、匹配占位、按分子聚合真实最小实现）、`src/evaluation.py`（分子级 MRR@25 占位实现，手动研究用，非测试文件）、`notebooks/archive/baseline/baseline.ipynb` + `kernel-metadata.json`（T4 离线，输出 `/kaggle/working/submission.csv`）。
- 本地手动验证：`adduct_counts` 与上一致；`baseline` 骨架产出 400 行×25 `CCO`，`rows_with_bad_width=0`。`adduct_to_neutral_mass` 未做加合物修正，匹配函数恒空，当前无 MRR、无提分，submission 仅格式占位。

## 2026-09-30 补充：001-2 最小闭环（train 实测与 split 清单，已人工验证）

- 发现 `data/raw/train.parquet` 为断点残留：2,597,322,752 B（预期 3,033,286,496 B），PyArrow 报 `Parquet magic bytes not found in footer`。用任务指定命令断点续传补全，现 3,033,286,496 B、`size_match: true`，字节/sha 记录于 `artifacts/runs/001-baseline/train_fetch.json`（`data/raw`、`artifacts` 均在 `.gitignore`）。
- `.venv/bin/python` + PyArrow 21.0.0 流式实测（只读标量列，未加载 ms2 数组）：`train.parquet` 共 2,539,608 行 / 21 row groups / 18 列；**无 `molecule_id`、`spectrum_id`**（独立验证了讨论 742042 的报告）。schema 与列细节见 `artifacts/runs/001-baseline/split_manifest.json` 的 `train` 段。
- 分组键为 shipped `inchikey14`（原文照用，未重算规范化）：去重后 275,810 个结构，`null`/空串 0 行，未分组 0 行。
- `ingest_lib` 共 11 个：`enveda-180` 1,153,785、`pluskal_ms2` 527,581、`riken` 347,171、`gnps` 220,849、`massbank` 101,727（余 6 个见 manifest）。
- `instrument_type` 命名杂乱：`timsTOF` 1,154,969、`Orbitrap` 745,562、`QTOF` 125,936 等约 80 种写法，`null` 33,053、空串 297、`nan` 字符串 972。timsTOF 是隐藏集仪器，但训练主体并非 timsTOF。
- `adduct` 约 120 种：`[M+H]+` 1,324,585、`[M-H]-` 465,688、`[2M+Na]+` 190,033、`[M+Na]+` 178,092、`[2M+H]+` 92,041、`[M+CH2O2-H]-` 83,188，完整分布见 manifest `train_profile`。
- 未见结构轨道：`seed=2026`，整组 shuffle 后取 `min(500, 5%)` = **500 结构 / 4,463 行**；参考谱库不得包含这些结构的标注谱。键清单在 manifest `unseen.keys`。
- 已见结构轨道：`seed_seen=2027`，从非留出多谱结构取 **500 结构**，每结构一种子随机挑 1 行做 query（500 行），其余做 library（3,712 行）；query/library 按行号天然不相交。ms2 层面的跨行精确去重本次未做（数组未加载），留给 B0。
- 本轮未跑 B0、未算 MRR；外部 checkpoint 相对该留出集的预训练暴露情况未知，后续引用须标注。

## 实际完成的修改与产物

- 新建根目录 `README.md`：比赛信息、计算资源判断、MCP 使用、CLI 下载/push/日志/正式代码提交方法。
- 新建 `AGENTS.md`：记录研究优先、TPU 偏好、不写测试文件、手动验证、计划与进展配对等约定。
- 新建 `.gitignore`：忽略原始数据、环境、缓存、凭据与大型模型产物。
- 建立 `docs/plan/` 与 `docs/progress/`，目前包含 001、002 两对同名文档。
- 用 Kaggle CLI 下载 `data/raw/sample_submission.csv` 与 `data/raw/test.parquet`。
- 建立只用于本轮人工读取的 `.venv`：Python 3.9.6、PyArrow 21.0.0。没有安装训练框架。
- 人工核对结果保存到 `data/references/local-data-inspection-2026-09-30.json`，包含 schema、分布、大小和 SHA-256。
- 官方页与关键讨论的精简来源记录见 `data/references/competition-evidence-2026-09-30.json`。

项目开始时为空目录且尚无 Git 仓库；本轮没有初始化 Git、创建测试文件、添加 CI、push notebook 或提交比赛。

## 实际人工核对

| 操作 | 观察结果 |
| --- | --- |
| `kaggle --version` | 本机版本 2.2.4 |
| MCP 比赛/页面/数据/讨论查询 | 成功，确认任务与规则 |
| CLI `competitions files` | 成功，列出 3 个文件；shell 需要额外联网权限 |
| CLI 下载两个小文件 | 成功，训练集未下载 |
| PyArrow 读取可见 test | 1,213 行，400 个唯一分子，12 列，全部 timsTOF |
| 可见 test 的 adduct | `[M+H]+` 959、`[M-H]-` 193、`[M+CH2O2-H]-` 31、`[M+Na]+` 22、`[M+NH4]+` 4、`[M+K]+` 2、`[M+Cl]-` 2 |
| CSV 读取 | 400 行，列 `molecule_id,smiles`，ID 与 test 一致；每行 25 个 `CCO` 占位候选 |
| MCP 输出与日志读取 | 对官方教程成功返回 `files/log/next_page_token`，可解析日志事件 |
| CLI 查询本人比赛 notebooks | 返回 `Not found`；只记录此查询结果，不推断其他比赛/账号状态 |
| CLI push/logs/submit 参数 | 根据本机帮助、安装源码和官方文档核对；没有实际 push 或 submit |
| 文档人工复核 | 两组 plan/progress 文件名配对、内部链接存在、证据 JSON 可读；未创建检查脚本或测试文件 |

`test.parquet` 各列顶层 null 数为 0；本次未检查所有数组内元素、峰配对、质量范围或训练数据质量，不能据此声称完整数据质量检查通过。

## 研究结论与实验结果分别记录

已确认的研究背景：

- 指标为 MRR@25，最多 25 个候选，分子级聚合多张谱。
- 官方评分规范化涉及 RDKit 2026.03.3；API 旧指标名称不能替代现行 scorer。
- 可见 test 是占位集；隐藏 Class 1/2/3 需要不同研究能力。
- 外部云端私有训练获主办方允许；最终仍为 Kaggle 离线代码提交。
- 跨库同结构重复需要显式处理；已见结构与未见结构分别评估。

以上为最初准备阶段的状态，当时尚无实验结果。后续 B0/B2 数值记录如下；本项目仍没有正式规范化评分或排行榜分数。公开 notebook 的执行日志仅用于验证工具读取能力，不作为本项目实验。

## 2026-09-30 补充：001-3 B0 真实检索（已本地手动执行，仅流程自检）

- 扩展 `src/retrieval.py` 为真实 B0（保持占位接口名）：15 类 adduct 质量表（含可见 7 类 + `[2M+H]+/[2M+Na]+/[2M-H]-/[M-H2O+H]+/[M]+/[M+C2H4O2-H]-/[M-2H2O+H]+/[M+2H]2+`，float64，`M=(mz*|z|-delta)/n`，未知回退为 precursor 本身）、ppm 精确匹配（默认 20）、余弦（归一化强度、top-N 默认 128、`global_topn` 与 `window_50da` 二选一实现、本次取 `global_topn`、碎片 ppm 默认 20）。库过滤与同谱/整组留出由调用方按 manifest 保证。
- 新建一次性手动脚本 `artifacts/runs/001-baseline/b0_run.py`（非测试文件）：unseen（库=非留出 2535145 行，query=留出 4463 行/500 结构）、seen（库=3712，query=500）；unseen 每 query 按 `|ppm|` 取最近 200 个再做余弦（本地可行性截断，如实记录）。输出 `artifacts/runs/001-baseline/b0_metrics.json`。
- 本地手动执行（`.venv/bin/python artifacts/runs/001-baseline/b0_run.py`，M5/32GB，约 242s）：scalar 3.4s、索引 0.9s、unseen 质量过滤 0.9s、Pass2 取峰 93.3s、seen 检索 0.6s、Pass3 余弦 143.0s。未 push、未提交、未下载权重。
- 两轨道（占位口径：精确字符串命中，RDKit 未接入，分数仅自检）：
  - unseen：MRR@25 0.0、Top1 0.0、Top25 0.0（真值不在库，符合整组留出预期）、分子覆盖率 0.988、无候选比 0.012、重复 0、空槽 0；谱级质量候选 mean 873.7/p50 658/max 3477/无候选 0.017；截断后 mean 168.1/p50 200；独立库行 114774、余弦 750139 次；未知 adduct query 0.47%/库 0.27%；多 SMILES 组 0。
  - seen：MRR@25 0.992、Top1 0.988、Top25 0.996、覆盖率 0.996、无候选 0.004、重复 0、空槽 0；谱级质量候选 mean 8.8/p50 6/max 100/无候选 0.004；manifest 行号核对 0 mismatch。
- 观察：同结构余弦显著高于跨结构（抽查 0.82 vs 0.0005）；20ppm 下 unseen 仍 98.8% 分子有干扰候选，说明质量 alone 选择性有限；seen 高分仅表示“有参考谱”上限，不代表未见泛化，未用占位 test 算泛化。
- Blocker/局限：unseen 全量 20ppm 平均候选约 874（最大 3477），纯 stdlib 全量余弦不可行，本次按 `|ppm|` 最近 200 截断；稀有约 110 种 adduct 走未知回退（占比 <0.5%）；50Da 窗口实现保留但本次未跑对照；跨行精确去重（ms2 层）未做，留给 B1/B2。

## 2026-10-01 补充：001-4 B1 权重核验 + B2 多谱融合对照（已本地手动执行，仅流程自检）

- B1 下载：`kaggle datasets download prvsiyan/casmi26-fp-models-v2 -p artifacts/runs/001-baseline/weights --unzip`，v1（2026-09-16，CC0），两文件各 144110255 B（与数据页一致），`artifacts/`、`*.pt` 均在 `.gitignore`，未入 Git、未提交。
  - `fp_single_s2.pt` sha256 `c19aa290…b2e52b614a`；`fp_merged_m1.pt` sha256 `261529c8…c507cce829`（两权重不同，data.pkl sha 各异）。
  - 本地无 torch（`.venv` 仅 PyArrow 21.0.0），按任务不强装 CUDA/XLA：文件头 `PK\x03\x04`（torch.save zip），`torch.load` 记录 `ModuleNotFoundError` 失败。标准库 zip/data.pkl 字节扫描实测：顶层容器 `{model: OrderedDict, step, nbits, d, layers}`（无 `opt`/优化器状态，纯推理权重）；single `step=20000`、merged `step=24000`，同 `nbits=6930/d=512/layers=6`；state dict 含 6 层 Transformer blocks（0–5）+ head，与 002 计划 FPNet 描述（d=512/6层）一致。
  - 训练来源：权重数据页无描述、无训练划分说明；作者 baseline v32 同时挂载比赛 train 与权重。**10-01 复核更正：共同挂载不足以证明权重用该 train 训练；训练来源与具体子集仍待证实。**
  - 留出与比赛训练池的交集（shipped inchikey14）：unseen 500/500、seen 500/500，两轨道互不相交。这是留出从 train 构造的事实，**不是 checkpoint 实际训练暴露的测量**；暴露状态仍 unknown，不能称严格未见泛化。
- B2 融合对照：一次性手动脚本 `artifacts/runs/001-baseline/b1b2_run.py`（同 b0 前例放 artifacts，非测试文件），库为 3712 全池化（同 B0 seen，跨结构干扰保留），seen 500 结构 4212 谱 leave-one-out，无新权重，占位口径。本地手动执行约 103s（fetch 96.5s、LOO 检索 5.0s）。`single` 复现 B0 seen 精确一致（MRR 0.992，`b0_reproduction_match=true`）。
  - 总体（分子级占位口径）：single 0.992 / scorefusion 0.996 / merged占位 0.995 / best_oracle 上限 0.998；平均候选数 single 1.24、scorefusion 1.25（20ppm 在 3712 池内选择性极强，同结构谱占命中多数）。
  - `[M+H]+`（n=313）：single 已 0.9968，融合无增益（scorefusion/merged 0.9952，天花板效应；1 例无候选融合也修不好）。
  - `[M-H]-`（n=75）：single 0.9867/Top1 0.9733 → scorefusion 与 merged 占位均 1.0/1.0。**10-01 撤回“两种融合在负模式都有效”的解释**：融合读入的谱与参考库重叠，且可能包含其他模式/adduct，无法据此归因。
  - 其他 adduct（n=112）：single 0.9821 → scorefusion 0.9955（修好无候选例+排名）、merged 占位 0.9911（只修排名）。
  - merged 占位口径说明：组内全谱峰池化取 top128，库只剔除指定 query，自合并输入包含仍在库中的其他谱。**10-01 更正：存在直接输入重叠，连相对趋势也不能用来选融合方法。** scorefusion 同样把原 library 谱重新作 query。早期“剔除全组”版本的 MRR=0 另保留为调试历史。
- 产物：`artifacts/runs/001-baseline/b1b2_report.json`（权重版本/加载性/重叠数/B2 对照表/耗时/seed）。未 push、未提交、未训练。
- Blocker/局限：无 torch 本地无法真正前向验证权重（需 T4/TPU 远端）；merged 占位≠真 FPNet-merged 推理；分数均为占位口径（RDKit 未接入）；unseen 轨道未做融合（真值不在库，融合无意义）。

## 遇到的问题

- 沙箱内 CLI 首次出现 DNS 访问错误；允许只读网络请求后查询和下载成功，不是凭据失效。
- `uv` 默认缓存路径不在可写范围；改用项目 `.cache/uv` 后成功创建环境和安装 PyArrow。
- PyArrow 在沙箱中报告 CPU cache/NEON 探测权限警告，但本次 Parquet 读取和结果保存成功。
- MCP 数据摘要只报告两列的汇总，无法代表 Parquet schema；已通过实际小文件读取补足。
- 训练列的页面描述与社区实际读取报告存在差异；本轮没有读取完整 train，不作确认。
- CLI push 不同步本地模块，历史版本日志查询也存在 2.2.4 实现限制；README 已说明。

## 下一步交接

转到 [004 计划](../plan/004-t4-candidate-ranking.md)：分离结构候选库与参考谱库，建立固定独立 Q/L 与 NP/timsTOF 主观察集，接入官方 scorer 和真实 FPNet，再研究训练/融合。保留旧结果，不覆盖旧 JSON 成新口径。

## 2026-10-01：研究方法复核补充

- `coverage=0.988` 实际为“有至少一个非空候选”的比例；当前 unseen 候选只来自已剔除真值的参考谱库。它不表示 98.8% 真值召回，零 MRR 也不能证明神经排序无效。
- seen 使用的背景库仅为评估 500 结构的其余 3,712 谱，平均输出候选约 1.24；尚未建立独立测量或跨行去重，0.992 不与排行榜分数比较。
- `src/evaluation.py` 排名前去重会压缩错误重复的名次，与官方 v13 不同；规范化与预测去重需要实现，评分本身保留真实提交顺序。当前 source 尚未修正，见 004 交接。
- 本轮标量扫描 NP 来源为 1,184 谱/250 shipped 结构、全部 timsTOF，且与两类原留出结构交集均 0。原 unseen 中 2,054/4,463 谱为 timsTOF（322/500 结构），seen query 中 372/500 为 timsTOF。新划分需覆盖该目标域。
- 原始 `b1b2_report.json` 个别说明称 merged 剔除全组，与实际代码冲突；保留历史文件并以本更正解释，不据该字段构造下一实验。
