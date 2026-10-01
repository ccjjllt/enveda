# 004 — Colab T4 与下一阶段研究：实际进展

对应计划：[004-t4-candidate-ranking](../plan/004-t4-candidate-ranking.md)。更新：2026-10-01。

**状态：深入研究、历史结果复核、T4 实际设备核对及实现交接已完成；R0 前向 notebook 已编写但未远端执行。未训练模型，没有新增有效 MRR 或排行榜分数。**

## 实际完成的研究

- 更新比赛与公开榜证据：分页读取 2,001 队，榜首展示分数 0.471、第五条 0.423、第十四条 0.409；保存读取日期、公开榜口径与非原子分页限制。这些不是最终奖金/金牌线。
- 读取官方 metric 当前 v13 与主办方去立体答复，确认 scorer 不压缩重复排名；hidden 答案上游去立体，与 scorer 内部操作要分开。
- 审阅本地 B0/B2 实验代码和报告，识别非空率误作候选覆盖、B2 query/library 重叠、checkpoint 暴露推论不足等问题，更新 001 进展解释，保留原始数值。
- 手动标量扫描：NP 示例实际 1,184 谱 / 250 shipped 结构、全 timsTOF；原 seen/unseen 各 500 结构均未覆盖这 250 个结构。没有把该观察宣称为隐藏分布。
- 读取 FPNet v32 的模型、预处理、训练脚本和额外数据依赖，确认 6,930 维输出必须使用 `fp_bits.npy` 的 10,407→6,930 选位映射。
- 对照官方模型资源，区分 forward 模拟重排与 de novo 生成。优先 FPNet/候选排序，随后小批 ICEBERG/GLACIER 重排，另留分子式候选 + 生成器探索。

## 实际 Colab 运行

当前 Codex 的 Colab MCP 连接工具返回成功，但元信息和可调用绑定仍只有 `open_colab_browser_connection`，没有 cell 执行工具。为执行已授权的 T4 研究，曾短暂使用浏览器界面。

| 项目 | 实际观察 |
| --- | --- |
| notebook | [“scratchpad”的副本](https://colab.research.google.com/drive/1gxH98kv2Bs0ysUezAMOYWRKGZmo_hSJ3)，由官方 scratchpad 保存到用户 Drive 的副本 |
| 设备 | Tesla T4，PyTorch 报总设备内存 **14.563 GiB** |
| Python | 3.13.15 |
| PyTorch / CUDA build | 2.11.0+cu128 / 12.8，`cuda_available=true` |
| 页面主机 RAM | 约 12.67 GB，总量为页面读数，不是 GPU HBM |
| 执行内容 | 仅导入 torch、读取版本/设备并打印 JSON，没有模型前向、优化器或训练循环 |
| 初次输入问题 | 浏览器自动缩进造成一次 IndentationError；改成一行代码后第二次执行得到上述真实输出 |

保存副本时，尝试连接原 MCP 实例曾出现“活跃连接过多”；官方原 scratchpad 仍占用单连接。上述 T4 设备是**新保存副本通过界面运行**取得的证据，不能写成 Codex MCP 已能执行该 notebook。

用户随后指出 Computer Use 太慢。本轮立即停止界面操作，之后只做本地研究与交接；不再通过 GUI 上传文件或执行模型。没有通过界面断开/删除 T4，也没有取消 Kaggle 排队作业；后续执行者需按实际会话核对当前状态。T4 分配成功不等于 FPNet 已训练。

## 实际代码与产物

- `notebooks/colab/004-t4-fpnet-feasibility.ipynb`：11 个 cell，其中 7 个代码 cell，全部 `execution_count=null`、无输出。可在 Colab 上手动运行：环境 → 固定 v1 权重 metadata → 外部参考定义与真实谱 → single 模型真实前向 → JSON 结果。
- `artifacts/004-research/reference-fpnet-v32.py`：从 v32 cell 17/18 原样摘录模型/输入适配，保留出处；未执行，不把未知 notebook 代码许可与权重 Dataset CC0 混同。
- `artifacts/004-research/np_forward_sample.json`：从本地 train 分块抽取 **16 谱 / 16 结构，正负各 8、全 timsTOF、5 类 adduct**；273,764 B，只用于前向兼容性，不是评估集。
- `data/references/competition-update-2026-10-01.json`：比赛、评分、新讨论、排行榜和奖项证据。
- `data/references/model-strategy-2026-10-01.json`：模型路线、位映射、输入契约、资源链接与 T4 预算的证据等级。
- `data/references/colab-t4-environment-2026-10-01.json`：从本轮界面实际输出转录的设备记录，标明 UI 执行方式。
- `docs/plan/004-t4-candidate-ranking.md`：完整的研究假设、实验顺序、数据划分、代码职责及接手步骤。

R0 notebook 默认只做 single 权重前向，merged 只核对 CPU 加载和 metadata，避免把单谱输入当成已验证的 merged 模型路径。权重采用 `torch.load(weights_only=True)`，不自动退回不受限 pickle。输出将保存为 `/content/004-r0-output/r0_feasibility.json`，**本轮尚未产生这个远端输出**。

原始谱、权重、第三方原始定义仍在忽略目录；notebook 没有嵌入原始谱或凭据。未修改全局 Git 身份、未提交 commit、未新增测试文件/CI。`src/` 的方法问题已写入交接，尚未实现修复。

## 手动核对范围

- notebook JSON 可读取，7 个代码 cell 完成标准库 AST 语法解析；不等于 Colab 前向已经成功。
- NP 样本数、结构数、模式、仪器及峰数组有限值已手动核对；没有运行模型或评估该样本准确率。
- 两份研究证据 JSON 可读取，计划/进展配对，文档内部链接和修改格式作了人工检查。
- 已审阅 R0 的模型加载与张量适配代码；没有安装本地 torch 来冒充 T4 结果。

## 研究结论与效果

可以确认下一方向是 **NP/timsTOF 主观察集 + 合法完整候选池 + 真实 FPNet/多谱排序**。现有全库随机留出和旧 B2 对照不足以判断目标域收益。需要把候选真值覆盖、候选内排序和总体 MRR 分开。

不能确认：现有权重对隐藏数据的效果、任何训练/微调增益、完整 T4 吞吐或 9 小时推理预算、公开榜排名。当前没有这些结果。公开榜和作者报告仅提供研究背景。

## 004-1b 交付（2026-10-01）：splits/candidates 实现与本地实测

对应计划“目标域与数据划分”节。本轮只做代码与本地手动验证；不训练、不 push、无新增 MRR/排行榜分数。

实际新增代码（手动研究模块，非测试文件，无 pytest/unittest 依赖）：

- `src/splits.py`：仪器族归一化（5 类 timsTOF/Orbitrap/QTOF/IT/other，逐字复刻 v32 `instr_family`；qTof/qTOF 等同义因含 `tof` 归入 QTOF，timsTOF 优先不被截胡）、重复谱组检测（ms2 两列分块 sha256，不全量加载 ms2；精确重复才成组）、`build_np_manifest()`（NP250 跨全来源按 shipped inchikey14 整组留出，行号清单落盘；canonical key 占位复制 shipped key，`caliber="shipped_placeholder"`，待 RDKit 官方规范化重算）。
- `src/candidates.py`：`StructureLibrary`（C）与 `ReferenceSpectrumLibrary`（L）分离对象；float64 质量索引（bisect，不转 float32）；`mass_recall_stats()` 按 5/10/20ppm 统计真值召回与窗口内候选数；`pack_bits/unpack_bits/PackedFingerprintStore` packed 占位（6930 位→867B，不提供整池 unpack 到 GPU）。

本地手动验证（`.venv` Python 3.9.6 + pyarrow 21.0.0，`PYTHONPATH=src`，命令均为手动发起）：

- AST 语法解析通过；仪器同义逐项核对（qTof/qTOF/qtof/Qtof/QTOF/LC-ESI-QTOF→QTOF；Orbitrap/Q-Exactive/ESI-QFT/LC-ESI-ITFT→Orbitrap；Ion Trap/Linear Ion Trap/iontrap/LC-ESI-QQ/QQQ→IT；None/nan/N/A/空→other）。记录到的 v32 口径边缘：`LIT`（11 行）与 `QIT`（1261 行）因不含 `trap`/`qq` 子串归入 other，已在 manifest 规则字符串中注明 verbatim，未私自改成 IT；是否增补独立缩写映射待定（改动将偏离 checkpoint 的 instrument embedding，必须另行记录）。
- `build_np_manifest()` 实测 109s：NP keys 250；np-lib 行 1184（全 timsTOF，与计划一致）；跨全来源 holdout 行 **56783**（训练保留 2482825）；每结构行数 min 5 / median 174.5 / max 1468；holdout 来源 drug_plus 145 / enveda-np-examples 1184 / gnps 10263 / masaryk 83 / massbank 6505 / mona 5276 / msdial 2393 / pluskal_ms2 4470 / riken 26464（无 enveda-180/spectraverse）；仪器族 IT 3637 / Orbitrap 29638 / QTOF 20445 / other 1879 / timsTOF 1184；重复谱组 2088 组 / 4303 行。产物 `artifacts/004-research/np_manifest.json`（2,368,804B，已忽略）。NP250 与旧 seen/unseen 各 500 key 交集均为 0（按 shipped key 复核）。
- 结构候选库实测约 3.3s：275,810 去重结构；公式质量 mass_ok 275,810（`stripped_charge_suffix` 509 含 `+2/+3` 多电荷 12 个已处理）、unknown 0；此前 `C12H30N2+2` 类 12 个在修复前记 unknown，修复后全部入库。
- 质量窗口真值召回（候选质量=分子式单同位素理论质量 float64，query 质量=前体+adduct float64）：
  - NP1184 目标域（timsTOF）：5ppm 1154/1184=0.9747；10ppm 1181/1184=0.9975；20ppm 1182/1184=0.9983；窗口内候选数 median 8/13/16，p90 45/74/105，max 224/227/384；真值 ppm 误差 median 1.68 / p90 3.26。20ppm 仅 2 谱 miss：row 2538939（`[M-H2O-H]-` 未知加合回退，误差 126744ppm）与 row 2538518（`[M-H]-` 误差 21.01ppm，恰为 NP 前体误差 max 21.06ppm 的同一谱）。产物 `artifacts/004-research/mass_recall_np1184.json`。
  - 全 holdout 56783（跨来源对照）：5/10/20ppm 召回 0.7267/0.8255/0.9078；unknown adduct 204；真值误差 median 0.90 / p90 18.95。目标域外的稀有加合与大误差是主要缺口，不把全 holdout 召回当目标域结论。产物 `artifacts/004-research/mass_recall_holdout56783.json`。
- `git status` 仅新增 `src/splits.py`、`src/candidates.py`（另有本轮既有未提交 `src/chemistry.py`，保留未动）；未创建测试文件/框架；未 push；凭据未入仓。

待办（未执行，不虚构结果）：canonical key 的 RDKit 去立体+互变异构重算与等价结构复核；固定 Q 单谱规则（single 不依赖真值选一张 Q，融合只读 Q；同极性/同 adduct 先融合）尚未实现，当前召回为 per-spectrum 统计；R1 的 mass-only→cosine→FPNet 同一 Q/L/C 对照、packed 指纹与真实前向、R2–R4 均未开始。

## 004-1a/1c 交付（2026-10-01）：chemistry/评分修正与 FPNet 入库（review 通过）

- `src/chemistry.py`（新建，主 agent 实现）：`prepare_prediction_smiles`（去立体预处理，保留同位素/电荷）、`official_match_key`（v13 复现，无 RDKit 抛异常）、`canonical_group_key`（split 用，顺序已记录）、`load_fp_bits_mapping`/`apply_fp_bits`（标准库解析 npy，不依赖 numpy；长度/上界校验）。AST 通过。
- `src/evaluation.py`（重写，主 agent 实现）：RR 按最终位置计分（`CC;CC;CCO`→1/3 已手工验证），`nonempty_rate` 与 `recall@k` 分开，逐分子表明细，`rejected`（零候选/超25）口径，caliber 标注（本地无 RDKit，走 `placeholder_exact_string`）；001 旧函数保留作 deprecated 历史对照。
- `src/models/fingerprint.py`（subagent，经 review）：FPNet 定义与 feature contract，许可头明确 unknown（权重 CC0 不套代码）；`prep_peaks`/adduct26/instrument5 与 v32 摘录逐项相等已复核；本地无 torch/numpy 时导入期明确失败（符合要求，Colab 有这两者）。
- `src/ranking.py`（subagent，经 review）：`dot_score`、`score_single_spectrum`、`aggregate_mean_scores`、真正的 `round_robin_fuse`（已验证 `['a','x','b','c']+['x','y']→['a','x','b','y','c']`，旧 retrieval 是顺序拼接）、超 25 截断计数；`SimpleLinearRanker` 占位。
- `src/train_fingerprint.py`（subagent，AST 通过，未执行 torch 路径）：head_only/unfreeze_last2/from_scratch 三模式，200–500 步预算入口，FP16 注释，BCE+窗口排序损失，保存 optimizer/config/split。
- `notebooks/colab/004-t4-candidate-ranking.ipynb`（12 cells）与 `notebooks/inference/004-offline-inference-skeleton.ipynb`（4 cells）：JSON 合法、代码 AST 通过，无硬编码路径/ID，无嵌入谱/权重。均未远端执行。
- review 发现并确认：ranking 报告中的 `dot` 实为 `dot_score`（命名出入，功能正常）；`round_robin_fuse` 入参为 dict（初次误传 list 系调用错误，非代码问题）。

待办（未执行）：fp_bits.npy 下载联调、固定 Q 单谱规则、R1–R4 真实前向/训练、RDKit 规范化重算，均需远端手动执行。

## R0 真实前向结果（2026-10-01，用户手动执行，产物已落盘）

- 产物 `artifacts/004-research/r0_feasibility.json`（6.5KB，已忽略不入 Git），`completed_utc=2026-10-01T16:54:57Z`。
- 环境：Tesla T4（15,637,086,208 B ≈ 14.56 GiB），torch 2.11.0+cu128 / CUDA build 12.8，Python 3.13.15，numpy 2.1.3。
- 权重：kagglehub 下 `prvsiyan/casmi26-fp-models-v2/versions/1`，11.3s；`fp_single_s2.pt` 144,110,255 B，sha 与本地一致（`c19aa290…`），`weights_only=True`，CPU 加载 0.12s，`{model,step,nbits,d,layers}`，step=20000，参数 36,019,474。
- 前向：`state_dict strict` 0 缺失/0 多余；16 真实谱 → `[16,6930]`，shape 相符，110,880 值全 finite；首跑 0.56s（含建模搬运），后 3 跑各 0.041s；`research_effect: not evaluated`（未算 MRR，无提分声称）。
- 结论：single 权重与输入 contract 一致，T4 单步前向可行；后 3 跑 0.041s/16谱是小 batch 诊断速度，不估计完整训练吞吐。下一步 R1 需 `fp_bits.npy`（候选侧 10,407→6,930 映射）与固定 Q/L/C。
- `fp_bits.npy`（2026-10-01，用户在 Colab 取 `prvsiyan/coconut-casmi26-candidates/versions/2`）：`sha256=a61e372fc41091bf5c7467e08134081266262665077cb9e2c56d73e4887f3013`。Colab 侧已在 R1 前置位；本地 `artifacts/004-research/fp_bits.npy` 副本待补（55KB）。维度校验（6930/int64/上界10407）以用户 Colab 输出为准，shape 回执待补。

## 接手顺序

1. 阅读 004 计划与本进展；先用能够实际发现 Colab notebook 工具的客户端连接明确的私有 notebook，按用户要求不继续低效 GUI 自动化。
2. 在该 runtime 写入/上传 R0 notebook 和两份本地输入，运行真实前向并取回 JSON。参考代码来源固定 v32；重建环境时不能遗漏该文件。
3. 先实现 chemistry/评分、固定 Q/L/C、`fp_bits.npy` 和候选库，再做 FPNet 排名；R0 只是前向研究产物，不能代替训练脚本。
4. 用实际数据开展短程训练，依据测得的成本决定完整训练；按计划分别解释公开 checkpoint 微调与严格整组留出的从头训练对照。
5. 整理一个真正离线 Kaggle 推理 notebook，取得本项目第一份隐藏重跑结果，再以失败分析决定追加哪条增量路线。
