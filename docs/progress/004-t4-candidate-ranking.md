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

## 候选侧指纹链验证（2026-10-01，Colab T4 手动执行）

- rdkit 2026.03.6（Colab 实测值；官方 scorer 2026.03.3，版本差已记录）。
- `fp_bits.npy`（v2，sha 见上条）：shape (6930,)/int64/min 2/max 10405，合法。
- 按 v32 `fp_and_mass` 原样（Morgan r2/r3 4096、RDKitFP 2048 maxPath 6、MACCS 167，concat 后 `[BITS]` 选位）：CCO→27 bits/mass 46.0419、醋酸 35/60.0211、咖啡因 548/194.0804、甘油 57/92.0473，质量与理论值一致。
- `c1cccccO`→None 经排查是测试用例笔误（环数字 `1` 只出现一次，环未闭合，本来非法）；同分子的 `Oc1ccccc1` 一次通过。解析器无问题，v32 的异常→None 语义保留。
- 发现 `coconut-casmi26-candidates` v2 自带预计算池：`coco_fp.npy` 378,349,391 B、`coco_mass.npy` 3,491,240 B、`coco_meta.pkl` 41,450,156 B（2026-09-15，MCP 实查）。R1 可直接用该池，不必本地重算 27 万结构指纹；shape 待 Colab 侧验证。

## R1 mass 阶段试点（2026-10-01，Colab T4，16 谱）

- Q 为 `np_forward_sample.json` 16 谱（`Q["spectra"]`，此前 `len(Q)=5` 只是外层 dict 键数）；C 为 coco 池 436,389（packed 867B）。
- 加合表与 `neutral_mass` 逐字采用 v32。20ppm 窗口：5–215 候选，中位约 33；最大两例 2538448（215）、2538450（156）；最小 2538446（5）。43.6 万→数十，质量过滤有效。
- 真值是否在窗内尚未核对（需 `coco_meta.pkl` 身份映射）；FPNet 打分尚未跑。这只是 mass 阶段观察，不作排序结论。

## R1 mass→FPNet 试点：coco 池真值召回为 0（2026-10-01/02，Colab T4）

- 同一 Q（16 NP 谱）/C（coco 436,389）/20ppm 窗：16 个真值结构**全不在池**（`truth_in_pool=False`），`truth_rank` 无从谈起。key 格式已排除（两边同为 14 位大写），是该快照真实缺失。
- 本地已导出 `artifacts/004-research/train_structures.jsonl`（2,539,608 行→275,810 去重结构，23MB，已忽略）：下一步 C 换训练结构池，真值必在池内，R1 才可测量排序增量。

## R1 训练结构池建成（2026-10-01/02，Colab T4）

- `train_fp.npy (275810, 867)` packed、`train_mass.npy`、`train_keys.npy`，分块 25k×4 worker，`fail=0`。C 换训练池后 16 试点真值必在池内，R1 可测排序增量。

## R1 训练池 FPNet 排序试点（2026-10-01/02，Colab T4，16 NP 谱）

- 同 Q（16 谱）同 C（训练结构池 275,810）同 20ppm 窗：16/16 真值在窗（mass 阶段召回 1.0）；FPNet single 点积排序 ranks：1×8、2×4、5、7、8、13。MRR≈0.659、Top1=0.50、Top25=1.0（本轮手工计算，仅试点口径）。
- 窗口大小 2–136（中位约 20），质量过滤后 FPNet 在几十候选内排序，链路真实跑通。
- 边界：16 分子试点；这些结构是训练集内结构（checkpoint 训练见过，属已见结构排序能力，非未见泛化）；cosine 中间档未跑（需谱库 L 接入，待补）；coco 池真值缺失结论保留（C 不同， recalls 不可比）。

## R1 全量 NP1184（2026-10-01/02，Colab T4，同 Q/C/20ppm）

- 1184/1184 真值在池；1183 有排名（1 谱真值在池但落在 20ppm 窗外，mass 缺口 1 例）。
- FPNet single：MRR=0.6001、Top1=0.4392（分母 1184 谱）；与 16 试点 0.659 同量级，试点未严重偏离。
- 用时约 8s/1184 谱（含前向+窗内点积），吞吐不是瓶颈。边界同试点：已见结构排序；cosine 档未跑；融合未做（R2）。

## R2 多谱融合（2026-10-01/02，Colab T4，NP250 分子级，同训练池）

- single（每分子峰数最多一张谱）：MRR=0.6754、Top1=0.5280，250/250 有排名。
- mean_all（全谱 logit 均值）：0.6490/0.4880；mean_ad（同 adduct 先均值再跨组均值）：0.6370/0.4760。朴素均值融合**不如 single**（已见结构试点口径，非最终结论）。
- rrfuse 首版全 None 是格子 bug（下标与 key 字符串比较），修正后重跑：rrfuse MRR=0.6300、Top1=0.4720，245/250 有排名（5 分子真值没进任何成员 top25，融合表天然缺失）。

## R2 结论（NP250 分子级，已见结构试点）

single 0.6754/0.5280 优于 mean_all 0.6490、mean_ad 0.6370、rrfuse 0.6300。本轮三种融合无一超过 single；朴素融合（均值/轮转）在此设定下无效。merged 模型对照未跑（需合并峰表路径验证，待补）。

## R3 训练分片就绪（2026-10-01/02，本地导出）

- `artifacts/004-research/train_shard_20k.parquet`（39MB，已忽略）：seed 2026，从训练保留 2,482,825 行抽 20,000 行，行号清单 `train_shard_20k_rows.json`；NP250 整组 holdout 零交集。R3 200–500 步预算估计用，不作完整训练。

## R3 预算估计（2026-10-01/02，Colab T4，head-only）

- head 可训练参数 16,298,770；目标 17,604/17,604 结构可算；`float(loss)` 有一条无害 UserWarning（标量转换）。
- bs=32：step 均值/中位 0.029s，峰值显存 **0.40 GiB**，25 步 loss 0.2967→0.2468（下降，头在学）。
- 推算：8 万步约 39 分钟，显存远非瓶颈；Colab 时间内可跑完整 head-only。权重未保存，此 25 步不定为训练完成。

## R3 20 万谱分片就绪（2026-10-01/02，本地导出）

- `artifacts/004-research/train_shard_200k_p{0..3}.parquet`（142/141/71/22MB，已忽略）：seed 2026，从训练保留抽 200,000 行（4×50k），行号清单 `train_shard_200k_rows.json`；NP 整组零交集。用户定：5000 步 × 20 万谱，head-only，NP250 同口径验证。

## 合订本（2026-10-02）：R1→R3 单 notebook

- 昨日实例误删后，按用户要求把 R0 验证过的全部 cell 合订为 `notebooks/colab/004-r1-r3-consolidated.ipynb`（15 cells / 9 代码，AST 已验）：Drive 挂载 → 依赖/环境 → kagglehub 下权重与候选集 → fp 链自检 → 模型加载 → 训练池重建 → R1 全量 → R2 融合 → R3 5000 步训练 + 每 1000 步 NP 评估 + 落盘。训练输入改为 Drive 版 4 分片（q0–q7 八小块作废，不再使用）。
- 产物统一落 `/content/004-output/`（`r1_np1184.json`、`r2_np250.json`、`head_5000.pt`、`r3_log.json`），带回本地 `artifacts/004-research/` 后 review。

## R3 5000 步 head-only 结果（2026-10-02，Colab T4，训练 20 万谱分片）

- 产物已回本地：`head_5000.pt`（62MB）、`r3_log.json`、`r1_np1184.json`（69KB）；`r2_np250.json` 下载失败，聚合输出与昨日一致（single 0.6754/0.5280、mean_all 0.6490、mean_ad 0.6370、rrfuse 0.6300/245），逐分子表待补。
- step0（公开 head）：MRR=0.6001/Top1=0.4392，与 R1 全量一致，对照可信。
- 训练 loss 0.177→0.144 下降，但 **NP 验证 MRR 掉到 0.475→0.498 区间**（1000/2000/3000/4000/5000 步：0.4750/0.4839/0.4910/0.4973/0.4919；Top1 约 0.32–0.33），始终未回到 0.60。
- 解释（假设，非结论）：通用分片训练（enveda-180 合成库主导）让 head 偏向多数域，NP/timsTOF 目标域排序受损——属域偏移/遗忘候选，需通用留出对照确证；未见结构泛化另议。
- 用时：5000 步约 204s（含 6 次 NP1184 评估）。此 head 不作为有效微调权重采用。

## 合订本重跑（2026-10-03，新 T4 实例，确定性复现）

- 旧实例误删后用合订本（`ab8f306` 之后含路径兼容/rdkit/concat/shard 修补）重跑全流程：新实例初为 CPU torch，切 T4 后正常；数据集走 Colab 缓存挂载（`/kaggle/input/`），软链兼容一次后通过。
- 四产物已回本地：`r1_np1184.json`（69KB）、`r2_np250.json`（29KB，逐分子表 250 行×4 方法齐）、`head_5000.pt`（62MB）、`r3_log.json`。数值与上一轮**完全一致**（seed 7 确定性）：R1 MRR=0.6001，R2 single 0.6754/mean_all 0.6490/mean_ad 0.6370/rrfuse 0.6300，R3 step0 0.6001→5000 步 0.4919、loss→0.1440。
- R3 结论不变：head-only 通用训练伤 NP 目标域，该 head 不采用。下一步按计划做通用留出对照确证域偏移，或混合域采样重训。

## C 配对分析（本地，r2 逐分子表 × manifest 元信息）

- single 赢 39、mean_all 赢 28、平 183（多数两边都是第 1）；两组在谱数（5.54 vs 5.14）、adduct 数（6.05 vs 6.36）上基本无分层——融合没有可识别的 systematically 有效子群。
- single 排名 >10 的硬例 11 个（key 已列表中），是 forward 重排（ICEBERG/GLACIER）的候选对象，不是融合能修的。
- 通用 Q 已导出 `artifacts/004-research/generic_queries.json`（500 结构/1411 谱，10MB，seed 2027，NP 整组排除）：A 诊断用。
- A+B 已追加进合订本（`0ff93b5`，共 20 格）：A 用 `head_5000.pt`（需传 Drive，62MB）对比公开/已训 head 在通用 Q 与 NP Q 上的表现；B 从公开 head 重起 timsTOF 50% 混合训练 5000 步（seed 11），NP 同口径评估，产物 `head_mixed_5000.pt` + `r3_mixed_log.json`。一次新实例跑完。

## A 诊断 + B 混合重训（2026-10-03，新实例一次跑完）

- A（同 session 四测，1184 NP 谱 + 1411 通用谱）：NP_pub 0.6001/0.4392，NP_tr 0.4919/0.3277；**GEN_pub 0.7846/0.6967，GEN_tr 0.4765/0.3388**（1402/1411 有排名）。
- 结论修正：训练连通用域一起打掉（0.78→0.48），**不是单纯 NP 遗忘**。首要嫌疑是本轮实现漏了计划规定的第二项损失——v32 用 `BCE + 同质量窗口 hard-negative 排序损失`，R3 格只跑了 BCE。loss 降而 ranking 崩与此一致。
- B（timsTOF 50% 混合，seed 11，产物 `head_mixed_5000.pt` 62MB 已回本地）：NP 0.6001→0.4685/0.4967/0.4837/0.4899/0.4815，同样没起来。混合采样救不了缺失的排序损失。
- 下一步：补排序损失（同质量窗口 hard-negative，负例排除真值等价结构）后重训 head-only 对照；再定是否解冻末两层。

## R3 排序损失对照（2026-10-03，Colab T4，seed 12，BCE+rank）

- 产物已回本地：`head_rank_5000.pt`（62MB）、`r3_rank_log.json`。5000 步约 667s（含 6 次双分裂评估）。
- NP：0.6001 → 0.6025/0.6215/0.5976/0.6099/**0.6217**（Top1 0.4392→0.4519）。
- GEN：0.7844 → 0.7516/0.7497/0.7518/0.7538/**0.7611**（Top1 0.6967→0.6704）。
- 结论：排序损失是起效成分——纯 BCE 双域全崩，加上 rank 后 NP 转正（+0.022）、通用小幅回落（-0.023）。增益小但方向对；此 head 可作为当前最优 head 保留，不称最终方案。
- 下一步候选：R4 简单重排器（便宜）、解冻末两层短跑、forward 重排硬例。由失败分析定。

## part2 合订本（2026-10-03，一次跑完剩余项）

- `notebooks/colab/004-part2-consolidated.ipynb`（19 格，已提交）：R1-cosine 档（20 万分片谱库）、R4（GEN 拟合→NP 评估 logistic 重排）、解冻末两层 2000 步（lr 1e-4，BCE+rank，NP/GEN 双验证）、merged 对照（合并峰表 mirror `_merge_peaks`）、forward 有界试点（无已核实权重 URL 则跳过并记录，不碰 NIST）。
- de novo 生成与 Kaggle 提交不在本子（权重决策未定 / 不同平台）。产物 `/content/004-output2/` 共 7 个文件。

## part2 回执（2026-10-03，一次跑完，产物已回本地）

- R1-cosine 档：1184 查询中 1097 有窗（中位 26 候选），但真值 top25 命中 0。根因是设计错误：L 用 20 万分片（采样已排除 NP 整组），库内无真值结构参考谱，真值在构造上不可检索。此 0 为无效实验，不作模型结论；重做需 NP 非 timsTOF 谱作 L。
- R4（GEN 拟合→NP）：MRR=0.6257/Top1=0.5025。注意口径修正：R4 用的是 rank-head 的 fp_dot，同 head single 为 0.6217，**R4 真实增量仅 +0.004**；之前记的 +0.026 是相对公开 head 的混杂比较，特此更正。
- merged 对照（merged 模型 + 合并峰表）：MRR=0.6650/Top1=0.5040，低于同 Q 的 public-single 0.6754/0.5280。merged 视角在 NP250 上无增益（与作者 Class-2 上 merged 不如 single 的方向一致）。
- 解冻末两层 2000 步（lr 1e-4，BCE+rank）：NP 0.6217→0.6147，GEN 0.7611→0.7737。基本持平，无增益；不继续加码。
- forward：按许可规则跳过（无已核实权重 URL），已记录。
- 当前最优仍是 rank-head single（NP 0.6217）+ R4（0.6257），增益都在小数点后两位。

## R1-cosine 重做有效（2026-10-03，Colab，NP 独立谱库，纯余弦 top-25）

- Q 1184 timsTOF 谱 × L 2453 非 timsTOF 谱（250 结构对齐），约 11 分钟跑完。
- **cosine：MRR=0.7076、Top1=0.6622，1016/1184 有排名**。超过同 Q 的 FPNet single（0.6217）与 R4（0.6257）。
- 结论：已见结构 + 有独立参考谱时，经典谱库余弦是最强单通道；神经指纹的价值在无参考谱（Class-2）场景。最终管线应分流：谱库命中走 cosine，无命中走 FPNet。中间曾误贴量化项目输出，已定位为贴串，作废。

## 首次提交（2026-10-03，Kaggle GPU，本项目第一个隐藏重跑分等待中）

- 提交 notebook `ccjjllt/enveda-submission` v1（FPNet single + 20ppm，coco-only 回退路径，因 GPU 镜像 Python 3.13 装不上 cp312 轮子）。
- 开发运行 34s 完成，400 行×25 候选，44KB diverse SMILES，非 CCO。产物 `artifacts/submission-v1/submission.csv`（已忽略）。
- 提交 ref `56791264`，`SubmissionStatus.PENDING`，public/private 待出。今日还剩 4 次提交。
- 边界：coco 池 43.6 万（训练结构因无 RDKit 未入池）；隐藏 Class-2/3 覆盖未知，分数出来前不声称效果。

## v1 零分复盘 + v2/v3/v4（2026-10-03）

- v1（`ccjjllt/enveda-submission` v1，GPU）：34s 跑完但 public 0.000，榜单他人正常（0.471）。复盘：dev 输出 399/400 行首候选 CCO——coco 池缺真值，窗口全空。教训：静默 CCO。
- 根因链：GPU 镜像 Python 3.13 装不上 cp312 轮子 → 无 RDKit → 训练 27.6 万结构进不了池。
- 修复：本地（RDKit 2026.03.6，同 Colab 版本）算好 275,810 指纹，传私有数据集 `ccjjllt/casmi26-train-fp-pool`（fp 239MB + mass + keys + smiles + meta；fp_bits 同哈希 `a61e37…`）。
- v2（CPU）：fail-fast 生效——CPU 镜像同样 Python 3.13，轮子装不上，ERROR  loud（无静默零分）。
- v3（GPU，T4）：训练池预计算 + coco，无 RDKit，可运行；逐成员 unpack 慢，RUNNING 约 48 分钟未完（9h 内，继续等）。
- v4（已提交仓库未 push）：逐分子批量 unpack（2 次调用代替 ≤2000 次），同逻辑加速；v3 占 session（batch 上限 2），v3 完才推。

## 接手顺序

1. 阅读 004 计划与本进展；先用能够实际发现 Colab notebook 工具的客户端连接明确的私有 notebook，按用户要求不继续低效 GUI 自动化。
2. 在该 runtime 写入/上传 R0 notebook 和两份本地输入，运行真实前向并取回 JSON。参考代码来源固定 v32；重建环境时不能遗漏该文件。
3. 先实现 chemistry/评分、固定 Q/L/C、`fp_bits.npy` 和候选库，再做 FPNet 排名；R0 只是前向研究产物，不能代替训练脚本。
4. 用实际数据开展短程训练，依据测得的成本决定完整训练；按计划分别解释公开 checkpoint 微调与严格整组留出的从头训练对照。
5. 整理一个真正离线 Kaggle 推理 notebook，取得本项目第一份隐藏重跑结果，再以失败分析决定追加哪条增量路线。
