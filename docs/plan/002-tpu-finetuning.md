# 002 — TPU 微调研究：Colab T4 核对兼容性，再适配 v5e-1

对应进展：[002-tpu-finetuning](../progress/002-tpu-finetuning.md)。创建日期：2026-09-30；更新日期：2026-10-01。

## 背景与前序结论

继承 001 的分子级指标、两类留出、候选覆盖率分析和离线提交方式。截至提交 `5e22537`，完整 train 已实测，split manifest、B0/B2 本地运行和两份 FPNet 权重下载检查已完成；split 使用原始 `inchikey14`，评分尚未接入官方 RDKit 规范化，merged 对照仍为占位实现。现有记录不能直接作为正式泛化基准，权重也未完成真实加载/前向。

用户保留 TPU 偏好。2026-09-30 曾优先评估 Kaggle v5e-8；2026-10-01 TPU 短运行 v1 已 push，最后记录为 `QUEUED` 且日志为空。因排队改为 **Colab 优先：先用 T4 核对已有 PyTorch 模型、权重和输入，再适配 Colab v5e-1**。Kaggle 保留为候补算力及最终推理平台，不等待队列才继续准备，也不在此推断旧作业已取消或训练失败。

003 接入研究之后，用户已于 2026-10-01 授权进入 Colab T4 训练与手动实验；当前具体路线与实现交接以 [004 计划](004-t4-candidate-ranking.md)为准。已有真实 T4 环境输出，但 Codex 的 Colab 执行工具仍未暴露，用户已要求停止低效 Computer Use；不要把“已连接”当成训练已完成。003 的先前“不启动训练”仅描述当时任务范围，不撤销本次授权。

Kaggle v5e-8 为 8 芯片、每芯片 16 GB HBM；128 GB 是分布式总容量，不是普通单设备代码能直接分配的内存。Colab v5e-1 单芯片 16 GB。Kaggle T4 ×2 每卡 16 GB；普通数据并行复制模型。主办方允许私有云处理/训练；最终 Kaggle CPU/GPU notebook 关闭互联网并在 9 小时内输出，训练平台与最终推理平台分开。

已有可复用资源以 PyTorch 为主，优先保留权重与 GPU 推理路径，尝试 PyTorch/XLA。迁移到 JAX/Flax 应有明确研究收益，避免为了平台重写整个已有模型。

## 目标与假设

- 目标：在一致数据划分和候选池下，判断 TPU 上训练/微调的谱到分子指纹模型是否改善 MRR@25，且能在 Kaggle 离线加载。
- 假设 A：timsTOF 域适配、正负模式及 adduct 条件信息比单纯增加参数更有价值。
- 假设 B：单谱与多谱/合并谱的互补信息能提高排序，需对不同融合方式做实际比较。
- 假设 C：TPU 固定形状批量训练可能有合适的吞吐，但编译、预处理和显存限制可能抵消收益。
- 不能预先保证提分或 TPU 更快；训练平台和模型效果分别评估。

## 首选模型与候选资源

| 资源 | 已读取情况 | 定位 |
| --- | --- | --- |
| [Analog Propagation 基线](https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline) v32 | FPNet 为 6 层 Transformer，d_model 512、8 头、128 峰；含中性损失、前体/adduct/仪器/碰撞能等条件 | 第一轮复用架构与训练逻辑 |
| [casmi26-fp-models-v2](https://www.kaggle.com/datasets/prvsiyan/casmi26-fp-models-v2) v1 | 两份权重已下载，各 144,110,255 B；标准库读取包结构与元数据，与 d=512/6 层/6,930 位指纹相符；未用 torch 加载 | Colab T4 首先核对真实加载/前向，再确定微调方案；具体训练子集与 holdout 暴露仍未知 |
| [Kaggle 指纹模型](https://www.kaggle.com/models/jahidhsn/casmi26-fingerprint-model) PyTorch/default/1 | 页面标 MIT、可微调，文件约 29.4 MB；架构未读 | 次要候选，不因名称直接选用 |
| [DreaMS](https://github.com/pluskal-lab/DreaMS) | 官方 MIT 仓库、预训练权重、微调教程；TPU 路径未验证 | 后续冻结表征/局部解冻的对照 |
| [MIST](https://github.com/samgoldman97/mist) | 官方 MIT 仓库、预训练 quickstart；需要子分子式预处理 | 方法参考，暂不优先移植 |

模型页面许可与代码许可不自动覆盖全部训练数据；保存来源、许可、下载版本和训练暴露信息。001 已确认两类留出各 500 个结构都在比赛训练池，但这不能证明 checkpoint 实际训练见过全部留出；具体训练子集仍未知。使用该权重时不能声称严格未见结构泛化。若没有兼容或可用的 checkpoint，明确称为“从头训练”，不能称微调。官方 de novo 教程从头初始化模型，不能用其存在证明已有可微调生成权重。

## 实验设计

| 编号 | 设置 | 作用 |
| --- | --- | --- |
| T0 | 001 固定权重的原始指纹模型 | 比较基准 |
| T1 | 固定编码器，仅训练指纹/排序头 | 小预算适配 |
| T2 | 解冻末若干层，小学习率；保留通用谱与 timsTOF 的混合采样 | 衡量域适配与遗忘 |
| T3 | 在预算允许时全量微调；单谱/合并谱分别观察 | 比较效果与成本 |

先用现有 128 峰设置；256 峰只作为单独实验，不能默认与已有 checkpoint 完全兼容。复用 BCE 与同质量窗口 hard-negative 排序损失时，负例池仅由训练侧构建，处理同结构和同义结构，避免假负例。一次只改变关键因素。

比较 MRR@25、Top-1/25、候选覆盖率、不同 adduct/仪器/谱数切片，以及总训练时间、编译时间、稳定后吞吐、显存、离线推理时间。不同设备先固定训练步数和数据进行模型比较，再单独比较相同时间预算；不要将硬件与算法效应混淆。

## 拟实现的代码

现有 `src/data.py`、`src/retrieval.py`、`src/evaluation.py` 和基线 notebook 来自 001。002 已有 Kaggle TPU 环境/单步占位入口并 push v1，尚无训练循环。以下按首个可用模型补齐，不把已有骨架当成已训练模型：

- `src/models/fingerprint.py`：与选中 checkpoint 一致的模型定义与加载。
- `src/preprocessing.py`：固定峰数/少量长度桶、元数据编码、指纹目标及数组分片；CPU 执行 RDKit 和 Parquet 解码。
- `src/train_xla.py`：XLA 训练、BF16、固定 batch、必要的梯度累积与恢复训练；先适配 Colab v5e-1 单芯片。仅在候补方案需要时补充 Kaggle v5e-8 多设备数据并行/分片，记录全局与每设备 batch。
- `src/inference.py`：CPU/CUDA 加载导出权重、多谱融合、候选排序与输出。
- `configs/002-tpu-finetuning.yaml`：模型/数据/训练配置、seed、权重路径和产物位置。
- `notebooks/archive/tpu-training/002-tpu-finetuning.ipynb` 与 metadata：保留已 push v1 的 `TpuV5E8` 短运行骨架，作为 Kaggle 候补入口；后续修改需另记版本，不为查看排队状态重复 push。
- `notebooks/archive/colab/002-tpu-finetuning.ipynb`：后续模型研究时创建，先实现 T4 权重加载/真实前向与数据核对，再增加 v5e-1 适配；003 的通用 MCP 配置不承担模型训练。
- `notebooks/archive/inference/`：离线 Kaggle 入口及 metadata；显式挂载代码、checkpoint、候选库、wheels。

不要只替换 `.cuda()` 就宣称 TPU 兼容。按当前 XLA 文档适配设备、优化器步进、attention 算子、AMP、同步与保存；记录实际运行版本。

## 实施步骤

1. 先完成并阅读 003 的 Colab 接入核实与手动连接记录，使用经核实可行的路径，不预设一定新增 MCP 服务；当前接入任务不启动训练。开始后续模型研究时，在 Colab T4 核对 Python/torch/CUDA、真实设备和可用显存，加载两份已下载 checkpoint，手动执行少量真实前向并检查输出维度、有限值和输入约定。
2. 保留相同权重与输入，在 Colab v5e-1 核对 Python/torch/torch_xla 或 JAX 版本、物理设备型号/数量与逻辑设备模式；记录加载和算子兼容问题，不把 `.cuda()` 替换当作 TPU 适配完成。Kaggle 队列和配额只在启用候补方案时按需查询，不因切换平台自动取消旧作业。
3. 在 Colab 使用私有数据存储/上传及持久化目录，记录数据和权重版本；不把本机目录或 Kaggle 挂载路径直接照搬。CPU 侧分批生成固定形状数组。峰 m/z 与连续嵌入至少使用 float32，精确质量/ppm 过滤保留 float64；不要把全部原始输入强制转成 BF16。补齐官方评分口径与候选覆盖率，再建立 T0。
4. 在实际研究预算内人工启动短程训练，记录初始化、编译、稳定迭代、损失和峰值内存。检查频繁重编译、CPU 回退和输入瓶颈；据结果选择 batch/峰数。
5. 确定预算后执行 T1/T2，按固定划分评估。先确认 Colab 私有产物的保存/恢复路径；将 checkpoint、优化器、配置、split manifest 与 seed 保存为可跨会话读取的产物，在会话上限前完成保存。若使用 Kaggle 候补则单独核对其输出保存方式；多芯片比较同时记录有效全局 batch 与学习率，不能把芯片数增多直接等同提速或提分。
6. 导出 CPU 可加载的标准权重与配置；在 Kaggle CPU/T4 人工核对同一批输入的输出、排名与数值差异。XLA 与 CUDA 不要求逐比特一致，但候选排序异常必须解释。
7. 按 MRR、切片和耗时决定是否推进 T3、融合或 DreaMS。将有效版本连同离线依赖带回 Kaggle，按正式代码提交路径执行。
8. 同步更新 progress，记录成功、失败、未验证项和下一轮方向。没有效果也保留结论与成本。

全程不创建测试文件或自动化测试门控；短运行、权重加载与评分均手动发起。连接可用、设备分配、真实前向、训练成功、离线指标和排行榜结果分开记录；Kaggle CLI/MCP 继续用于候补运行与最终提交。

## 参考

- [Kaggle 加速器 metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)
- [Kaggle TPU 会话与配额说明](https://www.kaggle.com/docs/tpu)
- [PyTorch/XLA 多设备分片](https://docs.pytorch.org/xla/master/perf/spmd_basic.html)
- [主办方允许私有云训练](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741876#3525815)
- [Google v5e 规格](https://docs.cloud.google.com/tpu/docs/v5e)
- [PyTorch/XLA 重编译说明](https://docs.pytorch.org/xla/release/r2.9/perf/recompilation.html)
- [PyTorch/XLA AMP](https://docs.pytorch.org/xla/master/perf/amp.html)
- [JAX TPU 安装](https://docs.jax.dev/en/latest/installation.html)
- [Colab 资源限制说明](https://research.google.com/colaboratory/faq.html)
