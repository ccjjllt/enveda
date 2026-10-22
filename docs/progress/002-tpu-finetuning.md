# 002 — TPU 微调研究：实际进展

对应计划：[002-tpu-finetuning](../plan/002-tpu-finetuning.md)。最后更新：2026-10-01。

**状态：Kaggle TPU 短运行 v1 已 push，最后记录为 QUEUED、日志为空；没有训练结果。当前转为 Colab 优先，003 正核实接入方式，后续先核对 T4 模型兼容性，再适配 v5e-1。**

## 2026-10-01：当前策略与文档交接

- 用户因 Kaggle TPU 排队选择 Colab T4 / v5e-1。更新 `AGENTS.md`、`README.md` 和 002 计划：Colab T4 先核对已有 PyTorch 模型、权重和真实前向；保留 TPU 偏好，随后适配 v5e-1；Kaggle 留作候补与最终离线推理。
- 当前任务核实 Colab 接入并按需配置，不启动模型训练。用户进一步提出：Kaggle 已关联 Colab，是否可直接通过现有 MCP/CLI 控制；此能力由 003 核实，通用 Colab MCP 保留为拟评估方案，不预先认定必须增加服务。配置与连接结果分别写入 [003 计划](../plan/003-colab-mcp.md) 与 [003 进展](003-colab-mcp.md)，不能由本页的平台选择推断 MCP 已可调用或 Colab 已分配设备。
- 人工对照提交 `5e22537`、001 进展以及 `artifacts/runs/001-baseline/train_fetch.json` / `split_manifest.json`：完整 train 已下载并实测为 2,539,608 行、18 列、21 row groups；初步 split 使用原始 `inchikey14`。B0/B2 已有本地流程结果，两份 FPNet 权重已下载，但尚无 torch 真实加载/前向，评分也未接入官方 RDKit 规范化。README 中“未下载 train”“没有 src/入口”“尚未 push TPU”等过期当前状态已更正。
- B0/B2 数值和局限保留在 001 记录，不当成 002 的 T0/T1 结果；尤其 merged 占位与真实 FPNet-merged 推理不同，预训练实际子集未知。
- 本次 002 文档同步仅修改四份协作文档，未修改代码、001 文档、客户端全局配置或云端作业；未重新查询、运行或取消既有 Kaggle v1。人工查看改动 diff，并执行 `git diff --check`，未发现空白格式问题；003 的新增入口由对应任务写入。下面的 QUEUED 与配额均为历史快照，不代表实时状态。

## 2026-10-01 补充：TPU v1 已 push（最后观察为 QUEUED）

- 用户明确授权使用 Kaggle TPU 后，经 CLI 人工 push：`kaggle kernels push -p notebooks/archive/tpu-training --accelerator TpuV5E8`，返回 `Kernel version 1`，地址 `ccjjllt/enveda-tpu-finetuning`。metadata 已补真实用户名，`is_private true`、`enable_tpu true`、`machine_shape TpuV5E8`、离线、比赛输入本比赛。
- push 后约 6 分钟内 `kaggle kernels status` 均为 `QUEUED`，`logs` 暂空；MCP `get_notebook_session_status` 返回空。这是当时的观察；本次文档更新未刷新状态。配额查询成功不等于分配成功，实际设备/版本/内存待 runtime 日志确认。
- 本次 push 前 TPU 配额 0/72000s 未用，GPU 约剩 3.2h，刷新 10-03 00:00 UTC。短运行只做环境/设备/挂载/单步占位，不训练。

## 2026-09-30 补充：参考固定与 TPU 骨架（已人工验证）

- 固定 `inversion/casmi-denovo-tutorial-notebook v9`（谱→SMILES 生成，编码器 6 层 12 头，从头训练，不作微调起点）、`prvsiyan/analog-propagation-casmi-2026-baseline v32`（FPNet d=512/6 层/8 头/128 峰，BCE+排序损失，merge_p=0.6）、`prvsiyan/casmi26-fp-models-v2 v1` CC0（两文件各约 144MB，T1 首选）、`jahidhsn/casmi26-fingerprint-model` MIT 约 29.4MB（架构与训练来源未知，次选）；DreaMS/MIST 均为 MIT，TPU 路径未验证。Notebook 许可字段 MCP 无返回，标未知。
- 当时新建 `notebooks/archive/tpu-training/002-tpu-finetuning.ipynb`（环境/设备核对 + 单步占位前向，不训练）+ `kernel-metadata.json`（`TpuV5E8`，离线私有）。本地提取 cells 验证：torch/xla 缺失时优雅跳过，无训练循环。**截至该 09-30 步骤**未 push、未申请设备；10-01 已 push v1，见上节。

## 实际完成内容

- 历史策略：初版遗漏 Kaggle TPU；09-30 补充后曾优先评估 Kaggle v5e-8、Colab v5e-1 备用。10-01 因 Kaggle 排队改为 Colab T4 先兼容性核对、随后 v5e-1；最终仍在 Kaggle CPU/GPU 离线推理。
- 查到主办方允许私有云存储、处理与训练数据的明确回复；将通用私有云许可应用于 Colab 是本项目判断。
- 核对 v5e 单芯片 16 GB HBM、T4 每卡 16 GB，以及当前 Kaggle 账户动态配额。
- 通过 Kaggle MCP 读取官方 de novo 教程 v9 与 Analog Propagation baseline v32 源码；读取公开指纹 checkpoint Dataset 与 Kaggle Models 的元信息。
- 核对 DreaMS/MIST 官方仓库、公开权重入口和方法定位。
- 在 README 与 002 计划中写明 PyTorch/XLA 优先路线、数据精度、固定形状、手动短运行、持久化及离线导出方案。

当前代码与权重状态：001 已创建 `src/data.py`、`src/retrieval.py`、`src/evaluation.py` 和基线 notebook，并下载两份 FPNet 权重；002 已有 TPU 环境/单步占位入口并 push v1。尚无真实训练循环、正式模型推理入口或微调产物。Colab 连接与 notebook 状态以 003 记录为准；无比赛提交记录，既有云作业的实时消耗未在本次文档更新中复查。

## 已确认事实与仍属判断的内容

| 内容 | 证据状态 |
| --- | --- |
| 本机 M5 / 32 GB、Colab 可用 T4 / v5e-1 | 用户提供；尚无本项目 Colab 模型运行实测 |
| 当前会话 Kaggle MCP | 已成功调用 |
| Colab 接入 | 003 已安装官方服务并配置三个客户端，初始握手成功；浏览器/动态工具/runtime 未连接验证，不等于训练已启动 |
| Kaggle 配额 | 09-30 快照 GPU 剩约 9.56h；10-01 push 前约剩 3.2h、TPU 0/72,000s 已用。刷新记录为 10-03 00:00 UTC；历史快照，不是 Colab 配额 |
| Kaggle TPU 型号参数 | 官方 metadata 为 `TpuV5E8`；本机 CLI 已 push v1，最后观察 QUEUED，实际分配尚无日志确认 |
| 公开 FPNet 架构、固定峰数和训练脚本 | 已读取代码，未执行 |
| 单谱/合并谱两份 checkpoint | 001 已下载并核对文件大小/哈希和包内元数据；未用 torch 真实加载/前向 |
| TPU 可作为本项目训练平台 | 技术可行性判断；具体模型兼容性、吞吐和显存尚未证实 |
| TPU 微调可以提高比赛分数 | 研究假设，尚无本项目证据 |

公开教程中关于显存/吞吐的注释及社区模型分数不作为本机/本项目测量。公开 checkpoint 的训练来源、划分、预训练暴露和依赖版本需要继续查清。

## 实验结果

002 尚无 T0/T1/T2/T3 结果，没有训练 loss 曲线、模型 MRR@25/Top-k、编译时间、稳定吞吐、HBM 峰值或导出权重。001 的 B0/B2 本地占位评分口径不填入本轮结果。不能声称模型已适配 TPU、已微调或能提分。

## 后续方向

2026-10-01 最新研究已转入 [004 计划](../plan/004-t4-candidate-ranking.md)与[004 进展](004-t4-candidate-ranking.md)。用户已授权 T4 训练，真实设备为 Tesla T4 / 14.563 GiB，但本轮仅执行环境信息单元；没有 T0/T1/T2 模型结果。按用户要求停止 Computer Use，后续执行使用可用 MCP 工具。

1. 继续 003 的浏览器连接与动态工具核对；已确认账号绑定增加的 Kaggle GPU 时长可沿用 Kaggle 工具，但使用 Colab 自身 runtime 需 Colab 侧连接。当前范围不含模型训练，既有 Kaggle 排队作业不作自动取消处理。
2. 后续在 Colab T4 真实加载现有 FPNet 权重并检查前向、输入约定和私有持久化；继承 001 数据与初步 split，补齐官方规范化评分及重复谱处理，再建立可信的 T0。
3. 以同一权重和输入适配 Colab v5e-1 PyTorch/XLA，手动核对算子、编译与保存，再在明确研究预算下执行短程训练。Kaggle 算力仅作候补，最终推理仍回 Kaggle。
4. 若现有模型存在难处理的 XLA 算子或动态形状问题，先评估简化代价，再选择 GPU 继续或单独研究 JAX 实现；记录原因，不为平台迁移无限扩展范围。
5. 完成领域微调和同配置对照后，再决定扩大训练、添加 DreaMS 或开启库外分子的 de novo 主题。

## 2026-09-30：Kaggle TPU 路线补充

以下为 09-30 的历史补充。用户指出为何没有使用 Kaggle TPU；当时人工核对了官方 CLI metadata、Kaggle TPU 文档、Google v5e 规格及 XLA 分片说明，并重新读取账户配额。README、AGENTS 与 002 计划消除了“必须先配置 Colab 才能开始 TPU 实验”的错误前提，补入 Kaggle 训练入口和参数示例。该技术结论保留，但当前平台顺序已按 10-01 排队情况调整。

Kaggle TPU 文档中的旧硬件教程已标过时；型号采用 metadata 的 `TpuV5E8`，实际分配需运行确认。每芯片 16 GB、8 芯片合计 128 GB 是硬件容量说明，不代表单设备可用 128 GB，也不能据此承诺 8 倍吞吐。该 09-30 步骤只做查询和文档修正，没有启动训练、建立测试文件或产生性能结果；10-01 的 push 记录另列在上方。
