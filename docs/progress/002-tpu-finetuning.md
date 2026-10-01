# 002 — TPU 微调研究：实际进展

对应计划：[002-tpu-finetuning](../plan/002-tpu-finetuning.md)。最后更新：2026-09-30。

**状态：TPU 短验证已 push v1，等待设备分配；训练未开始。**

## 2026-10-01 补充：TPU v1 已 push（QUEUED）

- 用户明确授权使用 Kaggle TPU 后，经 CLI 人工 push：`kaggle kernels push -p notebooks/tpu-training --accelerator TpuV5E8`，返回 `Kernel version 1`，地址 `ccjjllt/enveda-tpu-finetuning`。metadata 已补真实用户名，`is_private true`、`enable_tpu true`、`machine_shape TpuV5E8`、离线、比赛输入本比赛。
- push 后约 6 分钟内 `kaggle kernels status` 均为 `QUEUED`，`logs` 暂空；MCP `get_notebook_session_status` 返回空。配额查询成功不等于分配成功，实际设备/版本/内存待 runtime 日志确认。
- 本次 push 前 TPU 配额 0/72000s 未用，GPU 约剩 3.2h，刷新 10-03 00:00 UTC。短运行只做环境/设备/挂载/单步占位，不训练。

## 2026-09-30 补充：参考固定与 TPU 骨架（已人工验证）

- 固定 `inversion/casmi-denovo-tutorial-notebook v9`（谱→SMILES 生成，编码器 6 层 12 头，从头训练，不作微调起点）、`prvsiyan/analog-propagation-casmi-2026-baseline v32`（FPNet d=512/6 层/8 头/128 峰，BCE+排序损失，merge_p=0.6）、`prvsiyan/casmi26-fp-models-v2 v1` CC0（两文件各约 144MB，T1 首选）、`jahidhsn/casmi26-fingerprint-model` MIT 约 29.4MB（架构与训练来源未知，次选）；DreaMS/MIST 均为 MIT，TPU 路径未验证。Notebook 许可字段 MCP 无返回，标未知。
- 新建 `notebooks/tpu-training/002-tpu-finetuning.ipynb`（环境/设备核对 + 单步占位前向，不训练）+ `kernel-metadata.json`（`TpuV5E8`，离线私有）。本地提取 cells 验证：torch/xla 缺失时优雅跳过，无训练循环。未 push，未申请设备，未消耗配额。

## 实际完成内容

- 初版把用户的 TPU 偏好直接解释为 Colab 优先，遗漏 Kaggle TPU；本次补充后建议先评估 Kaggle v5e-8，Colab v5e-1 备用，最终在 Kaggle CPU/GPU 离线推理。
- 查到主办方允许私有云存储、处理与训练数据的明确回复；将通用私有云许可应用于 Colab 是本项目判断。
- 核对 v5e 单芯片 16 GB HBM、T4 每卡 16 GB，以及当前 Kaggle 账户动态配额。
- 通过 Kaggle MCP 读取官方 de novo 教程 v9 与 Analog Propagation baseline v32 源码；读取公开指纹 checkpoint Dataset 与 Kaggle Models 的元信息。
- 核对 DreaMS/MIST 官方仓库、公开权重入口和方法定位。
- 在 README 与 002 计划中写明 PyTorch/XLA 优先路线、数据精度、固定形状、手动短运行、持久化及离线导出方案。

未创建 `src/`、训练脚本、Colab notebook 或 Kaggle 推理 notebook；未下载模型权重；没有远程计算消耗或比赛提交。

## 已确认事实与仍属判断的内容

| 内容 | 证据状态 |
| --- | --- |
| 本机 M5 / 32 GB、Colab 可用 v5e-1 | 用户提供；未对硬件/Colab runtime 实测 |
| 当前会话 Kaggle MCP | 已成功调用 |
| 当前会话 Colab MCP | 未发现可调用工具；仅启用备用路线时需要，不影响准备 Kaggle TPU 实验 |
| Kaggle GPU 总计 30h，补查已用约 20.44h；TPU 20h 未用、未预留 | MCP 2026-09-30 补查快照，刷新 10-03 00:00 UTC；不是 Colab 配额 |
| Kaggle TPU 型号参数 | 官方 metadata 为 `TpuV5E8`；本机 CLI 接收 `--accelerator` 并传递 `machine_shape`，未 push 或实际分配 |
| 公开 FPNet 架构、固定峰数和训练脚本 | 已读取代码，未执行 |
| 单谱/合并谱两份 checkpoint | 已读取 Dataset v1 元信息，未下载/加载 |
| TPU 可作为本项目训练平台 | 技术可行性判断；具体模型兼容性、吞吐和显存尚未证实 |
| TPU 微调可以提高比赛分数 | 研究假设，尚无本项目证据 |

公开教程中关于显存/吞吐的注释及社区模型分数不作为本机/本项目测量。公开 checkpoint 的训练来源、划分、预训练暴露和依赖版本需要继续查清。

## 实验结果

尚无 T0/T1/T2/T3 结果，没有 loss 曲线、MRR@25、Top-k、编译时间、稳定吞吐、HBM 峰值或导出权重。不能声称模型已适配 TPU、已微调或能提分。

## 后续方向

1. 继续 001 的真实基线与留出集准备；首个 TPU 短运行优先评估 Kaggle v5e-8 的实际设备、框架、数据输入与 checkpoint 保存路径。
2. 选择来源清楚、可加载的 checkpoint，以 PyTorch/XLA 进行首次手动短运行。
3. 若现有模型存在难处理的 XLA 算子或动态形状问题，先评估简化代价，再选择 GPU 继续或单独研究 JAX 实现；记录原因，不为平台迁移无限扩展范围。
4. 完成领域微调和同配置对照后，再决定扩大训练、添加 DreaMS 或开启库外分子的 de novo 主题。

## 2026-09-30：Kaggle TPU 路线补充

用户指出为何没有使用 Kaggle TPU。本次人工核对了官方 CLI metadata、Kaggle TPU 文档、Google v5e 规格及 XLA 分片说明，并重新读取账户配额。README、AGENTS 与 002 计划已消除“必须先配置 Colab 才能开始 TPU 实验”的错误前提，补入 Kaggle 训练入口和参数示例。

Kaggle TPU 文档中的旧硬件教程已标过时；型号采用当前 metadata 的 `TpuV5E8`，实际分配仍待运行确认。每芯片 16 GB、8 芯片合计 128 GB 是硬件容量说明，不代表单设备可用 128 GB，也不能据此承诺 8 倍吞吐。本轮仍只做查询和文档修正，没有启动训练、建立测试文件或产生性能结果。
