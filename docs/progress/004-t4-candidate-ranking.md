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

## 接手顺序

1. 阅读 004 计划与本进展；先用能够实际发现 Colab notebook 工具的客户端连接明确的私有 notebook，按用户要求不继续低效 GUI 自动化。
2. 在该 runtime 写入/上传 R0 notebook 和两份本地输入，运行真实前向并取回 JSON。参考代码来源固定 v32；重建环境时不能遗漏该文件。
3. 先实现 chemistry/评分、固定 Q/L/C、`fp_bits.npy` 和候选库，再做 FPNet 排名；R0 只是前向研究产物，不能代替训练脚本。
4. 用实际数据开展短程训练，依据测得的成本决定完整训练；按计划分别解释公开 checkpoint 微调与严格整组留出的从头训练对照。
5. 整理一个真正离线 Kaggle 推理 notebook，取得本项目第一份隐藏重跑结果，再以失败分析决定追加哪条增量路线。
