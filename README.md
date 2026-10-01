# Enveda CASMI 2026 研究项目

从串联质谱（MS/MS）预测分子结构，围绕候选检索、排序、多谱融合和模型微调开展研究。核实日期：**2026-09-30**。

比赛：[`enveda-CASMI26-molecule-id-mass-spectra`](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)。最初提供的第一个链接显示文字为 ARC Prize，但实际链接属于 Enveda；本项目只研究 Enveda。

## 比赛要点

| 项目 | 已核实内容 |
| --- | --- |
| 任务 | 为每个 `molecule_id` 综合多张谱，输出最多 25 个有序 SMILES 候选 |
| 指标 | **MRR@25**：首个正确候选排名为 r 时得 1/r；前 25 名未命中为 0；对分子平均 |
| 匹配 | 官方用 RDKit **2026.03.3** 规范化互变异构体，再按 InChIKey 前 14 字符匹配；以官方评分代码为准 |
| 提交 | `submission.csv`，两列 `molecule_id,smiles`；同一行候选用分号分隔 |
| 最终运行 | 只能通过 Kaggle Notebook 提交；关闭互联网；CPU/GPU 运行上限 9 小时 |
| 外部资源 | 允许公开、免费可得的外部数据和预训练模型；遵守来源许可及比赛规则 |
| 奖金与提交次数 | 总奖金 50,000 美元；每天最多 5 次提交，最多选择 2 次最终提交；队伍最多 5 人 |
| 参赛/合队截止 | 2026-12-07 23:59 UTC，即北京时间 12-08 07:59 |
| 最终截止 | 2026-12-14 23:59 UTC，即北京时间 12-15 07:59 |

来源：[概览与评测](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview)、[官方评分代码](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank)、[规则](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules)、[数据](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data)。API 元信息里可能仍显示旧的指标名称，采用评测说明中的 MRR@25。

公开可下载不等于模型训练数据许可有效：主办方特别限制违反 NIST 等谱库许可分发的衍生权重；已明确准许 MassSpecGym 及相应 GLACIER 权重，并单独确认 CFM-ID 4。具体使用具体版本，不能把例外外推到所有同系列模型。[许可答疑](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742193)、[GLACIER 确认](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744338)、[CFM-ID 4 确认](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743774)

当前可见 `test.parquet` 是从训练集抽取的**占位数据**，最终会替换成隐藏测试集。不能把可见 test 上的记忆、精确检索或高命中率作为研究效果。官方描述隐藏数据约 400 个分子、1,500 张谱，全部来自 timsTOF，覆盖 10 类 adduct；这些描述不等于当前占位文件的实际统计。

| 文件 | 官方文件大小 | 本轮实际核对 |
| --- | ---: | --- |
| `train.parquet` | 3,033,286,496 B | 尚未下载，不将页面描述当成已读取的 schema |
| `test.parquet` | 4,848,729 B | 1,213 张谱、400 个分子、12 列、7 类 adduct，全部 timsTOF |
| `sample_submission.csv` | 43,619 B | 400 行，每行 25 个占位候选；ID 与可见 test 一致 |

两个小文件已下载到 `data/raw/`。字段、文件哈希和人工读取结果保存在 [本地数据核对记录](data/references/local-data-inspection-2026-09-30.json)。训练集的分子会跨来源重复，未见结构实验需按规范化结构整组留出，不能随机切分谱行。

官方训练来源表合计 2,539,575 张谱，正文报告 275,810 个结构；这是页面报告值，尚未通过完整训练文件核对。官方页面与讨论的精简证据见 [比赛来源记录](data/references/competition-evidence-2026-09-30.json)。

隐藏集包含三类问题：已有公开参考谱；结构数据库内但没有公开参考谱；结构数据库外的 de novo 分子。主办方确认第三类不在 PubChem 与 COCONUT，类别比例没有公开。因此同时保留“已见结构、独立谱”的检索评估与“未见结构”的泛化评估；不能假设全部隐藏分子都与训练结构不重叠，也不能仅靠候选库检索覆盖第三类。[主办方说明](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742274#3527095)

## 算力选择：优先评估 Kaggle TPU 训练，Kaggle GPU 推理

**可以直接使用 Kaggle TPU 研究训练/微调。建议先评估 Kaggle v5e-8，Colab v5e-1 作为备用或单芯片调试环境，Kaggle T4 ×2 用于兼容性对照和最终推理。** 先前只比较 Colab TPU 与 Kaggle GPU，遗漏了 Kaggle TPU；用户偏好 TPU 并不等于必须使用 Colab。平台选择、实际速度和提分仍需实验验证。

主办方已在[云端处理与训练讨论](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741876#3525815)确认可以私有存储比赛训练数据、在云端训练并把产物带回 Kaggle。将此许可应用到私有 Colab 训练是本项目的判断；帖子并未专门点名 Colab。最终仍要把权重、候选库、代码和依赖准备为 Kaggle 的离线输入。

| 环境 | 项目中的用途 | 实际限制 |
| --- | --- | --- |
| 本地 M5 / 32 GB | 编写代码、读取小文件、分批预处理、CPU 化学结构处理、人工检查 | 足够承担这些工作；大规模训练和完整模型评估放远端。3 GB Parquet 解压并展开成 Python 对象后可能占用更多内存 |
| Kaggle T4 ×2 | PyTorch 参考实现、必要的微调、最终离线推理 | 每张 T4 16 GB；两张不会自动成为一张 32 GB 卡。普通单卡代码只用一张，DDP 需要显式实现 |
| Kaggle TPU v5e-8 | 优先评估的 TPU 训练平台；可沿用已有 CLI/MCP 和 Kaggle 数据输入 | 8 芯片、每芯片 16 GB，合计 128 GB 分布式 HBM；需配置多设备数据并行或分片，不能当作单设备可直接分配的 128 GB。设备分配与实际运行尚未验证 |
| Colab v5e-1 | 备用训练配额、单芯片调试或 Kaggle 资源不可用时补充 | 一个 v5e 芯片、16 GB HBM；需适配 XLA/JAX、固定形状和数据管线。CPU 内存、时限、配额以实际 Colab runtime 为准 |

硬件资料：[Google v5e](https://docs.cloud.google.com/tpu/docs/v5e)、[NVIDIA T4](https://www.nvidia.com/en-us/data-center/tesla-t4/)。不能把 Google Cloud 整台主机的内存规格当作 Colab 分配值，也不能把 Kaggle 的 `TpuV5E8` 当成用户的 Colab v5e-1。

已有公开代码主要使用 PyTorch。优先考虑 **PyTorch/XLA 保留模型与权重格式**；只有收益明确时才迁移到 JAX/Flax。RDKit、SMILES 处理、动态候选搜索保留在 CPU；TPU 负责批量张量训练。采用分桶/填充、固定 batch、BF16、必要的梯度累积，并单独记录编译时间和稳定训练吞吐。先做手动短程实验，再决定完整训练预算。

Kaggle TPU 可直接复用现有远程工作流，减少跨平台数据和权重搬运。8 芯片不保证 8 倍速度；普通数据并行会复制模型，要让模型状态跨芯片共享容量，需采用参数/优化器分片等策略。[PyTorch/XLA 分片说明](https://docs.pytorch.org/xla/master/perf/spmd_basic.html)

[Kaggle TPU 文档](https://www.kaggle.com/docs/tpu)列出每周最多 20 小时、单次最多 9 小时；其中旧硬件教程已标注过时，设备型号以当前 metadata 文档和实际 runtime 为准。训练安排需要保存 checkpoint 并可跨会话恢复。这里的训练会话与比赛最终 CPU/GPU 离线提交是两个阶段；最终提交设备限制不是放弃 TPU 训练的理由。

2026-09-30 本次补查的 Kaggle 账户配额：GPU 总计 30 小时，已用约 20.44 小时，剩约 **9.56 小时**；TPU 总计 **20 小时**、已用 0、预留 0。刷新时间为 2026-10-03 00:00 UTC（北京时间 08:00）。这是账户当时的动态查询值，与 Colab 配额独立，不保证后续可用量或某种设备一定可分配。

当前会话已有 Kaggle MCP 和可用 CLI，**无需接入 Colab 就能准备 Kaggle TPU 训练**。只有实际启用 Colab 备用方案时，才需要配置其 notebook/runtime/日志/私有产物连接。本轮未启动训练或验证设备分配。

## 项目组织与研究约定

```text
docs/
├── plan/
│   ├── 001-baseline.md
│   └── 002-tpu-finetuning.md
└── progress/
    ├── 001-baseline.md
    └── 002-tpu-finetuning.md
data/
├── references/      # 精简来源与人工核对结果
└── raw/             # 本地数据，不入 Git
```

后续按需创建 `src/`、`notebooks/`、`configs/` 和 `artifacts/`，不预先搭建复杂框架。实验产物、日志和权重放 `artifacts/` 等忽略目录，并在进展文档记录位置。

- 同一轮计划与进展使用**相同编号和主题名**。计划可覆盖几轮紧密相关实验。
- 计划写清背景、目标与假设、实验方案、代码改动、实现步骤和前序结论。
- 进展记录真实修改、产物、问题和实际结果；代码状态与研究效果分开。
- **不写测试文件，不引入自动化测试或 CI 门控；所有验证手动发起。** 研究指标仍需认真计算、记录和解释。
- 实验记录至少包含数据划分、权重与数据版本、配置/seed、设备、耗时、MRR@25、Top-1/Top-25、候选覆盖率、notebook 版本及日志位置。

协作约定见 [AGENTS.md](AGENTS.md)。本轮只是研究准备与工具核实，没有训练结果或排行榜分数。

## Kaggle MCP：远程查看比赛、代码、运行状态与日志

用户已在 Codex 与 OpenCode 配置 Kaggle MCP。本轮在 **Codex 当前会话**成功读取比赛信息、页面、文件列表、讨论和公开 notebook 日志；没有单独启动 OpenCode 验证其连接。沿用已有凭据，不把 token 写进项目。官方说明见 [Kaggle MCP](https://www.kaggle.com/docs/mcp)。

MCP 由 agent 调用，下面的名称不是终端命令。不同客户端可能显示为 `mcp__kaggle__工具名` 或 `kaggle.工具名`；参数对象外层都是 `request`。

| 目的 | 工具 | 关键参数 |
| --- | --- | --- |
| 比赛信息、页面 | `get_competition`、`list_competition_pages` | `competitionName` |
| 数据列表/摘要 | `list_competition_data_files`、`get_competition_data_files_summary` | `competitionName`，列表可分页 |
| 最新讨论 | `list_competition_topics` | `competitionName`、`sortBy: "New"` |
| 讨论正文与回复 | `get_forum_topic` | `forumTopicId`、`includeComments: true` |
| 公开代码 | `search_notebooks` | `competition`、`sortBy: "VoteCount"` |
| notebook 信息/源码 | `get_notebook_info` | `userName`、`kernelSlug` |
| 运行状态 | `get_notebook_session_status` | `userName`、`kernelSlug` |
| 输出列表和日志 | `list_notebook_session_output` | `userName`、`kernelSlug`、`pageSize` |
| 下载指定产物 | `download_notebook_output` | `ownerSlug`、`kernelSlug`、`filePath`、可选 `versionNumber` |
| 账户 GPU/TPU 配额 | `get_accelerator_quota` | 空 `request` |

例如读取日志：

```json
{
  "request": {
    "userName": "inversion",
    "kernelSlug": "casmi-denovo-tutorial-notebook",
    "pageSize": 20
  }
}
```

对 `list_notebook_session_output` 的实际返回包含 `files`、`log` 和 `next_page_token`。`log` 可能是 JSON 字符串，里面是带 `stream_name`、`time`、`data` 的日志事件。先解析 MCP 的 `content[].text`，再解析 `log`，提取末尾或错误上下文；`pageSize` 只影响文件列表，不截断日志。不要把整个长日志或临时签名下载 URL 写入研究文档。

`get_notebook_info`、`get_notebook_session_status`、`list_notebook_session_output` 可用字符串 `versionLabel` 指定版本；产物下载使用数字 `versionNumber`。下载工具返回 URL/下载信息时，仍需显式保存文件，不能声称已经落地。当前没有单独的通用 notebook 日志 MCP 工具，`get_episode_agent_logs` 用于模拟比赛，不适用于这里。

## Kaggle CLI：本地开发 → push → 查看日志 → 提交

本机版本：**Kaggle CLI 2.2.4**；本轮已通过现有认证完成文件列表查询和两个小文件下载。命令以本机 `--help` 为准，官方仓库 main 可能比安装版本更新。以下命令从项目根目录执行；`YOUR_USERNAME`、版本号和 notebook 路径需替换。

### 查看、下载和本地读取

```bash
kaggle --version
kaggle competitions files -c enveda-CASMI26-molecule-id-mass-spectra --format json
kaggle competitions download enveda-CASMI26-molecule-id-mass-spectra \
  -f sample_submission.csv -p data/raw
kaggle competitions download enveda-CASMI26-molecule-id-mass-spectra \
  -f test.parquet -p data/raw

# 后续确实需要本地训练数据时再下载，文件约 3.03 GB
kaggle competitions download enveda-CASMI26-molecule-id-mass-spectra \
  -f train.parquet -p data/raw

kaggle kernels list --competition enveda-CASMI26-molecule-id-mass-spectra
kaggle kernels pull inversion/casmi-denovo-tutorial-notebook \
  -p notebooks/reference -m
```

本轮创建了仅用于人工数据读取的 `.venv`（Python 3.9.6、PyArrow 21.0.0），没有安装训练框架。可复现这个轻量环境：

```bash
uv --cache-dir .cache/uv venv --python /usr/bin/python3 .venv
uv --cache-dir .cache/uv pip install --python .venv/bin/python 'pyarrow==21.0.0'
.venv/bin/python -c "import pyarrow.parquet as pq; p = pq.ParquetFile('data/raw/test.parquet'); print(p.schema_arrow); print(p.metadata.num_rows)"
```

这些版本只用于本次读取；后续 TPU/CUDA 训练单独选择匹配的 Python、框架与运行环境。已经存在 `.venv` 时直接使用，不必重建。Codex 沙箱若报 DNS 或网络访问受限，需允许该条 CLI 联网操作；MCP 可连通不代表 shell 网络权限相同，不要先误判为凭据失效。

### 准备自包含 notebook 与 metadata

先编写 `notebooks/baseline/baseline.ipynb`，在同目录创建 `kernel-metadata.json`：

```json
{
  "id": "YOUR_USERNAME/enveda-baseline",
  "title": "Enveda Baseline",
  "code_file": "baseline.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": false,
  "machine_shape": "NvidiaTeslaT4",
  "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"],
  "dataset_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

也可以在已创建的目录执行 `kaggle kernels init -p notebooks/baseline` 生成模板再修改。此处只提供说明，当前仓库尚无可 push 的 baseline notebook。

官方 [metadata 文档](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)明确 `NvidiaTeslaT4` 表示 **T4 ×2**。本机 `--accelerator` 将覆盖 `machine_shape`。上面的硬件参数已核对文档与 CLI 源码，但本轮没有启动 GPU 分配。

Kaggle TPU 训练使用独立训练入口，metadata 设置 `"enable_gpu": false`、`"enable_tpu": true`、`"machine_shape": "TpuV5E8"`，保留私有 notebook 和所需数据来源。完成 TPU 训练入口后可执行：

```bash
# 保存并运行 TPU 训练版本，会消耗 TPU 配额；当前尚未创建此入口
kaggle kernels push -p notebooks/tpu-training --accelerator TpuV5E8
```

该参数已核对官方文档和本机 CLI，尚未实际申请 TPU。训练产物保存后作为最终 CPU/GPU 推理 notebook 的输入；不能把 TPU 训练 notebook 直接当作本比赛的最终提交入口。

**`kernels push` 只上传 `code_file` 内容和 metadata，不会把同目录的 `src/` 或整个仓库同步上去。** 多文件代码、模型、候选库和离线 wheels 需要打包为 Kaggle Dataset 等输入并填写来源，或生成自包含 notebook。入口应定位实际挂载的 `test.parquet`，按读取到的 `molecule_id` 生成输出，不能硬编码当前 400 个 ID 或本地绝对路径。最终输出路径为 `/kaggle/working/submission.csv`。

### Push、查看状态与取回输出

```bash
# 保存一个新版本并启动执行；会消耗远程算力
kaggle kernels push -p notebooks/baseline --accelerator NvidiaTeslaT4

kaggle kernels status YOUR_USERNAME/enveda-baseline
kaggle kernels logs YOUR_USERNAME/enveda-baseline
kaggle kernels logs YOUR_USERNAME/enveda-baseline --follow
kaggle kernels files YOUR_USERNAME/enveda-baseline
kaggle kernels output YOUR_USERNAME/enveda-baseline -p artifacts/001-baseline

# 可只下载关心的输出，避免拉回所有大型权重
kaggle kernels output YOUR_USERNAME/enveda-baseline \
  -p artifacts/001-baseline \
  --file-pattern 'submission\.csv$|metrics\.json$|\.ckpt$'
```

`push` 成功表示保存并触发执行；运行完成、生成有效输出、排行榜提交成功是后续独立状态。记录 push 返回的版本号，再检查日志和文件。本机 2.2.4 没有 `push --no-run`。安装源码中 `status/logs/output` 未把解析出的版本号传入请求，这些 CLI 命令只可靠读取最新运行；历史版本改用 MCP 的版本参数。

`kernels output` 在 API 返回日志时另存 `<kernel-slug>.log`，即使设置 `--file-pattern` 也会保存这份日志。

### 提交指定 notebook 版本

这是**代码比赛**，提交已成功运行的 notebook 版本，而不是直接上传本地 CSV：

```bash
# 版本 3 仅为例子；替换成已经人工核对的实际版本
kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra \
  -k YOUR_USERNAME/enveda-baseline -v 3 \
  -f submission.csv -m "001 baseline"
kaggle competitions submissions enveda-CASMI26-molecule-id-mass-spectra
```

这里 `-f submission.csv` 是 notebook 的输出文件名。相应 MCP 是 `create_code_competition_submission`，字段为 `competitionName`、`kernelOwner`、`kernelSlug`、`kernelVersion`、`fileName`、`submissionDescription`。普通文件上传 API 与此不同。提交后需要查看隐藏重跑的状态/日志和评分，不能用开发运行成功替代提交结果。[官方 CLI 教程](https://github.com/Kaggle/kaggle-cli/blob/main/docs/tutorials.md)

## 开始研究

先执行 [001 基线计划](docs/plan/001-baseline.md)，建立两类 holdout 和候选检索/排序基线；再执行 [002 TPU 微调计划](docs/plan/002-tpu-finetuning.md)。当前实际完成内容见 [001 进展](docs/progress/001-baseline.md) 与 [002 进展](docs/progress/002-tpu-finetuning.md)。

可复用参考：[官方 de novo 教程](https://www.kaggle.com/code/inversion/casmi-denovo-tutorial-notebook)、[公开 Analog Propagation 基线](https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline)。公开运行、代码注释或作者分数不算本项目实测结果。使用前记录版本，并分别确认代码、模型权重及外部候选库的许可和来源。
