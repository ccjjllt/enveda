# Enveda CASMI 2026 研究项目

从串联质谱（MS/MS）预测分子结构，围绕候选检索、排序、多谱融合和模型微调开展研究。比赛信息核实日期：**2026-09-30**；项目状态更新：**2026-10-01**。

比赛：[`enveda-CASMI26-molecule-id-mass-spectra`](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)。最初提供的第一个链接显示文字为 ARC Prize，但实际链接属于 Enveda；本项目只研究 Enveda。

## 当前研究入口：004（2026-10-01）

目标为最终奖金与金牌。最新方向是 **目标域观察集 + 完整结构候选召回 + FPNet 排序 + 独立多谱融合**，随后研究 ICEBERG/GLACIER 正向谱重排及小预算库外生成。具体假设、实现职责与接手步骤见 [004 研究交接计划](docs/plan/004-t4-candidate-ranking.md)，实际工作见 [004 进展](docs/progress/004-t4-candidate-ranking.md)。

- 已确认 Colab 实际为 Tesla T4、14.563 GiB，Python 3.13.15、torch 2.11.0+cu128；本轮只运行了设备信息单元，尚未训练或执行真实 FPNet 前向。
- 已准备 [T4 真实谱前向 notebook](notebooks/colab/004-t4-fpnet-feasibility.ipynb) 和本地忽略目录下的 16 谱输入/原始参考定义，供手动研究继续使用。
- 旧 B0 的 0.988 是非空输出率；B2 的 0.995–0.996 存在 query/library 重叠，不能用来证明融合有效。原始记录保留，解释已更正。
- 250 个天然产物示例结构没有进入原两类留出集；下一轮显式加入 NP/timsTOF 主观察集。FPNet 的 6,930 位输出需原始 `fp_bits.npy` 映射，不能仅下载 checkpoint 就开始候选打分。
- 当前 Codex 的 Colab MCP 仍只暴露连接工具。用户已指出 Computer Use 太慢，已停止 GUI 自动化；继续研究和实现交接，远端执行等待实际可用的 notebook MCP 工具。

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

主办方最新答复明确：使用或训练于 NIST 的方案会被取消资格，包括只争奖牌的情形。优先采用已明确准许的 MassSpecGym 及相应 GLACIER 权重；CFM-ID 4 是另一个已确认具体案例，不外推到其他模型。[NIST 最新说明](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744470#3530300)、[GLACIER 确认](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744338)、[CFM-ID 4 确认](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743774)

当前可见 `test.parquet` 是从训练集抽取的**占位数据**，最终会替换成隐藏测试集。不能把可见 test 上的记忆、精确检索或高命中率作为研究效果。官方描述隐藏数据约 400 个分子、1,500 张谱，全部来自 timsTOF，覆盖 10 类 adduct；这些描述不等于当前占位文件的实际统计。

| 文件 | 官方文件大小 | 已有人工核对 |
| --- | ---: | --- |
| `train.parquet` | 3,033,286,496 B | 已完整下载；2,539,608 行、21 row groups、18 列，无 `molecule_id` / `spectrum_id` |
| `test.parquet` | 4,848,729 B | 1,213 张谱、400 个分子、12 列、7 类 adduct，全部 timsTOF |
| `sample_submission.csv` | 43,619 B | 400 行，每行 25 个占位候选；ID 与可见 test 一致 |

三个文件均已下载到 `data/raw/`。两个小文件的字段、哈希和人工读取结果保存在 [本地数据核对记录](data/references/local-data-inspection-2026-09-30.json)；训练文件的下载修复、哈希和实测 schema 保存在 `artifacts/001-baseline/train_fetch.json` 与 `split_manifest.json`，详见 [001 进展](docs/progress/001-baseline.md)。训练集的分子会跨来源重复，未见结构实验需按规范化结构整组留出，不能随机切分谱行。当前清单仍使用原始 `inchikey14`，尚未重算官方互变异构规范化键。

官方训练来源表合计 2,539,575 张谱，实际文件为 2,539,608 行，相差 33 行；此处保留差异，不把表格合计覆盖到实测值。原始 `inchikey14` 实测去重数为 275,810，与官方正文结构数一致，但不等于已重算规范化结构数。官方页面与讨论的精简证据见 [比赛来源记录](data/references/competition-evidence-2026-09-30.json)。

隐藏集包含三类问题：已有公开参考谱；结构数据库内但没有公开参考谱；结构数据库外的 de novo 分子。主办方确认第三类不在 PubChem 与 COCONUT，类别比例没有公开。因此同时保留“已见结构、独立谱”的检索评估与“未见结构”的泛化评估；不能假设全部隐藏分子都与训练结构不重叠，也不能仅靠候选库检索覆盖第三类。[主办方说明](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742274#3527095)

## 算力选择：Colab T4 先核对兼容性，随后适配 TPU

**当前优先 Colab T4：完成 FPNet/候选排序的真实研究，再根据收益决定 v5e-1 适配。Kaggle 算力保留为候补及最终离线推理平台。** 此前 Kaggle TPU 短运行 v1 最后记录为 `QUEUED`、日志为空，因此调整顺序；该快照不是实时状态，本轮未取消该作业。用户已授权 T4 训练与手动研究，当前执行结果与待实现项见 004；003 的初始配置阶段并未训练。

主办方已在[云端处理与训练讨论](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741876#3525815)确认可以私有存储比赛训练数据、在云端训练并把产物带回 Kaggle。将此许可应用到私有 Colab 训练是本项目的判断；帖子并未专门点名 Colab。最终仍要把权重、候选库、代码和依赖准备为 Kaggle 的离线输入。

| 环境 | 项目中的用途 | 实际限制 |
| --- | --- | --- |
| 本地 M5 / 32 GB | 编写代码、读取小文件、分批预处理、CPU 化学结构处理、人工检查 | 足够承担这些工作；大规模训练和完整模型评估放远端。3 GB Parquet 解压并展开成 Python 对象后可能占用更多内存 |
| Colab T4 | 当前主研究平台：权重前向、候选排序与训练 | 已实际取得 Tesla T4 / 14.563 GiB、CUDA 12.8；目前只跑过环境信息，模型吞吐与训练显存待实测 |
| Colab v5e-1 | 保留 TPU 训练偏好，完成 GPU 兼容性核对后优先适配 | 一个 v5e 芯片、16 GB HBM；需适配 XLA/JAX、固定形状和数据管线。CPU 内存、时限、配额以实际 Colab runtime 为准 |
| Kaggle T4 ×2 | Colab 不可用时的候补算力、最终离线推理 | 每张 T4 16 GB；两张不会自动成为一张 32 GB 卡。普通单卡代码只用一张，DDP 需要显式实现 |
| Kaggle TPU v5e-8 | 已有短运行入口；作为候补保留，不再等待其排队才推进 Colab | 8 芯片、每芯片 16 GB，合计 128 GB 分布式 HBM；需配置多设备数据并行或分片。已 push v1，实际设备分配与运行尚无日志确认 |

硬件资料：[Google v5e](https://docs.cloud.google.com/tpu/docs/v5e)、[NVIDIA T4](https://www.nvidia.com/en-us/data-center/tesla-t4/)。不能把 Google Cloud 整台主机的内存规格当作 Colab 分配值，也不能把 Kaggle 的 `TpuV5E8` 当成用户的 Colab v5e-1。

已有公开代码主要使用 PyTorch。优先考虑 **PyTorch/XLA 保留模型与权重格式**；只有收益明确时才迁移到 JAX/Flax。RDKit、SMILES 处理、动态候选搜索保留在 CPU；TPU 负责批量张量训练。采用分桶/填充、固定 batch、BF16、必要的梯度累积，并单独记录编译时间和稳定训练吞吐。先做手动短程实验，再决定完整训练预算。

Kaggle TPU 候补方案可复用现有远程工作流，减少跨平台数据和权重搬运。8 芯片不保证 8 倍速度；普通数据并行会复制模型，要让模型状态跨芯片共享容量，需采用参数/优化器分片等策略。[PyTorch/XLA 分片说明](https://docs.pytorch.org/xla/master/perf/spmd_basic.html)

[Kaggle TPU 文档](https://www.kaggle.com/docs/tpu)列出每周最多 20 小时、单次最多 9 小时；其中旧硬件教程已标注过时，设备型号以当前 metadata 文档和实际 runtime 为准。训练安排需要保存 checkpoint 并可跨会话恢复。这里的训练会话与比赛最终 CPU/GPU 离线提交是两个阶段；最终提交设备限制不是放弃 TPU 训练的理由。

历史配额记录：2026-09-30 补查时 Kaggle GPU 剩约 9.56 小时、TPU 20 小时未用；2026-10-01 push 前记录 GPU 剩约 3.2 小时、TPU 0/72,000 秒已用。刷新时间记录为 2026-10-03 00:00 UTC（北京时间 08:00）。这些是当时查询值，与 Colab 配额独立，不保证当前余额、队列状态或设备可分配性。

Kaggle MCP 和 CLI 已验证可用，Kaggle TPU 入口也已 push；这些能力保留。2026-10-01 已安装 Google 官方 Colab MCP，并追加到 Codex、OpenCode、Grok 的用户级配置，可跨比赛复用；初始握手成功，浏览器连接后的完整工具与训练尚未验证。具体说明与实际核对分别记录在 [003 计划](docs/plan/003-colab-mcp.md) 和 [003 进展](docs/progress/003-colab-mcp.md)。账号关联、工具可调用、设备分配和训练成功是不同状态。

## Kaggle 关联 Colab 账号，能否直接使用现有工具？

需要区分实际算力在哪里：

| 功能 | 计算在哪运行 | 如何接入 |
| --- | --- | --- |
| Kaggle `Link to Colab` 的 Pro 订阅奖励 | Kaggle；活动为合格 Pro/Pro+ 账户增加 Kaggle GPU 时长，不消耗 Colab compute units | 已生效的额度继续用现有 Kaggle MCP/CLI，无需 Colab MCP |
| Kaggle `Open in Colab` / Kaggle Jupyter Server | 仍为 Kaggle；Colab 只是编辑界面 | 仍受 Kaggle 会话、资源和队列约束，不能据此绕过 Kaggle TPU 排队 |
| Colab 自己的托管 T4 / v5e-1 runtime | Colab | 使用 Colab 界面或连接 Colab 的工具；现有 Kaggle MCP/CLI 没有已确认可选择该后端的公开参数 |

[Kaggle 官方说明](https://www.kaggle.com/docs/notebooks)将 Pro 奖励列为可调整的实验活动，文档描述额外 15/30 GPU 小时；实际是否领取成功以账号额度为准。2026-10-01 本次只读查询仍为 GPU 总计 30 小时、剩约 47 分钟，TPU 20 小时未用，尚未体现附加 GPU 时长。这不证明没有绑定，也不自动判断订阅资格。

为了使用 Colab 自己的算力，**不需要自行开发新 MCP**：本项目直接使用 [Google 官方 colab-mcp](https://github.com/googlecolab/colab-mcp)。它是本地 stdio 到浏览器的连接桥，和 Kaggle 的远程 API 机制不同。

## Colab MCP：一次全局配置，多个比赛复用

本机已安装 `colab-mcp 1.0.1`，固定官方提交 `b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1`；独立工具环境使用 Python 3.13.14，项目 `.venv` 未变。可执行文件为 `/Users/zhaohuaxiang/.local/bin/colab-mcp`。本机不需要安装 CUDA 或 TPU 驱动来运行这座连接桥。

已向下列文件**合并新增** `colab-mcp`，保留已有 Kaggle 和其他设置；备份在各配置原目录，未复制到项目。

- Codex：`~/.codex/config.toml`
- OpenCode：`~/.config/opencode/opencode.jsonc`
- Grok CLI：`~/.grok/config.toml`

Codex 与本机 Grok CLI 的新增配置均为：

```toml
[mcp_servers.colab-mcp]
command = "/Users/zhaohuaxiang/.local/bin/colab-mcp"
args = []
enabled = true
startup_timeout_sec = 30
tool_timeout_sec = 120
```

OpenCode 的新增配置片段为：

```json
{
  "mcp": {
    "colab-mcp": {
      "type": "local",
      "command": ["/Users/zhaohuaxiang/.local/bin/colab-mcp"],
      "enabled": true,
      "timeout": 120000
    }
  }
}
```

这是合并片段，不能用它覆盖已有配置。其他电脑替换可执行文件绝对路径；官方也提供 `uvx --python 3.13 git+https://github.com/googlecolab/colab-mcp` 启动方式。本次为固定版本且避免每个客户端首次启动临时下载，先安装后引用同一可执行文件。Python 3.13 要求只针对本地 MCP，不要求 Colab 训练环境也用同一版本。

首次使用：

1. 保存工作后重启/重新加载对应客户端的 MCP；当前已打开的聊天不一定自动获得新工具。先使用一个客户端核对完整连接。
2. 告诉该客户端：`请调用 colab-mcp 的 open_colab_browser_connection，连接后列出实际可用工具；先不要执行训练或申请加速器。`
3. 工具会打开默认浏览器中的 Colab 页面。用户自行完成 Google 登录并确认页面上的 MCP 连接提示。服务等待连接最多 60 秒；超时后可在客户端重试。
4. 连接成功后核对是否出现 notebook 编辑/执行等实际工具；不要只凭初始连接工具存在就认定完整可用。Google 官方要求客户端支持 `notifications/tools/list_changed`。
5. 保存一个按比赛命名的私有 notebook，并确认目标页面。需要计算时在 Colab 选择 T4 或 TPU；不要把 MCP 连接成功当作设备已经分配。004 已另外通过 UI 确认 T4 环境，当前不再默认用 GUI 自动化执行实验。

目前已核对：Codex 的连接工具已返回成功，但仍只暴露这一工具；Grok `mcp doctor colab-mcp --json` 成功发现初始工具；OpenCode 本机实现支持工具变更通知。**三个客户端均没有完整 notebook MCP 操作成功的本项目记录。** 当前 Codex 不能据连接成功直接下发训练，后续从可用客户端的实际工具继续；已取得的 UI 环境结果另记在 004。

配置和安装可供多个比赛复用。每个客户端启动自己的进程和随机本地端口；官方实现每进程仅接受一个浏览器连接，不是供三个客户端共同控制同一 notebook 的共享服务。建议同一 notebook 一次由一个客户端控制；并行比赛用独立 notebook、连接和产物目录。

三个配置的调用预算均设为约 120 秒（OpenCode 单位为毫秒，其实现也将此值用于工具执行）。这不能保证一次调用等待数小时训练完成；后续按实际 Colab 工具能力启动任务并分次读取状态/日志，不把超时后的任务默认当成已停止。

保持关联浏览器页、本地客户端和电脑在线，才能维持这条 MCP 控制通道；断线不等于训练必然停止，也不保证会继续。长训练需要持久 checkpoint 和日志，MCP 的单次调用超时不等于训练会话时限。官方当前没有 `--tpu`、`--notebook-url` 或固定端口配置参数，实际 notebook 工具在连接后动态获取。

配置依据：[OpenAI 官方 MCP 文档](https://developers.openai.com/codex/mcp)、[OpenCode MCP 文档](https://opencode.ai/docs/mcp-servers/)、本机 `~/.grok/docs/user-guide/07-mcp-servers.md`、[Colab MCP 源码](https://github.com/googlecolab/colab-mcp/tree/b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1/src/colab_mcp)。

## 项目组织与研究约定

```text
docs/
├── plan/
│   ├── 001-baseline.md
│   ├── 002-tpu-finetuning.md
│   ├── 003-colab-mcp.md
│   └── 004-t4-candidate-ranking.md
└── progress/
    ├── 001-baseline.md
    ├── 002-tpu-finetuning.md
    ├── 003-colab-mcp.md
    └── 004-t4-candidate-ranking.md
data/
├── references/      # 精简来源与人工核对结果
└── raw/             # 本地数据，不入 Git
```

已建立 `src/`、`notebooks/baseline/`、`notebooks/tpu-training/` 及本地 `artifacts/`。后续按实际实验需要补充模块和配置，不预先搭建复杂框架。实验产物、日志和权重放 `artifacts/` 等忽略目录，并在进展文档记录位置。

- 同一轮计划与进展使用**相同编号和主题名**。计划可覆盖几轮紧密相关实验。
- 计划写清背景、目标与假设、实验方案、代码改动、实现步骤和前序结论。
- 进展记录真实修改、产物、问题和实际结果；代码状态与研究效果分开。
- **不写测试文件，不引入自动化测试或 CI 门控；所有验证手动发起。** 研究指标仍需认真计算、记录和解释。
- 实验记录至少包含数据划分、权重与数据版本、配置/seed、设备、耗时、MRR@25、Top-1/Top-25、候选覆盖率、notebook 版本及日志位置。

协作约定见 [AGENTS.md](AGENTS.md)。截至提交 `5e22537`，已完成训练数据实测、初步 split、B0/B2 本地运行和指纹权重下载检查；这些指标尚未接入官方 RDKit 规范化，B2 的 merged 路径也是占位比较，不能当作正式泛化结果。尚无真实模型前向、微调结果或排行榜分数，详见 [001 进展](docs/progress/001-baseline.md)。

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

本机版本：**Kaggle CLI 2.2.4**；已通过现有认证完成文件列表查询、三个比赛文件下载及 TPU 短运行 v1 push。命令以本机 `--help` 为准，官方仓库 main 可能比安装版本更新。以下命令从项目根目录执行；`YOUR_USERNAME`、版本号和 notebook 路径需替换。

### 查看、下载和本地读取

```bash
kaggle --version
kaggle competitions files -c enveda-CASMI26-molecule-id-mass-spectra --format json
kaggle competitions download enveda-CASMI26-molecule-id-mass-spectra \
  -f sample_submission.csv -p data/raw
kaggle competitions download enveda-CASMI26-molecule-id-mass-spectra \
  -f test.parquet -p data/raw

# 当前已下载完整；其他环境需要时执行，文件约 3.03 GB
kaggle competitions download enveda-CASMI26-molecule-id-mass-spectra \
  -f train.parquet -p data/raw

kaggle kernels list --competition enveda-CASMI26-molecule-id-mass-spectra
kaggle kernels pull inversion/casmi-denovo-tutorial-notebook \
  -p notebooks/reference -m
```

001 使用的轻量 `.venv` 为 Python 3.9.6、PyArrow 21.0.0，承担人工数据读取与 B0/B2 本地计算，未安装 torch；这也是两份 FPNet 权重尚未完成真实加载/前向的原因。可复现基础读取环境：

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

也可以在新建的 notebook 目录执行 `kaggle kernels init -p <目录>` 生成模板再修改。当前 `notebooks/baseline/` 已有入口和 metadata；上面是说明示例，不应覆盖现有配置。基线入口此前仅产出格式占位结果，真实 B0/B2 是通过本地手动脚本运行的，不能因文件存在就视为正式提交入口已就绪。

官方 [metadata 文档](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)明确 `NvidiaTeslaT4` 表示 **T4 ×2**。本机 `--accelerator` 将覆盖 `machine_shape`。上面的硬件参数已核对文档与 CLI 源码，但本轮没有启动 GPU 分配。

Kaggle TPU 候补路线使用独立入口，metadata 设置 `"enable_gpu": false`、`"enable_tpu": true`、`"machine_shape": "TpuV5E8"`，保留私有 notebook 和所需数据来源。现有入口只做环境/设备/挂载及单步占位，尚无训练循环；v1 已通过下述命令 push：

```bash
# 仅在需要新版本时执行；重复 push 会再创建运行并可能消耗 TPU 配额
kaggle kernels push -p notebooks/tpu-training --accelerator TpuV5E8
```

已有作业为 [`ccjjllt/enveda-tpu-finetuning`](https://www.kaggle.com/code/ccjjllt/enveda-tpu-finetuning) v1，2026-10-01 最后记录为 `QUEUED`、日志为空，未确认设备分配；当前不因排队重复 push。未来训练产物保存后作为最终 CPU/GPU 推理 notebook 的输入；不能把 TPU 训练 notebook 直接当作本比赛的最终提交入口。

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

当前执行 [003 Colab 接入计划](docs/plan/003-colab-mcp.md)，先核实 Kaggle 已关联账号能否复用现有工具，再决定通用 Colab MCP 的配置；实际结果见 [003 进展](docs/progress/003-colab-mcp.md)。本轮范围是连接能力核实与配置，不是训练。

后续继承 [001 基线计划](docs/plan/001-baseline.md) 与 [001 进展](docs/progress/001-baseline.md) 中的数据、划分和 B0/B2 记录，补齐官方评分规范化与真实权重前向；按 [002 TPU 微调计划](docs/plan/002-tpu-finetuning.md) 先做 Colab T4 兼容性核对，再推进 v5e-1 适配。平台变更、已 push 的 Kaggle 作业和未验证项见 [002 进展](docs/progress/002-tpu-finetuning.md)。

可复用参考：[官方 de novo 教程](https://www.kaggle.com/code/inversion/casmi-denovo-tutorial-notebook)、[公开 Analog Propagation 基线](https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline)。公开运行、代码注释或作者分数不算本项目实测结果。使用前记录版本，并分别确认代码、模型权重及外部候选库的许可和来源。
