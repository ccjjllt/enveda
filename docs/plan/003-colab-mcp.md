# 003 — 通用 Colab MCP 接入与训练平台切换

对应进展：[003-colab-mcp](../progress/003-colab-mcp.md)。创建：2026-10-01。

## 背景与前序继承

项目当前提交 `5e22537`。001 已有训练数据实测、两类初步 split、B0/B2 占位评分口径及两份 FPNet 权重；RDKit 官方规范化、跨行重复谱处理、window_50da 对照、仪器归一化、真实模型前向尚有缺口。不得将本地占位指标提升为正式泛化结论。

002 的 Kaggle TPU v1 已 push，但最后记录 QUEUED、日志空。用户明确转向 Colab T4 或 v5e-1，并希望 Codex、OpenCode、Grok CLI 可使用同一个通用 MCP 实现，在多个比赛中复用。当前轮次只做接入与说明，不开展训练，也不自动取消既有 Kaggle 作业。

## 目标与假设

- 采用 Google 官方 `googlecolab/colab-mcp`，不自行开发新的训练服务。
- 安装一次官方实现，在三个客户端分别配置用户级 stdio MCP；不用每个比赛复制配置或密钥。
- 区分 Kaggle-Colab 账号关联、Kaggle GPU 时长奖励、Colab 编辑器接 Kaggle 后端，以及 Colab 自己的托管 runtime。
- 服务安装、客户端配置、初始握手、浏览器连接、动态工具刷新、设备分配和模型训练分别记录状态。
- 假设三个本地客户端能通过其 MCP 实现使用该服务；Google 额外要求动态工具目录更新，因此逐项核实，不把通用 MCP 支持当作完整兼容性证明。

## 接入设计

链路：`本地客户端 → stdio colab-mcp → 本地随机端口 WebSocket → Colab 浏览器页面 → notebook/runtime`。

官方初始工具是 `open_colab_browser_connection`；浏览器接入后，真实工具由 Colab 前端提供，服务向客户端发送 `notifications/tools/list_changed`。连接工具最多等待 60 秒，客户端调用超时给足此窗口。

每个实例只接受一条浏览器连接；三个客户端可以引用相同安装路径，但各自启动独立进程。并行使用不同 notebook 的可行性由实现推断，未作为本轮实测结果。同一 notebook 由一个客户端负责，避免互相改动和中断执行。

## 计划修改

- `README.md`：通用客户端配置、首次连接、跨比赛复用、Kaggle/Colab 集成边界。
- `AGENTS.md`、002 的计划与进展：按用户要求优先推进 Colab，保留 Kaggle 候补/最终推理。
- `docs/progress/003-colab-mcp.md`：记录安装来源、全局配置路径、手动核对与未完成连接。
- 用户级客户端配置：仅新增 `colab-mcp`；保留其他 MCP、模型设置、凭据及注释，私有备份放原目录。
- 独立用户级 uv 工具环境：本地 Python ≥3.13，不替换项目 `.venv` 或训练依赖。

不新增测试文件、训练框架、代理转发层、自动化门控或持续后台服务。

## 实施步骤

1. 读取官方 README/源码与三客户端帮助，确认协议、版本、配置路径和动态工具要求。
2. 核实 Kaggle 已关联 Colab 的功能：若仅要使用已到账的 Kaggle GPU 奖励，现有 Kaggle CLI/MCP 即可；若要用 Colab 自己的 T4/v5e-1，继续本计划。
3. 固定官方源码提交，以独立 uv 工具安装；确认入口可运行。
4. 备份并合并三份用户级配置，核对仅新增 Colab 项；不输出原配置中的 secrets。
5. 手动执行初始 MCP 握手和有针对性的客户端诊断，仅列工具，不申请云硬件。
6. 用户重载客户端后调用连接工具；在默认浏览器完成登录和连接确认。核对真实工具目录刷新，这是仍需实际交互的一步。
7. 后续先连接私有 notebook，手动核对读写与一小段环境信息；选择 Colab 托管 T4，检查已有 FPNet 的真实权重加载，再推进 v5e-1 适配。此项按 002 的实验范围开展。
8. 按比赛建立独立输入/产物目录，保存代码版本、split manifest、权重来源、checkpoint 和日志。不要将该桥接服务当作浏览器关闭后也保证运行的作业调度系统。

## 账号关联的已核实边界

Kaggle 文档中的 `Link to Colab` 可以用于 Pro/Pro+ 订阅奖励，增加的是 Kaggle GPU 小时，不使用 Colab compute units；活动和额度以账户实际结果为准。`Open in Colab` 则是把 Colab 编辑器连接到 Kaggle Jupyter Server，仍使用 Kaggle 设备。这两者都不意味着 `kaggle kernels push` 已能指定 Colab 托管后端。

本机 Kaggle CLI 2.2.4 和当前 MCP schema 未发现 Colab runtime/provider/CU 选择入口。该核实范围不推断未来产品永远不会支持。

## 来源

- [Google 官方 Colab MCP](https://github.com/googlecolab/colab-mcp)
- [固定源码提交](https://github.com/googlecolab/colab-mcp/tree/b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1)
- [Kaggle notebook 集成说明](https://www.kaggle.com/docs/notebooks)
- [Kaggle Jupyter Server 官方公告](https://www.kaggle.com/discussions/product-announcements/567740)
- [Codex MCP](https://developers.openai.com/codex/mcp)
- [OpenCode MCP](https://opencode.ai/docs/mcp-servers/)
- Grok：本机随安装附带的 `~/.grok/docs/user-guide/07-mcp-servers.md`
