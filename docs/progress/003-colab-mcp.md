# 003 — 通用 Colab MCP 接入：实际进展

对应计划：[003-colab-mcp](../plan/003-colab-mcp.md)。更新：2026-10-01。

**状态：官方服务已安装，三个客户端全局配置已写入；当前 Codex 会话调用浏览器连接工具已返回成功。当前工具目录仍只有连接工具，尚不能调用 notebook 执行工具。004 已另通过 UI 确认 T4 设备，未开展模型训练。**

## 实际完成

- 读取 Google 官方 README 与源码，确认 stdio、本地浏览器桥接、随机端口、每实例单连接以及 `notifications/tools/list_changed` 要求。
- 安装 `colab-mcp 1.0.1`，源码固定为 `b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1`（Apache-2.0）。独立 uv 工具环境使用 CPython 3.13.14、FastMCP 2.14.5、MCP SDK 1.30.0；项目 `.venv` 未修改。
- 官方 Git/常规 GitHub archive 访问超时后，终止本轮对应下载进程，改从 GitHub 官方 codeload 下载固定提交归档，再用 `uv tool install --python ... /private/tmp/colab-mcp-b9ab3899.tar.gz` 安装运行依赖；未安装项目 dev/test 依赖。
- 入口：`/Users/zhaohuaxiang/.local/bin/colab-mcp`。新配置引用已安装的绝对路径，无需每次启动从 Git 下载。
- 三份配置在原文件基础上仅合并 `colab-mcp`；新配置无 token。写入前逐份校验语义只新增这一项，保留其他设置。

| 客户端 | 本机版本 | 修改的用户级配置 |
| --- | --- | --- |
| Codex CLI | 0.159.3 | `~/.codex/config.toml` |
| OpenCode | 1.18.34 | `~/.config/opencode/opencode.jsonc` |
| Grok CLI | 1.0.44 / 5b807183dd79 | `~/.grok/config.toml` |

私有备份在各文件原目录，文件名追加 `.bak.colab-20261001-d45176a4`，权限 0600，不入仓库。备份可能包含原有凭据，不要复制进项目或分享。恢复时只移除新增项或人工合并，不要覆盖配置文件之后的新改动。

## 人工核对结果

| 操作 | 实际结果 | 不能据此推断的内容 |
| --- | --- | --- |
| `colab-mcp --help` | 正常返回；当前参数只有日志与默认 proxy 开关 | 没有静态 `--tpu` / `--notebook-url` 参数 |
| MCP SDK 手动 `initialize` + `list_tools` | 服务 `ColabMCP`，协议 2025-11-25，初始工具 `open_colab_browser_connection` | 未调用连接工具，无 Colab 前端/运行时连接 |
| `codex mcp get colab-mcp` | 成功读出 stdio、入口、startup 30s / tool 120s | 未在 Codex 聊天验证动态工具更新 |
| `grok mcp doctor colab-mcp --json` | command/server/handshake 均通过，发现 1 个工具、healthy=true | 不证明浏览器连接后会刷新完整工具目录 |
| OpenCode 配置与实现阅读 | JSON 配置可解析；本机实现收到 tools/list_changed 后会重新取工具 | 尚未在 OpenCode 会话连接 Colab |

Grok 诊断结束时另有子进程组清理权限 warning；基础握手成功。直接 SDK 核对正常退出；随后只读检查发现一个父进程为 Codex 的短暂 Colab MCP 进程，再次按该 PID 复查时已消失，未主动杀进程。当前未验证长会话重连及多客户端并行。

## Kaggle 已关联 Colab 的核实结论

- 已生效的 Colab Pro 奖励增加的是 Kaggle GPU 小时，用现有 Kaggle MCP/CLI 就能继续使用这些 Kaggle 资源；不必为奖励安装 Colab MCP。
- `Open in Colab` / Kaggle Jupyter Server 功能的后端仍是 Kaggle，不能绕过 Kaggle TPU 排队。
- 使用 Colab 自己托管的 T4/v5e-1，需要 Colab 侧连接；本次采用 Google 官方服务，没有自行实现新 MCP。
- 本轮只读 Kaggle quota 快照：GPU 总计 108000 秒、已用约 105165 秒，剩约 47 分钟；TPU 总计 72000 秒、已用 0。没有体现额外 GPU 奖励；不据此推断用户未绑定或订阅无效。
- 没有更改 Kaggle/Colab 账号绑定，也没有取消或重启既有 Kaggle TPU v1。上游 QUEUED 为此前记录，本轮不宣称已重新确认该作业状态。

## 文档和代码状态

已更新 README、AGENTS、002 配对文档，并创建本轮 003 配对文档及精简安装记录。保留 `5e22537` 的代码与训练数据/权重，不改训练实现、不创建测试文件。未替用户提交 Git commit。

001 的占位 MRR、shipped inchikey14 分组、跨行重复谱及权重训练暴露问题仍需按研究计划处理。MCP 接入成功不修复这些方法缺口，也没有产生任何新训练结果。

## 下一步需要的实际交互

1. 用户重载一个客户端，调用 `open_colab_browser_connection`，自行完成 Google 登录/页面连接确认。
2. 确认调用返回连接成功，并能发现动态 notebook 工具；若 Codex/Grok 只保留初始工具，排查刷新支持，不声称已经完整兼容。
3. 保存私有比赛 notebook，确认实际 runtime 属于 Colab；后续按 002 先做 T4 权重前向，再研究 v5e-1。未启动这些步骤。

## 2026-10-01：连接与剩余额度查询

- 当前聊天的工具目录仍没有 Colab 工具，不能据全局配置和初始握手声称本聊天已连接账号或 notebook。
- 通过浏览器只读看到既有 `Untitled0.ipynb` 页面，按钮为“连接 v5e-1 TPU / 连接到新的运行时”；该观察时尚未连接此运行时，没有点击申请设备。
- 尝试查看资源选项时，电脑使用工具两次提示用户已切换 Chrome 页面，因此停止继续操作，未读取到账户 CU 余额、订阅或剩余时间。没有将 Kaggle 配额当作 Colab 余额。
- 官方 Colab FAQ 说明免费限额动态变化且不公开；付费 CU 可用于按实际消耗率估算时间，但不是固定可保证的剩余运行小时。当前无本账号余额实测值。

## 2026-10-01：当前会话连接成功，动态工具尚未暴露

- 本轮当前工具目录已出现 `mcp__colab_mcp__open_colab_browser_connection`，说明 MCP 已加载进聊天。
- 实际调用该工具，返回 `structuredContent.result=true`、`isError=false`，确认浏览器代理连接成功。
- 随后读取工具元信息与可调用绑定，两处仍只有 `open_colab_browser_connection`；没有可调用的 notebook 读取/编辑/执行工具。不能将连接成功写成已能训练或完成完整兼容性验证。
- 当前未执行任何 Colab 单元格、未读取设备或 CU 余额、未启动训练。下一步是客户端工具目录刷新/排查，之后再手动执行最小环境信息单元确认执行链路。

## 2026-10-01：OpenCode / Grok 操作能力判断

- OpenCode 1.18.34 已配置；本机实现及官方 MCP 源码均有工具变更通知处理和重新取工具的逻辑，具备官方 Colab MCP 要求的关键客户端能力，可优先做浏览器连接后的操作核对。但尚未在本机 OpenCode 实际读取或执行 Colab notebook。
- Grok CLI 1.0.44 已配置，前述 doctor 初始握手成功；尚无足够证据确认其在 Colab 连接后刷新出完整工具目录，更没有 notebook 操作实测。不能仅凭 healthy=true 宣称可训练。
- 两客户端各自启动本地 MCP 实例；Codex 当前连接成功不会自动把该连接共享给 OpenCode/Grok。首次核对只连接并读取 notebook 信息，执行能力另以手动小单元确认，不直接启动训练。

## 2026-10-01：004 研究执行补充

用户已授权进入 T4 训练与手动研究。004 中再次检查，当前 Codex 仍只有连接工具。曾通过浏览器将 scratchpad 保存到 Drive 副本并取得真实 T4 环境输出；新副本连接原 MCP 时提示已有活跃连接，不能将 UI 运行写成 MCP 执行成功。随后用户要求避免慢速 Computer Use，本轮已停止界面操作。模型尚未前向或训练，具体记录与下一步实现见 004 配对文档。
