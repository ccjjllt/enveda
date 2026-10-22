# 031 — 项目目录整理与冗余清理：实际进展

日期：2026-10-22（北京时间）。本轮完成本地整理，没有启动远端训练、Kaggle运行或比赛提交。

## 接手研究结论

已阅读027–030及相关交接记录。实际最佳仍是v14/v465 public **0.456**（ref57462581，既有kernel v8）；030完整N1联合ICE＋代理GL版本已准备，尚未运行或提交。此前用户选择配额恢复后手动继续；本轮不改变这一安排。

028扩池只有本地构建结果，029a遗漏N1开关是需要保留的失败机制。原始数据、权重、离线指标、评分回包、日志及失败实验都保留；不能把目录整理当作模型有效或分数提高。

## 实际目录与文档修改

- 51个手动研究脚本按轮次分组，其中49个移到 `scripts/NNN/`，两个未标轮次的旧脚本移到 `scripts/legacy/`；新增[脚本导航](../../scripts/README.md)。修正脚本根目录定位、跨轮次导入、动态脚本读取和命令引用。
- 当前 `notebooks/028/`、`029/`、`030/` 保持位置；15个旧目录分到 `notebooks/submissions/` 与 `notebooks/archive/`。全部27份迁移notebook保留原字节及署名，新增[notebook导航](../../notebooks/README.md)。
- 82个本地产物目录分到 `artifacts/runs/`、`shared/`、`submissions/`、`legacy/`。当前源码/查询快照位于 `artifacts/runs/030/`；共享候选池位于 `artifacts/shared/`。
- `src/`公共模块、`data/`、`datasets/`和三个 `.venv*` 环境保留原位置；同步公共模块的本地输入默认路径，补充目录说明。
- README从442行收敛为当前研究入口与导航；旧研究结论完整分流到[研究历史](../research-history.md)，比赛/数据/硬件/Colab/Kaggle操作分流到[远端工作流](../remote-workflow.md)，新增[001–031配对索引](../README.md)。AGENTS修正仍指向006的接手说明。

具体旧→新路径见[迁移表](../workspace-paths.json)。原始上游源码、历史notebook、`data/references/`及artifacts内机器记录保留历史字节和哈希；人类阅读文档中的本地路径已同步。

历史JSON的 `peak_file_ref` 仍是旧路径，因此增加小型读取函数 [workspace_paths.py](../../src/workspace_paths.py)，在相关008/009/013/016脚本读取时映射到新位置。015 direct消融回放仅在内存里规范历史哈希字典的路径键，不改历史记录或内容哈希。原始快照中的旧脚本不批量改写；需要重跑时使用迁移后的研究入口并参考本轮映射。

## 已清理内容

实际删除5,737个缓存或冗余文件，按文件逻辑大小共 **646,415,602 B（616.47 MiB）**；文件系统实际释放空间以系统统计为准。

| 类别 | 内容与核对依据 |
| --- | --- |
| 包管理缓存 | 根 `.cache/`、015 ranker的 `environment-cache/`、020 entropy试点的 `uv-cache/`；约344.2 MiB，可重新下载，现有环境无指向这些目录的符号链接或硬链接 |
| 重复分发包 | 根artifacts的 `f1-bundle`、`f4a-bundle`、`g1-validation`、`mist-single`、`r0-fixcheck` 五个tar.gz，共269.9 MiB；逐成员SHA256分别核对81/81、10/10、45/45、6/6、43/43，无缺失或独有内容，同名解包目录保留在 `artifacts/legacy/` |
| 系统与字节码缓存 | 4个 `.DS_Store` 与21个 `__pycache__/`，可按需重新生成；三个虚拟环境内容未清理 |

压缩包原SHA256与删除原因保留在迁移表。今后需要传给Colab时可从解包目录重新生成，例如在项目根手动运行：

```bash
tar -czf /private/tmp/f1-bundle.tar.gz -C artifacts/legacy/f1-bundle .
```

015的离线 `environment-wheels/` 与环境来源记录保留。`datasets/train-fp-pool/`和共享候选池即使载荷相同，也保留各自上传/研究入口。没有删除原始Parquet、模型权重、研究缓存、结果或日志。

## 人工核对记录与边界

- 迁移前后逐文件SHA256对照：**2,289个保留文件、6,289,875,922 B全部一致**，包含上游源码、权重、研究产物与迁移notebook。临时核对清单在 `/private/tmp/enveda031-inventory.json`，可携带摘要在迁移表。
- 从项目根手动运行 `python3 -B scripts/030/build_030_n1_challenger.py`，生成的notebook、metadata、source-manifest与整理前**3文件逐字节一致**。notebook SHA256仍是 `d64c97cf657e5916fb7ece3ac4b7a2e69d8cf4e51f8018dbcf558d5107e5a5b3`；私有、断网配置及 `code_file` 正常，32个Python cell语法解析成功。
- 手动内存解析73个Python源文件；使用现有 `.venv-015`（Python3.11.16）实际读取59份历史谱数组，定位成功且保存的SHA256相符；015消融的5个历史输入哈希全部一致。未重训或重算模型指标。
- 手动核对338个Markdown本地链接、226个原Git追踪文件的迁移后位置及 `git diff --check`，均通过；没有创建测试文件、测试框架或CI，也没有新增训练随机种子、设备或得分。
- 已知历史问题：029 metadata的 `code_file` 在整理前就指向不存在的旧文件名；该版本本来就因N1缺陷禁止直接推送，本轮保留原始配置供030构建器继承。当前030的输出文件名和metadata已核对，不受此问题影响。

后续研究从[030进展](030-n1-challenger-and-score-strategy.md)继续，完整模拟器留出验证与028独立召回对照仍待实际执行。
