# 比赛规则、数据与远端工作流

整理日期：2026-11-16（北京时间）。本页只保留当前仍适用的比赛事实与 Kaggle 操作；001–064 的历史运行记录已删除，结论见[已放弃研究总览(archive/research-archive.md)。

## 比赛要点

| 项目 | 已核实内容 |
| --- | --- |
| 任务 | 为每个 `molecule_id` 综合多张谱，输出最多 25 个有序 SMILES 候选 |
| 指标 | **MRR@25**：首个正确候选排名为 r 得 1/r；前 25 未命中为 0；对分子平均 |
| 匹配 | 官方 RDKit **2026.03.3** 规范化互变异构体，再按 InChIKey 前 14 字符匹配 |
| 提交 | `submission.csv`，两列 `molecule_id,smiles`；同行候选用分号分隔 |
| 最终运行 | 只能提交 Kaggle Notebook 版本；关闭互联网；单次上限 9 小时 |
| 外部资源 | 允许公开、免费可得的数据与预训练模型；遵守来源许可与比赛规则 |
| 奖金/提交 | 总奖金 50,000 美元；每天最多 5 次提交；最多选择 2 次最终提交 |
| 截止 | 参赛/合队 2026-12-07 23:59 UTC（北京 12-08 07:59）；最终 2026-12-14 23:59 UTC |

来源：[概览与评测](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview)、[官方评分代码](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank)、[规则](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules)、[数据](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data)。

主办方禁止 NIST 及其训练模型；长期公开的 **GNPS-NIST14-MATCHES** 是明确例外，可用于库匹配和训练，不可外推到其他 NIST 衍生资产。GLACIER/ICEBERG、DreaMS、ChEBI/LIPID MAPS 已获具体确认；仅继承 `train.parquet` 非商用标记的资产可用。见[主办方说明](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744470#3530300)与[模型/结构库确认](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745841#3533343)。

## 数据

| 文件 | 大小 | 已核对 |
| --- | ---: | --- |
| `train.parquet` | 3,033,286,496 B | 2,539,608 行、21 row groups、18 列，无 `molecule_id` / `spectrum_id` |
| `test.parquet` | 4,848,729 B | 1,213 张谱、400 分子、12 列、7 类 adduct，全部 timsTOF |
| `sample_submission.csv` | 43,619 B | 400 行，每行 25 个占位候选 |

三个文件都在 `data/raw/`，不入 Git。可见 `test.parquet` 是训练集占位副本，只用于检查运行与输出；最终隐藏测试约 400 分子、1,500 谱，覆盖 10 类 adduct。隐藏集同时包含“已有公开参考谱”“库内但无参考谱”“库外 de novo”三类，比例未公开；不能从 public 总分推断类别比例。

训练分子会跨来源重复。未见结构验证必须按规范化结构整组留出，不能随机切分谱行；已见结构的独立谱检索另报。本地数据读取与 schema 记录见 `data/references/local-data-inspection-2026-09-30.json`。

## 算力与配额

- **当前只用 Kaggle 远端 GPU（T4×2）**。本机 M5/32 GB 只做代码、文档与小文件检查，不再承担训练或完整推理。
- 062a 已核实 CPU 路线不可行：冻结管线强制 `nvidia-smi`，Engine 2 两个排序器在 GPU 上现训，CPU 会触发 8.55h 时间守卫；本机也没有隐藏测试，无法产生有效公榜分数。
- 周 GPU 配额按 **session 存活时间** 扣，不按 GPU 繁忙度。CPU 预处理慢会直接烧掉 GPU 配额；GPU session 内的循环必须向量化/批量化，并先在小样本实测吞吐。
- 2026-11-16 记录：本周 GPU 33.76h 已用 / 30h，剩余 **0.00h**，刷新 **2026-11-21 08:00 北京**。TPU 20h 仍在，但本管线需要 GPU，不改走 TPU。
- 每日提交额度 5 次，08:00 北京重置；提交次数与 GPU 配额是两套限制。

## 提交工作流

当前入口是 068a（slug `ccjjllt/enveda-068a-n1c-lam035`），底座 065a。流程固定为：

1. 只改一个已核实的旋钮，`COMMIT_MODE='smoke'`，保持断网、私有、T4×2。
2. `scripts/063/chain.sh` 推送；GPU 会话满额时每 5 分钟重试。
3. `scripts/063/watch.sh` 等 COMPLETE，只拉取 `submission.csv` 和审计 JSON，检查 400 行、ID 集合、候选列表合法性、top-1 与底座差异。
4. 审计通过后 `kaggle competitions submit -k <slug> -v <version> -f submission.csv`；记录 ref、实际 notebook 版本与公榜分数。
5. 代码成功、远端 COMPLETE、公榜提分分别记录，不把前一步当作后一步。

### 常用命令

```bash
# 查看与下载数据
kaggle competitions files -c enveda-CASMI26-molecule-id-mass-spectra --format json
kaggle competitions download enveda-CASMI26-molecule-id-mass-spectra \
  -f test.parquet -p data/raw

# 推送与查看（push 会启动执行并消耗配额）
kaggle kernels push -p notebooks/068a
kaggle kernels status ccjjllt/enveda-068a-n1c-lam035
kaggle kernels logs ccjjllt/enveda-068a-n1c-lam035
kaggle kernels output ccjjllt/enveda-068a-n1c-lam035 \
  -p /workspace/kout_068a --file-pattern '^submission\.csv$'

# 提交已完成的 notebook 版本（代码赛，不能直传 CSV）
kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra \
  -k ccjjllt/enveda-068a-n1c-lam035 -v 1 \
  -f submission.csv -m "068a N-1c LAM=0.35"
kaggle competitions submissions enveda-CASMI26-molecule-id-mass-spectra
```

`kernels push` 只上传 `code_file` 与 metadata，不同步整个仓库。入口必须是自包含 notebook，代码、权重和依赖作为输入挂载；输出写到 `/kaggle/working/submission.csv`。本赛是纯 code 赛（`is_kernels_submissions_only=true`），直接上传本地 CSV 会 400 拒绝；必须提交已 COMPLETE 的 notebook 版本。`kernels push` 成功只表示触发执行，运行完成、产物合法、提交成功、公榜出分是四个独立状态。

### 已踩过的坑

- `COMMIT_MODE='fast'` 的占位输出曾被误交（054，0.459 来自 dummy SMILES 偶然命中）；此后统一 `smoke`，审计后才 `-k/-v`。
- `kernels output` 会限流/挂死；watcher 只拉关心的单文件，失败时退避重试。
- CLI 2.2.4 的 `status/logs/output` 只可靠读取最新运行；历史版本用 MCP 的 `versionLabel`/`versionNumber`。
- 没有 cancel 命令；停作业只能网页点，或等 9 小时上限。`RUNNING` 且日志无新增通常是在烧配额。
- 长循环必须拆 cell，保证每 10 分钟内能看到新输出；单格无输出不要盲等。
- 日志重复投递是聚合现象，以最后一条新时间戳为准。

## Kaggle MCP 与凭据

Kaggle MCP 可读取比赛、数据、讨论、公开 notebook、运行状态、日志和 GPU/TPU 配额。常用工具：`get_competition`、`list_competition_topics`、`search_notebooks`、`get_notebook_info`、`get_notebook_session_status`、`list_notebook_session_output`、`download_notebook_output`、`get_accelerator_quota`。参数外层都是 `request`；日志在 `content[].text` 内再解析 `log` JSON。

凭据只放在用户认证配置中，不写入仓库、文档、notebook 或日志。研究账号是 **ccjjllt**；本机 CLI 默认可能是其他账号，执行前在进程内选择认证，不改全局默认身份。MCP 能连通不代表 shell 网络权限相同。
