# 比赛规则、数据与远端工作流

更新日期 2026-11-25。比赛规则、数据边界与 Kaggle 操作如下，历史清理见 [archive/research-archive.md](../archive/research-archive.md)。

## 比赛与数据要点

- **任务与指标**：为每分子输出最多 25 个有序 SMILES（分号分隔）；**MRR@25**，经 RDKit `2026.03.3` 去立体与互变异构归一化后按 `InChIKey14` + 重原子组成匹配。
- **运行约束**：纯 code 赛（`is_kernels_submissions_only=true`），提交已 `COMPLETE` 的断网 Notebook 版本。比赛给 CPU 和 GPU 版本各 9 小时上限。当前已验证的 0.506 管线采用 T4×2。每日 5 次提交（北京 08:00 重置），最多选 2 个最终提交。合队截止 2026-12-07 23:59 UTC，最终截止 2026-12-14 23:59 UTC。[比赛规则](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/)
- **外部数据边界**：允许 CC-BY-NC 及更宽松开源资源与公开预训练模型（MIST/MassSpecGym、DreaMS、GLACIER/ICEBERG、CFM-ID 4、ChEBI/LIPID MAPS、COCONUT、Enveda-180、GNPS-NIST14-MATCHES）；**严禁 NIST 本体及其衍生权重**，禁止无生成器的纯中间表。
- **本地数据（`data/raw/`，不入 Git）**：`train.parquet`（2.8 GB，2,539,608 行 / 275,810 结构）、`test.parquet`（4.6 MB，1,213 谱 / 400 分子，实为训练集 1:1 占位副本）、`sample_submission.csv`（43 KB）。隐藏测试约 400 分子 / 1,500 谱 / 10 类 adduct。

## 算力与操作流程

- 正式推理使用 Kaggle 断网 GPU。训练优先使用 `jiyujia` 的 Kaggle 配额，按该账号的即时余额安排。Colab 代理、账号、Drive 和断点操作见 [项目协作约定](../AGENTS.md)。
- **提交流程**：`COMMIT_MODE='smoke'`、断网、私有、T4×2 → `ops/063/chain.sh` 推送 → `watch.sh` 等 `COMPLETE` 后拉 `submission.csv` 校验 400 行/ID/1–25 合法候选 → `kaggle competitions submit -c enveda-CASMI26-molecule-id-mass-spectra -k <slug> -v <version> -f submission.csv -m "..."`。
- **关键坑点**：
  - `COMMIT_MODE='fast'` 会输出占位 SMILES；只允许用 `smoke` 产出通过校验后再 `-k/-v` 提交。
  - 历史同管线的隐藏重跑约 4.5–5.5 小时且消耗 GPU 配额。隐藏评测使用已提交的 Notebook 版本从头运行，硬件随版本配置。[Kaggle 隐藏重跑说明](https://www.kaggle.com/code-competition-debugging)
  - 凭据不写入仓库/日志；CLI 默认账号可能非 `ccjjllt`，执行前在进程内切换认证；MCP 可直接用于只读查询与配额/提交核对。
