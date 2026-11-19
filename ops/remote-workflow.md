# 比赛规则、数据与远端工作流

更新日期：2026-11-19。保留当前适用的比赛规则、数据边界与 Kaggle 操作；历史清理见 [archive/research-archive.md](../archive/research-archive.md)。

## 比赛与数据要点

- **任务与指标**：为每分子输出最多 25 个有序 SMILES（分号分隔）；**MRR@25**，经 RDKit `2026.03.3` 去立体与互变异构归一化后按 `InChIKey14` + 重原子组成匹配。
- **运行约束**：纯 code 赛（`is_kernels_submissions_only=true`），必须提交已 `COMPLETE` 的断网 Notebook 版本（T4×2，≤9h），每日 5 次提交（北京 08:00 重置），最多选 2 个最终提交。截止：合队 2026-12-07 23:59 UTC，最终 2026-12-14 23:59 UTC。
- **外部数据边界**：允许 CC-BY-NC 及更宽松开源资源与公开预训练模型（MIST/MassSpecGym、DreaMS、GLACIER/ICEBERG、CFM-ID 4、ChEBI/LIPID MAPS、COCONUT、Enveda-180、GNPS-NIST14-MATCHES）；**严禁 NIST 本体及其衍生权重**，禁止无生成器的纯中间表。
- **本地数据（`data/raw/`，不入 Git）**：`train.parquet`（2.8 GB，2,539,608 行 / 275,810 结构）、`test.parquet`（4.6 MB，1,213 谱 / 400 分子，实为训练集 1:1 占位副本）、`sample_submission.csv`（43 KB）。隐藏测试约 400 分子 / 1,500 谱 / 10 类 adduct。

## 算力与操作流程

- **算力现状**：只用 Kaggle 远端 GPU（T4×2）；周配额按 session 存活时间计，当前账号周 GPU 刷新时间为 **2026-11-21T00:00:00Z**（北京 11-21 08:00）。本机 M5/32 GB 只做代码、文档与轻量审计。
- **提交流程**：`COMMIT_MODE='smoke'`、断网、私有、T4×2 → `ops/063/chain.sh` 推送 → `watch.sh` 等 `COMPLETE` 后拉 `submission.csv` 校验 400 行/ID/1–25 合法候选 → `kaggle competitions submit -c enveda-CASMI26-molecule-id-mass-spectra -k <slug> -v <version> -f submission.csv -m "..."`。
- **关键坑点**：
  - `COMMIT_MODE='fast'` 会输出占位 SMILES；只允许用 `smoke` 产出通过校验后再 `-k/-v` 提交。
  - 隐藏重跑约 4.5–5.5 小时且消耗 GPU 配额；无 GPU 配额时提交可能长期 `PENDING`。
  - 凭据不写入仓库/日志；CLI 默认账号可能非 `ccjjllt`，执行前在进程内切换认证；MCP 可直接用于只读查询与配额/提交核对。
