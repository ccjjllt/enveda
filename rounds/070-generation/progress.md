# 070 进展

## 2026-11-17 070a commit

- `jiyujia/enveda-070a-full25` v1 COMPLETE 但 overlay 未生效：`neural_scores.py:32 assert mid in query`——evidence 含 400 分子而 commit run 的 query 谱只覆盖 12 分子，首个缺谱分子即断言失败，无输出、submission 原样。根因是 scorer 要求 evidence 分子必须在 query 中（083a 因 evidence 源自 query 未触发）。
- 修复：evidence 只含 query 覆盖的分子（commit 12 个，hidden rerun 全 400），其余分子保持原样；v2 已推送。审计待拉取：若可见改动只有 ~10 列表或 top-1 零变化，按 kill 规则判死不提交。
- v2 COMPLETE：`neural_full25_audit.json`：**reranked 12，top1_changed 0**（8 个因 `lib_max>=0.90` 冻结 leader）；diff 确认 12 列表变化、top-1 零变化。**070a 按 kill 规则判死，不提交。** 神经分与底座顺序在头部几乎共线，换帅门控全军覆没。
- 070b（ICE+对齐联合）已实现并跑完：`jiyujia/enveda-070b-icealign` v1 COMPLETE；`ice_align_full25_audit.json`：**ice_scored 400，align_scored 400，reranked 12，top1_changed 0**（4 个因 lib 冻结）。双新证据全覆盖 400 分子仍只改 12 列表、零换帅，与 070a/072/073/076 同模式。**070b 同样判死，不提交。**
- 结论：集合内重排路线（5 个独立打分函数：DreaMS kNN / timsTOF ICE / 对齐头 / 管线神经 / ICE+对齐联合）全部零换帅。底座头部顺序已被现有证据锁定，`w_mid` 与 RRF 融合没有可挖增量。

## 2026-11-17 11:00-11:35 P0a：MIST(MassSpecGym)+MolForge 链验证

目标：在 train timsTOF `[M+H]+` 150 个结构上跑通 MIST 指纹预测 → MolForge 解码 top-10，测 top-1/top-10 命中率，作为 P0 尾槽生成的阈值与期望值依据。kernel `jiyujia/enveda-p0a-mist-chain`（T4，联网，只读 train.parquet）。

迭代记录（每版均有日志 `[p0a] STAGE`）：

| 版本 | 结果 | 根因/修复 |
| --- | --- | --- |
| v1 | subform FAIL | 子进程缺 `PYTHONPATH` → 修 |
| v2 | subform FAIL | MGF 缺 `ID=`，`KeyError: 'ID'` → MGF 补 `ID=<name>` |
| v3 | subform FAIL | `mist.utils.chunked_parallel` 缺失：`mmf/src/mist` 是 regular 包（有 `__init__.py`）遮蔽 `main_v2` namespace 包 |
| v4 | 诊断确认 | `mist diag /kaggle/working/mmf/src/mist/__init__.py` 证实遮蔽 |
| v5 | subform FAIL | 只留 main_v2 后 `ModuleNotFoundError: pytorch_lightning.loggers.base`（Kaggle PL 无旧 API） |
| v6 | subform FAIL | in-process stub 失败：`tune_utils` 硬 import `ray` |
| v7 | subform FAIL | 子进程隔离 + 剔除 `tune_utils` 仍因子进程无 PL stub 而失败 |
| v8 | **subform OK `n_json=150, rc=0`** | 拷包到 `/kaggle/working/mistfix` + 空 `__init__.py` + 剔除 `tune_utils`，runner 内先 stub `LightningLoggerBase`/`rank_zero_experiment` 再 `runpy.run_path` |

v8 已进入 `run` 阶段（MIST 编码 + MolForge beam=10 解码 150 分子，batch-size=1，阈值 t=0.172）。结果待拉取，判读口径：若 top-10 命中率明显高于 0（文献修正值 top-1 10.73%），P0 尾槽生成值得做；若为 0 或与 FRIGID 一样全灭，P0 终止。

## 2026-11-17 11:29 公开前沿快照（CLI `--download` 实查）

- 公榜 5429 队；≥0.492 **538** 队、≥0.494 **83** 队、≥0.495 **63** 队。
- Rogers Johnson 仍 **0.492 / 第 125 名**，`LastSubmissionDate 2026-11-16 00:27`——q1–q5（nbr10/nbr25/wexp2/posflat/placebo）跑完**未提交**，无新信号。
- 榜首仍 0.549。公开侧仍无已核实 >0.492 配方。
