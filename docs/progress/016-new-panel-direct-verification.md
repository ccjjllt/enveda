# 016 — 新面板直接检索验证：实际进展（一）

日期：2026-10-17（北京时间）。对应[016计划](../plan/016-new-panel-direct-verification.md)。接手基点 `fb8ed10`。本轮完成 015 复核收尾与新面板冻结；GPU 候选生成尚未启动，无新提交。

## 015 收尾核对（CPU）

```bash
.venv-015/bin/python scripts/run_015_candidate_ranker.py --replay
.venv-015/bin/python scripts/analyze_015_direct_ablation.py --replay
```

- 五方案 OOF 与 direct-only 消融重放均为零误差（输入/模型 hash 不变）。
- 补 direct-only 全量拟合：新增 `scripts/fit_015_direct_full.py`（复用 015 同输入/权重/参数，不改原脚本与产物），42580 行、零警告，模型 `artifacts/015-analog/direct-only-ablation/models/lr_direct_only_experimental_full.joblib`（SHA `e564918e…06d069`），记录 `full_fit.json`。重载预测差 0。仅供新面板预测，未部署、无 OOF 成绩。

## 新面板冻结（CPU）

代码 `scripts/build_016_panel.py`，手动命令：

```bash
/opt/homebrew/bin/python3.11 scripts/build_016_panel.py --probe
/opt/homebrew/bin/python3.11 scripts/build_016_panel.py
```

- 排除：旧 816 raw 键 → 005 映射后 800 canonical（53 无映射回退，多为 R3 MassBank 键）；旧谱 hash 015 的 2561 + 补算 R3-50/Heldout-250/小面板 334，共约 2895。
- train 274186 canonical（与 005 一致）→ 新鲜 273436；跨来源池 2285（timsTOF 且有外部参考），enveda-only 180156，external-only 90995。
- 冻结 400：跨来源 200（含负例 50、多聚体 query 20，按 `SHA256("20261017|016|…")` 排序取用，无短缺）+ 自然缺库 200（enveda-only 单谱 100 + 多谱 100）。
- 核对：400 query 峰 hash 与旧集合零碰撞；19 个新面板 canonical 出现在旧 015 top50 候选键（20545）中——候选出现过不等于作为 query 用过，报告备查，不否决。
- 全量 8.1 秒。产物 `artifacts/016-panel/{exclusion_list,panel_manifest,pool_stats}.json`（ignored）。新面板只称“本项目此前未用”。

## GPU 安排的澄清

- 本地只有 ccjjllt 的两个 token，无 jiyujia 凭据；用户已定改用 ccjjllt GPU（剩约 27h），无私有输入可见性问题。
- GPU 阶段待办（未启动）：可追溯参考索引 + 40 例计时（CPU），新面板 FP/ICE 候选生成单 session 打包（ccjjllt push），冻结四模型预测与分组 MRR，再定提交。

## Kaggle 只读

本轮未做新查询；有效提交仍 v9r2/v10 public 0.323。下一步先完成参考索引与计时，再议 GPU 任务。
