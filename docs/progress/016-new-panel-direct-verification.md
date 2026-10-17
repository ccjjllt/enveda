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
- GPU 阶段待办：新面板 FP/ICE 候选生成单 session 打包（ccjjllt push），冻结四模型预测与分组 MRR，再定提交。

## 可追溯参考索引 + 40 例计时（CPU，已完成）

代码 `scripts/build_016_reference_index.py`，手动命令：

```bash
/opt/homebrew/bin/python3.11 scripts/build_016_reference_index.py --phase index --probe
/opt/homebrew/bin/python3.11 scripts/build_016_reference_index.py --phase index
/opt/homebrew/bin/python3.11 scripts/build_016_reference_index.py --phase timing
```

- 索引：train 2539608 行分批读（batch 16384），005 组键 + 015 冻结清洗/中性质量口径；保留 **2511008** 谱（98.9%），跳过 26386 清洗空谱与未知/稀有加合物（`[M+Br]-`、`[M+Li]+`、Ca 系列等，从不猜值）。`index.parquet` 219MB（一行一谱：组键、train 行号、中性质量、前体、adduct/mode/instrument/来源、峰 hash、清洗峰数、信号保留、分片/偏移）+ 156 个 npz 清洗峰分片 558MB（延迟 mmap）。全量 78.7 秒。
- 构建内存峰值 RSS 4.48GiB（<8GiB 硬限，高于 2GiB 目标；主因逐批 parquet 物化，未把 253 万谱展开为常驻 Python 对象）。
- 计时 40 例（跨来源 20 + 自然缺库 20，`SHA256("20261017|016-timing40|…")` 冻结，含负/多聚体）：±200Da 同极性最近 2000 检索 + direct-only 联合打分，p50 0.452s、p95 0.483s、max 1.624s（<2s ✓），40 query 行全部在索引中（缺失 0）；平均 quality>0 率 0.389；每 query eligible 约 220 万（质量窗本身极宽，最近 2000 截断率 ~99.9%，按质量距离截断，记录备查）；RSS 3.88GiB（索引常驻 + 全分片缓存，主因是全量 train 峰表一次误读已修复为分片读取）。
- 负结果：分片 LRU-32 缓存反而颠簸（141 秒/10.5GB），改回无界缓存（156 分片共 558MB）；LRU 不作为生产配置。
- 产物 `artifacts/016-index/{index.parquet,shards/,index_build.json,timing40.json}`（ignored）。下一步 GPU 任务可直接挂载该索引，T4 只跑神经计算。

## Colab 验证输入打包（本地 CPU，已完成）

代码 `scripts/build_016_colab_inputs.py`，v10 冻结口径（cells 4/5/fwd_batch 从提交 notebook JSON 原样 exec，SHA 记录；两处一词设备补丁 `.cuda→cpu` 已声明；chunk G=20 不影响逐分子分数）：

```bash
/opt/homebrew/bin/python3.11 scripts/build_016_colab_inputs.py --probe
/opt/homebrew/bin/python3.11 scripts/build_016_colab_inputs.py
```

- 400 面板谱本地 CPU s2 前向 → 10ppm 全窗 FP 打分（torch CPU bmm）→ 005 union 图谱 canonical-max → top100 → worker 版 `inputs.json`。400/400 ok，empty/mass_fallback/exception 均为 0。
- 面板 FP 基线（MRR@25，top100 内）：跨来源组 0.9548（198/200 命中 25 内）、自然缺库组 0.7434（185/200）。跨来源组 FP 已近天花板，后续看 ICE/direct 有无增量；自然缺库组才是主战场。产物 `artifacts/016-colab/panel-016-ice-bundle.tar.gz`（3.5MB，inputs.json 9.4MB + code/ice + install.sh + worker）。
- Colab notebook `notebooks/colab/016-panel-ice-colab.ipynb`（10 格，AST 通过）：Drive 取包 + gen/inten → T4 断言 → 安装 → 干跑 2 分子 → iceberg split/k=100 全量（cache 断点续跑，约 60–90 分钟）→ 打包 scores 取回。T4 只跑 ICE，前向已在本地完成。
- 排障记录：首版 inputs 缺 `scores_top100`（worker 336 行 fusion 用），Colab 干跑 `KeyError`；已补对齐分数并本地干跑验证。另确认 iceberg 仅覆盖正模式（`is_covered` 要求 positive），面板前 2 分子（负离子）0 jobs 属预期行为；全量干跑 400 分子 280 覆盖、56699 jobs、30980 候选（top100/分子），预计 T4 约 20–30 分钟。notebook 本体无需修改，只换 bundle 包。

## Kaggle 只读

本轮未做新查询；有效提交仍 v9r2/v10 public 0.323。下一步先完成参考索引与计时，再议 GPU 任务。
