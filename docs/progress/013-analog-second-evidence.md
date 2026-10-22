# 013 — 第二证据引擎支撑审计：实际进展（一）

日期：2026-10-17（北京时间）。对应[013计划](../plan/013-analog-second-evidence.md)。本轮完成 Kaggle 只读核实、三份公开 notebook 拉取与第一个 CPU 支撑审计；排序器尚未训练，无新提交。

## 014后续核查（优先于下方v1解释）

用户询问新submission后，014独立复算支撑率并补做固定36例诊断及R3碎片迁移：原direct未排query本谱，确认NP6＋G1 6个首谱与抽样参考原始峰数组完全相同，B域独立谱尚未建立。原shift为整谱平移后二选一，36例中29例最佳匹配≥80%由单峰对贡献；前体/二聚体完整单体峰会产生高相似。统一canonical和精确hash排除未改变该36例analog赢家，不据此称全部analog受重复污染。全部50个R3在全量train raw/canonical无参考已补足核实。R3固定纯负22例碎片MRR0.720671→0.716883、Top1/Top25不变，未确认迁移收益。当前无新的候选排序器/提交入口，014未新增submission。完整命令与结果见[014进展](014-submission-readiness-and-support-audit.md)。

## Kaggle 只读核实（ccjjllt 账号）

```bash
kaggle competitions submissions -c enveda-CASMI26-molecule-id-mass-spectra
kaggle competitions leaderboard -c enveda-CASMI26-molecule-id-mass-spectra --show
kaggle competitions files -c enveda-CASMI26-molecule-id-mass-spectra
kaggle kernels list --search "enveda CASMI26" --sort-by scoreDescending --page-size 10
kaggle quota
```

- 提交：v10 ref57293821 COMPLETE/0.323；v9r2 ref57028399 COMPLETE/0.323；v8 0.307；v7 0.206；v1 0.000。v9（57027339）public 为空。
- 榜单：榜首 0.495（CHU1PC、pikachu）、第五 0.468（Oliver Ford），与 012 快照一致，差距仍 0.145。
- 配额：GPU used 2.61h /剩 27.39h，TPU 0/20h，刷新 2026-10-24T00:00Z。
- 拉取三份源码到 `artifacts/runs/013-notebooks/`（只读，未执行其下载/权重逻辑）：
  - `prvsiyan/analog-propagation-casmi-2026-baseline`（58 cells，733K）：V33 AP-2，四证据通道 + mass-shift analog over train ∪ COCONUT，自报 V33 提交 56956461 public 0.344（正文自述，未独立验证）。
  - `beraterolelk/0-336-sota-enveda-casmi26-analog-ranker`（27 cells，73K）：CONFIG PPM_WIN 10.5、ANALOG_WIN ±200Da、N_ANALOG 80、SIM_POWER 3.2、GBM(6,500,0.03,80)；adduct→中性质量表、entropy/cosine 双核、代表谱每结构一条（3 条更差 0.49 vs 0.52）；候选特征刻意排除来源列以防 Class-2 泄漏。
  - `seyitkaangunes/casmi26-e006-pubchem-promo-rel-150-lb-0-446`（49 cells，171K）：E006 + PubChem promotion rel>150，自报 LB 0.446（未独立验证）。

## 支撑审计 v1：方法与边界

代码 `scripts/013/build_013_analog_pilot.py`，手动命令：

```bash
/opt/homebrew/bin/python3.11 scripts/013/build_013_analog_pilot.py --probe
/opt/homebrew/bin/python3.11 scripts/013/build_013_analog_pilot.py
```

- 查询谱：`artifacts/legacy/f1-bundle/inputs.json` NP250 + `artifacts/legacy/g1-validation/inputs.json` G1250（首谱，mz/it/prec/adduct 口径）+ `artifacts/runs/007-r3/observations_50.json` R3-50（`peaks/*.npz`）。共 550，4 个 R3 未知 adduct 无中性质量而跳过，有效 546。
- 参考：`data/raw/train.parquet` 2539608 行，按 stride 50 抽样约 5 万（实际 50681），adduct→中性质量用 Berat NZD 表同口径（n,z,d），未知载荷不猜。分组为 `inchikey14` 代理，不等价完整 canonical 别名排除（已披露）。
- 清洗：floor 0.002、top256、归一、0.01Da 余弦；analog 取 ±200Da 内非同组 max，shift 取 best-of（direct, shift=query_neutral − ref_neutral）。
- 每 query 限最近 2000 参考（按质量距离），全量墙钟 34.5 秒；旧脚本未测RSS，<8GB仅为计划目标，不能作为本次实测。无神经前向、无训练、无 push。
- 环境：Python 3.11.16，pyarrow 25.0.1，numpy，RDKit 未参与本脚本（仅质量换算常数），本地 CPU。配置seed为20261017，但rng未使用，实际采样是确定性stride50。

## 结果（支撑率，非排序增益）

| 域 | n | direct≥0.3 | direct≥0.5 | analog≥0.3 | analog-shift≥0.3 | mean direct/analog/shift |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| NP250 | 250 | 156 | 138 | 198 | 242 | 0.509/0.599/0.767 |
| G1-250 | 250 | 19 | 17 | 226 | 246 | 0.064/0.697/0.821 |
| R3（质量已知） | 46 | 0 | 0 | 26 | 32 | 0.000/0.356/0.511 |

产物 `artifacts/runs/013-analog/pilot_support.json`（546 行 + 4 errors）。

人工核对：546 ok / 4 errors 均为 R3 未知 adduct（与 007 的 4 unknown 一致）；全量 train 计数抽查 `ACWBQPMHZXGDFX` 347 谱、`AAWZDTNXLSGCEK` 224 谱、R3 `CLSDHBVHQKWEJR` 0 谱，确认 R3 direct 0 非抽样假象。probe 5 例与全量口径一致。

## 判断与下一步

1. 支撑审计说明analog相似度可计算，下一步需要候选判别实验：R3有32/46 shift≥0.3，G1为246/250。但max-over-2000背景偏高（G1 mean shift0.821）；014还发现前体/完整单体峰主导现象，不能把支撑率当结构analog成功或MRR增益。R3全50个真值在train全量0参考的结论已由014补足验证。
2. 下一步才是排序试点：固定 FP+ICE 对照，同式 top50 槽位内加 direct/adduct-shift/analog-shift + 碎片 + 质量误差，GroupKFold（canonical 组）学 HGB，logistic 消融；A 域（移出同组）与 B 域（留出本谱）分开报告，阈值不沿 public sweep。
3. 碎片独立组（R3 纯 `[M-H]-` 22，`SHA256("20261017|013|key")` 排序冻结）与最小生成链（非 oracle 公式→真实 MIST→B=10）按子任务设计排队，参数/预算沿用 012 冻结值，不重调 λ。
4. 本轮无新排行榜提升、无 push、无 submission。完整 PubChem 7.2GB 与 MS2DeepScore2 权重均未下载。
