# 016 — 新面板直接检索验证与提交准备

日期：2026-10-17（北京时间）。用户指令：下一步优先做直接谱库检索在新来源上的验证，确认收益后再制作新 submission；GPU 实验/训练用 jiyujia 账号资源，配额有限需拉满利用率。对应进展 `docs/progress/016-new-panel-direct-verification.md`。接手基点 `fb8ed10`。

## 背景与继承

- 最佳仍 public 0.323（v9r2/v10）。015：NP 允许同结构参考 LR 组合 0.668997→0.725342，direct-only 0.723109；移除目标参考与 G1 均未确认改善。015 重放零误差已复核；direct-only 全量拟合本轮已补（42580 行、零警告，`artifacts/015-analog/direct-only-ablation/full_fit.json`），仅供新面板预测，未部署、无 OOF 成绩。
- 新面板只称“本项目此前未用”，不称预训练未见；A 式移除评价沿用 015 干预口径，不作部署分支。
- GPU 铁律（README）：配额按 session 存活扣；CPU 预处理全部前置挂载，T4 只跑神经计算；批量 unpackbits、多格输出、限时里程碑，超时即停。

## 实验契约（预先冻结）

1. 排除清单两级：canonical 级（旧 800 + 16 NP query 键差 + R3/小面板 truth + 新面板 truth）与谱 hash 级（旧 2561 + 新面板全部 query 谱 hash）。inchikey14 与 005 归一 canonical 不一致时以 005 表为准并记录合并数；新面板 canonical 另与旧全部候选键做交叉报告。
2. 新面板目标 400 canonical，先冻结 seed/名单/四个对照再产生候选，禁按结果重采样：
   - 跨来源组 200：query 取 timsTOF 代表谱，同极性，参考限外部来源；正负约 3:1（含多聚体 query ≥20）；负参考薄（109）允许负例下探至 30 并记录。
   - 自然缺库组 200：仅 enveda 单来源，单谱优先约 100 + 多谱无外部约 100；只报 A 式评价，不与跨来源组混算 MRR。
   - 数量不足/无候选者如实降额，不凑数；不得加入 truth 候选。
3. 参考索引改为可追溯 parquet（一行一谱：canonical 组键、train 行号、中性质量、adduct/mode/instrument/ingest_lib、峰 hash、清洗审计），峰延迟读；必记最近 2000 截断率、空证据计数、来源密度。先 40 例手动计时/内存（RSS<8GiB 目标<2GiB、单 query p95<2 秒 CPU），再定完整库。
4. 新面板候选必须真实 FP/ICE top100（v9 同口径代表谱、ddof1、fused_merge），只在前 50 同式槽位用冻结模型预测重排。四个对照：原 v9、LR baseline 全量、LR direct-only 全量、LR 组合全量；HGB 不进入候选成品。
5. GPU（jiyujia 账号）单 session 打包：CPU 产物（索引、FP 池、候选表）作输入挂载；一次跑完新面板 FP 前向 + ICE 预测 + submission.csv + ice_meta；沿用 v10 多格/partial/里程碑。若 jiyujia 推流缺 ccjjllt 私有输入（train-fp-pool、cp313-wheelhouse、massbank539-expansion），按工作量选用：原账号公开/加协作者（分钟级）→ 重传并改 metadata pin（中等）→ 换公开等价 + 重验（最大）。用户定夺后执行。
6. 只有新面板出现可解释、可部署增量（主要部署域无明显退化）才集成新版本并完整离线运行核查；达到可运行态后按授权做一次验证性 code submission。public 不当调参集。

## 待修改代码与交付

- `scripts/build_016_panel.py`（排除清单 + 新鲜池统计 + 冻结面板 manifest，CPU）。
- `scripts/build_016_reference_index.py`（可追溯索引 + 40 例计时，CPU）。
- 新面板 GPU notebook（v10 变体，只增面板谱输入与候选输出分支，不改 v9/v10 排序）。
- 产物 `artifacts/016-panel/`、`artifacts/016-index/`；精简记 `data/references/research-results-016.json`。禁测试文件/CI；运行成功、离线信号、public 提升分开记录。
