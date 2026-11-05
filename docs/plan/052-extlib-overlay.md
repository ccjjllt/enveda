# 052 自建新版谱库叠加（051 底座）

## 目标
在 051（alexchilton Neural Ensemble，公榜 0.480）上叠加自建外部谱库匹配层，对强匹配把结构提到 rank 1。对标 heliosli 0.482 的 xl 层（其 `casmi26-extlib-v1` 仍私有）。

## 许可核查
| 库 | 许可 | A 线 | B 线/奖金 |
|---|---|---|---|
| GNPS CC0 / GNPS2 CC0 | CC0-1.0 | ✓ | ✓ |
| Zenodo novel top128 | CC-BY-4.0 | ✓ | ✓（需署名） |
| MSnLib v8 | CC-BY-4.0 | ✓ | ✓（需署名） |
| MassBank orig permissive CC0/CC-BY | 逐条 CC0 或 CC-BY | ✓ | ✓（排除 dl-de） |
| takumuhata/casmi26-extlib | CC0-1.0 | ✓ | ✓ |
| senanuretin MassBank 2026.03 CC0 | CC0，但本轮排除 train 后 0 条 | — | — |

底座 051 含 sanpier（unknown）→ 本提交仍属 **A 线冲牌**，不领奖。

## 方法
1. 汇总上述库的 MS2，RDKit 规范化 SMILES/InChIKey14，剔除 train 的 275810 个 ik14。
2. 强度归一、按 m/z 排序；前体窗口 10 ppm + 0.002 Da；sqrt-intensity 贪心一对一余弦；匹配峰数 ≥ K。
3. 在 2500 个 Enveda-180 timsTOF 训练分子上扫 TAU/K：命中即假阳性（库已去 train）。取假阳 ≤1% 下尽量敏感 → **TAU=0.55, K=3**（8/2500=0.32%）。`lib_max≥0.9` 跳过。
4. 三例自测：库内自查询须复现 key+cosine，否则跳过叠加。
5. 命中：结构已在列表则挪到第 1；否则插入第 1 并截断到 25。

## 离线预期
对 051 submission + test.parquet：hits=6，全部为插入（不在原 top25），top1 改 6/400。余弦偏弱（0.55–0.57），有误插风险；仍按校准阈值提交一次看公榜。

## 交付
- 数据集 `ccjjllt/casmi26-extlib-052`
- 020 kernel 新版本 = 051 notebook + xl 单元
- docs/plan+progress 052；中文 commit
