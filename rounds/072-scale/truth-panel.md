# 可见 400 分子真值面板（truth panel）

日期：2026-11-17。用途：072a 起把完整管线跑在**带真值**的可见集上，用分子级指标（MRR / Recall@k / 池召回）分解损失，而不是靠公榜 ±0.003 猜。

面板结论一句话：**可见 1213 张谱全部在 `train.parquet` 中找到逐位相同的副本，400 个分子全部得到唯一、无歧义、可独立验证的真值结构**，可直接作为离线评测集。

## 1. 方法

### 1.1 判据

`train.parquet` 里的 `ms2_mzs` 与测试谱最多 4 位小数，因此用「峰值列表四舍五入后取哈希」作为身份判据：

```
h4  = blake2b( np.round(mz, 4).tobytes() )                  # 主判据
h3  = blake2b( np.round(mz, 3).tobytes() )                  # 容错回退判据
h4i = blake2b( np.round(mz,4) | np.round(intensity,4) )     # 强度一致性旁证
```

峰位哈希天然包含峰数，且对浮点尾数不敏感。**前体不作为身份判据，只做流式预筛窗口（±0.01 Da）**，避免把前体重校准误差误判成不匹配。

### 1.2 流式扫描（本机 32 GB 内存约束）

`train.parquet` 2,539,608 行 / 21 个 row group / 3.0 GB。做法：

1. `ParquetFile.iter_batches(row_groups=[rg], batch_size=16384, columns=[仅 12 列])`，逐 row group、逐 batch 解码；
2. 只读 `precursor_mz` 一列到 numpy，用 `np.searchsorted` 对**已排序的测试前体数组**做窗口判断（`hi > lo` 即命中）；整批无命中就直接 `continue`，不解码峰表、不进 Python；
3. 命中批次用 `RecordBatch.filter(mask)` 裁剪后再 `to_pylist()`，只为命中的行计算哈希。

因此**同一时刻只有 1 个 batch（约 1.6 万行）的峰表在内存里**，从未把全库峰数组物化。实测峰值 RSS 1.46 GB、墙钟 13.6 s。

### 1.3 聚合

- 每个测试谱：命中哈希 → 该哈希背后的 `inchikey14` 集合；
- 每个 `molecule_id`：所有谱的 `inchikey14` 取并集；
- `status` 定义：`unique`（并集只有 1 个且全部谱命中）、`partial_unique`（1 个但未全部命中）、`ambiguous`（≥2 个）、`unmatched`；
- 歧义时 `inchikey14` 留空、原始候选写入 `candidate_inchikeys`，**不替数据做选择**。

## 2. 覆盖结果

| 项 | 数量 | 占比 |
| --- | --- | --- |
| 测试谱 | 1213 | — |
| 测试分子 | 400 | — |
| 谱级 `exact4` 命中 | **1213** | 100% |
| 谱级仅 3 位小数才命中 | 0 | 0% |
| 谱级未命中 | 0 | 0% |
| 谱级强度也逐位相同 | 1213 | 100% |
| 分子级 `unique` | **400** | **100%** |
| 分子级 `ambiguous` | 0 | 0% |
| 分子级 `unmatched` | 0 | 0% |
| `partial_unique` | 0 | 0% |

其他一致性：

- 每张测试谱在训练库中**恰好只有 1 个候选 `inchikey14`**（`n_candidate_inchikeys` 恒为 1，最大也是 1）；
- 400 个真值 `inchikey14` **两两不同**，没有两个 `molecule_id` 指向同一结构；
- 每张测试谱在训练库中的副本数恒为 1（1:1 拷贝，无额外重复谱）；
- 加合物在测试谱与匹配到的训练行上 **1213/1213 完全一致**。

所以不需要执行「若唯一率 <60% 再做 80% 峰匹配」的第二遍；该分支已判定为不必要（哈希唯一命中率 100%）。

## 3. 加合物分布

按谱：

| adduct | spectra | exact4 命中 | 仅 3 位命中 | 未命中 | 强度亦相同 |
| --- | --- | --- | --- | --- | --- |
| `[M+H]+` | 959 | 959 | 0 | 0 | 959 |
| `[M-H]-` | 193 | 193 | 0 | 0 | 193 |
| `[M+CH2O2-H]-` | 31 | 31 | 0 | 0 | 31 |
| `[M+Na]+` | 22 | 22 | 0 | 0 | 22 |
| `[M+NH4]+` | 4 | 4 | 0 | 0 | 4 |
| `[M+Cl]-` | 2 | 2 | 0 | 0 | 2 |
| `[M+K]+` | 2 | 2 | 0 | 0 | 2 |

按分子（一个分子可能有多张不同加合物的谱，故合计 513 > 400）：

| adduct | molecules | unique | ambiguous | unmatched |
| --- | --- | --- | --- | --- |
| `[M+H]+` | 366 | 366 | 0 | 0 |
| `[M-H]-` | 109 | 109 | 0 | 0 |
| `[M+CH2O2-H]-` | 17 | 17 | 0 | 0 |
| `[M+Na]+` | 16 | 16 | 0 | 0 |
| `[M+NH4]+` | 3 | 3 | 0 | 0 |
| `[M+Cl]-` | 1 | 1 | 0 | 0 |
| `[M+K]+` | 1 | 1 | 0 | 0 |

分子内加合物多样性：单加合物 292、双加合物 103、三加合物 5（共 108 个多谱多加合物分子）。极性：仅正离子 276、仅负离子 27、正负都有 97。

## 4. 独立验证（不只依赖哈希）

哈希只能证明「与训练某行相同」，不能证明「标签化学上成立」。因此另做三项独立核对，全部通过：

1. **前体 vs 分子式**：用 `molecular_formula` 算单同位素质量，按测试加合物加质子/钠/钾/铵/氯/甲酸根质量，与测试 `precursor_mz` 比较：|ppm| 中位数 **2.05**、p90 3.22、最大 5.97，1213/1213 ≤ 20 ppm。若标签或加合物错配，误差会到几百 ppm 量级。
2. **SMILES → InChIKey 回环**（rdkit）：400 条 `smiles` 全部可解析，`MolToInchiKey` 前 14 位 400/400 等于 `inchikey14`。
3. **同分异构压力**：370 个不同分子式承载 400 个结构，其中 24 个分子式含 ≥2 个不同 `inchikey14`（最多 5 个异构体）。说明面板确实要求区分同分异构体，不是「按分子式查表」就能拿分。

元素覆盖：C/H/N 全部 400，O 392，S 109，F 94，Cl 44，Br 19。

## 5. 面板边界（使用规则）

- **只代表「训练库里有参考谱」的那一类分子**。可见 400 的命中率不能外推为隐藏集召回率；隐藏集含训练库未见过的结构。
- **前体范围窄**：245.09–460.16 Da。近 460 Da 以上的加合物/中性质量处理在该面板上无证据。
- **只有 7 类加合物**，而隐藏集有 10 类（见 072 `plan.md` H2）；`[M+FA-H]-` 等缺失类别不能用该面板评估。
- 仪器全部 `timsTOF`、来源全部 `enveda-180`，跨仪器泛化无法评估。
- 因此该面板的合法用途是：**定位确定性缺陷**（候选池是否召回真值、真值名次、加合物中性化是否正确），不是宣称泛化提升，也不是拿来反复调阈值直到离线分数好看——离线提分必须再到公榜验证。

## 6. 产物与复现命令

产物（`artifacts/` 已 gitignore，不入库）：

| 路径 | 内容 |
| --- | --- |
| `artifacts/truth/visible_truth.parquet` | 400 行真值：`molecule_id, inchikey14, smiles, n_matched_spectra, n_candidate_inchikeys, matched_spectra_fraction, status, n_spectra_total, candidate_inchikeys, molecular_formula, inchikey, n_train_witness_spectra, ingest_lib, adducts` |
| `artifacts/truth/spectrum_matches.parquet` | 1213 行谱级证据：`match_level, n_candidate_inchikeys, candidate_inchikeys, n_train_spectra, intensity_identical, train_adducts` 等 |
| `artifacts/truth/summary.json` | 覆盖统计、加合物表、歧义/未命中清单、耗时与峰值内存 |
| `artifacts/truth/verify.json`、`run.log`、`verify.log` | 独立验证结果与原始日志 |

脚本（入库）：`rounds/072-scale/probe/build_visible_truth.py`、`rounds/072-scale/probe/verify_visible_truth.py`。

精确命令（本机，`~/.venvs/data` 虚拟环境，Python 3.11.15，装有 pyarrow 26.0.1 / pandas 3.1.1 / numpy 2.4.6 / rdkit）：

```bash
cd /Users/zhaohuaxiang/ccjjllt/enveda

/usr/bin/time -l ~/.venvs/data/bin/python \
  rounds/072-scale/probe/build_visible_truth.py \
  > artifacts/truth/run.log 2>&1

~/.venvs/data/bin/python \
  rounds/072-scale/probe/verify_visible_truth.py \
  > artifacts/truth/verify.log 2>&1
```

实测资源：墙钟 **13.6 s**，峰值 RSS **1.46 GB**（`/usr/bin/time -l` 报 1,534,787,584 bytes max resident），扫描 2,539,608 行、命中窗口内候选行 442,695 行、最终哈希命中 1213 行。

## 7. 待办

- 072a：把该 parquet 作为真值接进管线导出，计算分子级 MRR / Recall@k / 池召回，并按加合物分解损失。
- 若后续替换/扩充面板，必须重跑本节两个脚本并更新本文件，不得手工改真值表。
