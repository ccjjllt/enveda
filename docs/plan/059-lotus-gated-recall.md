# 059 计划：056 底座 + 门控 LOTUS 天然产物同分子式异构体召回（Engine1 / 池外召回兜底线）

## 背景
- A 线最佳：**056** = Gold Apex V22 + `V8='MASS'` + popwenman v80，ref **58089862** 公榜 **0.481**。
- 057（V23）0.476 退步；058（056 + eason V50 多碰撞能 rank-1 救援）020 v38 COMPLETE，2026-11-08 14:57 北京提交 ref **58118277**，PENDING。
- 058 已占"库 rank-1 救援"槽位，059 不再叠库覆盖；按常设顺序走 Engine1 / 池外召回（门控），保持与 056 的干净 A/B（不叠 057/058）。

## 公开扫描（2026-11-08 18:10 北京）
| 候选 | 结论 |
|---|---|
| chuanyuchangbio/casmi26-public-0482-20261108、youyouwou/casmi26-v142rr0482 | 字节一致（md5 9cd896aa…），= 054 失败配方（V8='BOTH' + fast + 无 v80，054/054b 0.459）。 |
| chuanyuchangbio/casmi26-public-0481-20261108、popwenman 0482、guhongbin massdose | 056 同方（MASS+v80）+ `COMMIT_MODE='fast'`，非增量。 |
| heliosli/casmi26-0-482 | dataset_sources 仍有空/私有槽，不可复现。 |
| stardjyeah/casmi26-v9-pubchem-pool | **不是**门控召回：最后一个 cell 用 PubChem 质量窗（pc_mass.npy，该版本未挂对应数据集）+ 训练库邻居 Tanimoto 直接**整表重写** submission（033 式朴素替换），且底座是无 V8 的 0.480 线 + W_OURS。弃用。 |
| stardjyeah/casmi26-v9-pubchem | 同上的诊断版（只写 probe/v9_report）。 |
| stardjyeah/casmi26-e3-gated(-extlib) | 0.480 底座 + TAU0.60 extlib 覆盖，052/055/058 已覆盖该机制。 |
| zkheart/casmi26-full-pipeline-v1 | 独立小管线（COCONUT/PubChemLite），非 Gold Apex 线。 |

## 池覆盖诊断（本地）
- Engine1 v2 池（ahmedberatozer/casmi26-v2-pool，710,701 结构；管线运行时另加 PubChemLite 356,820 → 1,137,233）。
- LOTUS（qwertykeypadapple，149,983，CC0）有 **87.8%** 已在 v2 池；COCONUT（prvsiyan，436,389）89.6% 在池。真正"池外"的 NP 库结构有限，所以 059 的价值主要是**让神经打分器重新审视同分子式 NP 异构体**（056 只在列表空位时用 LOTUS 回填，实际 0 次）。
- 训练集结构仅 4.2% 在 LOTUS；可见 12 个分子里只有 3–4 个在质量带 + 同分子式下有 LOTUS 候选 → 预计隐藏集触发稀少，期望增益小但风险受控。

## 方法（追加 cell，失败回退）
1. 资格：`V8_LIB < 0.60`（无库支持）的已处理分子。
2. 候选：LOTUS 中分子式 ∈ 列表前 3 名分子式、且中性质量误差在 v8 带内（`|ppm − MASS_CENTER| ≤ MASS_FREE` = −1.5±3 ppm），剔除已在列表中的 IK14；每分子最多 150 个。
3. 打分：与 056 列表完全相同的两个神经打分器（公开 v1 + 我们的 v80），在 列表 ∪ LOTUS 联合集合上逐分子 z 标准化后 0.5/0.5 平均。
4. 准入：LOTUS 结构在联合集合中排 **前 2**；插到 **第 6 名**（1–5 名含 leader 永不改动）；每分子 ≤2 个；全局改动分子 ≤20%（超出按与 leader 分差保留）。
5. 同轮校准：对 `V8_LIB ≥ 0.90`（强库命中、leader 基本正确）的分子用同一规则计算"误触发率"；≥5 个可测分子且误触发 >15% → 中止，056 文件原样保留；可测分子不足 5 个而有资格分子时也中止。
6. 任意异常 → 恢复 `submission_pre_059.csv`。审计：`g59_lotus_audit.json`。

## 与 033 / 057 的区别
- 033：028 扩池结构无条件注入 Engine1（−0.014）。059 只在低置信分子、同分子式、质量带、神经前 2、误触发代理通过时插入，且只到第 6 名。
- 057：无参照 leader 提升。059 断言 rank 1 不变。

## 提交规则
- `COMMIT_MODE='smoke'`、`V8='MASS'`、保留 popwenman v80；元数据与 056 相同（LOTUS 已挂）。
- 可见 smoke 全部 LIB=1.0 → 资格分子 0，只测代理；隐藏重跑才是实验（058 模式）。
- watcher `/workspace/k059/run.sh`：先 RUNNING 再 COMPLETE；必须拿到 submission.csv + `g59_lotus_audit.json`；status 以 `ok` 开头、LOTUS 已加载（>100k 行）、smoke 代理误触发 <0.5 才用 `-k/-v 39` 提交；拒 placeholder / 行数≠400。日上限触发则等到次日 08:05 北京重试。

## 许可
含 sanpier → **A 线冲牌**（与 056/058 同）。LOTUS 表 CC0。

## 致谢
qwertykeypadapple（LOTUS 候选表，LOTUS CC0）；上游 alex chilton、popwenman、hoowoo、sanpier、Dmitrii Gluzdov、romanrozen、zz3t3rn41。
