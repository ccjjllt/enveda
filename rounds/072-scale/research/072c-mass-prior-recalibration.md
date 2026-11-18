# 072c：mass prior 在当前底座上重新标定（结论：无余量）

日期：2026-11-18。目的：`MASS_CENTER / MASS_FREE / MASS_FULL / MASS_W` 是在 popwenman Gold Apex V22（0.481，`V8='MASS'`）上调出来的，从未在 068a 这个分支上复核。mass prior 是 0.492 配方里唯一真正有杠杆的活跃层（v2 日志：`candidates 688, penalised 447, lists_touched 394`），因此值得重新标定一次。

执行位置：Kaggle CPU kernel `jiyujia/enveda-072c-massgrid`（`KernelWorkerStatus.COMPLETE`），**不消耗 GPU 配额**。RDKit 由 `metric/rdkit-2026-3-3-wheel` 的 cp313 轮子离线安装（`rdkit 2026.03.3`）。

## 方法

在 120 个分子的带标签面板上，完整复刻 068a 第 20 格的算式：

- 中性质量：`nm = precursor_mz - _ADD[adduct]`，`_ADD` 为管线的 7 个加合物精确表，按分子取中位数，`nm>50` 保留 → **118/120** 个分子可用。
- 候选精确质量：RDKit `Descriptors.ExactMolWt`；`ppm = (exact - nm)/nm*1e6`。
- 罚分：`|ppm - CENTER| > FREE` 时 `pen = -W * min(1, (|ppm-CENTER| - FREE)/(FULL - FREE))`。
- 排序：第 1 名固定，2–25 名按 `score + pen` 降序。`score` 用管线 `_score_cand` 的主项 `1/(3+baseline_rank) + 0.75/(3+neural_rank)`；省略的 pop/spec/np 三项各自权重 ≤0.12，属于近似（已在结论里注明）。
- 网格：CENTER ∈ {-2.5,-2.0,-1.5,-1.0,0.0}、FREE ∈ {1.5,2,3,4,5}、FULL ∈ {6,8,10,12}、W ∈ {0.05,0.10,0.15,0.25,0.40}，共 **500** 组。

## 结果

| 项 | 数值 |
| --- | --- |
| 无 mass prior 的 MRR | 0.9128 |
| 出厂参数（C=-1.5, F=3, FULL=8, W=0.15） | **0.9144（Δ +0.00159）** |
| 500 组网格的最优 | **+0.00159**（C=0.0, F=5.0, FULL=12.0, W=0.40） |
| **可提升量** | **+0.00000** |

出厂参数已经落在网格最优点上，500 组里没有一组能超过它。

## 为什么天花板这么低

- **能动的分子只有 10 个**：真值名次 ≥2、`library_max<0.9`、且 ppm 可算，三者同时成立只有 10/120。其余分子的真值在榜首（prior 不碰第 1 名），或被库命中护栏跳过。
- mass prior 本身**确实带信息**：质量偏差对真值/诱饵的 AUC = **0.645**；真值 `|ppm-CENTER|` 中位 **1.36 ppm**、诱饵 **1.44 ppm**；3 ppm 内的比例 97.4% vs 77.3%。信息是真的，但决定性太弱（中位数只差 0.08 ppm），且可作用的分子太少。

## 结论

mass-prior 重标定**无余量**，这一层不能再提分。按验收规则，旋钮路线到此为止：

- 不推 GPU 运行，不提交（参数没有变化，提交只会再评一次 0.492）。
- 与既有证据一致：公开 59 臂表里 `V8: MASS→BASE = +0.000`、`MASS→RR = −0.021`、`MASS→BOTH = −0.020`，即这一层已经是全局最优点，改它就掉分。

**0.492 是本配方族的可复现上限。**

## 复现

```
jiyujia/enveda-072c-massgrid   (CPU, COMPLETE)
artifacts/kout/massgrid/mass_grid.json
```
