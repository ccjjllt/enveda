# 021 本地 NP250 验证（E006 去模拟器）

时间：2026-10-20（Asia/Shanghai）。面板：train 中 `enveda-np-examples` 1184 谱 / 250 结构。
评分：`src/casmi_metric.py`（官方 RDKit 2026.03.3 规范）。ICEBERG/GLACIER：**跳过**（box 无 GPU；无可用 NP250 缓存分数），结果为「E006 CPU 通道」相对比较，不等于完整 v11。

## 遮蔽定义

| 码 | 含义 | train 处理 |
|----|------|------------|
| A | 全可见 | 原 train（查询谱在库内，近似上界） |
| B | 本谱隐藏 | 去掉 1184 张 enveda-np-examples；**跨源同结构参考保留** |
| R | 参考全删 | 去掉与真值同一官方身份的所有谱；结构仍在池 |
| C | 结构移除 | R + Engine1/2/PubChem 池排除真值身份 |

面板：`/workspace/e021/panel_info.json`（B 余 2,482,335 谱；同身份删除 57,273）。

## 基线 MRR（final submission）

| 遮蔽 | MRR | top1 | top25 | 备注 |
|------|-----|------|-------|------|
| B | **0.9569** | 231 | 250 | 跨源参考极强；与 public 0.446 不可比 |
| C | **0.1168** | 4 | 89 | 池内无真值；e006_pre_rank2=0.1285，尾部融合略伤分 |
| A | 未跑完 | | | 11:59 启动后为给门扫/MB538 让 CPU 主动终止；A 为近似上界，优先级低 |

B→C 配对：Δ −0.840，wins 2 / losses 244 / ties 4。说明本面板上「库内参考」主导高分；去结构后召回崩塌。

## 增量（在 B_base 上，逐项）

### 1) 直接参考谱质量（post-hoc 重排）

对 submission 候选算与 mask-B train 参考谱的贪心 cosine（同极性、≥3 峰），高分且相对 top1 有 margin 则提到第 1。

| 配置 | MRR | Δ | wins | losses |
|------|-----|---|------|--------|
| q≥0.8, margin≥0.2, pos=1 | **0.9709** | **+0.0140** | 7 | 0 |
| q≥0.8, m≥0.1 | 0.9683 | +0.0113 | — | — |
| q≥0.7, m≥0.2 | 0.9689 | +0.0120 | — | — |
| q≥0.6, m≥0.05 | 0.9583 | +0.0013 | 7 | 7 |

**结论**：严格阈值（高 cosine + 足够 margin）在本面板无净损失；宽松阈值开始有 losses。属小面板、且 B 保留跨源参考，**外推到隐藏集需谨慎**（018 曾见扩库强化错误匹配）。

### 2) MassBank 538 扩池

hook：`/workspace/e021/hook_mb538.py`（Engine2 COCONUT 支路追加 mb539_*）。

**首跑失败**（15:01–15:17，`B_mb538.log`）：追加成功（`+ E021 MassBank538 500 new structures`），但 hook 在主进程调用 `canon_key()` 把 `"te"` 写入全局 `_g`，fork 出的指纹 worker 跳过 `_fp_init()`，275,808 个训练指纹全 None → Engine2 失败。已修（`if "m2" not in _g: _fp_init()`）。

**重跑成功**（16:47–17:27，`B_mb538r`，总 2367s）。相对 `B_base`：

| 指标 | 值 |
|------|-----|
| MRR | 0.9571（base 0.9569） |
| Δ | **+0.0001** |
| wins / losses / ties | **1 / 0 / 249** |
| 变化分子 | `np_LLQCRTZROWMVOL` RR 1/6 → 1/5 |
| bootstrap 95% CI | [0.000, +0.0004] |

Engine 列表里 MassBank 结构出现情况：top25 含 MB 的分子 8 个（相对 base 多出 6 个槽位），top1 **0**；中位最佳名次约 16。submission 有 53 行与 base 字符串不同，但几乎不影响 RR。

**结论**：扩池生效但对本面板几乎零收益；不并入 v12。

### 3) 门参数扫描

`sweep_gates.json`，B 遮蔽，reuse eng/pc，`E021_SKIP_LATE=1`（跳过 cell≥27）。默认值：LIB_TAU 0.9、REL_TH 600、PROMO_REL 150、POP_TAU 6.0、WINDOW 40；base MRR **0.9571**（231/250 top1）。配对比较 250 分子，bootstrap 2000 次。

| 配置 | MRR | Δ vs base | wins | losses | 95% CI |
|---|---|---|---|---|---|
| LIB_TAU 0.85 | 0.9591 | +0.0020 | 1 | 0 | [0.000, 0.006] |
| LIB_TAU 0.95 | 0.9549 | −0.0021 | 1 | 2 | [−0.007, 0.000] |
| REL_TH 400 | 0.9571 | 0 | 0 | 0 | — |
| REL_TH 800 | 0.9571 | 0 | 0 | 0 | — |
| PubChem 提升阈值 PROMO_REL 100 | 0.9524 | −0.0047 | 0 | 3 | [−0.011, 0.000] |
| PROMO_REL 250 | 0.9591 | +0.0020 | 1 | 0 | [0.000, 0.006] |
| PROMO_REL 关闭（1e18） | **0.9616** | **+0.0045** | 3 | 0 | [0.000, 0.011] |
| POP_TAU 3 | 0.9571 | 0 | 0 | 0 | — |
| POP_TAU 8 | 0.9571 | 0 | 0 | 0 | — |
| 候选窗口 WINDOW 80 | 0.9571 | 0 | 0 | 0 | — |

**结论**：REL_TH、POP_TAU、窗口在 B 上完全无影响（逐分子一致）。PubChem 提升阈值单调：越宽松越伤（100: 0/3），越严/关闭越好（关闭 3/0）。LIB_TAU 0.85 小正。所有非零 Δ 只涉及 1–3 个分子，CI 下界贴 0，属弱证据。注意 B 保留跨源库参考，PubChem 提升对「库外结构」可能有用而 B 测不出；C 中 PubChem 通道贡献 0/89 命中，也不支持它在库外有用，但 C 同时把真值移出了 PubChem 池，不能算反证。

### 1b) 直接参考质量在 C（真值无参考、不在池）上的伤害

| 配置 | MRR | Δ vs C_base 0.1168 | wins | losses |
|------|-----|---|---|---|
| q≥0.8, m≥0.2 | 0.1122 | −0.0046 | 0 | 5 |
| q≥0.7, m≥0.1 | 0.1109 | −0.0059 | 0 | 9 |

即：真值缺参考时，此门会把有强参考的错误结构提到第 1（与 018 机制一致）。B 上 +0.014 / C 上 −0.005，净效果取决于隐藏集中「有跨源参考」比例。

## C 中命中从哪来（事实核对）

C 的 89 个 top25 命中（4 个 top1）：
- PubChem 通道列表 top100 含真值：**0/89**；Engine2（casmi_engine）列表 top100：**0/89**。
- 89/89 已出现在 Engine1（v4n `casmi` 引擎）的 `baseline_submission.csv`。
- 真值不在 COCONUT/ChEBI-LIPID MAPS 池、不在 in_C train；v2-pool 中有 85 个命中真值，**85/85 已被 mask C 窗口过滤**。
- Engine1 有 class-3 生成步骤（`casmi/engine.py` 中 `self.generate(analogs, ...)`：对库内类似物做衍生/位置异构生成）。按排除法，C 的命中来自**该生成路径**（未逐个读取 is_gen 标记，属排除法结论）。例：scopoletin 真值排第 3，前两名是其位置异构体。

## ICEBERG / GLACIER

未找到可复用的 NP250 模拟分缓存；CPU 跳过。完整 v12 仍须在 Kaggle T4 上挂模拟器。

## 对 v12 的建议（按当前证据，2026-10-20 17:30 更新）

1. **主改动**：严格跨库直接参考规则（`q≥0.8, margin≥0.2`，提到第 1）。B +0.0140（7/0），C −0.0046（0/5）。净效果取决于隐藏集中有跨源参考的比例；上线前在 Kaggle 小样本或独立留出上核对，防 018 类假强匹配。
2. **可选次改动**：PubChem 提升阈值 150 → 250（+0.0020，1/0）或直接关闭（+0.0045，3/0）。证据弱且 B 测不到库外收益；若只想一次一个变量，v12 只上第 1 条，PROMO 留给 v13。
3. **不改**：REL_TH、POP_TAU、候选窗口（零效应）；LIB_TAU 0.85 的 +0.002 太小，不单独上。
4. **MassBank538 扩池**：重跑 Δ +0.0001（1/0），几乎零收益；**不并入 v12**。
5. 以上全为 CPU 通道数字，**不含 ICEBERG/GLACIER**；完整 v12 仍须在 Kaggle T4 上挂模拟器跑。B 的 0.96 不代表 public，面板小（250）存在乐观偏差。

## 产物

- 验证器：`src/eval_e006_local.py`，`src/casmi_metric.py`
- 运行根：`/workspace/e021/`（runs/B_base, C_base, B_mb538r, B_gates, posthoc, …；`mb538_compare.json`）
- Notebook：`/workspace/k020/enveda-020-e006-repro.ipynb`（与 v11 同文）

