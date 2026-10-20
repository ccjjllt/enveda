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

hook：`/workspace/e021/hook_mb538.py`（Engine2 COCONUT 支路追加 mb539_*）。**待跑**（等 A 结束后跑 B_mb538）。

### 3) 门参数扫描

`sweep_gates.json`：LIB_TAU / REL_TH / PROMO_REL / POP_TAU / WINDOW。C 全量 sweep 已中止（尾部 cell 27 单次约 6 min×16 过贵）。将用 `E021_SKIP_LATE=1` + reuse eng/pc 在 B 上补扫。**待跑**。

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

## 对 v12 的建议（按当前证据）

1. **优先**：把「直接参考谱质量门」做成候选级特征或 post-fusion 提升，默认阈值对齐 `q≥0.8, margin≥0.2`（本地 +14 mrr、7/0）；上线前在独立留出或 Kaggle 小样本核对，防 018 类假强匹配。
2. **次优先**：复现门扫与 MassBank538 本地 Δ 后再决定是否并入 v11 fork。
3. **不要**根据 B 的 0.96 乐观估计 public；B 有跨源参考，C 的 0.12 更接近「库外结构」难度。
4. 面板小（250）、与训练谱同批 NP 示例，存在乐观偏差；任何正 Δ 只作提交候选筛选。

## 产物

- 验证器：`src/eval_e006_local.py`，`src/casmi_metric.py`
- 运行根：`/workspace/e021/`（runs/B_base, C_base, posthoc, …）
- Notebook：`/workspace/k020/enveda-020-e006-repro.ipynb`（与 v11 同文）

