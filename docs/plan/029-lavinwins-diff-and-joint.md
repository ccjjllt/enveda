# 029 计划：对照 lavinwins/casmi26-top-1-gold，落地高价值差异

## 对照结论（2026-10-22 中午，代码级 diff）

对照对象：
- 公开 notebook：`lavinwins/casmi26-top-1-gold`（papermill 约 27 min；changelog 最高分 **0.464**）
- 我们基线：v465 fork（`/workspace/k027b` = `notebooks/028/enveda-028a-e1ext.ipynb` 同源），公榜 **0.456**

### 参数：先前记录有误

| 项 | 我们的 v465/028a | lavinwins **实际代码** | 备注 |
|---|---|---|---|
| `POP_LAM` | **0.15** | **0.15** | docstring 仍写 0.25，但赋值是 0.15 |
| `POP_UNION` | 200 | 200 | 相同 |
| `POOLPOP_MU` | **0.15** | **0.15** | 相同；不是 0.25 |
| Engine2 ChEBI | `sergiomalv/casmi-chebi-lmsd-structures` | 同左（ROLE 表仍保留 prvsiyan 旧库名） | 相同 |
| DreaMS 数据集 | **未挂载**（代码有 fill 分支） | 经 attributed assets / ROLE 声明 `hengck23-dreams-enveda-casmi26` | 我们缺挂载 |
| `GEN_N_ANALOG` | 默认（6） | **20** | lavinwins v17，changelog 未计分 |
| E006 `ice and gl` 分支 | 只用 GL（`[{}, gl]`） | N1 行上 **joint ICE+GL** | lavinwins v16：**0.456→0.464** |

lavinwins 版本表还记录：PubChem 通道先验 **0.25→0.15** 为 v10（0.455），与 guhongbin v465 阶梯一致。因此 **不应** 把 POP 改回 0.25 当作「lavinwins 风格」。

### lavinwins 另有、本轮不优先搬的部分

- fold-safe FPNet 五折集成、`SRC_BONUS`/`NP_BONUS`/`LOTUS_W`/同义词先验、attributed v4g 大包：changelog 在 joint 之前多停在 0.455–0.456，依赖额外 dmitrii 数据集，体积与许可核对成本更高，留作 029c+。

## 029 变体设计

### 029a（本轮离线备好，配额重置后再 push）

叠在 **028a** 上（含 Engine1 池扩展），三处改动：

1. **Joint ICE + proxy GLACIER**（factorial `rerank_row`）：仅当 `ice_fallback` 且 `mid ∈ N1_GL_MIDS` 时把 ICE 分数与 GL 一起送入 `rerank_multi`。出处：lavinwins v16 / flexonafft v47；公开 changelog +0.008。
2. **`GEN_N_ANALOG = 20`**（engine 初始化后）。
3. **kernel-metadata 增加** `hengck23/hengck23-dreams-enveda-casmi26`，激活已有 safe-tail DreaMS 填空。

保持 `POP_*=0.15`、`POOLPOP_MU=0.15`。Apache-2.0 沿袭 + 上游数据集原许可。

产物：
- `notebooks/029/enveda-029a-joint-dreams.ipynb`
- 推送目录：`/workspace/k029`（`code_file=enveda-020-e006-repro.ipynb`，同一私有 kernel）

### 029b（可选，低优先级）

仅 A/B `POP_LAM/POOLPOP_MU=0.25`（预期不优于 0.15）；仅在 029a 无增益且需排除「先验权重」时做。

## 配额重置后顺序（2026-10-24 08:00 北京时间）

1. push `/workspace/k027b`（028a），跑完并提交一次。
2. 再 push `/workspace/k029`（029a），跑完并提交一次。
3. 禁止在重置前做任何 GPU push。
