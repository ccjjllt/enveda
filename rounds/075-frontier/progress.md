# 075 进展

更新日期：2026-11-19（北京时间）。

## 已完成与实测结果

### 1. 前沿与管线调研（2026-11-18）
- 结合 `071` 深度审计，收敛为两份精简文档：[模型与资产清单](research/models-and-assets.md)、[管线信号与验证口径](research/pipeline-signals-and-eval.md)。
- 公榜复核（2026-11-19 10:42 北京）：第 1 名 `0.550`（Shehab Alshehabi）、第 20 名金牌线 `0.506`、第 30 名 `0.502`；前 30/40 名无任何公开 notebook。公开可复现前沿仍为 `0.492–0.494`（我方 `068a` 稳定 **0.492**）。

### 2. 本地制备核查（2026-11-18）
- **RDKit Murcko 兼容问题**：本机 `rdkit 2026.09.2` 调用 `MurckoScaffold.GetScaffoldForMol` / `MurckoScaffoldSmiles` 6/6 分子均报 `RuntimeError: bad any cast`。化学型/骨架分组必须在 Kaggle 远端（`RDKit 2026.03.3`）计算。
- **结构聚类稀疏度**：对 20,167 个 ≥2 谱结构抽样 5,000（Morgan2 2048-bit，Tanimoto ≥0.65 leader 聚类）得 **4,921 簇**（单结构 4,844、2–4 簇 77、≥5 簇 0，最大簇 3），结构库整体高度分散。

### 3. NX 同分异构判别器验证 → **公榜崩溃，已终止**
- `075c`（`jiyujia/enveda-075c-nx-instrument` v2 `COMPLETE`）：在 `072d` 150 分子面板上用 `ahmedberatozer/casmi26-nx-model` 做同式组重排，MRR `0.8286 → 0.9394`（`fold0 w=1.0`）。因 150 个结构均来自 `train.parquet`（`fold0` 仅留出 1 折），存在训练泄漏，只证明集成路径可用。
- **`075d`（`jiyujia/enveda-075d-nx-rerank` v3，ref `58519267`，2026-11-18 17:20 UTC 提交）**：将 `NX fold0 w=1.0` + `NX_LIB_GATE=0.9` 接入 `068a`（允许同式组内换 rank-1，commit smoke `active 0` 符合预期）。
  - **2026-11-19 读回：`COMPLETE`，public = `0.381`（相对 068a 0.492 暴跌 `−0.111`）**。
  - 结论：NX 在未见结构上替换 rank-1 严重误排（与 `075d` 原笔记历史注记 `e6 0.472→0.414` 同类崩塌）。**NX 路线正式关闭**，相关本地产物与脚本已归档清理。

## 资源与下一步

- **配额（2026-11-19 MCP 核实）**：当前账号周 GPU 已用 `121,537s / 108,000s`（0h 剩余），刷新时间 **`2026-11-21T00:00:00Z`**（北京 11-21 08:00）；全部历史提交均已 `COMPLETE`，无在途任务。
- **下一步**：GPU 刷新后若继续推进，按 [plan.md](plan.md) 先建 `075b` 化学型整组留出面板与零成本信号审计，再单臂评估 `MIST MSG` 候选打分，不恢复已关闭方向。
