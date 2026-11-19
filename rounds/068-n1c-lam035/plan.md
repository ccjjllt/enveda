# 068 计划：N-1c `LAM=0.35`（当前最佳 A 线基线）

- 底座：`065a`（`063a` 0.487 + N-1c `LAM=0.25` → 0.488）。
- 唯一改动：N-1c 默认参数 **`LAM` 0.25 → 0.35**（`TAU=0.6`、`V8='MASS'`、`LIB_HIT_GATE=0.78`、ExtLib `TAU=0.80`、`COMMIT_MODE='smoke'` 不变）。
- 入口：`rounds/068-n1c-lam035/068a/`（`ccjjllt/enveda-068a-n1c-lam035`）。
- 验证依据：Rogers Johnson `casmi26-l4-lam035`、alexchilton 提交 `58380172`、mathleteutd 消融表均核实 `LAM=0.35 → 0.492`（`LAM=0.50 → 0.484`）。
- 权限说明：A 线含 sanpier 等许可 unknown 模型，仅用于冲名次对照，不做纯干净许可宣称。
