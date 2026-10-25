# 036 — corrected PubChem 基线 + 神经重排 RRF

日期：2026-10-25（北京时间）。承接 035（v15/v18 公榜 0.464 持平）。实际结果见[同名进展](../progress/036-neural-rrf.md)。

## 背景与假设

公开 notebook [sanpier/pubchem-neural025-lb-0-466](https://www.kaggle.com/code/sanpier/pubchem-neural025-lb-0-466) 与本仓库同属 v4g/v465 引擎家族：内嵌的冻结基线（SHA256 `aa1a43cd…8743`）已含完整 N1 challenger（`N1_COMPLEMENT=False`），另有更丰富的 PubChem 先验（同义词等）和"按候选身份取证据"的 PubChem promotion 修正，上游报告其基线公榜 0.465；在最终 25 个候选上再做神经谱模型重排，`score = 1/(3+base_rank) + w/(3+neural_rank)`，w=0.25 报告公榜 0.466。

假设：直接采用该路径可从 0.464 提到约 0.466；权重是未验证的超参，同次运行导出多个权重变体可用少量提交测出更优的 w。

## 设计

- `scripts/036/build_036_neural_rrf.py` 从下载的上游 notebook 构建 `notebooks/036`：主 `submission.csv` 为 w=0.25，额外写 `submission_w015/w035/w050.csv`；候选集合不变，仅改顺序。上游源码与致谢原样内嵌。
- 新私有 kernel `ccjjllt/enveda-036-neural-rrf`，T4、断网、Py3.12 镜像，数据源与上游一致。
- 先提交 w=0.25（v19）确认复现，再按配额提交 w=0.35 对照。
- 不改其他规则；可见运行只是 smoke，不代表精度。
