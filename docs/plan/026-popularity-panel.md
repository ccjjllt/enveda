# 026 计划：按 PubChem 人气加权的面板 Q，以及公开 notebook 调研

## 面板 Q
- 构建方式与 025 的 P 相同，区别在抽样：P 在 PubChem 中均匀抽，Q 按 exp(pop) 加权抽样。
- pop = pc_lsid + pc_lpmid，与 notebook 里 PubChem 提升规则使用的人气一致（POP_TAU=5）。
- 先检验 v11 > v12、v11 > v13 是否成立。两条都成立，才用 Q 评估 PubChem 槽位数、LIB_TAU 和门控生成，并且只提交一次。

## 公开 notebook
- 用 kaggle CLI 查看得分高于 0.446 的公开 notebook，判断是否可以 fork。
