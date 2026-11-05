# 053 碎片 FREEZE 重排（051 底座）

## 目标
对标 heliosli rr3（私有模型 + 训练集结构冻结），用自研 048 断键碎片线性分在 051 top25 上做 FREEZE 重排：训练集 InChIKey14 钉死原位，其余槽位按碎片分重排。lib_max≥0.9 或列表长度&lt;2 不动。

## 为何不是短列表补全
可见 test 为占位，本地下载的 submission 大量为 `CCO`，不能代表隐藏测试。隐藏测试上 051 应产出完整候选列表；FREEZE 在 Kaggle 端对真实列表生效。若日志 `skipped_short` 很高，下一轮再做同式补全。

## 方法
1. 数据集 `ccjjllt/casmi26-053-fragfreeze`：train_ik14.pkl、frag_lin.json、frag_feats.py、ff_config.json。
2. 051 notebook 末尾加 overlay cell：读 submission.csv + test 谱，算碎片分，FREEZE 重排，写回；审计 `ff_overlay_audit.json` / 日志行 `053 FRAGFREEZE`。
3. 推送 020 新版本；watcher 在 COMPLETE 且日志含 FRAGFREEZE 后用 `-k/-v` 提交（不依赖本地下载的占位 submission 内容）。

## 许可
自研 048 干净；底座 051 含 unknown → **A 线冲牌**。

## 交付
docs/plan+progress、notebooks/053、中文 commit、watcher `/workspace/k053/run.sh`。
