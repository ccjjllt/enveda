# 公共研究代码

可执行研究脚本按轮次放在 [scripts](../scripts/README.md)；本目录保留公共模块和已有导入位置。

| 模块 | 用途 |
| --- | --- |
| `data.py`、`chemistry.py`、`splits.py`、`casmi_metric.py`、`evaluation.py` | 数据读取、结构规范化、划分和指标 |
| `candidates.py`、`candidate_expansion.py`、`retrieval.py` | 候选检索与扩库 |
| `ranking.py`、`molecule_ranker.py`、`analog_evidence.py`、`evidence_residual.py` | 排序、直接谱与analog证据 |
| `forward_rerank.py`、`fiora_adapter.py` | 前向模拟与重排 |
| `models/fingerprint.py`、`fp_train_data.py`、`train_fingerprint.py` | 指纹模型、训练数据与训练 |
| `eval_e006_local.py`、`panel_e180.py`、`panel_pc.py`、`study_e1_gen.py` | E006本地评估、面板与生成研究；边界见021–030进展 |
| `workspace_paths.py` | 读取历史JSON时按迁移表定位旧路径，不改写原始记录 |

模块存在不代表已验证有效，实际结果以[配对计划与进展](../docs/README.md)为准。
