# Notebook 入口

当前比赛候选为 [030](030/enveda-030-n1-challenger.ipynb)，实际研究状态见[030进展](../docs/progress/030-n1-challenger-and-score-strategy.md)。目录中的历史文件不表示该方案有效或应再次运行。

| 目录 | 用途与状态 |
| --- | --- |
| [030](030/) | 已得分v465基线上的完整N1联合证据候选；未远端运行或提交 |
| [028](028/) | Engine1扩池准备版本；仅本地构建核对，无提分结果 |
| [029](029/) | 历史混合准备版本；漏N1开关，勿直接推送；metadata供030构建器继承 |
| [submissions](submissions/) | 历次自有提交入口及v12附加cell；最佳v14来自027 fork，原始源码保存在本地 `artifacts/runs/030/own-current/` |
| [archive/colab](archive/colab/) | 004–016以及F4a、G1等历史研究入口 |
| [archive](archive/) | baseline、TPU、FIORA、FRIGID、GLACIER、ICEBERG试点与推理草稿 |

归档notebook保留原字节、历史输出路径和署名。旧本地路径的当前位置见[迁移表](../docs/workspace-paths.json)；需要重跑时从对应进展接手，区分本地目录与Colab/Kaggle内的路径。Kaggle metadata 的 `code_file` 仍相对于它所在目录，当前030可直接作为CLI入口。
