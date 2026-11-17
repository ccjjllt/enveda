# 研究轮次索引

一轮一目录：计划（`plan.md`）、实际进展（`progress.md`）、notebook 入口、专题调研都在同一目录下。接手先读对应轮次的 `plan.md` 与 `progress.md`，结束后更新 `progress.md`。

| 轮次 | 主题与状态 |
| --- | --- |
| [065-network-n1c](065-network-n1c/) | N-1c 网络重排，public 0.488；入口 `065a/`（ref `58329677`） |
| [066-libcos-midrank](066-libcos-midrank/) | 训练库连续余弦 mid-rank，0.488 中性；入口 `066a/` |
| [067-own-hn-midrank](067-own-hn-midrank/) | 自研 model3 mid-rank，0.488 中性；入口 `067a/` |
| [068-n1c-lam035](068-n1c-lam035/) | N-1c `LAM=0.35`，public **0.492**；入口 `068a/`（ref `58412029`） |
| [069-new-signals](069-new-signals/) | 新信号筛选（timsTOF ICEBERG / DreaMS / 对比对齐 / Enveda-180 谱库）；待提交入口 `082c/`；`research/` 下 5 份专题调研 |
| [070-generation](070-generation/) | 070 新方向：候选级强打分（070a/070b 已判死）、池外生成（P0 已按算术上限终止）、公开重扫、**谱库匹配精度标定（τ=0.72 precision 0.89）→ 082d 标定版入口 `082d/`** |

已终止方向的结论统一在[已放弃研究总览](../archive/research-archive.md)。notebook 入口都是自包含的，只依赖 Kaggle competition/dataset/kernel 输入；推送只上传入口目录：

```bash
kaggle kernels push -p rounds/068-n1c-lam035/068a
```

远端推送/看守脚本在 `../ops/063/`，比赛与远端操作在 `../ops/remote-workflow.md`。
