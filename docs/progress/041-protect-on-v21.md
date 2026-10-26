# 041 进度
- kernel v19 已 push（040 + 041 PROTECT cell）；watcher：k_watch.sh 19 "..." "PROTECT OK"（日志 /workspace/k041_watch.log）。
- kernel v19 COMPLETE，watcher 校验后自动提交 v22 ref 57655603（评分中）。042 续训 kernel 随即 push。
- 【结果】v22 公榜 0.398（v21 0.464 → −0.066），大幅回退。结论：Engine-1 首选明显差于神经重排后的首选，强行放回第 1 有害；top1 保护方向放弃，主线仍为 v21（040，W0.25 fp0.5）。
- 推论：神经模型对排名第 1 的贡献是正的（与 v20 W0.40 回退一起看，最优权重在 0.25 附近，而非"只调 2–25 名"）。后续重点转向 042 自研模型续训/集成。
