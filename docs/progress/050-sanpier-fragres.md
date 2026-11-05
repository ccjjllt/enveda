# 050 进展

- notebooks/050/build.py 生成 enveda-050-sanpier-fragres.ipynb（047 → 046A 等价单元 + 048 单元改为仅输出 FR_RANK）。
- 2026-11-05 00:06 北京时间：推送 ccjjllt/enveda-020-e006-repro v29，RUNNING。
- watcher：/workspace/k050/run.sh（nohup，日志 /workspace/k050/watch.log），轮询状态→重试下载输出→检查 `050 FRA` 与 top1_vs_046A>0→等待 08:05 北京时间→`kaggle competitions submit -k ... -v 29`。
- 050a 结果（v29，11-05 06:06 北京时间）：rows=400，NEURAL OK；`050 FRA` 日志 {'scored': 12, 'skipped': 388, 'changed': 12, 'changed_top1': 1, 'fr_used': 12, 'top1_vs_046A': 0}。仅 12 条被打分，top1 与 046A 完全一致 → 按门控**未提交**（公榜预期即 0.468）。
- 050b 计划：W_FR=0.30（`python3 build.py 0.30`），其它不变；11-05 12:07 北京时间推送 020 v31。watcher /workspace/k050b/run.sh（日志 /workspace/k050b/watch.log），完成后仅当 top1_vs_046A>0 且当日（UTC）提交数<2 时提交 -v 31。
- 若 050b 仍 top1_vs_046A=0：说明碎片残差只覆盖 12 条（fr_used=12）、天花板过低，停止 W_FR 扫描，转向 051 公开榜首方案复现线。
