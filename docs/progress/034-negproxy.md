# 034 进度
- v16（033）提交 ref 57595662，截至记录仍 PENDING。
- 本地面板：负代理 ICE 0.681 / GL 0.731（随机 0.227）。
- notebooks/034：033 notebook 第 23 cell 在 build_ice_input 后插入 NEGPROXY；离线逻辑检查：51 条代理谱，覆盖分子 369→396。
- GPU：kernel v13 已 push（运行中）。
- kernel v13 COMPLETE（约 20 分钟），日志：034 NEGPROXY spectra 1 molecules 12（可见 smoke 中 1 条代理谱），ICE/GL/N1 均正常。
- 22:00（北京时间）提交 v17，ref 57596597，PENDING；v16 ref 57595662 仍 PENDING。今日剩余 2 次提交。
- 评分：v17 public 0.446（v16 0.450 基础上再降 0.004；相对 v15 0.464 降 0.018）。负离子 [M-H]- 代理在本地面板有效但 LB 不增益，不采用。
- 结论：033/034 均退步，基线回到 v15（ref 57583178，0.464）。下一步应在 v15 基础上单独验证 034 代理（去掉 033 注入），或转向同式/缺参考错例修复。
