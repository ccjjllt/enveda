# 042 进度
- 续训 kernel（ccjjllt/enveda-039-own-spec，热启动，lr 1.5e-4，seed 1）北京时间 10-26 13:55 开跑，约 7.1h 完成。
- 结果：留出集 isomer MRR 0.7292@fp0.5（基线 0.7606），**无效**。原因：日志 `VAL init (random)` 为 0.145，热启动没生效——数据集 ccjjllt/enveda-own-spec 中 model.pt 位于根目录，而脚本 glob 的是 `**/own_spec/model.pt`，未匹配即静默从头训练（且 lr 只有 1.5e-4，比 039 的 3e-4 低，所以更差）。
- 修复：glob 改为 `/kaggle/input/**/model.pt`，OWN_INIT=1 时找不到权重直接 assert 失败；北京时间 10-27 00:20 重新 push（kernel version 5），预计 ≈07:30 完成。
- 完成后先确认日志 `warm start from` 与 `VAL init`≈0.76，再比较最终 MRR；涨则做 2 模型集成并替换 040 推理权重提交。
