# 061 进展

- 2026-11-09 12:05 北京：记录 060 ref 58142650 公榜 0.481（打平）。代码审读发现 `pxm_merge` 第 2–25 名神经权重硬编码 0.75（CUSTOM w_mid 基本不生效）。
- 构建 `notebooks/061a/enveda-061a-nwmid100.ipynb`（NW_MID 1.00）与 `notebooks/061b/enveda-061b-docprior020.ipynb`（POP_W/SPEC_W 0.20），均基于 060 笔记本（带同轮反事实审计，基线权重 0.10/0.10/0.12/0.75）。
- 2026-11-09 12:03 北京：推 `ccjjllt/enveda-020-e006-repro` **v41**（061a）RUNNING；watcher `/workspace/k061a/run.sh`（日志 watch.log）。
- 061b 推送报 "Maximum batch GPU session count of 2 reached"（另一个项目的 GPU kernel 占用）→ `/workspace/k061b/chain.sh` 等 061a 结束后自动推 `ccjjllt/enveda-061b-docprior020` 并启动 watcher。
- 2026-11-09 ~12:33 北京：v41 COMPLETE（约 30 分钟）。`g60_audit.json` status=ok，先验表 710,701 行、证据 400 分子；可见 smoke 12 个分子第 2–25 名全部改变（diff_full=12），diff_top1=0、diff_top3=3、diff_top5=6（060 为 1/3，重排幅度约翻倍）；与 056 提交文件对比 N_DIFF_TOP1=0。watcher 用 `-k/-v 41` 提交 ref **58151402**（今日剩余 4 次），PENDING。输出 `/workspace/kout_061a/`。
- 2026-11-09 12:36 北京：chain 推 `ccjjllt/enveda-061b-docprior020` **v1**（061b）RUNNING，watcher `/workspace/k061b/run.sh`。
