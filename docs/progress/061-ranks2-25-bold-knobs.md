# 061 进展

- 2026-11-09 12:05 北京：记录 060 ref 58142650 公榜 0.481（打平）。代码审读发现 `pxm_merge` 第 2–25 名神经权重硬编码 0.75（CUSTOM w_mid 基本不生效）。
- 构建 `notebooks/061a/enveda-061a-nwmid100.ipynb`（NW_MID 1.00）与 `notebooks/061b/enveda-061b-docprior020.ipynb`（POP_W/SPEC_W 0.20），均基于 060 笔记本（带同轮反事实审计，基线权重 0.10/0.10/0.12/0.75）。
- 2026-11-09 12:03 北京：推 `ccjjllt/enveda-020-e006-repro` **v41**（061a）RUNNING；watcher `/workspace/k061a/run.sh`（日志 watch.log）。
- 061b 推送报 "Maximum batch GPU session count of 2 reached"（另一个项目的 GPU kernel 占用）→ `/workspace/k061b/chain.sh` 等 061a 结束后自动推 `ccjjllt/enveda-061b-docprior020` 并启动 watcher。
