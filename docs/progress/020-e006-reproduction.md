# 020 — 公开 E006＋PubChem Promo（rel>150，public 0.446）原样复现

日期：2026-10-19（北京时间）。接手 `e2c298e`。用户已批准：原样 fork 为私有 kernel 并提交一次。本轮不改方法。对应[020计划](../plan/020-e006-reproduction.md)。

## 结论

**原样复现成功：v11 public 0.446（ref 57405363），与作者页面一致，比此前最佳 0.323 高 0.123。** 代码未改；四次 push 中三次失败都来自运行环境（源 metadata 锁定的是 CPU 镜像，Kaggle 默认新镜像为 Python 3.13）。最终版本为 v4：锁定 Kaggle GPU 镜像 v170（Python 3.12）。后续以该 fork 为基线。

## Fork 内容

- 来源：[seyitkaangunes/casmi26-e006-pubchem-promo-rel-150-lb-0-446](https://www.kaggle.com/code/seyitkaangunes/casmi26-e006-pubchem-promo-rel-150-lb-0-446) V1（作者页面显示 public 0.446，Apache-2.0）。上游为 antriksharya E006（0.442），唯一改动是 promotion 阈值 `rel > 250` → `rel > 150`。
- 私有 kernel：[ccjjllt/enveda-020-e006-repro](https://www.kaggle.com/code/ccjjllt/enveda-020-e006-repro)。notebook 与源逐字节一致，sha256 `ecc8f7aff23497db9819087c3d4177b23667cdea912487aff702aaa85b66419a`。
- metadata：`is_private=true`、`enable_gpu=true`、`enable_internet=false`、`machine_shape=NvidiaTeslaT4`（T4×2），docker image 沿用源 `gcr.io/kaggle-images/python@sha256:dafd4ce5…40b9`。competition source 为本赛；无 kernel/model source；16 个 dataset source 与源相同：
  prvsiyan/casmi26-ranker-features、prvsiyan/casmi26-fp-models-v2、dmitriigluzdov/casmi26-pubchem-popularity-prior、megayak/casmi26-simulated-ranker-rows、ahmedberatozer/casmi26-{fpnet-full1,glacier,iceberg,pubchem-tier,v2-pool,v3-models,v4b-models}、prvsiyan/chebi-lipidmaps-casmi26、prvsiyan/coconut-casmi26-candidates、hengck23/hengck23-dreams-enveda-casmi26、metric/rdkit-2026-3-3-wheel、eveliaveldrine/rdkit-wheels。
- 交互运行是 notebook 自带 smoke 模式（12 个分子，ICE_BUDGET=300）；只有比赛 rerun 才跑全部隐藏分子。

## 实际运行

| 版本 | push（北京时间） | 结果 |
|---|---|---|
| v1 | 15:15，`--accelerator NvidiaTeslaT4` | QUEUED 约 6 分钟，RUNNING 约 1 分钟，然后 ERROR。日志 32 s 处 E004 preflight 的 `nvidia-smi` 返回 exit 12，`assert 'T4'` 前已失败，即 session 里看不到 GPU。此前 RDKit 2026.03.3 安装、64 个 MANIFEST 文件校验均通过 |
| v2 | 15:23，原样重推（只依赖 metadata 的 GPU 设置） | 长时间 QUEUED 后同样在 GPU 自检失败 |
| v3 | 去掉 metadata 中锁定的 `gcr.io/kaggle-images/python`（CPU 镜像），用默认 GPU 环境 | 通过 GPU 自检，但默认环境为 Python 3.13，notebook 要求 3.12，失败 |
| v4 | 锁定 `gcr.io/kaggle-gpu-images/python@sha256:37c64f7d…`（v170，Python 3.12） | **COMPLETE** |

v1/v2 期间曾怀疑 GPU 配额用尽；用户确认配额尚有，只是排队。原记录：配额（SDK `quota_view`，16:05 查询）：GPU `totalTimeAllowed=21600s`，`timeUsed` 返回值格式异常 `36160.679641.0s`（若按 36160 s 读，已超过 21600 s），下次刷新 2026-10-24 08:00（北京时间）。v1 无 GPU 和 v2 长时间排队都可能与配额用尽有关，但返回值格式异常，未确认。TPU 0/72000 s。

## 提交

- v4 运行 COMPLETE 后，提交一次：ref 57405363，2026-10-20 02:15（北京时间），描述 `v11 020 E006 repro (rel>150)`。
- **public 0.446**，与作者页面一致，比此前最佳 0.323（v9r2 / v10）高 0.123。复现成功。

## 下一步

- 以 0.446 为新基线；后续改动在这个 fork 之上做，且只做原样复现以外的方法改动时再单独评估。
