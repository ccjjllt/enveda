# 029a 相对 028a 的补丁清单

源：`notebooks/028/enveda-028a-e1ext.ipynb` → `enveda-029a-joint-dreams.ipynb`

1. **Markdown 头**：说明 joint / DreaMS / GEN_N_ANALOG=20，并写明保持 POP=0.15。
2. **Engine cell**：`GEN_N_ANALOG = 20`；`E.cfg.gen_n_analog = ...`。
3. **E006 factorial `rerank_row`**：`ice and gl` 分支改为 N1 上 joint ICE+GL；audit 增加 `n1_proxy` / `joint_ice_proxy_gl`；报告增加 `n1_challenger`。
4. **kernel-metadata**：`dataset_sources` 增加 `hengck23/hengck23-dreams-enveda-casmi26`。

推送目录：`/workspace/k029`（与私有 kernel `ccjjllt/enveda-020-e006-repro` 对齐，镜像 digest 与 027/028 相同）。
**禁止在 GPU 配额重置前 push。**
