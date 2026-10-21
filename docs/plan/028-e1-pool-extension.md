# 028 计划：把 PubChemLite 与 ChEBI/LIPID MAPS 候选并入 Engine1 候选池（基于 v465）

- 现状：v465 只把 PubChemLite 和 ChEBI/LMSD 结构放进 Engine2 的池子；它们只能通过 Engine2 的前 40 名进入模拟器的输入（ICE_UNION）。
- 改动 028a：在 cell 16（pool symlink）之后插入一个 cell。
  1. 选出不在 v2-pool 中、且中性质量落在测试母离子（按加合物换算）15 ppm 以内的新结构。
  2. 为它们计算 Engine1 的 FPNet 位（`chem.raw_fingerprint` 加上 `fp_bits` 选位）和碎片质量（`frag.fragments_for_smiles`，总预算 1500 s，超出预算的结构不算碎片）。
  3. 以 src=4 并入 `work/` 下的池文件，按质量重新排序。
  4. 在 cell 19 把 POOL_POP 补零，并按同一顺序重排。
  - 任何失败都回退到原来的池。
- 效果：Engine1 会给这些结构打分、排序，进入 base 列表后，模拟器也能看到它们。
- 028b（可选）：把 N1A 扩展到更多加合物，等 GPU 时间允许再做。
- 每个变体只提交一次。
