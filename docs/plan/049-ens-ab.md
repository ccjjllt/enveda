# 049 计划：A×B 集成（046A sanpier 线 + 自研 model3）

- 动机：047 W_MID 扫描（0.60→0.466、0.90→0.467）已穷尽，046A 0.468 仍最佳；B 线 model3 自身 0.464，与 sanpier 模型误差来源不同，尝试互补。
- 做法：在 046A 流水线（W_TOP=0.25 / leader-safe W_MID=0.75 / 碎片 rank-1 规则）之前插入自研 model3 打分单元，只记录每分子候选的 own rank（不改 submission）；
  sanpier 单元三种融合序（V1 leader、W_MID 序、V2 碎片对照序）统一加项 `W_OWN/(3+own_rank)`，W_OWN=0.20（保守）。
- 失败回退：own 单元任何异常 → OWN_RANK 为空 → 与 046A 完全一致。
- 构建：`cd notebooks/049 && python build.py 0.20`（内部用 047/build.py 0.75 重建 046A 等价 notebook 后改写，并还原 047 产物）；数据源追加 ccjjllt/enveda-own-spec。
- 判定：公榜 >0.468 即新最佳；否则尝试 W_OWN 0.10/0.30 或仅作用于 rank 2-25。
