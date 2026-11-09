# 061 计划：056 底座 + 第 2–25 名大幅旋钮（061a 神经权重 0.75→1.00；061b 文档先验 0.10→0.20），rank 1 不动

## 背景
- A 线最佳仍为公榜 **0.481**：056（ref 58089862）= 059（ref 58124700）= 060（ref 58142650）。
- 改 top-1 均退步（057 0.476、058 0.473）；第 6 名插入（059）与文档先验 0.10→0.12（060）在 3 位小数下中性 → 小步不够，需要更大幅度的第 2–25 名重排才可能看到 ±0.001+。
- MRR@25：第 2 名 0.5、第 3 名 0.33，2–5 名重排仍有真实收益，且不碰 rank 1（已验证安全）。

## 关键发现（代码审读）
- 056 最终 Variant-M 合并（`pxm_merge`）对第 2–25 名用 `1/(3+基线名次) + 0.75/(3+神经名次) + 文档先验` 重新排序，**0.75 是硬编码**；上游 CUSTOM `w_mid` 只影响无证据的分子。
- 因此 liuzewen123 公开的 "casmi26-0482-v2weights"（= 056 + `CUSTOM w_mid 0.50` + fast）基本是空操作，不采纳。
- 真正决定第 2–25 名的旋钮是上述硬编码神经权重（历史上 0.50→0.75 曾带来 +0.004）和文档先验（0.477→0.480）。

## 公开扫描（2026-11-09 12:00 北京）
| 候选 | 结论 |
|---|---|
| heliosli/casmi26-0-482 | 已公开代码，但依赖一个不可见私有数据集（dataset_sources 含空项），busyaprime 0.477 底座 + rr3 同分子式重排；无法复现，不采纳。 |
| liuzewen123/casmi26-0482-v2weights | = 056 + CUSTOM w_mid 0.50 + fast；如上为近似空操作。 |
| ozertuu "0.485"（11-09 再跑） | 同 060 计划所述（extlib rank-1 提升，同 058 机制退步），不采纳。 |
| LB #1 | 0.537（Shehab Alshehabi），前 9 名 ≥0.495。 |

## 方法（两个单旋钮变体，均相对 056）
- **061a**：新增 `NW_MID`，`pxm_merge` 中 `0.75/(3+nr)` → `NW_MID/(3+nr)`，`NW_MID=1.00`（上游 "V4 bold consensus" 权重）；文档先验保持 0.10/0.10/0.12。020 **v41**。
- **061b**：`POP_W/SPEC_W` 0.10→**0.20**（060 的 0.12 中性，加大到 2 倍），`NW_MID=0.75`、`NP_W=0.12` 不变。新私有 kernel `ccjjllt/enveda-061b-docprior020`（v1）。
- 两者均保留 060 的同轮反事实审计（`g60_audit.json`，基线权重扩展为 `(0.10, 0.10, 0.12, 0.75)`）：任何 rank 1 差异即整体回退到 056 合并。
- `COMMIT_MODE='smoke'`、`V8='MASS'`、popwenman v80，元数据同 056/060。

## 运行与提交
- GPU 批量会话上限 2（pump-fun 项目 kernel 占用 1）→ 061a 先跑；`/workspace/k061b/chain.sh` 等 061a watcher 结束后推 061b 并启动其 watcher。
- watcher（`/workspace/k061a/run.sh`、`/workspace/k061b/run.sh`）：COMPLETE → 需 submission.csv + g60_audit.json；status=ok、可见集 top1 与 056 一致、审计 diff_top1=0 才 `-k/-v` 提交。
- 今日（11-09，北京 08:00 重置）上限 5 次，计划用 2 次。

## 预期与后续
- 若某变体 ≥0.482：沿该方向再走一步（NW_MID 1.25 / 文档先验 0.30）并组合两者。
- 若 ≤0.480：反向（NW_MID 0.60 / 先验 0.05）确认方向；若仍持平，A 线第 2–25 名旋钮基本饱和，转向 Engine1/池外召回或 B 线（许可干净）改进。

## 许可
含 sanpier → A 线冲牌。

## 致谢
alex chilton（w_mid 0.75）、ozertuu（文档先验旋钮）、baidalinadilzhan（文档先验层）、popwenman、hoowoo、sanpier、Dmitrii Gluzdov、romanrozen、zz3t3rn41、qwertykeypadapple。
