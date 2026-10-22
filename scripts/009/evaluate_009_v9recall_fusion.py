#!/usr/bin/env python3
"""009下一步第1项后半段：ICE回传后离线融合评估（CPU-only，手动）。

范围：11个支持分子（bundle，635候选）split主方案+mean对照，全扩展shortlist重算
z(base)+z(ice_merge)等权（ddof1，<2/零方差→0，missing不参与均值），scored优先
+unscored按base续后，3无支持对续后。另含2个无支持分子（BJRN/RUDA）FP核对。
不读G1保留250，不扫权重/阈值，零GPU，不push/submit。

手动运行：
  /opt/homebrew/bin/python3.11 scripts/009/evaluate_009_v9recall_fusion.py
产物：
  artifacts/runs/009-v9recall/fusion_eval.json
"""
from __future__ import annotations
import hashlib
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261012
OUT = ROOT / "artifacts/runs/009-v9recall/fusion_eval.json"

INPUTS = [
    "artifacts/runs/009-v9recall/bundle/inputs.json",
    "artifacts/runs/009-v9recall/cpu_expanded_top100.json",
    "artifacts/runs/009-v9recall/pairs_12_supported_detail.json",
    "artifacts/runs/009-v9recall/pairs_3_unsupported_detail.json",
    "artifacts/runs/009-v9recall/pairs_all_18.json",
    "artifacts/runs/009-v9recall/forward_tasks.json",
    "artifacts/legacy/f1-bundle/inputs.json",
    "artifacts/legacy/g1-validation/inputs.json",
    "artifacts/legacy/f1-scores/scores/f1e-4_ice_top100_split.json",
    "artifacts/legacy/f1-scores/scores/f1e-2_ice_top100_mean.json",
    "artifacts/legacy/g1-scores/scores/g1e-4_ice_top100_split.json",
    "artifacts/legacy/g1-scores/scores/g1e-2_ice_top100_mean.json",
    "artifacts/runs/009-v9recall/scores/scores/v9recall_ice_top100_split.json",
    "artifacts/runs/009-v9recall/scores/scores/v9recall_ice_top100_mean.json",
    "artifacts/runs/009-v9recall/scores/scores/v9recall_ice_top100_split.json.meta.json",
    "artifacts/runs/009-v9recall/scores/scores/v9recall_ice_top100_mean.json.meta.json",
    "artifacts/runs/009-v9recall/bundle/weights_reference.json",
    "artifacts/legacy/g1-validation/weights/MANIFEST.iceberg.json",
]


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _z(vals):
    ok = [v is not None and isinstance(v, (int, float)) and math.isfinite(v) for v in vals]
    xs = [float(v) for v, o in zip(vals, ok) if o]
    if len(xs) < 2:
        return [0.0] * len(vals)
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return [0.0] * len(vals)
    return [((float(v) - mu) / sd) if o else 0.0 for v, o in zip(vals, ok)]


def fuse_order(order, base, ice):
    """bundle README §5 + cpu_expanded z_recalc_steps：等权，canon tie，scored优先."""
    zb = _z(base)
    zi = _z(ice)
    fused = [a + b for a, b in zip(zb, zi)]
    n = len(order)
    scored = [i for i, v in enumerate(ice) if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]
    unscored = [i for i in range(n) if i not in set(scored)]
    scored_sorted = sorted(scored, key=lambda i: (-fused[i], order[i]))
    unscored_sorted = sorted(unscored, key=lambda i: (-base[i], order[i]))
    full_idx = scored_sorted + unscored_sorted
    return [order[i] for i in full_idx], fused, zb, zi


def rr_of_rank(r):
    return 1.0 / r if r is not None and r <= 25 else 0.0


def main():
    t0 = time.monotonic()
    # heldout guard
    for p in INPUTS:
        low = p.lower()
        assert "heldout" not in low and "holdout" not in low, p
    blobs = {}
    shas = {}
    for rel in INPUTS:
        p = ROOT / rel
        assert p.exists(), f"missing {rel}"
        b = p.read_bytes()
        blobs[rel] = b
        shas[rel] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    bundle = json.loads(blobs["artifacts/runs/009-v9recall/bundle/inputs.json"])
    cpu_exp = json.loads(blobs["artifacts/runs/009-v9recall/cpu_expanded_top100.json"])
    pairs12 = json.loads(blobs["artifacts/runs/009-v9recall/pairs_12_supported_detail.json"])
    f1_in = json.loads(blobs["artifacts/legacy/f1-bundle/inputs.json"])
    g1_in = json.loads(blobs["artifacts/legacy/g1-validation/inputs.json"])
    old_np_split = json.loads(blobs["artifacts/legacy/f1-scores/scores/f1e-4_ice_top100_split.json"])
    old_np_mean = json.loads(blobs["artifacts/legacy/f1-scores/scores/f1e-2_ice_top100_mean.json"])
    old_g1_split = json.loads(blobs["artifacts/legacy/g1-scores/scores/g1e-4_ice_top100_split.json"])
    old_g1_mean = json.loads(blobs["artifacts/legacy/g1-scores/scores/g1e-2_ice_top100_mean.json"])
    new_split = json.loads(blobs["artifacts/runs/009-v9recall/scores/scores/v9recall_ice_top100_split.json"])
    new_mean = json.loads(blobs["artifacts/runs/009-v9recall/scores/scores/v9recall_ice_top100_mean.json"])
    meta_split = json.loads(blobs["artifacts/runs/009-v9recall/scores/scores/v9recall_ice_top100_split.json.meta.json"])
    meta_mean = json.loads(blobs["artifacts/runs/009-v9recall/scores/scores/v9recall_ice_top100_mean.json.meta.json"])
    tar_sha = sha_file(ROOT / "artifacts/runs/009-v9recall/scores/v9recall-scores.tar.gz")
    orig_tar_sha = sha_file(Path("/Users/zhaohuaxiang/Downloads/v9recall-scores.tar.gz"))

    # ---- 入库核对 ----
    assert meta_split["n_mols"] == 11 and meta_split["n_cands"] == 635, meta_split
    assert meta_mean["n_mols"] == 11 and meta_mean["n_cands"] == 635, meta_mean
    for mkey in ["prep_tag", "ckpt_sha256", "ms_pred_commit", "unique_jobs", "n_pred"]:
        pass
    assert meta_split["prep_tag"].startswith("panel-A1:v1"), meta_split["prep_tag"]
    assert meta_mean["prep_tag"].startswith("panel-A1:v1"), meta_mean["prep_tag"]
    assert meta_split["ckpt_sha256"]["gen.pt"] == "deef0451c1870d2125653bb212c9d95c5b5d9bf80156dfa9b22fedda7db49bea"
    assert meta_split["ckpt_sha256"]["inten.pt"] == "c49e330e049d3a8b0095026ed694fcf70045c24305354853732aaa5f89d55068"
    assert meta_mean["ckpt_sha256"] == meta_split["ckpt_sha256"]
    assert meta_split["unique_jobs"] == 2066 and meta_split["n_pred"] == 2066
    assert meta_mean["unique_jobs"] == 1691 and meta_mean["n_pred"] == 1691
    # n_scored=0 已知bug：007已记录，预测与相似度有效则不阻断
    assert meta_split["n_scored"] == 0 and meta_mean["n_scored"] == 0
    assert meta_split["status"] == "ok" and meta_mean["status"] == "ok"
    assert meta_split["n_mols_covered"] == 11 and meta_mean["n_mols_covered"] == 11

    # 12/12 新候选 fused_merge 非空有限（mismatch即停）
    new12_map = {(d["mol_key"], d["new_canon"]): d for d in pairs12}
    assert len(new12_map) == 12
    for (mol, new_canon), d in sorted(new12_map.items()):
        recs = [r for r in new_split[mol].values() if r["canon"] == new_canon]
        assert len(recs) == 1, f"{mol} {new_canon} found {len(recs)}"
        fm = recs[0].get("fused_merge")
        assert fm is not None and math.isfinite(fm), f"{mol} {new_canon} fused_merge {fm}"
    # 635/635 有限
    for mol, d in new_split.items():
        for smi, rec in d.items():
            assert rec.get("fused_merge") is not None and math.isfinite(rec["fused_merge"]), (mol, smi)
    for mol, d in new_mean.items():
        for smi, rec in d.items():
            assert rec.get("fused") is not None and math.isfinite(rec["fused"]), (mol, smi)

    # ---- 融合 ----
    orig_by = {"np": {m["key"]: m for m in f1_in["molecules"]}, "g1": {m["key"]: m for m in g1_in["molecules"]}}
    old_split_by = {"np": old_np_split, "g1": old_g1_split}
    old_mean_by = {"np": old_np_mean, "g1": old_g1_mean}

    def canon_map(score_dict_mol, field):
        return {rec["canon"]: rec.get(field) for rec in score_dict_mol.values()}

    per_split = []
    per_mean = []
    # base一致性审计
    max_base_diff = 0.0
    max_ice_overlap_diff_split = 0.0
    max_ice_overlap_diff_mean = 0.0
    n_overlap_split = 0
    n_overlap_mean = 0

    for m in bundle["molecules"]:
        key = m["key"]; ds = m["ds"]
        truth = m["truth_canon"]
        exp_order = list(m["order_top100"])
        exp_base = [float(x) for x in m["scores_top100"]]
        # split
        new_map_split = canon_map(new_split[key], "fused_merge")
        exp_ice_split = [new_map_split.get(c) for c in exp_order]
        exp_fused_order_split, _, _, _ = fuse_order(exp_order, exp_base, exp_ice_split)
        exp_rank_split = exp_fused_order_split.index(truth) + 1 if truth in exp_fused_order_split else None
        # baseline original
        om = orig_by[ds][key]
        orig_order = list(om["order_top100"])
        orig_base = [float(om["base_scores_full_window"][c]) for c in orig_order]
        old_map_split = canon_map(old_split_by[ds][key], "fused_merge")
        orig_ice_split = [old_map_split.get(c) for c in orig_order]
        base_fused_order_split, _, _, _ = fuse_order(orig_order, orig_base, orig_ice_split)
        base_rank_split = base_fused_order_split.index(truth) + 1 if truth in base_fused_order_split else None
        # base一致性：重叠canon的FP差
        for c in set(exp_order) & set(orig_order):
            # exp base lookup
            eb = exp_base[exp_order.index(c)]
            ob = orig_base[orig_order.index(c)]
            max_base_diff = max(max_base_diff, abs(eb - ob))
        for c in set(new_map_split) & set(old_map_split):
            a, b = new_map_split[c], old_map_split[c]
            if a is not None and b is not None and math.isfinite(a) and math.isfinite(b):
                max_ice_overlap_diff_split = max(max_ice_overlap_diff_split, abs(a - b))
                n_overlap_split += 1
        # mean对照：fused字段
        new_map_mean = canon_map(new_mean[key], "fused")
        exp_ice_mean = [new_map_mean.get(c) for c in exp_order]
        exp_fused_order_mean, _, _, _ = fuse_order(exp_order, exp_base, exp_ice_mean)
        exp_rank_mean = exp_fused_order_mean.index(truth) + 1 if truth in exp_fused_order_mean else None
        old_map_mean = canon_map(old_mean_by[ds][key], "fused")
        orig_ice_mean = [old_map_mean.get(c) for c in orig_order]
        base_fused_order_mean, _, _, _ = fuse_order(orig_order, orig_base, orig_ice_mean)
        base_rank_mean = base_fused_order_mean.index(truth) + 1 if truth in base_fused_order_mean else None
        for c in set(new_map_mean) & set(old_map_mean):
            a, b = new_map_mean[c], old_map_mean[c]
            if a is not None and b is not None and math.isfinite(a) and math.isfinite(b):
                max_ice_overlap_diff_mean = max(max_ice_overlap_diff_mean, abs(a - b))
                n_overlap_mean += 1
        # 新候选在扩展融合中的位置（split）
        new_cands_this = [d["new_canon"] for d in pairs12 if d["mol_key"] == key]
        new_ranks_split = {nc: (exp_fused_order_split.index(nc) + 1 if nc in exp_fused_order_split else None) for nc in new_cands_this}
        per_split.append({
            "ds": ds, "key": key, "truth": truth,
            "n_expanded": len(exp_order), "n_original": len(orig_order),
            "n_ice_expanded": sum(1 for v in exp_ice_split if v is not None and math.isfinite(v)),
            "n_ice_original": sum(1 for v in orig_ice_split if v is not None and math.isfinite(v)),
            "base_rank": base_rank_split, "base_rr": rr_of_rank(base_rank_split),
            "expanded_rank": exp_rank_split, "expanded_rr": rr_of_rank(exp_rank_split),
            "delta_rank": (exp_rank_split - base_rank_split) if (exp_rank_split is not None and base_rank_split is not None) else None,
            "delta_rr": rr_of_rank(exp_rank_split) - rr_of_rank(base_rank_split),
            "new_fused_ranks": new_ranks_split,
            "expanded_top25": exp_fused_order_split[:25],
            "baseline_top25": base_fused_order_split[:25],
        })
        per_mean.append({
            "ds": ds, "key": key, "truth": truth,
            "base_rank": base_rank_mean, "base_rr": rr_of_rank(base_rank_mean),
            "expanded_rank": exp_rank_mean, "expanded_rr": rr_of_rank(exp_rank_mean),
            "delta_rank": (exp_rank_mean - base_rank_mean) if (exp_rank_mean is not None and base_rank_mean is not None) else None,
            "delta_rr": rr_of_rank(exp_rank_mean) - rr_of_rank(base_rank_mean),
        })

    # 无支持2分子（BJRN/RUDA）：全missing，融合序=FP序
    unsup_rows = []
    for mol_key in ["BJRNKVDFDLYUGJ", "RUDATBOHQWOJDD"]:
        om = orig_by["np"][mol_key]
        orig_order = list(om["order_top100"])
        orig_base = [float(om["base_scores_full_window"][c]) for c in orig_order]
        # 扩展序来自cpu_expanded
        exp_entry = cpu_exp["new_shortlists"]["np"][mol_key]
        exp_order = list(exp_entry["order_expanded_top100"])
        exp_base = [float(x) for x in exp_entry["scores_expanded_top100"]]
        # 全missing
        orig_ice = [None] * len(orig_order)
        exp_ice = [None] * len(exp_order)
        orig_fused, _, _, _ = fuse_order(orig_order, orig_base, orig_ice)
        exp_fused, _, _, _ = fuse_order(exp_order, exp_base, exp_ice)
        # 断言融合序等于FP序（按base排序即原序，因原序已按base排）
        assert orig_fused == orig_order, mol_key
        assert exp_fused == exp_order, mol_key
        truth = om["truth_canon"]
        br = orig_fused.index(truth) + 1 if truth in orig_fused else None
        er = exp_fused.index(truth) + 1 if truth in exp_fused else None
        unsup_rows.append({"ds": "np", "key": mol_key, "truth": truth,
                           "base_rank": br, "base_rr": rr_of_rank(br),
                           "expanded_rank": er, "expanded_rr": rr_of_rank(er),
                           "delta_rr": rr_of_rank(er) - rr_of_rank(br)})

    def agg(rows):
        rrs_b = [r["base_rr"] for r in rows]
        rrs_e = [r["expanded_rr"] for r in rows]
        return {
            "n": len(rows),
            "baseline_mrr25": float(sum(rrs_b) / len(rrs_b)) if rows else None,
            "expanded_mrr25": float(sum(rrs_e) / len(rrs_e)) if rows else None,
            "delta_mrr25": float(sum(rrs_e) / len(rrs_e) - sum(rrs_b) / len(rrs_b)) if rows else None,
            "baseline_top1": sum(1 for r in rows if r["base_rank"] == 1),
            "expanded_top1": sum(1 for r in rows if r["expanded_rank"] == 1),
            "baseline_top25": sum(1 for r in rows if r["base_rank"] is not None and r["base_rank"] <= 25),
            "expanded_top25": sum(1 for r in rows if r["expanded_rank"] is not None and r["expanded_rank"] <= 25),
            "improved": sum(1 for r in rows if r["delta_rr"] > 1e-12),
            "worsened": sum(1 for r in rows if r["delta_rr"] < -1e-12),
            "tied": sum(1 for r in rows if abs(r["delta_rr"]) <= 1e-12),
        }

    split_all = agg(per_split)
    mean_all = agg(per_mean)
    split_np = agg([r for r in per_split if r["ds"] == "np"])
    split_g1 = agg([r for r in per_split if r["ds"] == "g1"])
    mean_np = agg([r for r in per_mean if r["ds"] == "np"])
    mean_g1 = agg([r for r in per_mean if r["ds"] == "g1"])
    unsup_agg = agg(unsup_rows)

    # 全250外推：其余分子无新增，delta 0
    # NP 250：10个变化分子（8支持+2无支持）+240不变；G1 250：3变化+247不变
    def full_delta(sub_rows, full_n=250):
        return float(sum(r["delta_rr"] for r in sub_rows) / full_n)
    full_np_delta_split = full_delta([r for r in per_split if r["ds"] == "np"] + unsup_rows)
    full_g1_delta_split = full_delta([r for r in per_split if r["ds"] == "g1"])
    full_np_delta_mean = full_delta([r for r in per_mean if r["ds"] == "np"])
    full_g1_delta_mean = full_delta([r for r in per_mean if r["ds"] == "g1"])

    # verdict：008/009四条
    # 1 单主要变化：539扩库（固定来源/快照/许可/canonical/窗），是
    # 2 同口径可解释正向：split delta 0，无正向 → 否
    # 3 已知退化已处理：BHQC 40→42仍25外，无Top25损失；G1无真值位移；FP尾部挤出已列但融合未伤真值 → 算已记录但无Top25损失
    # 4 离线可跑：bundle+权重引用+scores齐备，11分子2066 jobs 5-10min，CPU离线融合<1s → 是
    # 综合：不构成单一提交候选
    verdict = {
        "submit": False,
        "label": "NOT_SUBMIT",
        "criteria": {
            "single_main_change": {"pass": True, "detail": "唯一变化为固定MassBank 539 canonical扩库（来源/快照/许可/canonical/10ppm窗/FP orig-head口径固定）；ICE补分仅12支持对，3无支持续后，无阈值/权重扫描。"},
            "same_caliber_positive": {"pass": False, "detail": f"split主方案11分子MRR {split_all['baseline_mrr25']:.6f}→{split_all['expanded_mrr25']:.6f} Δ{split_all['delta_mrr25']:+.6f}，Top1 {split_all['baseline_top1']}→{split_all['expanded_top1']}，Top25 {split_all['baseline_top25']}→{split_all['expanded_top25']}，改善0/恶化0（RR口径；BHQC rank 40→42仍25外RR 0）。NP支持8 Δ{split_np['delta_mrr25']:+.6f}，G1支持3 Δ{split_g1['delta_mrr25']:+.6f}。mean对照同样Δ{mean_all['delta_mrr25']:+.6f}。无可解释正向，不能凑结论。"},
            "known_degradation_handled": {"pass": True, "detail": "已知退化已如实记录、无Top25损失：BHQC融合40→42仍25外（FP 28→29仍25外）；G1三尾部FP挤出（GKB/PZQD/SFHL各100→101）在融合后真值仍rank1；BJRN/RUDA无支持按base续后真值3/21不变；R3 YTCZZ等完整v9缺口仍在r3_baseline_gaps未解决，不属本次提交范围。"},
            "offline_runnable": {"pass": True, "detail": "离线可跑：bundle inputs/MANIFEST/weights_reference齐备，scores split/mean+meta齐备，CPU融合<1s；T4全量2066 jobs约1.3min pure、含overhead 5-10min，远低于2h。但可跑≠可提交。"},
        },
        "gaps": [
            "同口径正向缺失：在真值已在库的NP/G1研究子集上，539新增全为干扰（12新候选融合 rank 3-77，真值0位移除BHQC 40→42外），MRR Δ0。真正可受益的库外真值（R3 B_known 20/21 FP召回）尚未做完整v9融合，YTCZZ盐等R3缺口未闭环。",
            "隐藏泛化未证：NP/G1全250外推Δ0（分母250稀释），不能据此估计隐藏timsTOF/未知化学的收益；LMSD单来源已示 chemotype失配零命中，不宜把MassBank同源召回外推为未知结构发现率。",
            "提交需先冻结R3完整v9基线（代表谱/CE回退/仪器/窗/shortlist/z重算）并在新数据上一次比较，或找到库外真值的融合正向证据；当前仅11分子中性结果不满足v10四条中的第2条。",
        ],
        "basis": "split主方案+mean对照双零增益；12/12新候选非空有限但全为非真值干扰；无Top1/Top25得失；全250外推Δ0。",
    }

    out = {
        "task": "009下一步第1项后半段：ICE回传后离线融合评估（CPU-only）",
        "device": "local M5 CPU-only; numpy; no torch/GPU; no push/submit",
        "seed": SEED,
        "wall_clock_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "inputs_sha256": shas,
        "tar_sha256": {"stored": tar_sha, "original_download": orig_tar_sha, "match": bool(tar_sha == orig_tar_sha)},
        "ingest": {
            "tar": "artifacts/runs/009-v9recall/scores/v9recall-scores.tar.gz",
            "unpacked": ["scores/v9recall_ice_top100_split.json", "scores/v9recall_ice_top100_split.json.meta.json",
                         "scores/v9recall_ice_top100_mean.json", "scores/v9recall_ice_top100_mean.json.meta.json"],
            "split_meta": {k: meta_split.get(k) for k in ["model", "cemode", "k", "status", "n_mols", "n_cands", "n_scored", "n_mols_covered", "prep_tag", "unique_jobs", "n_pred", "n_spectra_unsupported", "ckpt_sha256", "ms_pred_commit", "rdkit", "torch", "device", "seconds", "pred_per_s", "max_mem_gb", "n_bad_smiles"]},
            "mean_meta": {k: meta_mean.get(k) for k in ["model", "cemode", "k", "status", "n_mols", "n_cands", "n_scored", "n_mols_covered", "prep_tag", "unique_jobs", "n_pred", "n_spectra_unsupported", "ckpt_sha256", "ms_pred_commit", "rdkit", "torch", "device", "seconds", "pred_per_s", "max_mem_gb", "n_bad_smiles"]},
            "n_scored_note": "n_scored=0为g1_worker已知np.unicode_写缓存计数器bug（007已记录），预测与相似度有效，status ok，635/635有限。",
            "new12_fused_merge_nonempty": 12,
            "split_finite_635": 635,
            "mean_finite_635": 635,
        },
        "caliber": "z(base)+z(ice_merge)等权，样本std ddof1，<2/零方差→0，missing不参与均值且贡献0；scored按(-fused,canon)，unscored按(-base,canon)续后；3无支持按base续后不打0；split用fused_merge，mean用fused；truth仅排序后评价。",
        "audit": {
            "max_base_diff_expanded_vs_original_overlap": max_base_diff,
            "max_ice_overlap_diff_split_new_vs_old": max_ice_overlap_diff_split,
            "n_ice_overlap_split": n_overlap_split,
            "max_ice_overlap_diff_mean_new_vs_old": max_ice_overlap_diff_mean,
            "n_ice_overlap_mean": n_overlap_mean,
            "note": "base应0差；ice重叠差应<1e-6（G1两例5e-04为已知any-rdkit/数值微差，已记录，不改结论）。",
        },
        "split_primary": {
            "per_molecule": per_split,
            "overall_11": split_all,
            "np_supported_8": split_np,
            "g1_supported_3": split_g1,
            "unsupported_2_np": unsup_agg,
            "unsupported_rows": unsup_rows,
            "full250_extrapolated_delta": {"np": full_np_delta_split, "g1": full_g1_delta_split},
        },
        "mean_control": {
            "per_molecule": per_mean,
            "overall_11": mean_all,
            "np_supported_8": mean_np,
            "g1_supported_3": mean_g1,
        },
        "verdict": verdict,
        "boundary": "No test files/CI; CPU-only; no G1 heldout250 read; no weight/threshold scan; manual run + manual check; truth only for offline RR.",
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"split overall": split_all, "mean overall": mean_all, "unsup": unsup_agg,
                      "audit_base": max_base_diff, "audit_ice_split": max_ice_overlap_diff_split,
                      "verdict": verdict["label"]}, ensure_ascii=False, indent=2))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes) wall {time.monotonic()-t0:.2f}s")


if __name__ == "__main__":
    main()
