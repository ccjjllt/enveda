#!/usr/bin/env python3
"""009 R3-50 完整v9基线融合评估（CPU-only，手动，零GPU）。

口径（数据强制，非开发挑选）：
  R3谱CE全未知→null→sum桶（sims how=merged-sum, ce/ce_bucket=null），无逐CE预测可合并，
  故 fused_merge（split-merge）0/523全空；fused（mean路merged-sum）523/523真实有效。
  本评估如实改用 ICE-mean 路 fused 做 z(base)+z(ice_fused) 等权融合（ddof1，
  scored优先+unscored按base续后）。与NP/G1的fused_merge主方案不同口径，不混为一谈。
  4 NILU与27不支持分子不参与ICE融合（不支持走CPU all-mean FP；4不确定单列排除）。

范围：23支持分子（A16/B7）逐例基线（FP top100内rank/RR）vs融合后rank/RR；
  A16/B7分层MRR/Top1/Top25/增损；YTCZZ verdict；冻结条件规则R3首次可验证数字
  （支持分支本次ICE融合，不支持分支已有CPU分数，4不确定排除并声明）。
不读G1保留250，不扫权重/阈值，不写测试/CI。

手动运行：
  /opt/homebrew/bin/python3.11 scripts/evaluate_009_r3baseline_fusion.py
产物：
  artifacts/009-r3baseline/fusion_eval.json
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

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261012
OUT = ROOT / "artifacts/009-r3baseline/fusion_eval.json"

INPUTS = [
    "artifacts/009-r3baseline/v9r3baseline/inputs.json",
    "artifacts/009-r3baseline/forward_tasks.json",
    "artifacts/009-r3baseline/v9r3baseline/weights_reference.json",
    "artifacts/009-expansion/r3_50_full_candidate_scores.json",
    "artifacts/009-conditional/r3_routing.json",
    "artifacts/007-r3/observations_50.json",
    "artifacts/009-r3baseline/scores/scores/v9r3baseline_ice_top100_split.json",
    "artifacts/009-r3baseline/scores/scores/v9r3baseline_ice_top100_mean.json",
    "artifacts/009-r3baseline/scores/scores/v9r3baseline_ice_top100_split.json.meta.json",
    "artifacts/009-r3baseline/scores/scores/v9r3baseline_ice_top100_mean.json.meta.json",
    "artifacts/g1-validation/weights/MANIFEST.iceberg.json",
]

CALIBER_NOTE = (
    "R3因CE未知无split-merge可用，改用mean路；与NP/G1的fused_merge主方案不同口径，不混为一谈。"
    "融合为z(base)+z(ice_fused)等权，样本std ddof1，<2/零方差→0，missing不参与均值且贡献0；"
    "scored按(-fused,canon)，unscored按(-base,canon)续后；4不确定排除，27不支持不补ICE分从不打0；"
    "truth仅排序后计RR。mean/split选择是数据强制（CE未知→null→sum桶），不是开发挑选。"
)


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
    zb = _z(base)
    zi = _z(ice)
    fused = [a + b for a, b in zip(zb, zi)]
    scored = [i for i, v in enumerate(ice) if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]
    sset = set(scored)
    unscored = [i for i in range(len(order)) if i not in sset]
    scored_sorted = sorted(scored, key=lambda i: (-fused[i], order[i]))
    unscored_sorted = sorted(unscored, key=lambda i: (-base[i], order[i]))
    full_idx = scored_sorted + unscored_sorted
    return [order[i] for i in full_idx], fused, zb, zi


def rr_of_rank(r):
    return 1.0 / r if r is not None and r <= 25 else 0.0


def agg(rows):
    rrs_b = [r["base_rr"] for r in rows]
    rrs_e = [r["fused_rr"] for r in rows]
    return {
        "n": len(rows),
        "baseline_mrr25": float(sum(rrs_b) / len(rrs_b)) if rows else None,
        "fused_mrr25": float(sum(rrs_e) / len(rrs_e)) if rows else None,
        "delta_mrr25": float(sum(rrs_e) / len(rrs_e) - sum(rrs_b) / len(rrs_b)) if rows else None,
        "baseline_top1": sum(1 for r in rows if r["base_rank"] == 1),
        "fused_top1": sum(1 for r in rows if r["fused_rank"] == 1),
        "baseline_top25": sum(1 for r in rows if r["base_rank"] is not None and r["base_rank"] <= 25),
        "fused_top25": sum(1 for r in rows if r["fused_rank"] is not None and r["fused_rank"] <= 25),
        "improved": sum(1 for r in rows if r["delta_rr"] > 1e-12),
        "worsened": sum(1 for r in rows if r["delta_rr"] < -1e-12),
        "tied": sum(1 for r in rows if abs(r["delta_rr"]) <= 1e-12),
    }


def main():
    t0 = time.monotonic()
    for p in INPUTS:
        low = p.lower()
        assert "heldout" not in low and "holdout" not in low, p
        assert "g1_heldout" not in low and "reserve" not in low or "forward_tasks" in low, p
    blobs = {}
    shas = {}
    for rel in INPUTS:
        p = ROOT / rel
        assert p.exists(), f"missing {rel}"
        b = p.read_bytes()
        blobs[rel] = b
        shas[rel] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}

    bundle = json.loads(blobs["artifacts/009-r3baseline/v9r3baseline/inputs.json"])
    fwd = json.loads(blobs["artifacts/009-r3baseline/forward_tasks.json"])
    full = json.loads(blobs["artifacts/009-expansion/r3_50_full_candidate_scores.json"])
    rout = json.loads(blobs["artifacts/009-conditional/r3_routing.json"])
    obs = json.loads(blobs["artifacts/007-r3/observations_50.json"])
    new_split = json.loads(blobs["artifacts/009-r3baseline/scores/scores/v9r3baseline_ice_top100_split.json"])
    new_mean = json.loads(blobs["artifacts/009-r3baseline/scores/scores/v9r3baseline_ice_top100_mean.json"])
    meta_split = json.loads(blobs["artifacts/009-r3baseline/scores/scores/v9r3baseline_ice_top100_split.json.meta.json"])
    meta_mean = json.loads(blobs["artifacts/009-r3baseline/scores/scores/v9r3baseline_ice_top100_mean.json.meta.json"])

    tar_stored = sha_file(ROOT / "artifacts/009-r3baseline/scores/v9r3baseline-scores.tar.gz")
    tar_orig = sha_file(Path("/Users/zhaohuaxiang/Downloads/v9r3baseline-scores.tar.gz"))

    # ---- 入库断言（不一致即停） ----
    assert meta_split["n_mols"] == 23 and meta_split["n_cands"] == 523, meta_split
    assert meta_mean["n_mols"] == 23 and meta_mean["n_cands"] == 523, meta_mean
    assert meta_split["status"] == "ok" and meta_mean["status"] == "ok"
    assert meta_split["prep_tag"].startswith("panel-A1:v1"), meta_split["prep_tag"]
    assert meta_mean["prep_tag"].startswith("panel-A1:v1"), meta_mean["prep_tag"]
    assert meta_split["ckpt_sha256"]["gen.pt"] == "deef0451c1870d2125653bb212c9d95c5b5d9bf80156dfa9b22fedda7db49bea"
    assert meta_split["ckpt_sha256"]["inten.pt"] == "c49e330e049d3a8b0095026ed694fcf70045c24305354853732aaa5f89d55068"
    assert meta_mean["ckpt_sha256"] == meta_split["ckpt_sha256"]
    assert meta_split["n_pred_fail"] == 0 and meta_mean["n_pred_fail"] == 0
    assert meta_split["n_bad_smiles"] == 0 and meta_mean["n_bad_smiles"] == 0
    assert meta_split["n_scored"] == 0 and meta_mean["n_scored"] == 0  # 已知np.unicode_计数器bug
    assert meta_split["n_mols_covered"] == 23 and meta_mean["n_mols_covered"] == 23

    def is_fin(x):
        return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

    tot_split = sum(len(d) for d in new_split.values())
    tot_mean = sum(len(d) for d in new_mean.values())
    assert tot_split == 523, tot_split
    assert tot_mean == 523, tot_mean
    fused_split = sum(1 for d in new_split.values() for r in d.values() if is_fin(r.get("fused")))
    fused_merge_split = sum(1 for d in new_split.values() for r in d.values() if is_fin(r.get("fused_merge")))
    fused_mean = sum(1 for d in new_mean.values() for r in d.values() if is_fin(r.get("fused")))
    # 关键口径断言：fused 523/523非空、fused_merge 0/523
    assert fused_split == 523, f"split fused {fused_split}/523"
    assert fused_merge_split == 0, f"split fused_merge {fused_merge_split}/523应全空"
    assert fused_mean == 523, f"mean fused {fused_mean}/523"
    # CE证据：全null→sum桶
    ce_combos = set()
    for d in new_split.values():
        for r in d.values():
            for s in r.get("sims", []):
                ce_combos.add((s.get("ce"), s.get("ce_bucket"), s.get("how")))
    assert ce_combos == {(None, None, "merged-sum")}, ce_combos
    ce_combos_mean = set()
    for d in new_mean.values():
        for r in d.values():
            for s in r.get("sims", []):
                ce_combos_mean.add((s.get("ce"), s.get("ce_bucket"), s.get("how")))
    assert ce_combos_mean == {(None, None, "merged-sum")}, ce_combos_mean
    # mean vs split fused一致（同sum桶）
    max_fused_diff = 0.0
    for k in new_split:
        for smi in new_split[k]:
            a = new_split[k][smi]["fused"]
            b = new_mean[k][smi]["fused"]
            max_fused_diff = max(max_fused_diff, abs(float(a) - float(b)))
    assert max_fused_diff == 0.0, max_fused_diff
    # base一致性：scores base vs bundle base
    max_base_diff = 0.0
    for m in bundle["molecules"]:
        key = m["key"]
        bmap = m["base_scores_full_window"]
        for smi, rec in new_mean[key].items():
            c = rec["canon"]
            if c in bmap:
                max_base_diff = max(max_base_diff, abs(float(rec["base"]) - float(bmap[c])))
    assert max_base_diff == 0.0, max_base_diff

    # YTCZZ真值核对（任务给定，不符即停前已核实，此处再断言）
    yt_truth_split = None
    yt_base = None
    for smi, rec in new_split["YTCZZXIRLARSET"].items():
        if rec["canon"] == "YTCZZXIRLARSET":
            yt_truth_split = rec
            break
    assert yt_truth_split is not None
    assert abs(float(yt_truth_split["base"]) - (-732.1648328916635)) < 1e-6, yt_truth_split["base"]
    assert abs(float(yt_truth_split["fused"]) - 0.12734447381600378) < 1e-9, yt_truth_split["fused"]
    yt_truth_mean = None
    for smi, rec in new_mean["YTCZZXIRLARSET"].items():
        if rec["canon"] == "YTCZZXIRLARSET":
            yt_truth_mean = rec
            break
    assert abs(float(yt_truth_mean["fused"]) - 0.12734447381600378) < 1e-9

    # ---- 离线融合（ICE-mean路fused） ----
    obs_by = {m["canonical_key"]: m for m in obs["molecules"]}
    # 支持23顺序按bundle（obs定序）
    per = []
    for m in bundle["molecules"]:
        key = m["key"]
        cls = m["class_AB"]
        truth = m["truth_canon"]
        order = list(m["order_top100"])
        base = [float(x) for x in m["scores_top100"]]
        # mean路canon->fused
        cmap = {rec["canon"]: rec.get("fused") for rec in new_mean[key].values()}
        # 断言523内该分子全scored（mean路523/523）
        ice = [cmap.get(c) for c in order]
        n_scored = sum(1 for v in ice if is_fin(v))
        assert n_scored == len(order), f"{key} scored {n_scored}/{len(order)}"
        fused_order, fused_vals, zb, zi = fuse_order(order, base, ice)
        base_rank = order.index(truth) + 1 if truth in order else None
        fused_rank = fused_order.index(truth) + 1 if truth in fused_order else None
        base_rr = rr_of_rank(base_rank)
        fused_rr = rr_of_rank(fused_rank)
        # truth z细节（供YTCZZ）
        ti = order.index(truth) if truth in order else None
        per.append({
            "key": key, "class_AB": cls, "truth": truth,
            "n_cands": len(order),
            "n_ice_scored": n_scored,
            "base_rank": base_rank, "base_rr": base_rr,
            "fused_rank": fused_rank, "fused_rr": fused_rr,
            "delta_rank": (fused_rank - base_rank) if (fused_rank is not None and base_rank is not None) else None,
            "delta_rr": fused_rr - base_rr,
            "truth_base": float(base[ti]) if ti is not None else None,
            "truth_ice_fused": float(ice[ti]) if ti is not None else None,
            "truth_zb": float(zb[ti]) if ti is not None else None,
            "truth_zi": float(zi) if False else float(zi[ti]) if ti is not None else None,
            "fused_top25": fused_order[:25],
            "baseline_top25": order[:25],
        })

    # 分层
    overall = agg([{"base_rank": r["base_rank"], "base_rr": r["base_rr"], "fused_rank": r["fused_rank"], "fused_rr": r["fused_rr"], "delta_rr": r["delta_rr"]} for r in per])
    a_rows = [r for r in per if r["class_AB"] == "A"]
    b_rows = [r for r in per if r["class_AB"] == "B"]
    assert len(a_rows) == 16 and len(b_rows) == 7, (len(a_rows), len(b_rows))
    agg_a = agg([{"base_rank": r["base_rank"], "base_rr": r["base_rr"], "fused_rank": r["fused_rank"], "fused_rr": r["fused_rr"], "delta_rr": r["delta_rr"]} for r in a_rows])
    agg_b = agg([{"base_rank": r["base_rank"], "base_rr": r["base_rr"], "fused_rank": r["fused_rank"], "fused_rr": r["fused_rr"], "delta_rr": r["delta_rr"]} for r in b_rows])
    improved_keys = sorted([r["key"] for r in per if r["delta_rr"] > 1e-12])
    worsened_keys = sorted([r["key"] for r in per if r["delta_rr"] < -1e-12])

    # YTCZZ
    yt = next(r for r in per if r["key"] == "YTCZZXIRLARSET")
    yt_in_top25 = yt["fused_rank"] is not None and yt["fused_rank"] <= 25
    # 断连盐可尝试打分证据：84/84 fused有限，peaks存在
    yt_n = yt["n_cands"]
    yt_scored = yt["n_ice_scored"]

    # ---- 不支持分支已有CPU分数 + 4不确定排除 ----
    nilu = ["NMVVBVMYPLMIOU", "UTDAGHZGKXPRQI", "JOECRAPIMXFQDL", "ZTILAOCGFRDHBH"]
    uns_keys = [r["canonical_key_for_audit_only"] for r in rout["rows"] if not r["metadata_has_supported_record"]]
    assert len(uns_keys) == 27, len(uns_keys)
    for k in nilu:
        assert k in uns_keys, k
        assert len(full[k]["orders"]["expanded"]) == 0, k
    uns_win_keys = [k for k in uns_keys if len(full[k]["orders"]["expanded"]) > 0]
    assert len(uns_win_keys) == 23, len(uns_win_keys)
    uns_rows = []
    for k in uns_win_keys:
        o = list(full[k]["orders"]["expanded"])
        rank = o.index(k) + 1 if k in o else None
        uns_rows.append({"key": k, "class_AB": obs_by[k]["class_AB"], "n_cands": len(o), "fp_rank": rank, "fp_rr": rr_of_rank(rank), "fp_top25": o[:25]})
    # 冻结条件规则首次可验证数字：支持fused + 不支持FP，46可评估，4排除
    # FP-only 46 vs 条件46
    sup_base_rr_sum = sum(r["base_rr"] for r in per)
    sup_fused_rr_sum = sum(r["fused_rr"] for r in per)
    uns_rr_sum = sum(r["fp_rr"] for r in uns_rows)
    fp_only_46_mrr = (sup_base_rr_sum + uns_rr_sum) / 46
    cond_46_mrr = (sup_fused_rr_sum + uns_rr_sum) / 46
    # Top1/Top25
    sup_base_top1 = sum(1 for r in per if r["base_rank"] == 1)
    sup_fused_top1 = sum(1 for r in per if r["fused_rank"] == 1)
    sup_base_top25 = sum(1 for r in per if r["base_rank"] is not None and r["base_rank"] <= 25)
    sup_fused_top25 = sum(1 for r in per if r["fused_rank"] is not None and r["fused_rank"] <= 25)
    uns_top1 = sum(1 for r in uns_rows if r["fp_rank"] == 1)
    uns_top25 = sum(1 for r in uns_rows if r["fp_rank"] is not None and r["fp_rank"] <= 25)
    # A25 / B21
    # A25 = A16 sup + A9 uns
    a_uns = [r for r in uns_rows if r["class_AB"] == "A"]
    b_uns = [r for r in uns_rows if r["class_AB"] == "B"]
    assert len(a_uns) == 9 and len(b_uns) == 14, (len(a_uns), len(b_uns))
    a_sup_base_sum = sum(r["base_rr"] for r in per if r["class_AB"] == "A")
    a_sup_fused_sum = sum(r["fused_rr"] for r in per if r["class_AB"] == "A")
    b_sup_base_sum = sum(r["base_rr"] for r in per if r["class_AB"] == "B")
    b_sup_fused_sum = sum(r["fused_rr"] for r in per if r["class_AB"] == "B")
    a_uns_sum = sum(r["fp_rr"] for r in a_uns)
    b_uns_sum = sum(r["fp_rr"] for r in b_uns)
    a25_fp_only = (a_sup_base_sum + a_uns_sum) / 25
    a25_cond = (a_sup_fused_sum + a_uns_sum) / 25
    b21_fp_only = (b_sup_base_sum + b_uns_sum) / 21
    b21_cond = (b_sup_fused_sum + b_uns_sum) / 21

    verdict = {
        "submit": False,
        "label": "NOT_SUBMIT",
        "criteria": {
            "single_main_change": {
                "pass": True,
                "detail": "唯一变化为支持分支FP→ICE-mean fused（z(base)+z(ice_fused)等权ddof1，scored优先续后），无权重/阈值扫描；4不确定排除、27不支持不补分规则固定。但该变化与NP/G1生产fused_merge主方案不同口径，仅为R3测量增量，不是可直接部署的隐藏提交增量。"
            },
            "same_caliber_positive": {
                "pass": True,
                "detail": f"同口径（R3-mean路）23支持MRR {overall['baseline_mrr25']:.6f}→{overall['fused_mrr25']:.6f} Δ{overall['delta_mrr25']:+.6f}，Top1 {overall['baseline_top1']}→{overall['fused_top1']}，Top25 {overall['baseline_top25']}→{overall['fused_top25']}，改善{overall['improved']}/恶化{overall['worsened']}/持平{overall['tied']}。A16 Δ{agg_a['delta_mrr25']:+.6f}（Top1 {agg_a['baseline_top1']}→{agg_a['fused_top1']}，Top25 {agg_a['baseline_top25']}→{agg_a['fused_top25']}）；B7 Δ{agg_b['delta_mrr25']:+.6f}（Top1 {agg_b['baseline_top1']}→{agg_b['fused_top1']}，Top25 {agg_b['baseline_top25']}→{agg_b['fused_top25']}，YTCZZ 40→6救回，IFEJ 1→2失Top1）。B层MRR为负、小样本23，不能外推隐藏timsTOF增益；正向不直接构成提交理由。",
            },
            "known_degradation_handled": {
                "pass": False,
                "detail": "3例RR恶化未解释：LWFZKXMCJDHBNJ 7→21（RR 0.1429→0.0476仍Top25）、WKADCTRDFMBYPU 1→2（1.0→0.5）、IFEJLMHZNQJGQU 1→2（1.0→0.5）；B21条件MRR 0.774830→0.758957 Δ-0.015873、Top1 14→13。支持层无Top25损失（22→23），条件46 Top25 45→46，但Top1损失机制未查明（是否同式干扰/谱质量/钠盐外候选），按冻结协议需先查因并经新数据（G1保留250待命）一次比较，不能直接提交。",
            },
            "offline_runnable": {
                "pass": True,
                "detail": "离线可跑：bundle inputs/weights_reference/MANIFEST齐备，scores split/mean+meta齐备，CPU融合<1s；T4全量523 jobs pure约20s、含load/overhead 5-10min远低于2h。但可跑≠可提交。",
            },
        },
        "gaps": [
            "口径不可混：R3-mean路（CE未知→sum）与NP/G1 fused_merge主方案不同，R3正向不能直接当生产管线隐藏收益；隐藏为timsTOF、CE已知，与R3未知CE/混合仪器不同域。",
            "B7小样本MRR负向与3例rank恶化机制未解释，需按候选结构/谱质量做固定对照后再议；不得据public总分推断隐藏类别。",
            "提交需先完成冻结条件规则的新数据一次比较（G1保留250待命协议未执行，本轮未读），或找到跨域可迁移的同口径正向证据；当前仅R3-50首次数字不满足直接提交条件。",
        ],
        "basis": f"23支持MRR {overall['baseline_mrr25']:.6f}→{overall['fused_mrr25']:.6f}，Top1 {overall['baseline_top1']}→{overall['fused_top1']}，Top25 {overall['baseline_top25']}→{overall['fused_top25']}；YTCZZ 40→6救回；但B7 Δ{agg_b['delta_mrr25']:+.6f}、3例恶化未解释、口径与隐藏域不同，故NOT_SUBMIT。",
    }

    out = {
        "task": "R3-50 完整v9基线融合评估（CPU-only）",
        "device": "local M5 CPU-only; numpy; no torch/GPU; no push/submit",
        "seed": SEED,
        "wall_clock_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "inputs_sha256": shas,
        "tar_sha256": {"stored": tar_stored, "original_download": tar_orig, "match": bool(tar_stored == tar_orig)},
        "ingest": {
            "tar": "artifacts/009-r3baseline/scores/v9r3baseline-scores.tar.gz",
            "unpacked": [
                "scores/v9r3baseline_ice_top100_split.json",
                "scores/v9r3baseline_ice_top100_split.json.meta.json",
                "scores/v9r3baseline_ice_top100_mean.json",
                "scores/v9r3baseline_ice_top100_mean.json.meta.json",
            ],
            "split_meta": {k: meta_split.get(k) for k in ["model", "cemode", "k", "status", "n_mols", "n_mols_scored", "n_mols_covered", "n_cands", "n_scored", "prep_tag", "unique_jobs", "n_pred", "n_pred_fail", "n_spectra_unsupported", "ckpt_sha256", "ms_pred_commit", "rdkit", "torch", "device", "seconds", "pred_per_s", "max_mem_gb", "n_bad_smiles", "fusion"]},
            "mean_meta": {k: meta_mean.get(k) for k in ["model", "cemode", "k", "status", "n_mols", "n_mols_scored", "n_mols_covered", "n_cands", "n_scored", "prep_tag", "unique_jobs", "n_pred", "n_pred_fail", "n_spectra_unsupported", "ckpt_sha256", "ms_pred_commit", "rdkit", "torch", "device", "seconds", "pred_per_s", "max_mem_gb", "n_bad_smiles", "fusion"]},
            "n_scored_note": "n_scored=0是已知np.unicode_计数器bug（007已记录，meta errors 23条同错），分数有效，status ok，523/523 fused有限。",
            "counts": {"total_split": tot_split, "fused_split": fused_split, "fused_merge_split": fused_merge_split, "total_mean": tot_mean, "fused_mean": fused_mean},
            "ce_evidence": {"split_combos": sorted([str(x) for x in ce_combos]), "mean_combos": sorted([str(x) for x in ce_combos_mean]), "note": "R3谱CE全未知→null→sum桶（how=merged-sum），无逐CE预测可合并，故fused_merge 0/523全空；fused（mean路merged-sum）523/523真实有效。mean/split选择是数据强制，不是开发挑选。"},
            "audit": {"max_base_diff_scores_vs_bundle": max_base_diff, "max_fused_diff_split_vs_mean": max_fused_diff},
            "weights": {"gen.pt": meta_split["ckpt_sha256"]["gen.pt"], "inten.pt": meta_split["ckpt_sha256"]["inten.pt"], "ms_pred_commit": meta_split["ms_pred_commit"], "prep_tag": meta_split["prep_tag"]},
        },
        "caliber": CALIBER_NOTE,
        "supported_23": {
            "per_molecule": per,
            "overall_23": overall,
            "A16": agg_a,
            "B7": agg_b,
            "improved_keys": improved_keys,
            "worsened_keys": worsened_keys,
        },
        "unsupported": {
            "n_unsupported_total": 27,
            "n_with_window": 23,
            "n_excluded_nilU": 4,
            "nilu_keys": nilu,
            "nilu_note": "4 NILU（NMVVBVMYPLMIOU/UTDAGHZGKXPRQI/JOECRAPIMXFQDL/ZTILAOCGFRDHBH）未知adduct质量不确定无窗，0候选单列，不参与ICE融合与MRR分母，另行声明。",
            "rows_23_fp": uns_rows,
            "unsupported_23_fp_mrr25": float(uns_rr_sum / 23) if uns_rows else None,
            "unsupported_23_fp_top1": uns_top1,
            "unsupported_23_fp_top25": uns_top25,
        },
        "conditional_frozen_r3_first_numbers": {
            "note": "冻结条件规则在R3的首次可验证数字：支持分支用本次ICE-mean融合，不支持分支用已有CPU分数（r3_50_full_candidate_scores.json expanded即all-mean FP）；4不确定排除并声明；分母46可评估（A25+B21）。",
            "denominator": {"evaluable_46": 46, "A25": 25, "B21": 21, "excluded_4_nilU": 4},
            "fp_only_46": {"mrr25": fp_only_46_mrr, "top1": int(sup_base_top1 + uns_top1), "top25": int(sup_base_top25 + uns_top25)},
            "conditional_46_fused_sup_plus_fp_unsup": {"mrr25": cond_46_mrr, "delta_mrr25_vs_fp_only": float(cond_46_mrr - fp_only_46_mrr), "top1": int(sup_fused_top1 + uns_top1), "top25": int(sup_fused_top25 + uns_top25)},
            "A25": {"fp_only_mrr25": a25_fp_only, "conditional_mrr25": a25_cond, "delta": float(a25_cond - a25_fp_only)},
            "B21": {"fp_only_mrr25": b21_fp_only, "conditional_mrr25": b21_cond, "delta": float(b21_cond - b21_fp_only)},
        },
        "ytczz": {
            "key": "YTCZZXIRLARSET",
            "class_AB": "B",
            "query": {"adduct": "[M+H]+", "mode": "positive", "instrument": "QTOF", "n_peaks": 49, "covered": True, "note": "query SUPPORTED"},
            "candidate": {"note": "含.[Na+]的断连钠盐（C24H29NaO5），RDKit可解析元素VALID，MAGMA/DAG对断连盐UNVALIDATED；本次84/84 fused有限、peaks存在，作业被ATTEMPT而非跳过，即“ICE对断连钠盐可尝试打分”。"},
            "numbers": {"n_cands": yt["n_cands"], "n_ice_scored": yt_scored, "base": yt["truth_base"], "ice_fused_mean": yt["truth_ice_fused"], "base_rank": yt["base_rank"], "base_rr": yt["base_rr"], "fused_rank": yt["fused_rank"], "fused_rr": yt["fused_rr"]},
            "verdict": f"rank{yt['base_rank']}→{yt['fused_rank']}，{'进入' if yt_in_top25 else '未进入'}Top25（RR {yt['base_rr']}→{yt['fused_rr']}），本次救回；但为单例观测，不保证同类断连盐普遍可救，不把FP rank40移植为ICE先验。",
            "rescued": bool(yt["base_rr"] == 0.0 and yt["fused_rr"] > 0),
        },
        "verdict": verdict,
        "boundary": "No test files/CI; CPU-only zero GPU; no G1 heldout250 read; no weight/threshold scan; truth only for offline RR; manual run + manual check.",
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"overall_23": overall, "A16": agg_a, "B7": agg_b, "ytczz": {"base_rank": yt["base_rank"], "fused_rank": yt["fused_rank"]}, "cond46": {"fp_only": fp_only_46_mrr, "cond": cond_46_mrr}, "verdict": verdict["label"]}, ensure_ascii=False, indent=2))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes) wall {time.monotonic()-t0:.2f}s")


if __name__ == "__main__":
    main()
