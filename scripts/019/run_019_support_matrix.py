#!/usr/bin/env python3
"""019 同式候选的观测峰支持矩阵。手动 CPU，不训练、不提交。

覆盖 017 全库 crossings 的 38 个唯一错误对，并加入直接模型正确抬升、
且加合物可被 012 规则解释、原名次在 2–6、前体质量落在错误例范围内的对照。
断键规则与 019 小样本相同。同位素邻峰和相差整数个氢的同支持峰不重复计为独立证据。

少数支持的预定义来自三例代表，而不是这 38 对的结果：支持候选数 <= max(1, 组大小//4)。
VZL/KU/UVM 三例同时给出留出汇总。只有错误对和对照的少数支持方向都偏向正确结构时，
才把同一个候选分数当作一个排序特征，并只在这些开发 query 上和 v9、direct 比较。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/012"))
sys.path.insert(0, str(ROOT / "src"))
from analyze_012_fragment_pilot import load_frag
from analog_evidence import ADDUCT_NZD, neutral_mass

OUT = ROOT / "artifacts/runs/019-research"
PILOT = {"VZLCYTHOFMBCOV", "KUJPBAAEUZXUIJ", "UVMJRKUPBRAHRO"}
C13 = 1.003354835
H = 1.00782503207
LINK_TOL = 0.002


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def clean_mask(mz, it, prec, adduct):
    mask = it >= it.max() * 0.002
    if mask.sum() > 256:
        indices = np.flatnonzero(mask)
        keep = indices[np.argsort(-it[indices], kind="stable")[:256]]
        mask = np.zeros(len(mz), dtype=bool)
        mask[keep] = True
    mask &= np.abs(mz - prec) > 1.5
    if adduct in ADDUCT_NZD:
        n, z, delta = ADDUCT_NZD[adduct]
        if n > 1:
            mask &= np.abs(mz - (neutral_mass(prec, adduct) + delta) / z) > 1.5
    return mask


def explain(frag, smi, mz, adduct, cache):
    levels = frag.candidate_levels(smi, cache=cache)
    if levels is None:
        return None
    cfg = frag.DEFAULT_CFG
    shifts = np.array([c + dh * frag.H for c in frag.CARRIERS[adduct] for dh in cfg["h_shifts"]])
    target = (mz[None, :] - shifts[:, None]).ravel()
    result = np.full(len(mz), -1, dtype=int)
    for k, arr in enumerate(levels):
        if not len(arr):
            continue
        hit = (np.searchsorted(arr, target + cfg["tol"], "right")
               > np.searchsorted(arr, target - cfg["tol"], "left")).reshape(len(shifts), len(mz)).any(0)
        result[(result < 0) & hit] = k
    return result


def related(dmz):
    for k in (1, 2):
        if abs(dmz - k * C13) <= LINK_TOL or abs(dmz - k * H) <= LINK_TOL:
            return True
    return abs(dmz - 3 * H) <= LINK_TOL


def isotope_satellite(mz, it, support, retained):
    """Heavier retained peak whose supporters are inside a stronger lighter C13 parent."""
    satellite = np.zeros(len(mz), dtype=bool)
    parent = np.full(len(mz), -1, dtype=int)
    kept = np.flatnonzero(retained)
    for i in kept:
        best, best_it = None, -1.0
        for j in kept:
            dmz = mz[i] - mz[j]
            if dmz <= 0 or not (abs(dmz - C13) <= LINK_TOL or abs(dmz - 2 * C13) <= LINK_TOL):
                continue
            if it[j] < it[i] or not support[i] <= support[j]:
                continue
            if it[j] > best_it:
                best, best_it = int(j), float(it[j])
        if best is not None:
            satellite[i] = True
            parent[i] = best
    return satellite, parent


def independent_count(mz, members):
    members = list(members)
    n = len(members)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in range(n):
        for b in range(a + 1, n):
            if related(abs(mz[members[a]] - mz[members[b]])):
                ra, rb = find(a), find(b)
                parent[rb] = ra
    return len({find(i) for i in range(n)})


def spectrum_of(molecule):
    spectrum = max(molecule["spectra"], key=lambda s: len(s["mz"]))
    mz = np.asarray(spectrum["mz"], float)
    it = np.asarray(spectrum["it"], float)
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0) & (mz > 0)
    return spectrum, mz[ok], it[ok]


def build_matrix(frag, molecule, manifest, cache):
    spectrum, mz, it = spectrum_of(molecule)
    adduct = spectrum["adduct"]
    if adduct not in frag.CARRIERS:
        return {"status": "no_explicit_carrier", "adduct": adduct}
    formula = molecule["formulas"][molecule["key"]]
    candidates = [c for c in manifest["candidate_keys"] if molecule["formulas"].get(c) == formula]
    smiles = []
    parsable = []
    for cand in candidates:
        smi = molecule["rep_smiles"].get(cand)
        if not smi:
            parsable.append(False)
            smiles.append(None)
            continue
        smiles.append(smi)
        parsable.append(True)
    levels = []
    for cand, smi, ok in zip(candidates, smiles, parsable):
        levels.append(explain(frag, smi, mz, adduct, cache) if ok else None)
    failed = [c for c, lv in zip(candidates, levels) if lv is None]
    keep = [i for i, lv in enumerate(levels) if lv is not None]
    if molecule["key"] not in [candidates[i] for i in keep]:
        return {"status": "truth_unparsed", "adduct": adduct, "failed": failed}
    candidates = [candidates[i] for i in keep]
    mat = np.vstack([levels[i] for i in keep])
    prec = float(spectrum["prec"])
    retained = clean_mask(mz, it, prec, adduct)
    support = [frozenset(j for j in range(len(candidates)) if mat[j, i] >= 0) for i in range(len(mz))]
    satellite, parent = isotope_satellite(mz, it, support, retained)
    return {"status": "ok", "adduct": adduct, "mode": spectrum["mode"], "precursor_mz": prec,
            "ce_ev": spectrum.get("ce_ev"), "ce_orig": spectrum.get("ce_orig"),
            "formula": formula, "candidates": candidates, "mz": mz, "it": it, "retained": retained,
            "mat": mat, "support": support, "satellite": satellite, "isotope_parent": parent,
            "failed": failed, "baseline_order": manifest["baseline_order"]}


def minority_limit(group_size):
    return max(1, group_size // 4)


def pair_record(matrix, other, role, meta):
    candidates = matrix["candidates"]
    if other not in candidates:
        return {**meta, "role": role, "other": other, "status": "other_not_in_formula_group",
                "group_size": len(candidates)}
    ti = candidates.index(matrix["truth_key"])
    oi = candidates.index(other)
    mz, it = matrix["mz"], matrix["it"]
    retained, satellite = matrix["retained"], matrix["satellite"]
    support = matrix["support"]
    total = float(it.sum())
    n = len(candidates)
    limit = minority_limit(n)
    truth_set, other_set = frozenset([ti]), frozenset([oi])

    def frac(mask):
        return float(it[mask].sum() / total) if total else 0.0

    truth_only = np.array([(ti in s and oi not in s) for s in support], dtype=bool)
    other_only = np.array([(oi in s and ti not in s) for s in support], dtype=bool)
    minority = np.array([0 < len(s) <= limit for s in support], dtype=bool)
    independent = retained & ~satellite
    # Combinations: same support set, at least two peaks that are not C13/H neighbors.
    groups = defaultdict(list)
    for i in np.flatnonzero(independent):
        groups[support[i]].append(int(i))
    combos = []
    for supp, members in groups.items():
        if not supp or len(supp) > limit:
            continue
        n_ind = independent_count(mz, members)
        if n_ind < 2:
            continue
        intensity = frac(np.isin(np.arange(len(mz)), members) | np.array(
            [satellite[i] and matrix["isotope_parent"][i] in members for i in range(len(mz))], dtype=bool))
        combos.append({"support_count": len(supp), "independent_peaks": n_ind,
                       "raw_fraction": intensity,
                       "contains_truth": ti in supp, "contains_other": oi in supp,
                       "mz": [float(mz[i]) for i in members[:6]]})
    combo_truth = sum(c["raw_fraction"] for c in combos if c["contains_truth"] and not c["contains_other"])
    combo_other = sum(c["raw_fraction"] for c in combos if c["contains_other"] and not c["contains_truth"])
    # Candidate score uses only information available for every candidate.
    score = []
    for j in range(n):
        mask = independent & minority & np.array([j in s for s in support], dtype=bool)
        score.append(frac(mask))
    return {**meta, "role": role, "other": other, "status": "ok", "group_size": n,
            "minority_limit": limit, "retained_fraction": frac(retained),
            "truth_not_other_retained": frac(truth_only & retained),
            "other_not_truth_retained": frac(other_only & retained),
            "truth_not_other_minority": frac(truth_only & independent & minority),
            "other_not_truth_minority": frac(other_only & independent & minority),
            "shared_minority": frac(np.array([(ti in s and oi in s) for s in support], bool) & independent & minority),
            "minority_truth_peaks": int((truth_only & independent & minority).sum()),
            "minority_other_peaks": int((other_only & independent & minority).sum()),
            "combo_truth_not_other": combo_truth, "combo_other_not_truth": combo_other,
            "n_minority_combos": len(combos),
            "truth_score": score[ti], "other_score": score[oi],
            "feature_prefers_truth": bool(score[ti] > score[oi] + 1e-12),
            "feature_tie": bool(abs(score[ti] - score[oi]) <= 1e-12),
            "top_truth_only": peak_rows(mz, it, support, candidates, truth_only & independent, 5),
            "top_other_only": peak_rows(mz, it, support, candidates, other_only & independent, 5)}


def peak_rows(mz, it, support, candidates, mask, k):
    total = float(it.sum()) or 1.0
    rows = []
    for i in np.argsort(-it):
        if not mask[i]:
            continue
        rows.append({"mz": float(mz[i]), "raw_fraction": float(it[i] / total),
                     "support_count": len(support[i])})
        if len(rows) >= k:
            break
    return rows


def direction(rows, key_pos, key_neg):
    if not rows:
        return {"n": 0}
    pos = np.array([r[key_pos] for r in rows], float)
    neg = np.array([r[key_neg] for r in rows], float)
    delta = pos - neg
    return {"n": len(rows), "median_delta": float(np.median(delta)),
            "mean_delta": float(delta.mean()),
            "truth_ahead": int((delta > 1e-12).sum()),
            "other_ahead": int((delta < -1e-12).sum()),
            "tie": int((np.abs(delta) <= 1e-12).sum()),
            "median_truth_side": float(np.median(pos)),
            "median_other_side": float(np.median(neg))}


def feature_rule(error_rows, control_rows):
    """Both sides must point at the correct structure. Ties do not count as support."""
    def ok(rows, pos, neg):
        if len(rows) < 4:
            return False
        delta = np.array([r[pos] - r[neg] for r in rows], float)
        return float(np.median(delta)) > 0 and int((delta > 1e-12).sum()) >= int(np.ceil(0.5 * len(rows)))
    return ok(error_rows, "truth_not_other_minority", "other_not_truth_minority") and ok(
        control_rows, "truth_not_other_minority", "other_not_truth_minority")


def rank_of(order, key):
    return order.index(key) + 1 if key in order else None


def main():
    t0 = time.perf_counter()
    frag = load_frag()
    inputs = {m["key"]: m for m in json.loads((ROOT / "artifacts/runs/017-panel-inputs/inputs.json").read_text())["molecules"]}
    manifest = {q["key"]: q for q in json.loads((ROOT / "artifacts/runs/017-corrected-features/manifest.json").read_text())["queries"]}
    details = json.loads((ROOT / "artifacts/runs/017-corrected-features/predict_metrics_details.json").read_text())
    crossings = json.loads((ROOT / "artifacts/runs/017-review/corrected-full-crossings.json").read_text())
    pairs = []
    seen = set()
    for row in crossings:
        key = (row["case"], row["candidate"])
        if key in seen:
            continue
        seen.add(key)
        pairs.append(row)
    error_queries = {row["case"] for row in pairs}
    precs = []
    for key in error_queries:
        _, _, it_dummy = None, None, None
        spectrum, _, _ = spectrum_of(inputs[key])
        precs.append(float(spectrum["prec"]))
    prec_lo, prec_hi = min(precs) - 50, max(precs) + 50
    controls = []
    used = set()
    for row in details:
        v9_rank, direct_rank = row["ranks"]["v9"], row["ranks"]["lr_direct"]
        if row["case"] in error_queries or row["case"] in used:
            continue
        if not (v9_rank and direct_rank and direct_rank < v9_rank and 2 <= v9_rank <= 6):
            continue
        spectrum, _, _ = spectrum_of(inputs[row["case"]])
        if spectrum["adduct"] not in {"[M+H]+", "[M-H]-"}:
            continue
        prec = float(spectrum["prec"])
        if not prec_lo <= prec <= prec_hi:
            continue
        order = manifest[row["case"]]["baseline_order"]
        formula = inputs[row["case"]]["formulas"][row["case"]]
        truth_at = order.index(row["case"])
        ahead = [c for c in order[:truth_at] if inputs[row["case"]]["formulas"].get(c) == formula
                 and c in manifest[row["case"]]["candidate_keys"]]
        if not ahead:
            continue
        used.add(row["case"])
        controls.append({"case": row["case"], "other": ahead[-1], "group": row["group"],
                         "scenario": row["scenario"], "v9_rank": v9_rank, "direct_rank": direct_rank,
                         "direct_order": row["orders"]["lr_direct"]})
    cache = {}
    matrices = {}
    for key in list(error_queries) + [c["case"] for c in controls]:
        built = build_matrix(frag, inputs[key], manifest[key], cache)
        built["truth_key"] = key
        matrices[key] = built
    # Reproduction of the three published full-group counts.
    checks = []
    expected = {"VZLCYTHOFMBCOV": (103.0549, 5, 28), "KUJPBAAEUZXUIJ": (84.0809, 28, 50)}
    for key, (target, count, group) in expected.items():
        matrix = matrices[key]
        i = int(np.argmin(np.abs(matrix["mz"] - target)))
        checks.append({"case": key, "mz": float(matrix["mz"][i]), "support": len(matrix["support"][i]),
                       "expected_support": count, "group": len(matrix["candidates"]), "expected_group": group,
                       "match": abs(matrix["mz"][i] - target) < 0.002 and len(matrix["support"][i]) == count
                       and len(matrix["candidates"]) == group})
    records = []
    for row in pairs:
        matrix = matrices[row["case"]]
        meta = {"case": row["case"], "group": row["group"], "scenarios": sorted({
            r["scenario"] for r in crossings if r["case"] == row["case"] and r["candidate"] == row["candidate"]}),
                "v9_truth_rank": row["truth_ranks"]["v9"], "v9_other_rank": row["candidate_v9_rank"],
                "pilot_query": row["case"] in PILOT}
        if matrix["status"] != "ok":
            records.append({**meta, "role": "error", "other": row["candidate"], "status": matrix["status"]})
            continue
        records.append(pair_record(matrix, row["candidate"], "error", meta))
    for row in controls:
        matrix = matrices[row["case"]]
        meta = {"case": row["case"], "group": row["group"], "scenarios": [row["scenario"]],
                "v9_truth_rank": row["v9_rank"], "direct_rank": row["direct_rank"], "pilot_query": False}
        if matrix["status"] != "ok":
            records.append({**meta, "role": "control", "other": row["other"], "status": matrix["status"]})
            continue
        rec = pair_record(matrix, row["other"], "control", meta)
        if rec["status"] == "ok":
            rec["direct_truth_rank"] = rank_of(row["direct_order"], row["case"])
            rec["direct_other_rank"] = rank_of(row["direct_order"], row["other"])
        records.append(rec)
    ok_errors = [r for r in records if r["role"] == "error" and r["status"] == "ok"]
    ok_controls = [r for r in records if r["role"] == "control" and r["status"] == "ok"]
    heldout = [r for r in ok_errors if not r["pilot_query"]]
    use_feature = feature_rule(ok_errors, ok_controls)
    # Within-formula diagnostic only if the pre-specified direction test passes.
    diagnostic = None
    if use_feature:
        diagnostic = []
        for key, matrix in matrices.items():
            if matrix["status"] != "ok":
                continue
            order = matrix["baseline_order"]
            detail = next(r for r in details if r["case"] == key)
            scores = {}
            n = len(matrix["candidates"])
            limit = minority_limit(n)
            total = float(matrix["it"].sum()) or 1.0
            independent = matrix["retained"] & ~matrix["satellite"]
            for j, cand in enumerate(matrix["candidates"]):
                mask = independent & np.array([len(s) <= limit and j in s for s in matrix["support"]], bool)
                scores[cand] = float(matrix["it"][mask].sum() / total)
            group = defaultdict(list)
            for pos, cand in enumerate(order):
                formula = inputs[key]["formulas"].get(cand)
                group[formula or "unknown:" + cand].append(pos)
            new_order = list(order)
            for positions in group.values():
                ranked = sorted(positions, key=lambda p: (-scores.get(order[p], -1.0), p))
                for slot, src in zip(positions, ranked):
                    new_order[slot] = order[src]
            diagnostic.append({"case": key, "v9": rank_of(order, key),
                               "direct": detail["ranks"]["lr_direct"],
                               "feature": rank_of(new_order, key)})
    summary = {
        "seconds": round(time.perf_counter() - t0, 2),
        "unique_error_pairs": len(pairs),
        "error_queries": len(error_queries),
        "controls": [{"case": c["case"], "other": c["other"], "scenario": c["scenario"],
                      "v9_rank": c["v9_rank"], "direct_rank": c["direct_rank"]} for c in controls],
        "control_rule": "direct rank improves on v9; query not in the 28 error queries; adduct [M+H]+ or [M-H]-; "
                        "v9 truth rank 2-6; precursor within 50 Da of the error precursor range; "
                        "comparison candidate is the same-formula candidate immediately ahead in v9.",
        "minority_rule": "support count <= max(1, formula_group//4), fixed from the 5/28 versus 28/50 pilot contrast",
        "isotope_rule": "a heavier retained peak within 0.002 Da of +1 or +2 C13, no stronger than the lighter peak, "
                        "and with a subset support set, is not a second ion. Peaks in one support set within 0.002 Da "
                        "of an integer H or C13 spacing count as one ion.",
        "reproduction": checks,
        "error_exclusive": direction(ok_errors, "truth_not_other_retained", "other_not_truth_retained"),
        "error_minority": direction(ok_errors, "truth_not_other_minority", "other_not_truth_minority"),
        "error_combo": direction(ok_errors, "combo_truth_not_other", "combo_other_not_truth"),
        "error_minority_without_pilot": direction(heldout, "truth_not_other_minority", "other_not_truth_minority"),
        "control_minority": direction(ok_controls, "truth_not_other_minority", "other_not_truth_minority"),
        "control_combo": direction(ok_controls, "combo_truth_not_other", "combo_other_not_truth"),
        "feature_direction_test_passed": use_feature,
        "feature_pair_preference": {
            "error_truth": int(sum(r["feature_prefers_truth"] for r in ok_errors)),
            "error_tie": int(sum(r["feature_tie"] for r in ok_errors)),
            "error_other": int(sum(not r["feature_prefers_truth"] and not r["feature_tie"] for r in ok_errors)),
            "control_truth": int(sum(r["feature_prefers_truth"] for r in ok_controls)),
            "control_tie": int(sum(r["feature_tie"] for r in ok_controls)),
            "control_other": int(sum(not r["feature_prefers_truth"] and not r["feature_tie"] for r in ok_controls)),
        },
        "indistinguishable_error_pairs": [
            {"case": r["case"], "other": r["other"], "group_size": r["group_size"]}
            for r in ok_errors if r["truth_not_other_retained"] <= 1e-12 and r["other_not_truth_retained"] <= 1e-12],
        "queries_enumerated": len(cache),
        "rank_diagnostic": diagnostic,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "support-matrix-summary.json", summary)
    with (OUT / "support-matrix-pairs.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "seconds", "unique_error_pairs", "error_queries", "reproduction", "error_minority",
        "error_minority_without_pilot", "error_combo", "control_minority", "control_combo",
        "feature_direction_test_passed", "feature_pair_preference", "indistinguishable_error_pairs")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
