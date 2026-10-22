#!/usr/bin/env python3
"""018 同分子参考库对照：015 的 500 个查询，只换参考库。

实验一：同一批分子、同一套 FP+ICE 候选、同一个冻结 LR direct 模型，
比较 015 的 stride-50 抽样参考（约 5 万）和 016 的 251 万全库。
结构排除与查询谱 hash 排除沿用 015，不使用 016 面板的代表谱排除。

实验二：在全库特征上，按 015 的 canonical 五折、样本权重和 LR 参数
重训 direct 模型。500 分子报告 OOF；400 面板只用全数据模型做迁移，
该面板不进入训练。

相对特征对照由 --relative 单独打开，定义写死，不扫阈值。
手动 CPU 运行，不提交比赛。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import joblib
import numpy as np
from rdkit import RDLogger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path[:0] = [str(ROOT / "scripts/015"), str(ROOT / "scripts/016")]
from analog_evidence import clean_peaks, neutral_mass, spectral_evidence, tanimoto_to_refs
from build_015_candidate_features import INPUTS, aggregate_analog
from build_016_panel_features import prepare_eligible, select_pool
from build_016_reference_index import load_index, read_peaks
from run_015_candidate_ranker import FOLDS, SEED, rank_within_formula, rr, truth_rank

OUT = ROOT / "artifacts/runs/018-reference-condition"
SPARSE_DIR = ROOT / "artifacts/runs/015-analog/features"
OLD_DIRECT = ROOT / "artifacts/runs/015-analog/direct-only-ablation/models/lr_direct_only_experimental_full.joblib"
PANELS = {
    "full_source": ROOT / "artifacts/runs/017-corrected-features",
    "external": ROOT / "artifacts/runs/017-corrected-features-ext",
}
DIRECT_NAMES = ["fp_z", "ice_z", "ice_present", "v9_reciprocal_rank", "direct_quality"]
RELATIVE_NAMES = ["same_formula_quality_margin", "same_formula_positive_count"]
BASE_NAMES = ["fp_z", "ice_z", "ice_present", "v9_reciprocal_rank",
              "direct_quality", "analog_max_tanimoto", "analog_weighted_tanimoto",
              "analog_max_product", "analog_top5mean_product", "analog_best_spec_quality"]
FEATURE_NAMES = BASE_NAMES + RELATIVE_NAMES
LOGISTIC = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
SCENARIOS = ("B_seen", "A_absent")


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def smiles_map():
    import pyarrow.parquet as pq
    table = pq.read_table(ROOT / "artifacts/runs/005/structure_identity_train.parquet",
                          columns=["raw_key", "canonical_group_key", "output_smiles"])
    raw = table.column("raw_key").to_pylist()
    canon = table.column("canonical_group_key").to_pylist()
    smiles = table.column("output_smiles").to_pylist()
    mapped = {}
    for c, s in zip(canon, smiles):
        if c and s and (c not in mapped or s < mapped[c]):
            mapped[c] = s
    return mapped, len(raw)


def load_015_queries(max_queries):
    manifest = json.loads((SPARSE_DIR / "manifest.json").read_text())
    frozen = json.loads((SPARSE_DIR / "frozen_queries.json").read_text())
    exclusions = json.loads((SPARSE_DIR / "frozen_reference_exclusions.json").read_text())
    if manifest["feature_names"][:10] != BASE_NAMES or manifest["scenarios"] != list(SCENARIOS):
        raise ValueError("015 feature contract changed")
    molecules = {}
    for ds, path in INPUTS.items():
        molecules[ds] = {m["key"]: m for m in json.loads((ROOT / path).read_text())["molecules"]}
    if len(frozen) != len(manifest["queries"]):
        raise ValueError("015 frozen queries and manifest differ in length")
    queries = []
    for meta, fr in zip(manifest["queries"], frozen):
        if fr["case"] != f'{meta["dataset"]}:{meta["key"]}' or fr["dataset"] != meta["dataset"]:
            raise ValueError(("query mismatch", meta["key"], fr["case"]))
        mol = molecules[meta["dataset"]][meta["key"]]
        spectrum = mol["spectra"][fr["selected_spectrum"]]
        if spectrum["adduct"] != fr["adduct"] or spectrum["adduct"][-1:] not in "+-":
            raise ValueError(("adduct", meta["key"]))
        mz, intensity, audit = clean_peaks(np.asarray(spectrum["mz"], float),
                                           np.asarray(spectrum["it"], float),
                                           float(spectrum["prec"]), spectrum["adduct"])
        mass = neutral_mass(spectrum["prec"], spectrum["adduct"])
        if abs(mass - fr["neutral_mass"]) > 1e-6:
            raise ValueError(("mass", meta["key"]))
        queries.append({"mz": mz, "intensity": intensity, "adduct": spectrum["adduct"],
                        "precursor_mz": float(spectrum["prec"]), "neutral_mass": mass,
                        "group": meta["dataset"], "dataset": meta["dataset"], "key": meta["key"],
                        "case": fr["case"], "truth": meta["truth"], "rep_smiles": mol["rep_smiles"],
                        "candidates": meta["candidate_keys"], "formulas": meta["candidate_formulas"],
                        "baseline_order": meta["baseline_order"], "clean_audit": audit})
    if max_queries:
        queries = queries[:max_queries]
    return manifest, queries, set(exclusions["all_query_peak_hashes"]), set(exclusions["all_query_canonical_groups"])


def subset_index(idx, mask):
    return {k: v[mask] for k, v in idx.items()}


def pools_for(queries, idx, qhashes, targets):
    prepared = prepare_eligible(idx, qhashes, targets)
    out = []
    for q in queries:
        chosen = {}
        for scenario in SCENARIOS:
            chosen[scenario] = select_pool(q, scenario, idx, prepared, False)
        out.append(chosen)
    return out


def score_selected(q, positions, idx, shard_cache):
    scores, refs = {}, {}
    for i in positions:
        mz, intensity = read_peaks(shard_cache, int(idx["shard"][i]), int(idx["off"][i]), int(idx["n_clean"][i]))
        ref = {"mz": mz, "intensity": intensity, "adduct": str(idx["adduct"][i]),
               "precursor_mz": float(idx["prec"][i]), "canonical": str(idx["canon"][i]),
               "row": int(idx["row"][i]), "source": str(idx["lib"][i])}
        scores[i] = spectral_evidence(q, ref)
        refs[i] = ref
    return refs, scores


def dq_stats(q, dqs):
    dqs = [float(v) for v in dqs]
    wrong = [i for i, cand in enumerate(q["candidates"]) if cand != q["truth"]]
    truth_pos = q["candidates"].index(q["truth"]) if q["truth"] in q["candidates"] else None
    return {"wrong_candidates": len(wrong),
            "wrong_positive": int(sum(dqs[i] > 0 for i in wrong)),
            "max_wrong_quality": max((dqs[i] for i in wrong), default=None),
            "truth_quality": dqs[truth_pos] if truth_pos is not None else None,
            "truth_in_top50": truth_pos is not None}


def candidate_block(q, pool, refs, scores, smiles, baseline_rows):
    ordered = sorted(pool, key=lambda i: (-scores[i]["quality"], -scores[i]["cosine"], refs[i]["row"]))
    unique, seen = [], set()
    for i in ordered:
        if scores[i]["quality"] <= 0 or refs[i]["canonical"] in seen:
            continue
        unique.append(i)
        seen.add(refs[i]["canonical"])
        if len(unique) == 80:
            break
    qualities = np.asarray([scores[i]["quality"] for i in unique], float)
    ref_smiles = []
    for i in unique:
        smi = smiles.get(refs[i]["canonical"])
        if not smi:
            raise ValueError(("reference without smiles", q["case"], refs[i]["canonical"]))
        ref_smiles.append(smi)
    by_canon = defaultdict(list)
    for i in pool:
        by_canon[refs[i]["canonical"]].append(i)
    dqs, analogs, best_src = [], [], []
    for pos, cand in enumerate(q["candidates"]):
        sims = tanimoto_to_refs(q["rep_smiles"][cand], ref_smiles) if unique else np.array([])
        keep = np.asarray([refs[i]["canonical"] != cand for i in unique], dtype=bool)
        direct_rows = by_canon.get(cand, [])
        dq = max((scores[i]["quality"] for i in direct_rows), default=0.0)
        dqs.append(float(dq))
        analogs.append(aggregate_analog(sims[keep], qualities[keep]))
        if direct_rows:
            best = max(direct_rows, key=lambda i: (scores[i]["quality"], scores[i]["cosine"], -refs[i]["row"]))
            best_src.append(refs[best]["source"])
        else:
            best_src.append(None)
    formulas = q["formulas"]
    rows, margins, counts = [], [], []
    for pos, cand in enumerate(q["candidates"]):
        formula = formulas[pos]
        group = [j for j, value in enumerate(formulas) if formula and value == formula]
        if not group:
            group = [pos]
        others = [dqs[j] for j in group if j != pos]
        margin = dqs[pos] - (max(others) if others else 0.0)
        count = float(sum(dqs[j] > 0 for j in group))
        margins.append(margin)
        counts.append(count)
        base = baseline_rows[pos]
        if abs(float(base[3]) - 1.0 / (pos + 1)) > 1e-5:
            raise ValueError(("baseline rank drift", q["case"], pos))
        rows.append([float(base[0]), float(base[1]), float(base[2]), float(base[3]), dqs[pos],
                     *analogs[pos], margin, count])
    truth_rows = by_canon.get(q["truth"], [])
    best_wrong_source = None
    wrong = [i for i, cand in enumerate(q["candidates"]) if cand != q["truth"]]
    if wrong:
        best_wrong_source = best_src[max(wrong, key=lambda i: (dqs[i], -i))]
    audit = {**dq_stats(q, dqs), "selected": len(pool), "truth_reference_rows": len(truth_rows),
             "best_wrong_source": best_wrong_source,
             "pool_truth_quality": float(max((scores[i]["quality"] for i in truth_rows), default=0.0)),
             "pool_best_other_quality": float(max((scores[i]["quality"] for i in pool if refs[i]["canonical"] != q["truth"]), default=0.0)),
             "pool_positive": int(sum(scores[i]["quality"] > 0 for i in pool)),
             "pool_positive_other_structure": int(sum(scores[i]["quality"] > 0 and refs[i]["canonical"] != q["truth"] for i in pool))}
    return rows, audit


def sparse_baseline(manifest, n_queries):
    with np.load(SPARSE_DIR / "features.npz", allow_pickle=False) as z:
        X = np.asarray(z["X"], float)
        qi = np.asarray(z["query_index"])
        si = np.asarray(z["scenario_index"])
        pos = np.asarray(z["candidate_position"])
    buckets = defaultdict(dict)
    for row, qii, sii, p in zip(range(len(X)), qi, si, pos):
        if int(qii) < n_queries:
            buckets[(int(qii), int(sii))][int(p)] = row
    aligned = {}
    for qii in range(n_queries):
        n = len(manifest["queries"][qii]["candidate_keys"])
        for sii in range(2):
            rows = [buckets[(qii, sii)][p] for p in range(n)]
            aligned[(qii, sii)] = X[rows]
    return X, qi, si, pos, aligned


def displacement(full_pool, sparse_pool, full_idx, sparse_idx, truth, full_refs_canon):
    full_rows = {int(full_idx["row"][i]) for i in full_pool}
    sparse_rows = {int(sparse_idx["row"][i]) for i in sparse_pool}
    full_truth = {int(full_idx["row"][i]) for i in full_pool if str(full_idx["canon"][i]) == truth}
    sparse_truth = {int(sparse_idx["row"][i]) for i in sparse_pool if str(sparse_idx["canon"][i]) == truth}
    return {"sparse_selected": len(sparse_rows), "full_selected": len(full_rows),
            "sparse_kept_in_full": len(sparse_rows & full_rows),
            "sparse_displaced": len(sparse_rows - full_rows),
            "full_new_rows": len(full_rows - sparse_rows),
            "sparse_truth_rows": len(sparse_truth), "full_truth_rows": len(full_truth),
            "truth_rows_displaced": len(sparse_truth - full_truth),
            "full_eligible": full_refs_canon}


def build_features(args, out):
    t0 = time.perf_counter()
    manifest, queries, qhashes, targets = load_015_queries(args.max_queries)
    smiles, identity_rows = smiles_map()
    print(f"queries {len(queries)} hashes {len(qhashes)} targets {len(targets)} identity_rows {identity_rows}", flush=True)
    idx, n_index = load_index()
    idx["row"] = np.asarray(idx["row"], np.int64)
    has_smiles = np.fromiter((str(c) in smiles for c in idx["canon"]), dtype=bool, count=n_index)
    stride = (idx["row"] % 50) == 0
    print(json.dumps({"index": n_index, "with_smiles": int(has_smiles.sum()),
                      "stride50": int(stride.sum()), "stride50_with_smiles": int((stride & has_smiles).sum()),
                      "index_load_s": round(time.perf_counter() - t0, 1)}), flush=True)
    full_idx = subset_index(idx, has_smiles)
    sparse_idx = subset_index(idx, stride & has_smiles)
    del idx
    full_chosen = pools_for(queries, full_idx, qhashes, targets)
    sparse_chosen = pools_for(queries, sparse_idx, qhashes, targets)
    print(f"pools ready {time.perf_counter()-t0:.1f}s", flush=True)
    _, _, _, _, aligned = sparse_baseline(manifest, len(queries))
    shard_cache = {}
    all_rows, audits = [], []
    qindex, sindex, positions = [], [], []
    recon_abs = []
    for qi, q in enumerate(queries):
        full_union, sparse_union = [], []
        for scenario in SCENARIOS:
            full_union.extend(full_chosen[qi][scenario][0])
            sparse_union.extend(sparse_chosen[qi][scenario][0])
        full_refs, full_scores = score_selected(q, sorted(set(full_union)), full_idx, shard_cache)
        sparse_refs, sparse_scores = score_selected(q, sorted(set(sparse_union)), sparse_idx, shard_cache)
        for si, scenario in enumerate(SCENARIOS):
            base = aligned[(qi, si)]
            full_pool, full_eligible = full_chosen[qi][scenario]
            sparse_pool, sparse_eligible = sparse_chosen[qi][scenario]
            rows, audit = candidate_block(q, full_pool, full_refs, full_scores, smiles, base)
            sparse_rows, sparse_audit = candidate_block(q, sparse_pool, sparse_refs, sparse_scores, smiles, base)
            stored_dq = base[:, 4]
            recon_dq = np.asarray([r[4] for r in sparse_rows], float)
            if len(recon_dq) != len(stored_dq):
                raise ValueError(("candidate count", q["case"], scenario))
            delta = np.max(np.abs(recon_dq - stored_dq)) if len(stored_dq) else 0.0
            recon_abs.append(float(delta))
            move = displacement(full_pool, sparse_pool, full_idx, sparse_idx, q["truth"], full_eligible)
            move["sparse_eligible"] = sparse_eligible
            stored = dq_stats(q, stored_dq)
            audits.append({"case": q["case"], "dataset": q["dataset"], "scenario": scenario,
                           "sparse_reconstruction_max_abs_direct": float(delta),
                           "stored_sparse": stored,
                           "reconstructed_sparse": {k: sparse_audit[k] for k in (
                               "max_wrong_quality", "truth_quality", "wrong_positive", "pool_truth_quality",
                               "pool_best_other_quality", "truth_reference_rows")},
                           "full": audit, "displacement": move})
            all_rows.extend(rows)
            qindex.extend([qi] * len(rows))
            sindex.extend([si] * len(rows))
            positions.extend(range(len(rows)))
        if (qi + 1) % 25 == 0 or qi + 1 == len(queries):
            print(f"features {qi+1}/{len(queries)} wall={time.perf_counter()-t0:.0f}s "
                  f"recon_max={max(recon_abs):.4g}", flush=True)
    X = np.asarray(all_rows, np.float32)
    y = []
    for qii, pos in zip(qindex, positions):
        q = queries[qii]
        y.append(q["candidates"][pos] == q["truth"])
    y = np.asarray(y, np.uint8)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "features.npz", X=X, y=y,
                        query_index=np.asarray(qindex, np.int32),
                        scenario_index=np.asarray(sindex, np.uint8),
                        candidate_position=np.asarray(positions, np.int16))
    kept_manifest = {"feature_names": FEATURE_NAMES, "direct_feature_names": DIRECT_NAMES,
                     "relative_feature_names": RELATIVE_NAMES, "scenarios": list(SCENARIOS),
                     "queries": manifest["queries"][:len(queries)],
                     "source_manifest": str(SPARSE_DIR / "manifest.json"),
                     "reference_contract": "015 global query-hash exclusion and 500 canonical removal; smiles from train identity only; full index is 016 float32 cleaned peaks.",
                     "max_queries": args.max_queries}
    save(out / "manifest.json", kept_manifest)
    with (out / "per_query.jsonl").open("w") as handle:
        for row in audits:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {"queries": len(queries), "rows": int(len(X)), "finite": bool(np.isfinite(X).all()),
               "seconds": round(time.perf_counter() - t0, 1),
               "reconstruction_max_abs_direct": float(max(recon_abs) if recon_abs else 0.0),
               "reconstruction_median_query_max_abs_direct": float(np.median(recon_abs) if recon_abs else 0.0),
               "queries_recon_abs_gt_1e-3": int(sum(v > 1e-3 for v in recon_abs)),
               "queries_recon_abs_gt_1e-2": int(sum(v > 1e-2 for v in recon_abs)),
               "complete_feature_build": args.max_queries is None}
    save(out / "feature_summary.json", summary)
    print(json.dumps(summary), flush=True)
    return kept_manifest


def folds_and_weights(queries, qindex, sindex, pos):
    groups = sorted({q["truth"] for q in queries},
                    key=lambda g: hashlib.sha256(f"{SEED}|015|{g}".encode()).hexdigest())
    fold_of = {g: i % FOLDS for i, g in enumerate(groups)}
    row_fold = np.asarray([fold_of[queries[int(i)]["truth"]] for i in qindex], np.int8)
    weights = np.empty(len(qindex), np.float64)
    buckets = defaultdict(list)
    for row, qi, si, p in zip(range(len(qindex)), qindex, sindex, pos):
        buckets[(int(qi), int(si))].append((int(p), row))
    ordered = {}
    for qi, q in enumerate(queries):
        n = len(q["candidate_keys"]) if "candidate_keys" in q else len(q["candidates"])
        keys = q["candidate_keys"] if "candidate_keys" in q else q["candidates"]
        if len(keys) != n:
            raise ValueError("candidate length")
        for si in range(2):
            pairs = sorted(buckets[(qi, si)])
            if [p for p, _ in pairs] != list(range(n)):
                raise ValueError((q.get("key"), si, "positions"))
            rows = np.asarray([r for _, r in pairs], np.int64)
            ordered[(qi, si)] = rows
            weights[rows] = 0.5 / n
    return fold_of, row_fold, weights, ordered


def fit_lr(X, y, rows, weights):
    scaler = StandardScaler().fit(X[rows], sample_weight=weights[rows])
    transformed = scaler.transform(X[rows])
    estimator = LogisticRegression(**LOGISTIC)
    with threadpool_limits(limits=1):
        estimator.fit(transformed, y[rows], sample_weight=weights[rows])
    return {"scaler": scaler, "estimator": estimator,
            "coefficient": estimator.coef_[0].tolist(), "intercept": estimator.intercept_.tolist(),
            "n_iter": estimator.n_iter_.tolist(), "rows": int(len(rows)), "positives": int(y[rows].sum())}


def predict_lr(bundle, X):
    transformed = bundle["scaler"].transform(X)
    with threadpool_limits(limits=1):
        return bundle["estimator"].predict_proba(transformed)[:, 1]


def overtake(order, baseline, truth):
    if truth not in order:
        return {"truth_missing": 1, "ahead": 0, "ahead_from_v9_rank_2_5": 0}
    ranks = []
    for cand in order[:order.index(truth)]:
        if cand in baseline:
            ranks.append(baseline.index(cand) + 1)
    return {"truth_missing": 0, "ahead": len(ranks),
            "ahead_from_v9_rank_2_5": int(sum(2 <= r <= 5 for r in ranks))}


def query_record(q, orders):
    ranks = {name: truth_rank(order, q["truth"]) for name, order in orders.items()}
    rec = {"dataset": q.get("dataset", q.get("group")), "key": q["key"], "truth": q["truth"],
           "reciprocal_ranks": {name: rr(rank) for name, rank in ranks.items()},
           "ranks": ranks}
    rec["overtake"] = {name: overtake(order, q["baseline_order"], q["truth"]) for name, order in orders.items()}
    return rec


def summarize(records, model_names):
    def cell(rows):
        out = {"n": len(rows)}
        for name in model_names:
            values = np.asarray([r["reciprocal_ranks"][name] for r in rows], float)
            out[name] = {"mrr": float(values.mean()) if len(values) else None,
                         "top1": int((values == 1).sum()), "top25": int((values > 0).sum())}
        contrasts = {}
        pairs = [("old_direct_full", "v9"), ("old_direct_sparse", "v9"), ("oof_direct_full", "v9"),
                 ("oof_direct_full", "old_direct_full"), ("old_direct_full", "old_direct_sparse"),
                 ("full_fit_direct", "v9"), ("relative_oof", "oof_direct_full"), ("relative_full", "full_fit_direct")]
        available = set(model_names)
        for after, before in pairs:
            if after not in available or before not in available:
                continue
            a = np.asarray([r["reciprocal_ranks"][after] for r in rows], float)
            b = np.asarray([r["reciprocal_ranks"][before] for r in rows], float)
            delta = a - b
            boot = None
            if len(rows) >= 2:
                from run_015_candidate_ranker import bootstrap
                boot = bootstrap(delta, [r["truth"] for r in rows])
            contrasts[f"{after}_vs_{before}"] = {
                "delta_mrr": float(delta.mean()) if len(delta) else None,
                "improved": int((delta > 1e-12).sum()), "worsened": int((delta < -1e-12).sum()),
                "top1_gained": int(((b < 1) & (a == 1)).sum()), "top1_lost": int(((b == 1) & (a < 1)).sum()),
                "bootstrap": boot}
        overt = {}
        for name in model_names:
            if name == "v9":
                continue
            ahead = [r["overtake"][name]["ahead"] for r in rows]
            band = [r["overtake"][name]["ahead_from_v9_rank_2_5"] for r in rows]
            overt[name] = {"queries_with_someone_ahead_of_truth": int(sum(v > 0 for v in ahead)),
                           "queries_with_v9_rank_2_to_5_ahead": int(sum(v > 0 for v in band)),
                           "v9_rank_2_to_5_candidates_now_ahead": int(sum(band))}
        out["contrasts"] = contrasts
        out["overtake"] = overt
        return out
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["dataset"]].append(rec)
    return {domain: cell(rows) for domain, rows in sorted(grouped.items())}


def distribution_from_audits(path):
    grouped = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            grouped[(row["dataset"], row["scenario"])].append(row)
    report = {}
    for key, rows in sorted(grouped.items()):
        cell = {"n": len(rows)}
        for library, block in (("stored_sparse", "stored_sparse"), ("full", "full")):
            wrong_n = np.asarray([r[block]["wrong_candidates"] for r in rows], float)
            wrong_pos = np.asarray([r[block]["wrong_positive"] for r in rows], float)
            max_wrong = [r[block]["max_wrong_quality"] for r in rows]
            truth_q = [r[block]["truth_quality"] for r in rows]
            gaps, worse = [], 0
            usable = 0
            for tw, tq in zip(max_wrong, truth_q):
                if tq is None or tw is None:
                    continue
                usable += 1
                gaps.append(tq - tw)
                worse += int(tw > tq + 1e-12)
            displaced = np.asarray([r["displacement"]["sparse_displaced"] for r in rows], float)
            selected = np.asarray([r["displacement"]["sparse_selected"] for r in rows], float)
            truth_disp = np.asarray([r["displacement"]["truth_rows_displaced"] for r in rows], float)
            item = {
                "mean_wrong_positive_fraction": float(np.mean(wrong_pos / np.maximum(wrong_n, 1))),
                "queries_with_any_positive_wrong": int(sum(v > 0 for v in wrong_pos)),
                "mean_max_wrong_quality": float(np.mean([0.0 if v is None else v for v in max_wrong])),
                "mean_truth_minus_best_wrong": float(np.mean(gaps)) if gaps else None,
                "queries_best_wrong_exceeds_truth": worse,
                "truth_in_top50": int(sum(r[block]["truth_in_top50"] for r in rows)),
                "usable_gap_queries": usable}
            if "pool_best_other_quality" in rows[0][block]:
                item["mean_pool_best_other_quality"] = float(np.mean([r[block]["pool_best_other_quality"] for r in rows]))
                item["mean_pool_truth_quality"] = float(np.mean([r[block]["pool_truth_quality"] for r in rows]))
            if library == "full":
                item["mean_sparse_displaced"] = float(displaced.mean())
                item["mean_sparse_selected"] = float(selected.mean())
                item["mean_truth_rows_displaced"] = float(truth_disp.mean())
            cell[library] = item
        report[f"{key[0]}__{key[1]}"] = cell
    return report


def score_matrix(queries, X, ordered, predictors):
    records_by_scenario = {s: [] for s in SCENARIOS}
    for qi, q in enumerate(queries):
        view = {"key": q.get("key"), "truth": q["truth"], "dataset": q.get("dataset", q.get("group")),
                "baseline_order": q["baseline_order"],
                "candidate_keys": q["candidate_keys"] if "candidate_keys" in q else q["candidates"],
                "candidate_formulas": q["candidate_formulas"] if "candidate_formulas" in q else q["formulas"]}
        for si, scenario in enumerate(SCENARIOS):
            rows = ordered[(qi, si)]
            block = X[rows]
            orders = {"v9": view["baseline_order"]}
            for name, fn in predictors.items():
                scores = fn(block, view["candidate_formulas"])
                if len(scores) != len(view["candidate_keys"]):
                    raise ValueError((name, q.get("key"), len(scores)))
                orders[name] = rank_within_formula(view, scores)
            records_by_scenario[scenario].append(query_record(view, orders))
    return records_by_scenario


def frozen_predictor(bundle, names):
    index = [FEATURE_NAMES.index(n) if n in FEATURE_NAMES else None for n in names]
    # Old 015 model names are the first five of FEATURE_NAMES, indices 0..4.
    if names == DIRECT_NAMES:
        columns = [0, 1, 2, 3, 4]
    elif names == DIRECT_NAMES + RELATIVE_NAMES:
        columns = [0, 1, 2, 3, 4, 10, 11]
    else:
        raise ValueError(names)
    del index

    def predict(block, formulas):
        use = block
        if names == DIRECT_NAMES + RELATIVE_NAMES and block.shape[1] == 10:
            use = add_relative_columns(block, formulas)
        return predict_lr(bundle, np.asarray(use[:, columns], float))
    return predict


def add_relative_columns(block, formulas):
    dqs = np.asarray(block[:, 4], float)
    margin = np.zeros(len(dqs))
    count = np.zeros(len(dqs))
    groups = defaultdict(list)
    for i, formula in enumerate(formulas):
        groups[formula if formula else f"unknown:{i}"].append(i)
    for members in groups.values():
        values = dqs[members]
        positive = float((values > 0).sum())
        for local, i in enumerate(members):
            others = [values[j] for j in range(len(members)) if j != local]
            margin[i] = values[local] - (max(others) if others else 0.0)
            count[i] = positive
    return np.column_stack([np.asarray(block, float), margin, count])


def load_feature_pack(path):
    manifest = json.loads((path / "manifest.json").read_text())
    with np.load(path / "features.npz", allow_pickle=False) as z:
        pack = {name: z[name] for name in ("X", "y", "query_index", "scenario_index", "candidate_position")}
    return manifest, pack


def panel_queries_and_matrix(path):
    manifest = json.loads((path / "manifest.json").read_text())
    with np.load(path / "features.npz", allow_pickle=False) as z:
        X = np.asarray(z["X"], float)
        qi = np.asarray(z["query_index"])
        si = np.asarray(z["scenario_index"])
        pos = np.asarray(z["candidate_position"])
    queries = manifest["queries"]
    # Score helper expects candidate_keys; 017 manifest already uses that name.
    fold_dummy, _, _, ordered = folds_and_weights(
        [{"truth": q["truth"], "candidate_keys": q["candidate_keys"]} for q in queries], qi, si, pos)
    del fold_dummy
    return queries, X, ordered


def run_models(args, out):
    t0 = time.perf_counter()
    manifest, pack = load_feature_pack(out)
    queries = manifest["queries"]
    # Adapt query records to the fold helper.
    adapted = []
    for q in queries:
        adapted.append({**q, "candidates": q["candidate_keys"]})
    _, row_fold, weights, ordered = folds_and_weights(
        adapted, pack["query_index"], pack["scenario_index"], pack["candidate_position"])
    X = np.asarray(pack["X"], float)
    y = np.asarray(pack["y"])
    direct_x = X[:, :5]
    old = joblib.load(OLD_DIRECT)
    if old["spec"]["feature_names"] != DIRECT_NAMES:
        raise ValueError("frozen direct model features changed")
    oof = np.full(len(X), np.nan)
    fold_info = {}
    for fold in range(FOLDS):
        train = np.flatnonzero(row_fold != fold)
        test = np.flatnonzero(row_fold == fold)
        if len(train) == 0 or len(test) == 0:
            continue
        bundle = fit_lr(direct_x, y, train, weights)
        oof[test] = predict_lr(bundle, direct_x[test])
        fold_info[str(fold)] = {k: bundle[k] for k in ("coefficient", "intercept", "rows", "positives", "n_iter")}
        joblib.dump({"spec": {"feature_names": DIRECT_NAMES}, "scaler": bundle["scaler"],
                     "estimator": bundle["estimator"]}, out / "models" / f"lr_direct_full_fold{fold}.joblib")
    if not np.isfinite(oof).all():
        raise ValueError("OOF coverage incomplete")
    full_bundle = fit_lr(direct_x, y, np.arange(len(X)), weights)
    joblib.dump({"spec": {"feature_names": DIRECT_NAMES}, "scaler": full_bundle["scaler"],
                 "estimator": full_bundle["estimator"], "coefficient": full_bundle["coefficient"],
                 "intercept": full_bundle["intercept"]}, out / "models" / "lr_direct_full_fit.joblib")
    with np.load(SPARSE_DIR / "features.npz", allow_pickle=False) as z:
        sparse_X = np.asarray(z["X"], float)
        sparse_qi = np.asarray(z["query_index"])
    if args.max_queries:
        keep = sparse_qi < len(queries)
        sparse_X = sparse_X[keep]
    if sparse_X.shape[0] != X.shape[0]:
        raise ValueError((sparse_X.shape, X.shape))
    # Baseline columns must be identical; only reference evidence is allowed to change.
    if not np.allclose(sparse_X[:, :4], X[:, :4], atol=1e-5):
        raise ValueError("FP/ICE baseline columns changed between reference libraries")
    sparse_direct = sparse_X[:, :5]
    old_sparse_scores = np.full(len(X), np.nan)
    old_full_scores = np.full(len(X), np.nan)
    for rows in ordered.values():
        old_sparse_scores[rows] = predict_lr(old, sparse_direct[rows])
        old_full_scores[rows] = predict_lr(old, direct_x[rows])
    records = {s: [] for s in SCENARIOS}
    for qi, q in enumerate(adapted):
        view = {"key": q["key"], "truth": q["truth"], "dataset": q["dataset"],
                "baseline_order": q["baseline_order"], "candidate_keys": q["candidate_keys"],
                "candidate_formulas": q["candidate_formulas"]}
        for si, scenario in enumerate(SCENARIOS):
            rows = ordered[(qi, si)]
            orders = {"v9": q["baseline_order"],
                      "old_direct_sparse": rank_within_formula(view, old_sparse_scores[rows]),
                      "old_direct_full": rank_within_formula(view, old_full_scores[rows]),
                      "oof_direct_full": rank_within_formula(view, oof[rows])}
            records[scenario].append(query_record(view, orders))
    by_scenario = {s: summarize(records[s], ["v9", "old_direct_sparse", "old_direct_full", "oof_direct_full"])
                   for s in SCENARIOS}
    panel_report = {}
    if args.max_queries is None:
        full_predictor_bundle = {"scaler": full_bundle["scaler"], "estimator": full_bundle["estimator"]}
        for name, path in PANELS.items():
            panel_q, panel_X, panel_ordered = panel_queries_and_matrix(path)
            panel_records = score_matrix(panel_q, panel_X, panel_ordered, {
                "old_direct_full": lambda block, formulas: predict_lr(old, np.asarray(block[:, :5], float)),
                "full_fit_direct": lambda block, formulas: predict_lr(full_predictor_bundle, np.asarray(block[:, :5], float)),
            })
            panel_report[name] = {s: summarize(panel_records[s], ["v9", "old_direct_full", "full_fit_direct"])
                                  for s in SCENARIOS}
    dist = distribution_from_audits(out / "per_query.jsonl")
    report = {"seconds": round(time.perf_counter() - t0, 1),
              "old_direct_coefficient": old["estimator"].coef_[0].tolist(),
              "full_library_oof_coefficient_by_fold": fold_info,
              "full_library_full_fit": {k: full_bundle[k] for k in ("coefficient", "intercept", "rows", "positives", "n_iter")},
              "same_500": by_scenario, "distribution": dist, "panel_transfer": panel_report,
              "note": "old_direct_sparse applies the 015 full-data model to its own training features. "
                      "oof_direct_full is the retrained model on full-library features. "
                      "panel_transfer trains on all 500 and evaluates the already-used 017 panel; it is not a new blind set."}
    save(out / "model_report.json", report)
    print(json.dumps({"same_500": by_scenario, "distribution_keys": list(dist),
                      "panel": {k: list(v) for k, v in panel_report.items()}}, ensure_ascii=False)[:4000], flush=True)
    return report


def run_relative(args, out):
    t0 = time.perf_counter()
    manifest, pack = load_feature_pack(out)
    if manifest["feature_names"] != FEATURE_NAMES:
        raise ValueError("relative columns missing; rebuild features")
    adapted = [{**q, "candidates": q["candidate_keys"]} for q in manifest["queries"]]
    _, row_fold, weights, ordered = folds_and_weights(
        adapted, pack["query_index"], pack["scenario_index"], pack["candidate_position"])
    X = np.asarray(pack["X"], float)
    y = np.asarray(pack["y"])
    columns = [0, 1, 2, 3, 4, 10, 11]
    rel_x = X[:, columns]
    direct_x = X[:, :5]
    oof_rel = np.full(len(X), np.nan)
    oof_dir = np.full(len(X), np.nan)
    for fold in range(FOLDS):
        train = np.flatnonzero(row_fold != fold)
        test = np.flatnonzero(row_fold == fold)
        rel_bundle = fit_lr(rel_x, y, train, weights)
        dir_bundle = fit_lr(direct_x, y, train, weights)
        oof_rel[test] = predict_lr(rel_bundle, rel_x[test])
        oof_dir[test] = predict_lr(dir_bundle, direct_x[test])
    rel_full = fit_lr(rel_x, y, np.arange(len(X)), weights)
    dir_full = fit_lr(direct_x, y, np.arange(len(X)), weights)
    joblib.dump({"spec": {"feature_names": DIRECT_NAMES + RELATIVE_NAMES},
                 "scaler": rel_full["scaler"], "estimator": rel_full["estimator"],
                 "coefficient": rel_full["coefficient"]},
                out / "models" / "lr_direct_relative_full_fit.joblib")
    records = {s: [] for s in SCENARIOS}
    for qi, q in enumerate(adapted):
        view = {"key": q["key"], "truth": q["truth"], "dataset": q["dataset"],
                "baseline_order": q["baseline_order"], "candidate_keys": q["candidate_keys"],
                "candidate_formulas": q["candidate_formulas"]}
        for si, scenario in enumerate(SCENARIOS):
            rows = ordered[(qi, si)]
            orders = {"v9": q["baseline_order"],
                      "oof_direct_full": rank_within_formula(view, oof_dir[rows]),
                      "relative_oof": rank_within_formula(view, oof_rel[rows])}
            records[scenario].append(query_record(view, orders))
    report = {"seconds": round(time.perf_counter() - t0, 1),
              "relative_definition": "margin = candidate direct_quality minus max other same-formula direct_quality (0 if none); "
                                   "count = number of same-formula shortlist candidates with direct_quality > 0, including self. No threshold search.",
              "relative_full_fit": {k: rel_full[k] for k in ("coefficient", "intercept", "rows", "positives")},
              "direct_full_fit_repeat": {k: dir_full[k] for k in ("coefficient", "intercept")},
              "same_500": {s: summarize(records[s], ["v9", "oof_direct_full", "relative_oof"]) for s in SCENARIOS}}
    if args.max_queries is None:
        rel_bundle = {"scaler": rel_full["scaler"], "estimator": rel_full["estimator"]}
        dir_bundle = {"scaler": dir_full["scaler"], "estimator": dir_full["estimator"]}
        panels = {}
        for name, path in PANELS.items():
            panel_q, panel_X, panel_ordered = panel_queries_and_matrix(path)
            # 017 matrices have 10 columns. Relative columns are computed from direct_quality and formulas.
            panel_records = {s: [] for s in SCENARIOS}
            for qi, q in enumerate(panel_q):
                view = {"key": q["key"], "truth": q["truth"], "dataset": q["group"],
                        "baseline_order": q["baseline_order"], "candidate_keys": q["candidate_keys"],
                        "candidate_formulas": q["candidate_formulas"]}
                for si, scenario in enumerate(SCENARIOS):
                    rows = panel_ordered[(qi, si)]
                    block = panel_X[rows]
                    aug = add_relative_columns(block, q["candidate_formulas"])
                    orders = {"v9": q["baseline_order"],
                              "full_fit_direct": rank_within_formula(view, predict_lr(dir_bundle, block[:, :5])),
                              "relative_full": rank_within_formula(view, predict_lr(rel_bundle, aug[:, columns]))}
                    panel_records[scenario].append(query_record(view, orders))
            panels[name] = {s: summarize(panel_records[s], ["v9", "full_fit_direct", "relative_full"]) for s in SCENARIOS}
        report["panel_transfer"] = panels
    save(out / "relative_report.json", report)
    print(json.dumps(report["same_500"], ensure_ascii=False)[:3000], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--phase", choices=("all", "relative"), default="all")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    out = args.output_dir or (OUT / "probe" if args.max_queries else OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    if args.phase == "all":
        if not (out / "features.npz").exists():
            build_features(args, out)
        run_models(args, out)
    else:
        if not (out / "features.npz").exists():
            raise SystemExit("features missing; run --phase all first")
        run_relative(args, out)


if __name__ == "__main__":
    main()
