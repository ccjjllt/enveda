"""Manual CPU audit: compare base vs expanded pools under matched scoring rules.

Reuses saved FP logits/ICE scores, never trains or runs a neural model. Original
009 artifacts remain unchanged. R3 cached mean-FP, explicit sum-CE fallback and
literal v9 representative-FP/merge-only behavior are reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
SEED = 20261014
DRAWS = 20_000
SOURCES = {}


def read(rel):
    p = ROOT / rel
    b = p.read_bytes()
    SOURCES[rel] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    return json.loads(b)


def finite(v):
    return v is not None and math.isfinite(float(v))


def zscore(values):
    a = np.asarray([float(v) if finite(v) else np.nan for v in values])
    ok = np.isfinite(a)
    z = np.zeros(len(a))
    if ok.sum() > 1 and a[ok].std(ddof=1) > 0:
        z[ok] = (a[ok] - a[ok].mean()) / a[ok].std(ddof=1)
    return z


def fuse(order, scores, ice):
    """Native v9 tie rule is original shortlist position, not canonical key."""
    base = [scores[c] for c in order]
    iv = [ice.get(c) for c in order]
    total = zscore(base) + zscore(iv)
    scored = [i for i, v in enumerate(iv) if finite(v)]
    unscored = [i for i, v in enumerate(iv) if not finite(v)]
    ix = sorted(scored, key=lambda i: (-total[i], i))
    ix += sorted(unscored, key=lambda i: (-base[i], i))
    return [order[i] for i in ix]


def fp_order(scores):
    return sorted(scores, key=lambda c: (-scores[c], c))[:100]


def rank(order, truth):
    return order.index(truth) + 1 if truth in order else None


def rr(r):
    return 1 / r if r is not None and r <= 25 else 0.0


def compare(rows, variant="expanded"):
    if not rows:
        return {"n": 0}
    a = np.asarray([rr(r["ranks"]["base"]) for r in rows])
    b = np.asarray([rr(r["ranks"][variant]) for r in rows])
    d = b - a
    rng = np.random.default_rng(SEED)
    ci = np.quantile(d[rng.integers(0, len(d), (DRAWS, len(d)))].mean(1), [.025, .975])
    return {"n": len(rows), "base_mrr25": float(a.mean()), "expanded_mrr25": float(b.mean()),
            "delta_mrr25": float(d.mean()), "ci95": ci.tolist(),
            "base_top1": int((a == 1).sum()), "expanded_top1": int((b == 1).sum()),
            "base_top25": int((a > 0).sum()), "expanded_top25": int((b > 0).sum()),
            "improved": int((d > 1e-12).sum()), "worsened": int((d < -1e-12).sum()),
            "wrong_to_right": int(((a < 1) & (b == 1)).sum()),
            "right_to_wrong": int(((a == 1) & (b < 1)).sum()),
            "new_top25": int(((a == 0) & (b > 0)).sum()),
            "lost_top25": int(((a > 0) & (b == 0)).sum())}


def strata(rows, variants):
    groups = {"all50": rows, "known_mass46": [r for r in rows if not r["mass_uncertain"]],
              "A25": [r for r in rows if r["class_AB"] == "A"],
              "B_all25": [r for r in rows if r["class_AB"] == "B"],
              "B_known21": [r for r in rows if r["class_AB"] == "B" and not r["mass_uncertain"]],
              "supported23": [r for r in rows if r["has_supported_spectrum"]]}
    return {g: {v: compare(rs, v) for v in variants} for g, rs in groups.items()}


def canon_scores(raw, field):
    return {r["canon"]: r.get(field) for r in raw.values()}


def audit_np_g1():
    expanded = read("artifacts/runs/009-v9recall/cpu_expanded_top100.json")
    new = read("artifacts/runs/009-v9recall/scores/scores/v9recall_ice_top100_split.json")
    audit = {}
    for ds, bundle, prior in [
        ("np", "f1-bundle", "f1-scores/scores/f1e-4_ice_top100_split.json"),
        ("g1", "g1-validation", "g1-scores/scores/g1e-4_ice_top100_split.json"),
    ]:
        inputs = read(f"artifacts/{bundle}/inputs.json")["molecules"]
        old = read("artifacts/" + prior)
        rows = []
        for m in inputs:
            key, truth = m["key"], m["truth_canon"]
            bo = m["order_top100"]
            bs = m["base_scores_full_window"]
            old_iv = canon_scores(old[key], "fused_merge")
            b_ranked = fuse(bo, bs, old_iv)
            e = expanded["new_shortlists"][ds].get(key)
            eo = e["order_expanded_top100"] if e else bo
            es = dict(zip(eo, e["scores_expanded_top100"])) if e else bs
            # A displaced old top100 candidate may not have been recomputed in
            # the expanded bundle. Retain its old score for the base-pool
            # counterfactual rather than accidentally turning it into missing.
            new_iv = dict(old_iv)
            if key in new:
                new_iv.update(canon_scores(new[key], "fused_merge"))
            missing_new = [c for c in eo if c not in bo and c not in old_iv and c not in new_iv]
            supported = any(s["mode"] == "positive" and s["adduct"] in {"[M+H]+", "[M+Na]+"} for s in m["spectra"])
            if supported and missing_new:
                raise ValueError((ds, key, "new candidate ICE not available", missing_new))
            erank = fuse(eo, es, new_iv)
            # Re-score the unexpanded set with the same refreshed cache when
            # available, isolating expansion from small numerical/version drift.
            refreshed = fuse(bo, bs, new_iv) if key in new else b_ranked
            rows.append({"key": key, "truth": truth, "ranks": {
                "base": rank(b_ranked, truth), "expanded": rank(erank, truth),
                "refreshed_base": rank(refreshed, truth)},
                "original_vs_refreshed_top25_same": b_ranked[:25] == refreshed[:25],
                "base_top25": b_ranked[:25], "expanded_top25": erank[:25],
                "expanded_candidate_count": len(eo), "base_candidate_count": len(bo),
                "new_truth_in_original_window": truth in bs})
        audit[ds] = {"summary": compare(rows), "refreshed_baseline_delta": compare(rows, "refreshed_base"),
                     "truths_already_in_original_windows": sum(r["new_truth_in_original_window"] for r in rows),
                     "rank_changes": [r for r in rows if r["ranks"]["base"] != r["ranks"]["expanded"]],
                     "rows": rows}
    return audit


def audit_r3_cached():
    full = read("artifacts/runs/009-expansion/r3_50_full_candidate_scores.json")
    observations = read("artifacts/runs/007-r3/observations_50.json")["molecules"]
    bundle = read("artifacts/runs/009-r3baseline/v9r3baseline/inputs.json")["molecules"]
    split = read("artifacts/runs/009-r3baseline/scores/scores/v9r3baseline_ice_top100_split.json")
    mean = read("artifacts/runs/009-r3baseline/scores/scores/v9r3baseline_ice_top100_mean.json")
    bundle_by = {m["key"]: m for m in bundle}
    filed = read("artifacts/runs/009-r3baseline/fusion_eval.json")
    variants = ("expanded", "exclude_query_records", "exclude_query_contributors")
    outputs = {}
    score_match_max = 0.0
    for protocol in ("strict_merge_field", "explicit_sum_ce_fallback"):
        rows = []
        for m in observations:
            key = m["canonical_key"]
            supported = key in bundle_by
            iv = {}
            if supported:
                field = "fused_merge" if protocol == "strict_merge_field" else "fused"
                iv = canon_scores(split[key], field)
                for smi, r in split[key].items():
                    if finite(r.get("fused")):
                        score_match_max = max(score_match_max, abs(r["fused"] - mean[key][smi]["fused"]))
            ranks, orders = {}, {}
            for label in ("base",) + variants:
                sc = full[key]["scores"][label]
                order = fp_order(sc)
                if supported and any(c not in iv for c in order):
                    raise ValueError((key, label, "missing cached candidate"))
                ranked = fuse(order, sc, iv)
                ranks[label] = rank(ranked, key)
                orders[label] = ranked[:25]
            bs, es = full[key]["scores"]["base"], full[key]["scores"]["expanded"]
            rows.append({"key": key, "class_AB": m["class_AB"], "ranks": ranks,
                         "has_supported_spectrum": supported, "mass_uncertain": not bool(es),
                         "base_candidate_count": len(bs), "expanded_candidate_count": len(es),
                         "added_candidates": sorted(set(es) - set(bs)), "top25": orders})
        outputs[protocol] = {"strata": strata(rows, variants), "rows": rows}
    finite_counts = {field: sum(finite(r.get(field)) for d in split.values() for r in d.values())
                     for field in ("fused", "fused_merge")}
    support_records = [s for m in bundle for s in m["spectra"]
                       if s["mode"] == "positive" and s["adduct"] in {"[M+H]+", "[M+Na]+"}]
    outputs["contract"] = {
        "n_supported_molecules": len(bundle), "n_supported_query_records": len(support_records),
        "support_records_with_known_ev": sum(bool(s.get("ce_ev")) for s in support_records),
        "score_field_finite_counts": finite_counts, "split_mean_fused_maxdiff": score_match_max,
        "mean_fp_warning": "Both cached protocols retain mean-of-spectrum FP logits and median query mass; neither is the literal production v9 representative-spectrum FP baseline.",
        "sum_fallback_warning": "Using fused when fused_merge is missing is an explicit score-field fallback change. It is not forced mathematically by missing CE and is not present in the original v9 merge-only selection.",
        "filed_comparison": "009 R3 baseline_mrr compares expanded FP vs expanded FP+sum-CE ICE, not base pool vs expanded pool.",
    }
    # Verify the previously reported expanded FP -> expanded ICE values separately.
    fallback_rows = outputs["explicit_sum_ce_fallback"]["rows"]
    by_key = {r["key"]: r for r in fallback_rows}
    old_rows = filed["supported_23"]["per_molecule"]
    mismatches = [r["key"] for r in old_rows
                  if r["fused_rank"] != by_key[r["key"]]["ranks"]["expanded"]]
    outputs["contract"]["filed_fused_rank_mismatches"] = mismatches
    return outputs, observations, finite_counts


def literal_v9_representative(observations, finite_counts):
    """Rebuild representative FP from cached logits. R3 merge field is all null.

    Consequently original v9's merge-only policy has no usable ICE scores for
    these inputs; no absent neural predictions are fabricated. Candidate scores
    and mass windows are recomputed with the notebook's own mass conversion.
    """
    if finite_counts["fused_merge"] != 0:
        raise ValueError("This shortcut is valid only with all merge scores missing.")
    from candidate_expansion import fp_bits_for_smiles
    nb = read("notebooks/submissions/submission-v9/enveda-submission-v9.ipynb")
    mass_source = next("".join(c["source"]) for c in nb["cells"]
                       if "".join(c["source"]).startswith("MASS = dict"))
    env = {"np": np}
    exec(mass_source, env)
    zpath = ROOT / "artifacts/runs/009-expansion/r3_50_logits.npy"
    z = np.load(zpath, mmap_mode="r")
    SOURCES[str(zpath.relative_to(ROOT))] = {"sha256": hashlib.sha256(zpath.read_bytes()).hexdigest(), "shape": list(z.shape)}
    tp, cp = ROOT / "artifacts/shared/pool-train", ROOT / "artifacts/runs/005/coco"
    masses = np.concatenate([np.load(tp / "mass.npy"), np.load(cp / "coco_mass.npy")])
    tfp, cfp = np.load(tp / "fp.npy", mmap_mode="r"), np.load(cp / "coco_fp.npy", mmap_mode="r")
    tk = np.load(tp / "keys.npy", allow_pickle=True).tolist()
    cmeta = pickle.loads((cp / "coco_meta.pkl").read_bytes())
    ck = list(cmeta["keys"])
    if set(tk) & set(ck):
        raise ValueError("unexpected raw-key overlap")
    keys, ntr = tk + ck, len(tk)
    maps = []
    for domain in ("train", "coco"):
        tab = pq.read_table(ROOT / f"artifacts/runs/005/structure_identity_{domain}.parquet",
                            columns=["raw_key", "canonical_group_key"]).to_pydict()
        maps.append(dict(zip(tab["raw_key"], tab["canonical_group_key"])))
    union = set(maps[0].values()) | set(maps[1].values())
    record_path = ROOT / "artifacts/runs/008-expansion/candidate_index_records.jsonl"
    raw = record_path.read_bytes()
    SOURCES[str(record_path.relative_to(ROOT))] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    new = [r for r in (json.loads(s) for s in raw.decode().splitlines()) if r["canonical_key"] not in union]
    nm = np.asarray([r["mass_theo"] for r in new])
    bits_selection = np.load(cp / "fp_bits.npy")
    fingerprint = {}
    offset, rows = 0, []
    for m in observations:
        ss = m["spectra"]
        chosen = max(range(len(ss)), key=lambda i: ss[i]["n_peaks"])
        q = np.asarray(z[offset + chosen], float)
        offset += len(ss)
        s = ss[chosen]
        qm = float(env["neutral_mass"](np.array([s["precursor_mz"]]), [s.get("adduct")])[0])
        base, added = {}, {}
        if math.isfinite(qm):
            ix = np.flatnonzero(np.abs(masses - qm) <= qm * 1e-5)
            if len(ix):
                packed = np.stack([tfp[j] if j < ntr else cfp[j - ntr] for j in ix])
                dots = np.unpackbits(packed, axis=1)[:, :6930].astype(float) @ q
                for j, score in zip(ix, dots):
                    canon = maps[0 if j < ntr else 1][keys[j]]
                    base[canon] = max(base.get(canon, -np.inf), float(score))
            for j in np.flatnonzero(np.abs(nm - qm) <= qm * 1e-5):
                rec = new[j]
                if rec["canonical_key"] not in fingerprint:
                    fingerprint[rec["canonical_key"]] = fp_bits_for_smiles(rec["smiles_rep"], bits_selection)[0]
                added[rec["canonical_key"]] = float(np.asarray(fingerprint[rec["canonical_key"]], float) @ q)
        pools = {"base": base, "expanded": {**base, **added}}
        orders = {label: fp_order(scores) for label, scores in pools.items()}
        rows.append({"key": m["canonical_key"], "class_AB": m["class_AB"],
                     "has_supported_spectrum": any(t["ionization_mode"] == "positive" and t.get("adduct") in {"[M+H]+", "[M+Na]+"} for t in ss),
                     "mass_uncertain": not math.isfinite(qm), "query_mass": qm if math.isfinite(qm) else None,
                     "selected_spectrum_id": s["spectrum_id"], "selected_local_index": chosen,
                     "ranks": {label: rank(order, m["canonical_key"]) for label, order in orders.items()},
                     "top25": {label: order[:25] for label, order in orders.items()},
                     "base_candidate_count": len(base), "expanded_candidate_count": len(base) + len(added),
                     "added_candidates": sorted(added)})
    if offset != len(z):
        raise ValueError("logit row count mismatch")
    return {"protocol": "Literal v9 max-raw-peaks representative, notebook mass function, fixed base pool vs base+539, canonical max FP scoring, original merge-only ICE field. All R3 merge scores are missing, so this correctly falls back to representative FP. CPU float64 arithmetic and cached canonical identity replace GPU float32/recanonicalization; no neural forward.",
            "n_new_structures": len(new), "n_logit_rows": offset,
            "strata": strata(rows, ("expanded",)), "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/runs/010-recall-audit")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "audit.json"
    if target.exists():
        raise FileExistsError(target)
    start = time.monotonic()
    report = {"seed": SEED, "bootstrap_draws": DRAWS,
              "scope": "Local cached-score audit; no training, neural forward, GPU, downloads, or submission.",
              "interpretation": "NP/G1 already contain all truths in the base pool. Zero expansion delta is an interference/regression observation, not a failed novel-structure recall experiment. R3 source-selected same-library data are optimistic and cannot estimate hidden category prevalence or guarantee public gain.",
              "np_g1": audit_np_g1()}
    r3, observations, counts = audit_r3_cached()
    report["r3_cached_protocols"] = r3
    report["r3_literal_v9"] = literal_v9_representative(observations, counts)
    report["source_sha256"] = SOURCES
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["wall_seconds"] = time.monotonic() - start
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for ds, d in report["np_g1"].items():
        print(ds, d["summary"])
    for protocol in ("strict_merge_field", "explicit_sum_ce_fallback"):
        print(protocol, r3[protocol]["strata"]["known_mass46"])
    print("literal_v9", report["r3_literal_v9"]["strata"]["known_mass46"])
    print("wrote", target, "seconds", report["wall_seconds"])


if __name__ == "__main__":
    main()
