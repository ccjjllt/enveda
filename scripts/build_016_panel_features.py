"""016 panel candidate features (manual, local CPU only).

Mirrors 015 frozen logic (same 10 columns, same cleaning/matching/quality/
aggregation rules) for the 400-molecule 016 panel, with references from the
016 traceable index (2.51M) instead of the 015 stride sample:
  B_seen: exclude the 400 panel query peak hashes (exact), allow same-structure
          other spectra.
  A_absent: B + globally exclude the 400 panel target canonicals (evaluation
          intervention decided before scoring; not a deployable switch).
Baseline (FP z + ICE fused_merge z + present + reciprocal rank) uses the local
v10-caliber forward and the Colab ICE predictions; truth only for post-hoc
labels. Candidates: first 50 of panel order_top100, same-formula slots later.

Manual run: /opt/homebrew/bin/python3.11 scripts/build_016_panel_features.py [--probe]
Probe: 4 queries (2+2), no full write. Full writes
artifacts/016-panel-features/{features.npz,manifest.json,summary.json}.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analog_evidence import (canonical_morgan, clean_peaks, exact_peak_hash,
                             neutral_mass, spectral_evidence, tanimoto_to_refs)

sys.path.insert(0, str(ROOT / "scripts"))
from build_016_reference_index import load_index, read_peaks

OUT = ROOT / "artifacts/016-panel-features"
ENVIDA_LIBS = {"enveda-180", "enveda-np-examples"}
FEATURES = ['fp_z', 'ice_z', 'ice_present', 'v9_reciprocal_rank', 'direct_quality',
            'analog_max_tanimoto', 'analog_weighted_tanimoto', 'analog_max_product',
            'analog_top5mean_product', 'analog_best_spec_quality']
SEED = 20261017
WIN_DA = 200.0
MAXREF = 2000
TOPREF = 80


def zscore(values):
    a = np.asarray(values, dtype=float)
    finite = np.isfinite(a)
    z = np.zeros(len(a))
    if finite.sum() > 1:
        sd = a[finite].std(ddof=1)
        if sd > 1e-12:
            z[finite] = (a[finite] - a[finite].mean()) / sd
    return z


def aggregate_analog(similarities, qualities):
    if not len(similarities) or not np.any(qualities > 0):
        return [0.0] * 5
    products = similarities * qualities
    best = int(np.argmax(products))
    return [float(similarities.max()), float(np.dot(similarities, qualities) / qualities.sum()),
            float(products[best]), float(np.sort(products)[-5:].mean()), float(qualities[best])]


def smiles_map():
    import pyarrow.parquet as pq
    m = {}
    for path in ["artifacts/005/structure_identity_train.parquet",
                 "artifacts/005/structure_identity_coco.parquet"]:
        for r in pq.read_table(ROOT / path, columns=["canonical_group_key", "output_smiles"]).to_pylist():
            k, s = r["canonical_group_key"], r["output_smiles"]
            if k and s and (k not in m or s < m[k]):
                m[k] = s
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--cross-external-only", action="store_true",
                    help="cross_source queries: references restricted to non-enveda libs")
    args = ap.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else OUT
    t0 = time.time()
    panel = json.load(open(ROOT / "artifacts/016-panel/panel_manifest.json"))["panel"]
    inputs = {m["key"]: m for m in
              json.load(open(ROOT / "artifacts/016-colab/panel-016/inputs.json"))["molecules"]}
    ice = json.load(open(ROOT / "artifacts/016-panel-scores/scores/panel016_ice_top100_split.json"))
    smap = smiles_map()
    idx, n = load_index()
    order = np.argsort(idx["nm"], kind="mergesort")  # load_index already mass-sorts; keep explicit
    idx = {k: v[order] for k, v in idx.items()}
    mass = idx["nm"]
    qhashes = {p["query_peak_hash"] for p in panel if p.get("query_peak_hash")}
    targets = {p["canonical"] for p in panel}
    rng = np.random.default_rng(SEED)
    shard_cache = {}
    queries = []
    for p in panel:
        m = inputs[p["canonical"]]
        order100 = m["order_top100"]
        fp = np.asarray(m["scores_top100"], dtype=float)
        ice_d = ice.get(p["canonical"], {})
        iv = np.asarray([ice_d.get(c, {}).get("fused_merge", None)
                         if isinstance(ice_d.get(c), dict) else None for c in order100], dtype=float)
        fp_z, ice_z = zscore(fp), zscore(iv)
        base = {c: (float(fp_z[i]), float(ice_z[i]), int(np.isfinite(iv[i]))) for i, c in enumerate(order100)}
        s = m["spectra"][0]
        cmz, cit, audit = clean_peaks(np.asarray(s["mz"], float), np.asarray(s["it"], float),
                                      float(s["prec"]), s["adduct"])
        queries.append({"key": p["canonical"], "group": p["group"], "truth": p["canonical"],
                        "adduct": s["adduct"], "mode": s["mode"], "precursor_mz": float(s["prec"]),
                        "neutral_mass": neutral_mass(s["prec"], s["adduct"]),
                        "cmz": cmz, "cit": cit, "baseline_order": order100,
                        "candidates": order100[:50], "base": base,
                        "rep_smiles": m["rep_smiles"], "formulas": m["formulas"],
                        "ice_supported": any(v[2] for v in base.values())})
    if args.probe:
        queries = [q for q in queries if q["group"] == "cross_source"][:2] + \
                  [q for q in queries if q["group"] == "natural_missing"][:2]
    all_X, qindex, sindex, positions, manifest_q = [], [], [], [], []
    for qi, q in enumerate(queries):
        pools = {}
        for si, scen in enumerate(["B_seen", "A_absent"]):
            lo = int(np.searchsorted(mass, q["neutral_mass"] - WIN_DA, "left"))
            hi = int(np.searchsorted(mass, q["neutral_mass"] + WIN_DA, "right"))
            eli = [i for i in range(lo, hi)
                   if str(idx["adduct"][i])[-1:] == q["adduct"][-1:]
                   and str(idx["phash"][i]) not in qhashes
                   and (scen == "B_seen" or str(idx["canon"][i]) not in targets)
                   and not (args.cross_external_only and q["group"] == "cross_source"
                            and str(idx["lib"][i]) in ENVIDA_LIBS)]
            eli.sort(key=lambda i: (abs(float(mass[i]) - q["neutral_mass"]), int(idx["row"][i])))
            pools[scen] = eli[:MAXREF]
        union = sorted(set(pools["B_seen"]) | set(pools["A_absent"]))
        refs = {}
        for i in union:
            rmz, rit = read_peaks(shard_cache, int(idx["shard"][i]), int(idx["off"][i]),
                                  int(idx["n_clean"][i]))
            refs[i] = {"mz": rmz, "intensity": rit, "adduct": str(idx["adduct"][i]),
                       "precursor_mz": float(idx["prec"][i]), "canonical": str(idx["canon"][i])}
        scores = {i: spectral_evidence({"mz": q["cmz"], "intensity": q["cit"],
                                        "adduct": q["adduct"], "precursor_mz": q["precursor_mz"]},
                                       refs[i]) for i in union}
        for si, scen in enumerate(["B_seen", "A_absent"]):
            pool = pools[scen]
            ordered = sorted(pool, key=lambda i: (-scores[i]["quality"], -scores[i]["cosine"],
                                                  int(idx["row"][i])))
            unique, seen = [], set()
            for i in ordered:
                if scores[i]["quality"] <= 0:
                    continue
                c = refs[i]["canonical"]
                if c not in seen:
                    unique.append(i)
                    seen.add(c)
                    if len(unique) == TOPREF:
                        break
            qualities = np.asarray([scores[i]["quality"] for i in unique])
            ref_smiles = [smap.get(refs[i]["canonical"]) for i in unique]
            if any(s is None for s in ref_smiles):
                missing = {refs[unique[j]]["canonical"] for j, s in enumerate(ref_smiles) if s is None}
                raise ValueError(("refs without smiles", q["key"], sorted(missing)[:5]))
            by_canon = {}
            for i in pool:
                by_canon.setdefault(refs[i]["canonical"], []).append(i)
            for pos, cand in enumerate(q["candidates"]):
                sims = tanimoto_to_refs(q["rep_smiles"][cand], ref_smiles) if unique else np.array([])
                keep = np.asarray([refs[i]["canonical"] != cand for i in unique], dtype=bool)
                dq = max((scores[i]["quality"] for i in by_canon.get(cand, [])), default=0.0)
                fz, iz, ip = q["base"][cand]
                row = [fz, iz, ip, 1.0 / (q["baseline_order"].index(cand) + 1), dq]
                row += aggregate_analog(sims[keep], qualities[keep])
                all_X.append(row)
                qindex.append(qi)
                sindex.append(si)
                positions.append(pos)
        manifest_q.append({"key": q["key"], "group": q["group"], "truth": q["truth"],
                           "baseline_order": q["baseline_order"], "candidate_keys": q["candidates"],
                           "candidate_formulas": [q["formulas"].get(k) for k in q["candidates"]],
                           "adduct": q["adduct"], "ice_supported": q["ice_supported"]})
        if (qi + 1) % 50 == 0 or qi == len(queries) - 1:
            print(f"features {qi+1}/{len(queries)} wall={time.time()-t0:.0f}s", flush=True)
    X = np.asarray(all_X, dtype=np.float32)
    y = np.asarray([manifest_q[qi]["candidate_keys"][pos] == manifest_q[qi]["truth"]
                    for qi, pos in zip(qindex, positions)], dtype=np.uint8)
    if not args.probe:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "features.npz", X=X, y=y,
                            query_index=np.asarray(qindex, dtype=np.int32),
                            scenario_index=np.asarray(sindex, dtype=np.uint8),
                            candidate_position=np.asarray(positions, dtype=np.int16))
        json.dump({"feature_names": FEATURES, "scenarios": ["B_seen", "A_absent"],
                   "queries": manifest_q}, open(out_dir / "manifest.json", "w"))
        json.dump({"queries": len(queries), "rows": len(all_X), "finite": bool(np.isfinite(X).all()),
                   "cross_external_only": args.cross_external_only,
                   "seconds": round(time.time() - t0, 1)}, open(out_dir / "summary.json", "w"), indent=2)
    print(json.dumps({"queries": len(queries), "rows": len(all_X),
                      "finite": bool(np.isfinite(X).all()),
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
