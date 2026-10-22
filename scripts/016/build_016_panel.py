"""016 new-panel freeze (manual, CPU only).

Builds two-level exclusion list (canonical + exact spectrum hash) from all
old panels, stats the fresh train pool, and freezes a stratified 400-canonical
panel by metadata/hash sort BEFORE any candidate generation. No result-based
resampling; no FP/ICE; no GPU; no submission.

Manual run: /opt/homebrew/bin/python3.11 scripts/016/build_016_panel.py [--probe]
Probe: stats + first 10 selections only, no manifest write.
Full: writes artifacts/runs/016-panel/{exclusion_list.json,pool_stats.json,panel_manifest.json}.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analog_evidence import exact_peak_hash
from workspace_paths import workspace_path

ROOT = Path(__file__).resolve().parents[2]
SEED = "20261017|016"
ENVIDA_LIBS = {"enveda-180", "enveda-np-examples"}
OUT = ROOT / "artifacts/runs/016-panel"


def rank_key(group, canon):
    return hashlib.sha256(f"{SEED}|{group}|{canon}".encode()).hexdigest()


def old_panels():
    """Return (canon_set, hash_set, report). Raw keys mapped to canonical later."""
    canons, hashes, report = set(), set(), {}

    def add_json(path, key_fields):
        try:
            d = json.load(open(ROOT / path))
        except FileNotFoundError:
            return 0
        mols = d.get("molecules", d if isinstance(d, list) else [])
        n = 0
        for m in mols:
            for f in key_fields:
                v = m.get(f)
                if isinstance(v, str) and v:
                    canons.add(v)
                    n += 1
                    break
        return n

    report["f1_NP250"] = add_json("artifacts/legacy/f1-bundle/inputs.json", ["truth_canon", "key"])
    report["g1_250"] = add_json("artifacts/legacy/g1-validation/inputs.json", ["truth_canon", "key"])
    report["heldout_250"] = add_json("artifacts/runs/009-heldout/inputs.json", ["truth_canon", "key", "truth"])
    report["r3_50"] = add_json("artifacts/runs/007-r3/observations_50.json", ["canonical_key"])
    # small panels
    report["v9recall_11"] = add_json("artifacts/runs/009-v9recall/bundle/inputs.json", ["truth_canon", "key"])
    report["r3baseline_23"] = add_json("artifacts/runs/009-r3baseline/v9r3baseline/inputs.json", ["truth_canon", "key"])
    try:
        d = json.load(open(ROOT / "artifacts/runs/013-r3-fragment/frozen_inputs.json"))
        cases = d.get("cases", d.get("molecules", []))
        n = 0
        for m in cases if isinstance(cases, list) else []:
            for f in ("truth", "canonical_key", "key"):
                if isinstance(m.get(f), str):
                    canons.add(m[f])
                    n += 1
                    break
        report["r3frag_50cases"] = n
    except FileNotFoundError:
        report["r3frag_50cases"] = "missing"
    # 015 frozen truths (case 'NP:<truth>') + candidate keys + query hashes
    try:
        q = json.load(open(ROOT / "artifacts/runs/015-analog/features/frozen_queries.json"))
        qlist = q if isinstance(q, list) else q.get("queries", [])
        n = 0
        cand_keys = set()
        for item in qlist:
            case = item.get("case", "")
            if ":" in case:
                canons.add(case.split(":")[-1])
                n += 1
            for k in (item.get("candidates") or [])[:50]:
                if isinstance(k, str):
                    cand_keys.add(k)
        report["015_queries_truth"] = n
        report["015_candidate_keys_top50"] = len(cand_keys)
    except FileNotFoundError:
        report["015_queries_truth"] = "missing"
        cand_keys = set()
    try:
        e = json.load(open(ROOT / "artifacts/runs/015-analog/features/frozen_reference_exclusions.json"))
        h = e.get("all_query_peak_hashes", [])
        hashes.update(h)
        report["015_query_hashes"] = len(h)
        report["015_truth_groups_n"] = len(e.get("all_query_canonical_groups", []))
    except FileNotFoundError:
        report["015_query_hashes"] = "missing"
    # heldout + R3 + small-panel spectrum hashes where peaks are stored
    report["extra_hashes"] = compute_extra_hashes(hashes)
    return canons, hashes, cand_keys, report


def compute_extra_hashes(hashes):
    import pyarrow.parquet as pq  # noqa: placeholder to keep imports local
    added = {}
    # R3 npz peaks
    try:
        o = json.load(open(ROOT / "artifacts/runs/007-r3/observations_50.json"))
        n = 0
        for m in o["molecules"]:
            for s in m.get("spectra", [])[:1]:
                ref = s.get("peak_file_ref")
                ref = workspace_path(ref) if ref else None
                if ref and os.path.exists(ref):
                    z = np.load(ref)
                    keys = list(z.keys())
                    mk = next((k for k in keys if "mz" in k.lower()), keys[0])
                    ik = next((k for k in keys if "int" in k.lower()), keys[-1])
                    hashes.add(exact_peak_hash(np.asarray(z[mk]).ravel(), np.asarray(z[ik]).ravel()))
                    n += 1
        added["r3_npz"] = n
    except Exception as ex:
        added["r3_npz"] = f"error {ex}"
    # heldout/v9recall/r3baseline inline spectra (mz/it lists)
    for name, path in [("heldout", "artifacts/runs/009-heldout/inputs.json"),
                       ("v9recall", "artifacts/runs/009-v9recall/bundle/inputs.json"),
                       ("r3baseline", "artifacts/runs/009-r3baseline/v9r3baseline/inputs.json")]:
        try:
            d = json.load(open(ROOT / path))
            n = 0
            for m in d.get("molecules", []):
                for s in (m.get("spectra") or [])[:1]:
                    mz = s.get("mz") or s.get("ms2_mzs")
                    it = s.get("it") or s.get("intensities") or s.get("ms2_normalized_intensities")
                    if mz and it and len(mz) == len(it):
                        hashes.add(exact_peak_hash(np.asarray(mz), np.asarray(it)))
                        n += 1
            added[name] = n
        except Exception as ex:
            added[name] = f"error {ex}"
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    import pyarrow.parquet as pq

    raw_old, hashes, cand_keys, report = old_panels()
    # 005 canonical mapping
    t5 = pq.read_table(ROOT / "artifacts/runs/005/structure_identity_train.parquet",
                       columns=["raw_key", "canonical_group_key"])
    raw2canon = dict(zip(t5.column("raw_key").to_pylist(), t5.column("canonical_group_key").to_pylist()))
    mapped = {raw2canon.get(k, k) for k in raw_old}
    report["old_raw_keys"] = len(raw_old)
    report["old_canonical_mapped"] = len(mapped)
    report["unmapped_fallback"] = sum(1 for k in raw_old if k not in raw2canon)

    cols = ["inchikey14", "ingest_lib", "instrument_type", "ionization_mode",
            "adduct", "precursor_mz", "num_peaks"]
    t = pq.read_table(ROOT / "data/raw/train.parquet", columns=cols)
    n = t.num_rows
    ik = np.asarray(t.column("inchikey14").to_pylist(), dtype=object)
    lib = np.asarray(t.column("ingest_lib").to_pylist(), dtype=object)
    inst = np.asarray(t.column("instrument_type").to_pylist(), dtype=object)
    mode = np.asarray(t.column("ionization_mode").to_pylist(), dtype=object)
    ad = np.asarray(t.column("adduct").to_pylist(), dtype=object)
    pr = np.asarray(t.column("precursor_mz").to_pylist(), dtype=np.float64)
    npk = np.asarray(t.column("num_peaks").to_pylist(), dtype=np.float64)
    canon = np.array([raw2canon.get(k, k) for k in ik], dtype=object)
    report["canon_fallback_rows"] = int(sum(1 for a, b in zip(ik, canon) if a == b and a not in raw2canon))
    is_env = np.array([x in ENVIDA_LIBS for x in lib])
    is_tims = np.array([x == "timsTOF" for x in inst])

    # per-canonical aggregates (fresh only to save time: first mark fresh)
    # aggregate via sort by canon
    order = np.argsort(canon, kind="mergesort")
    uniq, starts = np.unique(canon[order], return_index=True)
    fresh_stats = {}
    n_fresh = 0
    # vectorized group spans
    ends = np.append(starts[1:], n)
    for u, s, e in zip(uniq.tolist(), starts.tolist(), ends.tolist()):
        if u in mapped:
            continue
        idx = order[s:e]
        libs = set(lib[idx].tolist())
        fresh_stats[u] = {
            "n_spectra": int(e - s),
            "n_enveda": int(is_env[idx].sum()),
            "n_external": int((~is_env[idx]).sum()),
            "n_tims": int(is_tims[idx].sum()),
            "modes": sorted(set(mode[idx].tolist())),
            "libs": sorted(libs),
        }
        n_fresh += 1
    report["train_canonical_total"] = int(len(uniq))
    report["fresh_canonical"] = n_fresh
    # pools
    cross = [u for u, f in fresh_stats.items() if f["n_tims"] > 0 and f["n_external"] > 0]
    env_only = [u for u, f in fresh_stats.items()
                if f["n_external"] == 0 and f["n_enveda"] > 0]
    ext_only = [u for u, f in fresh_stats.items()
                if f["n_enveda"] == 0 and f["n_external"] > 0]
    single = [u for u in env_only if fresh_stats[u]["n_spectra"] == 1]
    report["pool_cross_source"] = len(cross)
    report["pool_enveda_only"] = len(env_only)
    report["pool_external_only"] = len(ext_only)
    report["pool_enveda_single_spectrum"] = len(single)

    # selection needs per-row best timsTOF query: build canon->rows map for pools only
    pool_set = set(cross) | set(env_only)
    canon_rows = {}
    for i in order.tolist():
        c = canon[i]
        if c in pool_set:
            canon_rows.setdefault(c, []).append(i)

    def best_tims(rows):
        cand = [i for i in rows if is_tims[i]]
        if not cand:
            return None
        return max(cand, key=lambda i: (npk[i] if np.isfinite(npk[i]) else -1))

    def pick(pool, k, group, filters=()):
        ranked = sorted(pool, key=lambda u: rank_key(group, u))
        out = []
        for u in ranked:
            if len(out) >= k:
                break
            if all(f(u) for f in filters):
                out.append(u)
        return out, ranked

    is_neg = lambda u: (lambda i: mode[i] == "negative")(best_tims(canon_rows[u]))
    is_multi = lambda u: str(ad[best_tims(canon_rows[u])]).startswith(("2M", "3M"))
    cross_sorted = sorted(cross, key=lambda u: rank_key("cross", u))
    # negative-first then multimer then rest to hit strata
    sel_cross, short = [], {"neg_short": 0, "multi_short": 0}
    negs = [u for u in cross_sorted if is_neg(u)][:50]
    sel_cross += negs
    rest = [u for u in cross_sorted if u not in sel_cross]
    multis = [u for u in rest if is_multi(u)][:20]
    sel_cross += multis
    rest = [u for u in rest if u not in sel_cross]
    sel_cross += rest[:200 - len(sel_cross)]
    if sum(1 for u in sel_cross if is_neg(u)) < 30:
        short["neg_short"] = sum(1 for u in sel_cross if is_neg(u))
    single_sorted = sorted(single, key=lambda u: rank_key("nat-single", u))[:100]
    multi_env = sorted([u for u in env_only if fresh_stats[u]["n_spectra"] > 1],
                       key=lambda u: rank_key("nat-multi", u))[:100]
    short["nat_single_got"] = len(single_sorted)
    short["nat_multi_got"] = len(multi_env)

    panel = []
    for u in sel_cross:
        i = best_tims(canon_rows[u])
        f = fresh_stats[u]
        panel.append({"canonical": u, "group": "cross_source",
                      "query_row": int(i), "query_lib": str(lib[i]),
                      "query_instrument": str(inst[i]), "query_mode": str(mode[i]),
                      "query_adduct": str(ad[i]), "query_precursor_mz": float(pr[i]),
                      "query_num_peaks": float(npk[i]),
                      "n_ref_external": f["n_external"], "n_ref_enveda": f["n_enveda"]})
    for u in single_sorted + multi_env:
        rows = canon_rows[u]
        i = max(rows, key=lambda j: (npk[j] if np.isfinite(npk[j]) else -1))
        f = fresh_stats[u]
        panel.append({"canonical": u, "group": "natural_missing",
                      "query_row": int(i), "query_lib": str(lib[i]),
                      "query_instrument": str(inst[i]), "query_mode": str(mode[i]),
                      "query_adduct": str(ad[i]), "query_precursor_mz": float(pr[i]),
                      "query_num_peaks": float(npk[i]),
                      "n_ref_external": f["n_external"], "n_ref_enveda": f["n_enveda"]})
    report["selected_cross"] = len(sel_cross)
    report["selected_natural"] = len(single_sorted) + len(multi_env)
    report["shortfalls"] = short

    # verify selected query peak hashes not in old hash set (read peaks for selected rows only)
    if not args.probe:
        tp = pq.read_table(ROOT / "data/raw/train.parquet",
                           columns=["ms2_mzs", "ms2_normalized_intensities"])
        mzs = tp.column("ms2_mzs")
        its = tp.column("ms2_normalized_intensities")
        collisions = []
        for p in panel:
            r = p["query_row"]
            h = exact_peak_hash(np.asarray(mzs[r].as_py(), dtype=np.float64),
                                np.asarray(its[r].as_py(), dtype=np.float64))
            p["query_peak_hash"] = h
            if h in hashes:
                collisions.append(p["canonical"])
            hashes.add(h)
        report["new_panel_overlap_old_candidates"] = sorted(
            {p["canonical"] for p in panel} & {raw2canon.get(k, k) for k in cand_keys})
        report["new_query_hash_collisions_with_old"] = collisions
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump({"seed": SEED, "old_canonical": sorted(mapped),
                   "old_hashes": sorted(hashes)},
                  open(OUT / "exclusion_list.json", "w"))
        json.dump(report, open(OUT / "pool_stats.json", "w"), indent=2)
        json.dump({"seed": SEED, "panel": panel}, open(OUT / "panel_manifest.json", "w"))
    print(json.dumps(report, indent=2, ensure_ascii=False)[:3000])
    print(f"panel={len(panel)} probe={args.probe} seconds={time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
