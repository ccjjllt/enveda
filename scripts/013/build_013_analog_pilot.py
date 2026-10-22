"""013 analog pilot v1: reference/analog support audit (CPU only, no training).

- Queries: f1-bundle NP250 + g1-validation G1-250 + 007-r3 R3-50 spectra (existing dev caches).
- Reference: sampled train.parquet spectra (pilot sample, NOT full 2.5M recall).
- Grouping: inchikey14 proxy; full canonical-alias exclusion is TODO (disclosed).
- Domains: A (remove same-group refs) vs B (keep independent spectra) reported separately.
- No FP/ICE rerank yet; this is support audit before ranker (012 sequence step 1).
- Manual run: /opt/homebrew/bin/python3.11 scripts/013/build_013_analog_pilot.py [--probe]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from workspace_paths import workspace_path

MASS = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196,
            P=30.97376163, S=31.97207100, F=18.99840322, Cl=34.96885268,
            Br=78.9183371, Na=22.9897692809, K=38.96370668)
E = 0.00054857990
PROTON = MASS['H'] - E
H2O = 2 * MASS['H'] + MASS['O']
NH4 = MASS['N'] + 4 * MASS['H']
FORMATE = MASS['C'] + 2 * MASS['H'] + 2 * MASS['O']
ACETATE = 2 * MASS['C'] + 4 * MASS['H'] + 2 * MASS['O']
# (n, z, d): neutral = (mz*z - d)/n ; copied caliber from Berat 0.336 public notebook (read-only)
ADDUCT_NZD = {
    "[M+H]+": (1, 1, PROTON), "[M+NH4]+": (1, 1, NH4 - E),
    "[M+Na]+": (1, 1, MASS['Na'] - E), "[M+K]+": (1, 1, MASS['K'] - E),
    "[M-H2O+H]+": (1, 1, PROTON - H2O), "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O),
    "[M+2H]2+": (1, 2, 2 * PROTON), "[M]+": (1, 1, -E),
    "[M-H]-": (1, 1, -PROTON), "[M-H2O-H]-": (1, 1, -PROTON - H2O),
    "[M+CH2O2-H]-": (1, 1, FORMATE - PROTON),
    "[M+C2H4O2-H]-": (1, 1, ACETATE - PROTON),
    "[M+Cl]-": (1, 1, MASS['Cl'] + E), "[M]-": (1, 1, E),
    "[M-2H]-": (1, 2, -2 * PROTON),
    "[2M+H]+": (2, 1, PROTON), "[2M+Na]+": (2, 1, MASS['Na'] - E),
    "[2M+H]+": (2, 1, PROTON), "[2M-H]-": (2, 1, -PROTON),
    "[2M+CH2O2-H]-": (2, 1, FORMATE - PROTON),
    "[2M+C2H4O2-H]-": (2, 1, ACETATE - PROTON),
}
TOL = 0.01
FLOOR = 0.002
TOPK = 256
ANALOG_WIN = 200.0
REF_SAMPLE = 50000
REF_SEED = 20261017


def neutral_mass(prec, adduct):
    nzd = ADDUCT_NZD.get(adduct)
    if nzd is None or prec is None:
        return np.nan
    try:
        p = float(prec)
    except Exception:
        return np.nan
    if not np.isfinite(p):
        return np.nan
    n, z, d = nzd
    return float((p * z - d) / n)


def clean(mz, it):
    mz = np.asarray(mz, dtype=np.float64)
    it = np.asarray(it, dtype=np.float64)
    m = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    mz, it = mz[m], it[m]
    if len(mz) == 0:
        return mz, it
    mx = it.max()
    if mx <= 0:
        return np.zeros(0), np.zeros(0)
    keep = it >= FLOOR * mx
    mz, it = mz[keep], it[keep]
    if len(mz) > TOPK:
        idx = np.argsort(it)[-TOPK:]
        idx = np.sort(idx)
        mz, it = mz[idx], it[idx]
    s = it.sum()
    if s > 0:
        it = it / s
    o = np.argsort(mz)
    return mz[o], it[o]


def cos_sim(qmz, qp, cmz, cp):
    i = j = 0
    dot = 0.0
    na = float((qp * qp).sum())
    nb = float((cp * cp).sum())
    if na <= 0 or nb <= 0:
        return 0.0
    n, m = len(qmz), len(cmz)
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -TOL:
            i += 1
        elif d > TOL:
            j += 1
        else:
            dot += qp[i] * cp[j]
            i += 1
            j += 1
    return float(dot / np.sqrt(na * nb))


def cos_shift(qmz, qp, cmz, cp, shift):
    a = cos_sim(qmz, qp, cmz, cp)
    if abs(shift) < 1e-9:
        return a
    b = cos_sim(qmz, qp, cmz + shift, cp)
    return a if a > b else b


def load_queries():
    out = []
    for path, ds in [("artifacts/legacy/f1-bundle/inputs.json", "NP"),
                     ("artifacts/legacy/g1-validation/inputs.json", "G1")]:
        d = json.load(open(path))
        for mol in d["molecules"]:
            key = mol.get("key") or mol.get("truth_canon")
            specs = mol.get("spectra") or []
            if not specs:
                continue
            s = specs[0]
            mz = s.get("mz") or s.get("ms2_mzs") or []
            it = s.get("it") or s.get("intensities") or s.get("ms2_intensities") or s.get("ms2_normalized_intensities") or []
            prec = s.get("prec") or s.get("precursor_mz")
            try:
                prec = float(prec) if prec not in (None, "", "None") else None
            except Exception:
                prec = None
            out.append({"case": f"{ds}:{key}", "dataset": ds, "truth14": key,
                        "mz": list(mz), "it": list(it),
                        "adduct": s.get("adduct"), "prec": prec})
    # R3 peaks from npz
    o = json.load(open("artifacts/runs/007-r3/observations_50.json"))
    for mol in o["molecules"]:
        key = mol.get("canonical_key")
        specs = mol.get("spectra") or []
        if not specs:
            continue
        s = specs[0]
        ref = s.get("peak_file_ref")
        ref = workspace_path(ref) if ref else None
        mz, it = [], []
        if ref and os.path.exists(ref):
            z = np.load(ref)
            # discover keys
            keys = list(z.keys())
            # common: mz/int or mzs/intensities
            mk = next((k for k in keys if "mz" in k.lower() or k == "mzs"), keys[0])
            ik = next((k for k in keys if "int" in k.lower()), keys[1] if len(keys) > 1 else keys[0])
            mz = list(np.asarray(z[mk]).ravel())
            it = list(np.asarray(z[ik]).ravel())
        out.append({"case": f"R3:{key}", "dataset": "R3", "truth14": key,
                    "mz": mz, "it": it, "adduct": s.get("adduct"),
                    "prec": float(s.get("precursor_mz")) if s.get("precursor_mz") else None})
    return out


def load_train_sample():
    import pyarrow.parquet as pq
    rng = np.random.default_rng(REF_SEED)
    # stride sample to ~REF_SAMPLE across 2.5M rows without loading peaks first
    t_light = pq.read_table("data/raw/train.parquet",
                            columns=["inchikey14", "adduct", "precursor_mz"])
    n = t_light.num_rows
    stride = max(1, n // REF_SAMPLE)
    idx = np.arange(0, n, stride, dtype=np.int64)
    # light arrays
    ik = np.asarray(t_light.column("inchikey14").to_pylist(), dtype=object)
    ad = np.asarray(t_light.column("adduct").to_pylist(), dtype=object)
    pr = np.asarray(t_light.column("precursor_mz").to_pylist(), dtype=np.float64)
    ik, ad, pr = ik[idx], ad[idx], pr[idx]
    nm = np.array([neutral_mass(p, a) for p, a in zip(pr, ad)], dtype=np.float64)
    ok = np.isfinite(nm)
    # now read peaks only for sampled rows via take
    t_full = pq.read_table("data/raw/train.parquet",
                           columns=["ms2_mzs", "ms2_normalized_intensities"])
    mzs = t_full.column("ms2_mzs")
    its = t_full.column("ms2_normalized_intensities")
    refs = []
    for j in np.where(ok)[0]:
        gi = int(idx[j])
        mz = np.asarray(mzs[gi].as_py(), dtype=np.float64)
        it = np.asarray(its[gi].as_py(), dtype=np.float64)
        if mz is None or len(mz) == 0:
            continue
        refs.append({"ik14": str(ik[j]), "adduct": str(ad[j]),
                     "nm": float(nm[j]), "mz": mz, "it": it})
    return refs, {"train_rows": n, "stride": int(stride), "sampled": len(refs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--max-refs-per-query", type=int, default=2000)
    args = ap.parse_args()
    t0 = time.time()
    queries = load_queries()
    print(f"queries loaded: {len(queries)}", flush=True)
    if args.probe:
        queries = queries[:5]
    if args.max_queries:
        queries = queries[:args.max_queries]
    refs, meta = load_train_sample()
    print(f"train sample: {meta}", flush=True)
    ref_nm = np.array([r["nm"] for r in refs])
    order = np.argsort(ref_nm)
    refs = [refs[i] for i in order]
    ref_nm = ref_nm[order]
    # clean refs once
    for r in refs:
        cmz, cit = clean(r["mz"], r["it"])
        r["cmz"], r["cit"] = cmz, cit
    rows = []
    for q in queries:
        qmz, qit = clean(q["mz"], q["it"])
        if len(qmz) == 0:
            rows.append({"case": q["case"], "error": "empty_query_spectrum"})
            continue
        # query neutral mass: prefer precursor+adduct else median of train? fallback skip shift
        qnm = neutral_mass(q["prec"] if q["prec"] else np.nan, q["adduct"] or "")
        if not np.isfinite(qnm):
            # fallback: use max-peak mz as rough center for window on precursor space
            rows.append({"case": q["case"], "error": "unknown_query_neutral_mass"})
            continue
        lo = np.searchsorted(ref_nm, qnm - ANALOG_WIN, "left")
        hi = np.searchsorted(ref_nm, qnm + ANALOG_WIN, "right")
        cand_idx = np.arange(lo, hi)
        if len(cand_idx) > args.max_refs_per_query:
            # closest in mass first (bounded pilot)
            d = np.abs(ref_nm[cand_idx] - qnm)
            cand_idx = cand_idx[np.argsort(d)[:args.max_refs_per_query]]
        direct_best = 0.0
        analog_best = 0.0
        analog_shift_best = 0.0
        n_direct = 0
        n_analog = 0
        for ci in cand_idx:
            r = refs[ci]
            if len(r["cmz"]) == 0:
                continue
            if r["ik14"] == q["truth14"]:
                n_direct += 1
                s = cos_sim(qmz, qit, r["cmz"], r["cit"])
                if s > direct_best:
                    direct_best = s
            else:
                n_analog += 1
                s = cos_sim(qmz, qit, r["cmz"], r["cit"])
                if s > analog_best:
                    analog_best = s
                ss = cos_shift(qmz, qit, r["cmz"], r["cit"], qnm - r["nm"])
                if ss > analog_shift_best:
                    analog_shift_best = ss
        rows.append({"case": q["case"], "dataset": q["dataset"], "truth14": q["truth14"],
                     "qnm": qnm, "n_direct": n_direct, "n_analog": n_analog,
                     "direct_best": direct_best, "analog_best": analog_best,
                     "analog_shift_best": analog_shift_best})
    # summary by dataset, domain A (analog only, same-group removed) is analog_* columns
    summary = {}
    for ds in ["NP", "G1", "R3"]:
        sub = [r for r in rows if r.get("dataset") == ds and "error" not in r]
        if not sub:
            continue
        summary[ds] = {
            "n": len(sub),
            "direct_cov_0_3": sum(1 for r in sub if r["direct_best"] >= 0.3),
            "direct_cov_0_5": sum(1 for r in sub if r["direct_best"] >= 0.5),
            "analog_cov_0_3": sum(1 for r in sub if r["analog_best"] >= 0.3),
            "analog_shift_cov_0_3": sum(1 for r in sub if r["analog_shift_best"] >= 0.3),
            "mean_direct": float(np.mean([r["direct_best"] for r in sub])),
            "mean_analog": float(np.mean([r["analog_best"] for r in sub])),
            "mean_analog_shift": float(np.mean([r["analog_shift_best"] for r in sub])),
        }
    os.makedirs("artifacts/runs/013-analog", exist_ok=True)
    out = {"meta": {"seconds": time.time() - t0, "train": meta, "tol": TOL,
                    "analog_win": ANALOG_WIN, "note": "pilot sample, inchikey14 proxy grouping"},
           "summary": summary, "rows": rows}
    json.dump(out, open("artifacts/runs/013-analog/pilot_support.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote artifacts/runs/013-analog/pilot_support.json in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
