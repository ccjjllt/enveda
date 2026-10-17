"""Manual, bounded audit of 013 support scores; never produces submission ranks.

Fixed 36-query metadata/hash sample, stride-50 train references, CPU only.
Keep the original 013 artifacts unchanged. Removing precursor +/-1.5 Da is a
diagnostic ablation, not a proposed production preprocessing rule.
"""
import collections
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import build_013_analog_pilot as pilot

OUT = Path("artifacts/013-support-audit")
SEED = "20261017|013-support-audit-v1"
PRECURSOR_RADIUS = 1.5
MAX_REFS = 2000


def peak_hash(mz, it):
    """Exact raw peak-array hash, float64 representation, no rounding."""
    h = hashlib.sha256()
    for a in (mz, it):
        v = np.asarray(a, dtype="<f8")
        h.update(np.asarray([len(v)], dtype="<i8").tobytes())
        h.update(v.tobytes())
    return h.hexdigest()


def all_query_hashes(first_queries):
    hashes = {peak_hash(q["mz"], q["it"]) for q in first_queries}
    for filename in ("artifacts/f1-bundle/inputs.json", "artifacts/g1-validation/inputs.json"):
        for m in json.load(open(filename))["molecules"]:
            for s in m["spectra"]:
                hashes.add(peak_hash(s["mz"], s["it"]))
    for m in json.load(open("artifacts/007-r3/observations_50.json"))["molecules"]:
        for s in m["spectra"]:
            with np.load(s["peak_file_ref"]) as z:
                mk = next(k for k in z.files if "mz" in k.lower())
                ik = next(k for k in z.files if "int" in k.lower())
                hashes.add(peak_hash(z[mk].ravel(), z[ik].ravel()))
    return hashes


def choose_queries(queries):
    strata = collections.defaultdict(list)
    for q in queries:
        if not np.isfinite(pilot.neutral_mass(q["prec"], q["adduct"])):
            continue
        negative = q["adduct"].endswith("-")
        group = "negative" if negative else "positive"
        if q["dataset"] == "G1" and not negative:
            group = "positive_dimer" if q["adduct"].startswith("[2M") else "positive_monomer"
        strata[(q["dataset"], group)].append(q)
    selected = []
    counts = {}
    for (ds, group), rows in sorted(strata.items()):
        n = 4 if ds == "G1" else 6
        rows.sort(key=lambda q: hashlib.sha256(f'{SEED}|{q["case"]}'.encode()).hexdigest())
        selected.extend(rows[:n])
        counts[f"{ds}:{group}"] = {"available": len(rows), "selected": min(n, len(rows))}
    return selected, counts


def remove_precursor(mz, it, precursor):
    keep = np.abs(mz - float(precursor)) > PRECURSOR_RADIUS
    return mz[keep], it[keep]


def read_refs(identity, r3_keys):
    file = pq.ParquetFile("data/raw/train.parquet")
    stride = max(1, file.metadata.num_rows // pilot.REF_SAMPLE)
    cols = ["inchikey14", "adduct", "precursor_mz", "ingest_lib",
            "ionization_mode", "ms2_mzs", "ms2_normalized_intensities"]
    refs, offset = [], 0
    full_sources, sampled_sources = collections.Counter(), collections.Counter()
    r3_raw_counts, r3_canonical_counts = collections.Counter(), collections.Counter()
    for batch in file.iter_batches(columns=cols, batch_size=8192):
        full_sources.update(batch.column("ingest_lib").to_pylist())
        for raw in batch.column("inchikey14").to_pylist():
            if raw in r3_keys:
                r3_raw_counts[raw] += 1
            canonical = identity.get(raw)
            if canonical in r3_keys:
                r3_canonical_counts[canonical] += 1
        idx = np.arange((-offset) % stride, batch.num_rows, stride, dtype=np.int64)
        for local, row in zip(idx, batch.take(idx).to_pylist()):
            nm = pilot.neutral_mass(row["precursor_mz"], row["adduct"])
            if not np.isfinite(nm) or not row["ms2_mzs"]:
                continue
            raw = str(row["inchikey14"])
            cmz, cit = pilot.clean(row["ms2_mzs"], row["ms2_normalized_intensities"])
            refs.append({"row": offset + int(local), "raw": raw,
                         "canonical": identity.get(raw), "nm": nm,
                         "prec": row["precursor_mz"], "adduct": row["adduct"],
                         "source": row["ingest_lib"], "mode": row["ionization_mode"],
                         "cmz": cmz, "cit": cit,
                         "hash": peak_hash(row["ms2_mzs"], row["ms2_normalized_intensities"])})
            sampled_sources[row["ingest_lib"]] += 1
        offset += batch.num_rows
    refs.sort(key=lambda r: r["nm"])
    return refs, {"train_rows": offset, "stride": stride, "sampled": len(refs),
                  "full_sources": dict(full_sources), "sampled_sources": dict(sampled_sources),
                  "r3_full_train_counts": {key: {"raw": r3_raw_counts[key], "canonical": r3_canonical_counts[key]}
                                           for key in sorted(r3_keys)},
                  "unmapped_sampled_refs": sum(r["canonical"] is None for r in refs)}


def matched_peaks(qm, qi, rm, ri, shift, qp, rp):
    denom = np.linalg.norm(qi) * np.linalg.norm(ri)
    if not denom:
        return []
    pairs, i, j = [], 0, 0
    while i < len(qm) and j < len(rm):
        diff = qm[i] - (rm[j] + shift)
        if diff < -pilot.TOL:
            i += 1
        elif diff > pilot.TOL:
            j += 1
        else:
            pairs.append({"query_mz": float(qm[i]), "ref_mz": float(rm[j]),
                          "contribution": float(qi[i] * ri[j] / denom),
                          "query_precursor_near": bool(abs(qm[i] - qp) <= PRECURSOR_RADIUS),
                          "ref_precursor_near": bool(abs(rm[j] - rp) <= PRECURSOR_RADIUS)})
            i += 1
            j += 1
    return sorted(pairs, key=lambda x: -x["contribution"])


def score(q, r, stripped=False):
    qm, qi = q["cmz"], q["cit"]
    rm, ri = r["cmz"], r["cit"]
    if stripped:
        qm, qi = remove_precursor(qm, qi, q["prec"])
        rm, ri = remove_precursor(rm, ri, r["prec"])
    shift = q["nm"] - r["nm"]
    direct = pilot.cos_sim(qm, qi, rm, ri)
    shifted = pilot.cos_sim(qm, qi, rm + shift, ri)
    return max(direct, shifted), shift if shifted > direct else 0.0


def winner(q, refs, stripped=False):
    if not refs:
        return None
    best_score, best, shift = -1.0, None, 0.0
    for r in refs:
        value, delta = score(q, r, stripped)
        if value > best_score:
            best_score, best, shift = value, r, delta
    return best, best_score, shift


def describe(q, result, stripped=False):
    if result is None:
        return {"score": 0.0, "reference": None}
    r, value, shift = result
    qm, qi, rm, ri = q["cmz"], q["cit"], r["cmz"], r["cit"]
    if stripped:
        qm, qi = remove_precursor(qm, qi, q["prec"])
        rm, ri = remove_precursor(rm, ri, r["prec"])
    pairs = matched_peaks(qm, qi, rm, ri, shift, q["prec"], r["prec"])
    # Annotation only: an oligomer precursor can dissociate to the intact
    # monomer ion, which precursor +/-1.5 Da removal does not remove.
    qn, qz, qd = pilot.ADDUCT_NZD[q["adduct"]]
    rn, rz, rd = pilot.ADDUCT_NZD[r["adduct"]]
    for pair in pairs:
        pair["query_oligomer_intact_monomer_near"] = bool(qn > 1 and abs(pair["query_mz"] - (q["nm"] + qd) / qz) <= pilot.TOL)
        pair["ref_oligomer_intact_monomer_near"] = bool(rn > 1 and abs(pair["ref_mz"] - (r["nm"] + rd) / rz) <= pilot.TOL)
    precursor_contribution = sum(p["contribution"] for p in pairs
                                if p["query_precursor_near"] or p["ref_precursor_near"])
    monomer_contribution = sum(p["contribution"] for p in pairs
                              if p["query_oligomer_intact_monomer_near"] or p["ref_oligomer_intact_monomer_near"])
    return {"score": value, "shift_used": shift,
            "reference": {k: r[k] for k in ("row", "raw", "canonical", "source", "mode", "adduct", "prec", "nm", "hash")},
            "query_peaks": len(qm), "reference_peaks": len(rm),
            "matched_peak_count": len(pairs), "matched_peaks": pairs,
            "precursor_pair_cosine_contribution": precursor_contribution,
            "oligomer_intact_monomer_fraction_of_score": monomer_contribution / value if value > 0 else 0.0,
            "largest_pair_fraction_of_score": pairs[0]["contribution"] / value if pairs and value > 0 else 0.0,
            "precursor_fraction_of_score": precursor_contribution / value if value > 0 else 0.0}


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    queries = pilot.load_queries()
    chosen, strata = choose_queries(queries)
    manifest = {"seed": SEED, "rule": "fixed metadata strata, ascending SHA256(seed|case); no support/rank/label-based selection",
                "strata": strata, "queries": [{k: q[k] for k in ("case", "dataset", "adduct", "prec")} for q in chosen]}
    (OUT / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2))
    identities = pq.read_table("artifacts/005/structure_identity_train.parquet",
                              columns=["raw_key", "canonical_group_key"]).to_pylist()
    identity = {r["raw_key"]: r["canonical_group_key"] for r in identities}
    refs, meta = read_refs(identity, {q["truth14"] for q in queries if q["dataset"] == "R3"})
    ref_mass = np.array([r["nm"] for r in refs])
    query_hashes = all_query_hashes(queries)
    hash_refs = collections.defaultdict(list)
    for r in refs:
        hash_refs[r["hash"]].append(r)
    duplicate_queries = []
    for q in queries:
        hits = hash_refs.get(peak_hash(q["mz"], q["it"]), [])
        if hits:
            duplicate_queries.append({"case": q["case"], "rows": [r["row"] for r in hits],
                                      "raw_keys": sorted({r["raw"] for r in hits})})
    original = {r["case"]: r for r in json.load(open("artifacts/013-analog/pilot_support.json"))["rows"]}
    output = []
    for q in chosen:
        q["cmz"], q["cit"] = pilot.clean(q["mz"], q["it"])
        q["nm"] = pilot.neutral_mass(q["prec"], q["adduct"])
        lo, hi = np.searchsorted(ref_mass, [q["nm"] - pilot.ANALOG_WIN, q["nm"] + pilot.ANALOG_WIN])
        idx = np.arange(lo, hi)
        idx = idx[np.argsort(np.abs(ref_mass[idx] - q["nm"]))[:MAX_REFS]]
        bounded = [refs[i] for i in idx if len(refs[i]["cmz"])]
        old = [r for r in bounded if r["raw"] != q["truth14"]]
        # Same bounded reference pool isolates identity/hash exclusions; do not refill it.
        independent = [r for r in bounded if r["canonical"] is not None and r["hash"] not in query_hashes]
        unseen = [r for r in independent if r["canonical"] != q["truth14"]]
        seen = [r for r in independent if r["canonical"] == q["truth14"]]
        orig = winner(q, old)
        same_ref = None if orig is None else (orig[0], *score(q, orig[0], True))
        old_direct = [r for r in bounded if r["raw"] == q["truth14"]]
        qcut, _ = remove_precursor(q["cmz"], q["cit"], q["prec"])
        output.append({"case": q["case"], "dataset": q["dataset"], "adduct": q["adduct"],
                       "prec": q["prec"], "query_n_peaks": len(q["cmz"]), "query_n_peaks_stripped": len(qcut),
                       "original_saved_score": original[q["case"]]["analog_shift_best"],
                       "original_analog": describe(q, orig),
                       "original_winner_stripped": describe(q, same_ref, True),
                       "original_pool_stripped_remax": describe(q, winner(q, old, True), True),
                       "canonical_unseen": describe(q, winner(q, unseen)),
                       "canonical_unseen_stripped_remax": describe(q, winner(q, unseen, True), True),
                       "counts": {"bounded_refs": len(bounded), "raw_analog_refs": len(old), "canonical_unseen_refs": len(unseen),
                                  "all_query_exact_hash_refs": sum(r["hash"] in query_hashes for r in bounded),
                                  "same_canonical_alias_refs_in_old_analog": sum(r["canonical"] == q["truth14"] for r in old)},
                       "direct_original": max((pilot.cos_sim(q["cmz"], q["cit"], r["cmz"], r["cit"]) for r in old_direct), default=0),
                       "direct_canonical_independent": max((pilot.cos_sim(q["cmz"], q["cit"], r["cmz"], r["cit"]) for r in seen), default=0)})
    summary = {}
    metrics = ("original_analog", "original_winner_stripped", "original_pool_stripped_remax", "canonical_unseen", "canonical_unseen_stripped_remax")
    for ds in ("NP", "G1", "R3", "all"):
        rr = [r for r in output if ds == "all" or r["dataset"] == ds]
        summary[ds] = {"n": len(rr), "empty_after_precursor_removal": sum(r["query_n_peaks_stripped"] == 0 for r in rr),
                       "original_winner_precursor_fraction_ge_0_8": sum(r["original_analog"]["precursor_fraction_of_score"] >= .8 for r in rr),
                       "original_winner_oligomer_intact_monomer_fraction_ge_0_8": sum(r["original_analog"]["oligomer_intact_monomer_fraction_of_score"] >= .8 for r in rr),
                       "original_winner_largest_pair_fraction_ge_0_8": sum(r["original_analog"]["largest_pair_fraction_of_score"] >= .8 for r in rr),
                       "original_winner_cross_polarity": sum(r["adduct"][-1] != r["original_analog"]["reference"]["adduct"][-1] for r in rr),
                       **{metric: {"mean": float(np.mean([r[metric]["score"] for r in rr])),
                                   "ge_0_3": sum(r[metric]["score"] >= .3 for r in rr),
                                   "ge_0_99": sum(r[metric]["score"] >= .99 for r in rr)} for metric in metrics}}
    max_diff = max(abs(r["original_analog"]["score"] - r["original_saved_score"]) for r in output)
    result = {"method": {"precursor_radius_da": PRECURSOR_RADIUS, "tol_da": pilot.TOL, "max_refs": MAX_REFS,
                         "identity": "existing RDKit canonical map, drop unmapped refs in canonical paths",
                         "exact_hash": "SHA256(length + little-endian float64 raw mz and raw intensity arrays), no rounding",
                         "query_exclusion": "union of all NP/G1/R3 query spectrum hashes, independent of truth",
                         "unseen_exclusion": "then remove canonical query structure; diagnostic uses truth for domain simulation only",
                         "stripped_scope": "diagnostic only; also may remove real near-precursor fragments; dimer monomer peaks remain",
                         "reference_pool": "same original bounded mass-nearest 2000, not refilled after exclusions"},
              "meta": meta, "exact_first_query_duplicate_hits_all_550": duplicate_queries,
              "all_query_hash_count": len(query_hashes), "sampled_refs_matching_any_query_hash": sum(r["hash"] in query_hashes for r in refs),
              "summary": summary, "max_original_recompute_abs_difference": max_diff,
              "wall_seconds": time.time() - start, "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "rows": output}
    (OUT / "support_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("summary", "max_original_recompute_abs_difference", "wall_seconds", "sampled_refs_matching_any_query_hash")}, indent=2))


if __name__ == "__main__":
    main()
