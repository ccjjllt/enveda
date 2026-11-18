#!/usr/bin/env python3
"""Build ground truth for the visible placeholder test set (400 molecules).

The competition states the visible test molecules are copies of training
spectra. This script verifies that claim and recovers the labels.

Method
  test side : hash = blake2b(round(mz, 4).tobytes()) per spectrum, plus a
              3-decimal variant and an intensity-inclusive variant.
  train side: stream train.parquet row-group by row-group, batch by batch,
              keep only rows whose precursor_mz is within 0.01 Da of some test
              precursor, then hash the same way.
  aggregate : per test molecule_id, collect the inchikey14 witnesses.

Outputs (artifacts/ is gitignored):
  artifacts/truth/visible_truth.parquet
  artifacts/truth/spectrum_matches.parquet
  artifacts/truth/summary.json
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TEST_PATH = "data/raw/test.parquet"
TRAIN_PATH = "data/raw/train.parquet"
OUT_DIR = "artifacts/truth"
PREC_TOL = 0.01          # Da window gating the train-side scan
BATCH_SIZE = 16384

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def peak_hash(mzs, ndigits: int, intensities=None) -> str:
    a = np.asarray(mzs, dtype=np.float64)
    h = hashlib.blake2b(np.round(a, ndigits).tobytes(), digest_size=16)
    if intensities is not None:
        b = np.asarray(intensities, dtype=np.float64)
        h.update(b"|")
        h.update(np.round(b, ndigits).tobytes())
    return h.hexdigest()


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ test
    test = pq.read_table(TEST_PATH).to_pandas()
    test["n_peaks"] = test.ms2_mzs.apply(len)
    test["h4"] = [peak_hash(m, 4) for m in test.ms2_mzs]
    test["h3"] = [peak_hash(m, 3) for m in test.ms2_mzs]
    test["h4i"] = [
        peak_hash(m, 4, i) for m, i in zip(test.ms2_mzs, test.ms2_normalized_intensities)
    ]
    test_h4_set = set(test.h4)
    test_h3_set = set(test.h3)
    test_h4i_set = set(test.h4i)
    log(
        f"test: {len(test)} spectra / {test.molecule_id.nunique()} molecules; "
        f"unique h4={test.h4.nunique()} h3={test.h3.nunique()} h4i={test.h4i.nunique()}"
    )

    test_prec_sorted = np.sort(test.precursor_mz.to_numpy(dtype=np.float64))

    def candidate_mask(prec: np.ndarray) -> np.ndarray:
        lo = np.searchsorted(test_prec_sorted, prec - PREC_TOL, side="left")
        hi = np.searchsorted(test_prec_sorted, prec + PREC_TOL, side="right")
        return hi > lo

    # --------------------------------------------------------------- scan
    # h4 -> inchikey14 -> evidence
    h4_hits: dict[str, dict[str, dict]] = defaultdict(dict)
    h3_train_seen: set[str] = set()
    h4i_train_seen: set[str] = set()
    n_rows_scanned = 0
    n_candidate_rows = 0

    pf = pq.ParquetFile(TRAIN_PATH)
    cols = [
        "inchikey14",
        "inchikey",
        "normalized_smiles",
        "molecular_formula",
        "ingest_lib",
        "adduct",
        "ionization_mode",
        "instrument_type",
        "precursor_mz",
        "num_peaks",
        "ms2_mzs",
        "ms2_normalized_intensities",
    ]

    for rg in range(pf.metadata.num_row_groups):
        rg_t0 = time.time()
        n_rg_cand = 0
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, row_groups=[rg], columns=cols):
            n_rows_scanned += batch.num_rows
            prec = batch.column("precursor_mz").to_numpy(zero_copy_only=False)
            mask = candidate_mask(prec)
            if not mask.any():
                continue
            sub = batch.filter(pa.array(mask))
            n_rg_cand += sub.num_rows
            n_candidate_rows += sub.num_rows

            mzs = sub.column("ms2_mzs").to_pylist()
            ints = sub.column("ms2_normalized_intensities").to_pylist()
            ik14 = sub.column("inchikey14").to_pylist()
            ikfull = sub.column("inchikey").to_pylist()
            smi = sub.column("normalized_smiles").to_pylist()
            formula = sub.column("molecular_formula").to_pylist()
            lib = sub.column("ingest_lib").to_pylist()
            adduct = sub.column("adduct").to_pylist()
            ionmode = sub.column("ionization_mode").to_pylist()
            instr = sub.column("instrument_type").to_pylist()
            pvec = sub.column("precursor_mz").to_pylist()

            for j in range(sub.num_rows):
                h4 = peak_hash(mzs[j], 4)
                if h4 not in test_h4_set:
                    # cheap diagnostics: does this row match some test row at 3 dp?
                    h3 = peak_hash(mzs[j], 3)
                    if h3 in test_h3_set:
                        h3_train_seen.add(h3)
                    continue
                h4i = peak_hash(mzs[j], 4, ints[j])
                if h4i in test_h4i_set:
                    h4i_train_seen.add(h4i)
                rec = h4_hits[h4].get(ik14[j])
                if rec is None:
                    rec = h4_hits[h4][ik14[j]] = {
                        "smiles": set(),
                        "inchikey": set(),
                        "formula": set(),
                        "lib": set(),
                        "adducts": set(),
                        "ionization_mode": set(),
                        "instrument": set(),
                        "precursors": [],
                        "n_spectra": 0,
                    }
                rec["smiles"].add(smi[j])
                rec["inchikey"].add(ikfull[j])
                rec["formula"].add(formula[j])
                rec["lib"].add(lib[j])
                rec["adducts"].add(adduct[j])
                rec["ionization_mode"].add(ionmode[j])
                rec["instrument"].add(instr[j])
                rec["precursors"].append(round(float(pvec[j]), 4))
                rec["n_spectra"] += 1

        log(
            f"rg {rg:2d}/{pf.metadata.num_row_groups - 1} "
            f"rows={pf.metadata.row_group(rg).num_rows} candidates={n_rg_cand} "
            f"({time.time() - rg_t0:.1f}s) cum_candidates={n_candidate_rows}"
        )

    log(f"scan done: rows={n_rows_scanned} candidates={n_candidate_rows} "
        f"matched_h4={len(h4_hits)} h3_only={len(h3_train_seen - test_h4_set)} "
        f"h4i={len(h4i_train_seen)}")

    # ---------------------------------------------------------- per spectrum
    spec_rows = []
    for row in test.itertuples(index=False):
        hit = h4_hits.get(row.h4, {})
        if hit:
            level = "exact4"
        elif row.h3 in h3_train_seen:
            level = "exact3"
        else:
            level = "none"
        cands = sorted(hit.keys())
        spec_rows.append(
            {
                "molecule_id": row.molecule_id,
                "spectrum_id": row.spectrum_id,
                "adduct": row.adduct,
                "ionization_mode": row.ionization_mode,
                "precursor_mz": float(row.precursor_mz),
                "n_peaks": int(row.n_peaks),
                "match_level": level,
                "n_candidate_inchikeys": len(cands),
                "candidate_inchikeys": ";".join(cands),
                "n_train_spectra": int(sum(v["n_spectra"] for v in hit.values())),
                "intensity_identical": bool(row.h4i in h4i_train_seen),
                "train_adducts": ";".join(
                    sorted({a for v in hit.values() for a in v["adducts"]})
                ),
            }
        )
    spec_df = pa.Table.from_pylist(spec_rows).to_pandas()
    log("spectrum match levels: " + json.dumps(spec_df.match_level.value_counts().to_dict()))

    # ---------------------------------------------------------- per molecule
    sid_to_h4 = dict(zip(test.spectrum_id, test.h4))
    mol_rows = []
    for mol, grp in spec_df.groupby("molecule_id", sort=True):
        cands = set()
        for s in grp.candidate_inchikeys:
            if s:
                cands.update(s.split(";"))
        n_spec = len(grp)
        n_matched = int((grp.match_level != "none").sum())
        frac = round(n_matched / n_spec, 4)
        if not cands:
            status = "unmatched"
        elif len(cands) == 1 and n_matched == n_spec:
            status = "unique"
        elif len(cands) == 1:
            status = "partial_unique"
        else:
            status = "ambiguous"

        smiles = formula = libs = inchikey_full = ""
        n_witness = 0
        if len(cands) == 1:
            ik = next(iter(cands))
            for sid in grp.spectrum_id:
                ev = h4_hits.get(sid_to_h4[sid], {}).get(ik)
                if ev:
                    smiles = "|".join(sorted(ev["smiles"]))
                    formula = "|".join(sorted(ev["formula"]))
                    libs = "|".join(sorted(ev["lib"]))
                    inchikey_full = "|".join(sorted(ev["inchikey"]))
                    break
        n_witness = int(grp.n_train_spectra.sum()) if len(grp) else 0

        mol_rows.append(
            {
                "molecule_id": mol,
                "inchikey14": next(iter(cands)) if len(cands) == 1 else "",
                "smiles": smiles,
                "n_matched_spectra": n_matched,
                "n_candidate_inchikeys": len(cands),
                "matched_spectra_fraction": frac,
                "status": status,
                "n_spectra_total": n_spec,
                "candidate_inchikeys": ";".join(sorted(cands)),
                "molecular_formula": formula,
                "inchikey": inchikey_full,
                "n_train_witness_spectra": n_witness,
                "ingest_lib": libs,
                "adducts": ";".join(sorted(set(grp.adduct))),
            }
        )

    mol_df = pa.Table.from_pylist(mol_rows).to_pandas()
    status_counts = {k: int(v) for k, v in mol_df.status.value_counts().to_dict().items()}
    log(f"molecule status: {json.dumps(status_counts)}")

    # ---------------------------------------------------------------- write
    truth = mol_df[
        [
            "molecule_id",
            "inchikey14",
            "smiles",
            "n_matched_spectra",
            "n_candidate_inchikeys",
            "matched_spectra_fraction",
            "status",
            "n_spectra_total",
            "candidate_inchikeys",
            "molecular_formula",
            "inchikey",
            "n_train_witness_spectra",
            "ingest_lib",
            "adducts",
        ]
    ]
    truth_path = os.path.join(OUT_DIR, "visible_truth.parquet")
    truth.to_parquet(truth_path, index=False)
    spec_path = os.path.join(OUT_DIR, "spectrum_matches.parquet")
    spec_df.to_parquet(spec_path, index=False)

    adduct_tbl = []
    for ad, g in spec_df.groupby("adduct", sort=True):
        adduct_tbl.append(
            {
                "adduct": ad,
                "spectra": int(len(g)),
                "matched_exact4": int((g.match_level == "exact4").sum()),
                "matched_exact3": int((g.match_level == "exact3").sum()),
                "unmatched": int((g.match_level == "none").sum()),
                "intensity_identical": int(g.intensity_identical.sum()),
            }
        )
    adduct_mol = []
    mol_expl = mol_df.assign(adduct=mol_df.adducts.str.split(";")).explode("adduct")
    for ad, g in mol_expl.groupby("adduct", sort=True):
        adduct_mol.append(
            {
                "adduct": ad,
                "molecules": int(g.molecule_id.nunique()),
                "unique": int((g.status == "unique").sum()),
                "partial_unique": int((g.status == "partial_unique").sum()),
                "ambiguous": int((g.status == "ambiguous").sum()),
                "unmatched": int((g.status == "unmatched").sum()),
            }
        )

    summary = {
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test_spectra": int(len(spec_df)),
        "test_molecules": int(mol_df.molecule_id.nunique()),
        "train_rows_scanned": int(n_rows_scanned),
        "train_candidate_rows": int(n_candidate_rows),
        "precursor_tolerance_da": PREC_TOL,
        "molecule_status": status_counts,
        "molecule_status_pct": {
            k: round(100.0 * v / len(mol_df), 2) for k, v in status_counts.items()
        },
        "spectrum_match_level": {
            k: int(v)
            for k, v in spec_df.match_level.value_counts().to_dict().items()
        },
        "spectrum_intensity_identical": int(spec_df.intensity_identical.sum()),
        "adduct_spectrum_table": adduct_tbl,
        "adduct_molecule_table": adduct_mol,
        "ambiguous_molecules": mol_df.loc[mol_df.status == "ambiguous", "molecule_id"].tolist(),
        "unmatched_molecules": mol_df.loc[mol_df.status == "unmatched", "molecule_id"].tolist(),
        "partial_unique_molecules": mol_df.loc[
            mol_df.status == "partial_unique", "molecule_id"
        ].tolist(),
        "wall_seconds": round(time.time() - T0, 1),
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1
        ),
        "outputs": {"truth": truth_path, "spectra": spec_path},
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    log("SUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
