#!/usr/bin/env python3
"""Independent checks on artifacts/truth/visible_truth.parquet.

The hash match already ties each test spectrum to one exact training row, but a
hash tells us nothing about chemical plausibility.  This probe re-derives the
monoisotopic mass from the matched molecular_formula and checks it against the
test precursor_mz under the test adduct.  A wrong label (or a wrong adduct)
shows up as a large ppm error.

Read-only; prints a report and exits non-zero only on a hard assumption break.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

TRUTH = "artifacts/truth/visible_truth.parquet"
SPEC = "artifacts/truth/spectrum_matches.parquet"
TEST = "data/raw/test.parquet"

MONO = {
    "C": 12.0,
    "H": 1.00782503207,
    "N": 14.0030740048,
    "O": 15.9949146196,
    "P": 30.97376163,
    "S": 31.97207100,
    "F": 18.99840322,
    "Cl": 34.96885268,
    "Br": 78.9183371,
    "I": 126.904473,
    "Na": 22.9897692809,
    "K": 38.96370668,
    "Si": 27.9769265325,
    "B": 11.0093054,
    "Se": 79.9165213,
    "As": 74.9215965,
    "Fe": 55.9349375,
    "Zn": 63.9291422,
    "Mg": 23.9850417,
    "Ca": 39.9625909,
    "Li": 7.01600455,
    "Al": 26.98153863,
    "Ni": 57.9353429,
    "Cu": 62.9295975,
    "Mn": 54.9380451,
    "Co": 58.9331950,
    "Sn": 119.9021947,
    "Ba": 137.9052472,
    "Hg": 201.9706430,
    "Pb": 207.9766521,
}

PROTON = 1.007276467
ELECTRON = 0.00054858
ADDUCT_SHIFT = {
    "[M+H]+": PROTON,
    "[M-H]-": -PROTON,
    "[M+Na]+": 22.9897692809 - ELECTRON,
    "[M+K]+": 38.96370668 - ELECTRON,
    "[M+NH4]+": 18.03382555 - ELECTRON,   # NH4+
    "[M+Cl]-": 34.96885268 + ELECTRON,
    "[M+CH2O2-H]-": 46.0054793 - PROTON,   # formic acid - H
    "[M+C2H4O2-H]-": 60.0211294 - PROTON,  # acetic acid - H
    "[M]+": -ELECTRON,
    "[M+2H]2+": PROTON,
    "[2M+Na]+": 22.9897692809 - ELECTRON,
}

TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def monoisotopic(formula: str) -> float:
    pos, total = 0, 0.0
    while pos < len(formula):
        m = TOKEN.match(formula, pos)
        if not m:
            raise ValueError(f"cannot parse formula at {pos}: {formula}")
        el, n = m.group(1), m.group(2)
        total += MONO[el] * (int(n) if n else 1)
        pos = m.end()
    return total


def main() -> int:
    truth = pd.read_parquet(TRUTH)
    spec = pd.read_parquet(SPEC)
    test = pd.read_parquet(TEST, columns=["spectrum_id", "molecule_id", "adduct", "precursor_mz"])

    print(f"truth rows={len(truth)}  spectra rows={len(spec)}  test rows={len(test)}")

    # 1. coverage
    print(f"distinct truth inchikey14      : {truth.inchikey14.replace('', np.nan).nunique()}")
    print(f"molecules with empty inchikey14: {(truth.inchikey14 == '').sum()}")
    print(f"status counts                  : {truth.status.value_counts().to_dict()}")
    dup = truth[truth.inchikey14 != ""].inchikey14.value_counts()
    dup = dup[dup > 1]
    print(f"inchikey14 shared by >1 molecule: {len(dup)} {list(dup.index)[:5]}")

    # 2. every spectrum matched, and matched molecule count per molecule
    print(f"spectra with match_level!=exact4: {(spec.match_level != 'exact4').sum()}")
    print(f"spectra with intensity_identical: {int(spec.intensity_identical.sum())}")
    print(f"spectra whose train adduct == test adduct: "
          f"{int((spec.train_adducts == spec.adduct).sum())} / {len(spec)}")
    mism = spec[spec.train_adducts != spec.adduct]
    if len(mism):
        print(mism[["spectrum_id", "adduct", "train_adducts"]].head(20).to_string())

    # 3. formula / adduct / precursor consistency
    s = spec.merge(truth[["molecule_id", "molecular_formula"]], on="molecule_id", how="left")
    s = s[s.molecular_formula.fillna("") != ""]
    ppm_rows = []
    for r in s.itertuples(index=False):
        try:
            m = monoisotopic(r.molecular_formula)
        except Exception:
            ppm_rows.append(np.nan)
            continue
        shift = ADDUCT_SHIFT.get(r.adduct)
        ppm_rows.append(np.nan if shift is None else (r.precursor_mz - (m + shift)) / (m + shift) * 1e6)
    s = s.assign(ppm=np.abs(ppm_rows))
    ok = s.ppm.dropna()
    print(f"\nprecursor-vs-formula check: n={len(ok)} "
          f"median|ppm|={ok.median():.2f} p90={ok.quantile(0.9):.2f} "
          f"max={ok.max():.2f}")
    for thr in (5, 20, 50, 200):
        print(f"  |ppm| <= {thr:4d}: {int((ok <= thr).sum())}/{len(ok)} "
              f"({100.0 * (ok <= thr).mean():.1f}%)")
    bad = s[s.ppm > 50]
    if len(bad):
        print("  spectra outside 50 ppm:")
        print(bad[["molecule_id", "adduct", "precursor_mz", "molecular_formula", "ppm"]]
              .sort_values("ppm", ascending=False).head(20).to_string(index=False))

    # 4. SMILES -> InChIKey round trip (rdkit optional)
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        bad_smi, bad_ik = [], []
        for r in truth.itertuples(index=False):
            m = Chem.MolFromSmiles(r.smiles)
            if m is None:
                bad_smi.append(r.molecule_id)
                continue
            if Chem.MolToInchiKey(m)[:14] != r.inchikey14:
                bad_ik.append(r.molecule_id)
        print(f"\nrdkit: unparseable smiles={len(bad_smi)}  inchikey14 mismatch={len(bad_ik)}")
        isomer = truth.groupby("molecular_formula").inchikey14.nunique()
        print(f"rdkit: formulas with >1 distinct molecule = {int((isomer > 1).sum())} "
              f"(max {int(isomer.max())} isomers per formula)")
    except ImportError:
        print("\nrdkit not installed, skipping SMILES round-trip check")

    # 5. per-adduct molecule coverage
    print("\nper-adduct molecules (from truth):")
    print(truth.assign(adduct=truth.adducts.str.split(";")).explode("adduct")
          .groupby("adduct").size().to_string())

    # 6. spectra per molecule vs matched
    print("\nmatched_spectra_fraction distribution:",
          truth.matched_spectra_fraction.value_counts().to_dict())
    print("n_matched_spectra total:", int(truth.n_matched_spectra.sum()),
          "/ spectra in test:", len(test))

    json.dump(
        {
            "distinct_truth_inchikey14": int(truth.inchikey14.replace("", np.nan).nunique()),
            "ppm_median": float(ok.median()),
            "ppm_p90": float(ok.quantile(0.9)),
            "ppm_max": float(ok.max()),
            "ppm_within_20": int((ok <= 20).sum()),
            "ppm_within_50": int((ok <= 50).sum()),
            "n_checked": int(len(ok)),
            "adduct_mismatch_spectra": int((spec.train_adducts != spec.adduct).sum()),
        },
        open("artifacts/truth/verify.json", "w"),
        indent=2,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
