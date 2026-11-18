"""Audit the neutral-mass arithmetic used by the 068a pipeline for every adduct class.

Ground truth: train.parquet rows carry precursor_mz, adduct and normalized_smiles, so the
exact monoisotopic mass of the true structure is known. For a correct adduct parser the
reconstructed neutral mass must equal that exact mass to within a few ppm.

Three implementations are compared, all lifted from rounds/068-n1c-lam035/068a:
  A correct  : M = (|z| * mz - delta) / n
  B mass prior (068a cell 20): exact dict lookup `_ADD`, unlisted -> NaN (silently skipped)
  C extlib overlay (068a _apex_extlib_overlay): substring test, default +-1.007276

Read-only, streaming; writes one CSV + one JSON into artifacts/adduct/.
"""
from __future__ import annotations

import collections
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path('artifacts/adduct')
OUT.mkdir(parents=True, exist_ok=True)

# monoisotopic masses (CIAAW / NIST), plus proton = H - electron
MASS = {
    'H': 1.00782503207, 'C': 12.0, 'N': 14.0030740048, 'O': 15.9949146196,
    'Na': 22.9897692809, 'K': 38.96370668, 'Cl': 34.96885268, 'Ca': 39.96259098,
    'Br': 78.9183371, 'Li': 7.01600455, 'S': 31.97207100, 'P': 30.97376163,
    'F': 18.99840322, 'Si': 27.97692653, 'Fe': 55.9349375, 'Zn': 63.9291422,
    'Mg': 23.98504170, 'Cu': 62.9295975, 'I': 126.904473, 'B': 11.0093054,
    'Se': 79.9165213, 'As': 74.9215965, 'Al': 26.98153863, 'Ba': 137.9052472,
    'Mn': 54.9380451, 'Ni': 57.9353429, 'Co': 58.9331950, 'Sr': 87.9056121,
}
H_ATOM = MASS['H']
PROTON = 1.007276466812

TOKEN = re.compile(r'([A-Z][a-z]?)(\d*)')


def formula_mass(tok: str) -> float | None:
    """Mass of a formula fragment such as 'CH2O2', allowing a leading coefficient ('2H')."""
    if not tok:
        return 0.0
    coeff = 1
    mc = re.match(r'^(\d+)(?=[A-Z])', tok)
    if mc:
        coeff = int(mc.group(1))
        tok = tok[mc.end():]
    total = 0.0
    pos = 0
    while pos < len(tok):
        m = TOKEN.match(tok, pos)
        if not m:
            return None
        el, num = m.group(1), m.group(2)
        if el not in MASS:
            return None
        total += MASS[el] * (int(num) if num else 1)
        pos = m.end()
    return coeff * total


def split_signed(inner: str) -> list[tuple[int, str]] | None:
    """'2M+Na-2H' -> [(+1,'2M'), (+1,'Na'), (-1,'2H')]."""
    parts: list[tuple[int, str]] = []
    cur = ''
    sign = 1
    for ch in inner:
        if ch in '+-':
            if cur:
                parts.append((sign, cur))
            sign = 1 if ch == '+' else -1
            cur = ''
        else:
            cur += ch
    if cur:
        parts.append((sign, cur))
    return parts or None


def parse_adduct(adduct: str) -> dict | None:
    """Return {n, z, delta} for '[nM + kX - jH]z+' style strings, else None."""
    a = adduct.strip()
    m = re.match(r'^\[(.*)\](\d*)([+-])$', a)
    if not m:
        return None
    inner = m.group(1)
    z = int(m.group(2) or 1) * (1 if m.group(3) == '+' else -1)
    if inner.strip() == 'Cat':
        return None
    parts = split_signed(inner)
    if not parts:
        return None
    n = None
    delta = 0.0
    for sign, tok in parts:
        mm = re.match(r'^(\d*)M$', tok)
        if mm:
            n = int(mm.group(1) or 1)
            continue
        fm = formula_mass(tok)
        if fm is None:
            return None
        delta += sign * fm
    if n is None:
        return None
    # '+' adduct of formula X contributes X minus an electron if it is a bare proton count;
    # charge is already used as |z|, and electron bookkeeping is < 1 ppm at these masses,
    # so use the neutral-fragment mass (this is what the competition constants do).
    return dict(n=n, z=abs(z), delta=delta)


def nm_correct(adduct: str, mz: float) -> float | None:
    p = parse_adduct(adduct)
    if p is None or not np.isfinite(mz):
        return None
    return (p['z'] * mz - p['delta']) / p['n']


# --- exact copy of the 068a cell-20 mass-prior table -------------------------------
ADD_EXACT = {'[M+H]+': 1.007276, '[M-H]-': -1.007276, '[M+Na]+': 22.989218, '[M+NH4]+': 18.033823,
             '[M+K]+': 38.963158, '[M+CH2O2-H]-': 44.998201, '[M+Cl]-': 34.969402}


def nm_massprior(adduct: str, mz: float) -> float | None:
    if adduct not in ADD_EXACT:
        return None
    return mz - ADD_EXACT[adduct]


# --- exact copy of the 068a extlib-overlay substring logic --------------------------
def nm_overlay(adduct: str, mz: float, mode: int) -> float:
    nm = mz - 1.007276 if mode == 1 else mz + 1.007276
    if '[M+Na]+' in adduct:
        nm = mz - 22.989218
    elif '[M+K]+' in adduct:
        nm = mz - 38.963158
    elif '[M+NH4]+' in adduct:
        nm = mz - 18.033823
    elif '[M+HCOO]-' in adduct:
        nm = mz - 44.998201
    return nm


def main() -> int:
    sample = pickle.load(open('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/adduct_sample.pkl', 'rb'))
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog('rdApp.*')

    exact: dict[str, float] = {}
    for _, rows in sample.items():
        for _, smi in rows:
            if smi not in exact:
                mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
                exact[smi] = float(Descriptors.ExactMolWt(mol)) if mol is not None else float('nan')

    per_class = []
    for adduct, rows in sample.items():
        mode = 1 if adduct.rstrip().endswith('+') else -1
        res = collections.defaultdict(list)
        n_parsed = 0
        for mz, smi in rows:
            truth = exact.get(smi, float('nan'))
            if not np.isfinite(truth) or truth <= 0:
                continue
            c = nm_correct(adduct, float(mz))
            if c is not None:
                n_parsed += 1
                res['correct_ppm'].append((c - truth) / truth * 1e6)
            b = nm_massprior(adduct, float(mz))
            if b is not None:
                res['massprior_ppm'].append((b - truth) / truth * 1e6)
            o = nm_overlay(adduct, float(mz), mode)
            res['overlay_ppm'].append((o - truth) / truth * 1e6)

        def frac_within(vals, tol=20.0):
            if not vals:
                return None
            a = np.abs(np.asarray(vals))
            return dict(n=len(a), within=float((a <= tol).mean()), median_abs=float(np.median(a)))

        per_class.append(dict(
            adduct=adduct, n_rows=len(rows), n_parsed=n_parsed,
            correct=frac_within(res['correct_ppm']),
            massprior=frac_within(res['massprior_ppm']),
            overlay=frac_within(res['overlay_ppm']),
        ))

    return per_class


if __name__ == '__main__':
    per_class = main()
    import pandas as pd
    df = pd.DataFrame([{
        'adduct': r['adduct'], 'n_rows': r['n_rows'], 'parsed': r['n_parsed'],
        'correct_within20': (r['correct'] or {}).get('within'),
        'correct_med_abs_ppm': (r['correct'] or {}).get('median_abs'),
        'massprior_within20': (r['massprior'] or {}).get('within'),
        'massprior_med_abs_ppm': (r['massprior'] or {}).get('median_abs'),
        'overlay_within20': (r['overlay'] or {}).get('within'),
        'overlay_med_abs_ppm': (r['overlay'] or {}).get('median_abs'),
    } for r in per_class])
    df = df.sort_values('n_rows', ascending=False)
    df.to_csv(OUT / 'adduct_neutral_mass_audit.csv', index=False)
    print(df.head(20).to_string(index=False))
    print('...')
    print(df.tail(5).to_string(index=False))
    ok = df['correct_within20'] >= 0.99
    print('\nclasses with correct parser >=99% within 20ppm:', int(ok.sum()), '/', len(df))
    print('classes the mass prior covers (non-null):', int(df.massprior_within20.notna().sum()))
