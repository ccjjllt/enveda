"""Spectrum <-> candidate features for a learned same-formula isomer reranker (v8, romanrozen).

Built on Spectral Parallax's MetFrag-lite enumeration (frag_rescore.py, imported as `frag`): every candidate gives the
neutral masses of the pieces left after removing 0, 1 or 2 bonds of its heavy-atom graph. Each peak of each spectrum is
matched against `piece + dh*H + charge carrier` (dh -2..+3, the carriers its adduct allows) at two tolerances, and the
features describe how much of the spectrum the candidate explains, how cheaply (fewest cleavages), how precisely (mass
error), which peaks (top-k, high mass), plus agreement between neutral losses seen in the spectrum and the candidate's
functional groups (sugars, acids, methoxy, ...). Features are averaged over the molecule's spectra, then the ranker sees
them both raw and centred within the candidate list (a list = the isomers being compared).

    features_for_list(spectra, smiles_list, cache) -> np.ndarray [n_candidates, N_FEATURES] (NaN rows: unparsable)
    FEATURE_NAMES
"""
from __future__ import annotations

import numpy as np

import frag_rescore as frag

H = frag.H
TOLS = (0.005, 0.012)
H_SHIFTS = tuple(range(-2, 4))
LOSSES = {                      # name: (neutral loss mass, SMARTS a candidate needs to lose it)
    'h2o': (18.010565, '[OX2H1]'),
    'co2': (43.989829, '[CX3](=O)[OX2H1,OX1-]'),
    'hcooh': (46.005479, '[CX3](=O)[OX2H1,OX1-]'),
    'nh3': (17.026549, '[NX3;H2,H1;!$(NC=O)]'),
    'ch3': (15.023475, '[CH3][OX2,NX3]'),
    'ch3oh': (32.026215, '[CH3][OX2]'),
    'c2h2o': (42.010565, '[CH3]C(=O)[O,N]'),
    'hexose': (162.052824, '[OX2;R][CX4;R]1[OX2;R][CX4;R][CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]1[OX2]'),
    'dhex': (146.057909, 'C[CX4;R]1[OX2;R][CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]1[OX2]'),
    'pentose': (132.042259, '[OX2][CX4;R]1[OX2;R][CX4;R][CX4;R]([OX2])[CX4;R]1[OX2]'),
    'glcA': (176.032088, 'OC(=O)[CX4;R]1[OX2;R][CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]1[OX2]'),
    'so3': (79.956815, 'S(=O)(=O)[OX2H1,OX1-,OX2]'),
    'co': (27.994915, '[CX3]=O'),
}
_LOSS_PATTERNS = None


def _patterns():
    global _LOSS_PATTERNS
    if _LOSS_PATTERNS is None:
        from rdkit import Chem
        _LOSS_PATTERNS = {k: Chem.MolFromSmarts(v[1]) for k, v in LOSSES.items()}
    return _LOSS_PATTERNS


SPEC_FEATURES = ['expl_k1_lin', 'expl_k2_lin', 'expl_k1_sqrt', 'expl_k2_sqrt', 'expl_k2_lin_tight', 'expl_k2_sqrt_tight',
                 'top5_k2', 'top10_k2', 'top10_k1', 'top3_k1', 'npeaks_k2', 'hi_mass_k2', 'lo_mass_k2', 'h0_share',
                 'mass_err_mda', 'k2_only_share']
CAND_FEATURES = ['log_n_k1', 'log_n_k2', 'heavy_atoms', 'n_rings', 'frac_aromatic', 'n_O', 'n_N']
LOSS_FEATURES = [f'loss_{k}_{t}' for k in LOSSES for t in ('agree', 'clash')]
FEATURE_NAMES = SPEC_FEATURES + CAND_FEATURES + LOSS_FEATURES
N_FEATURES = len(FEATURE_NAMES)


def prep_spectrum(s):
    """{mz, it, prec, adduct, mode} -> arrays used for matching (None when empty)."""
    mz = np.asarray(s.get('mz', ()), np.float64).ravel()
    it = np.asarray(s.get('it', ()), np.float64).ravel()
    n = min(len(mz), len(it))
    mz, it = mz[:n], it[:n]
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    prec = float(s.get('prec') or 0.0)
    if prec > 0:
        ok &= mz <= prec + 2.0
    mz, it = mz[ok], it[ok]
    if len(mz) < 2:
        return None
    order = np.argsort(-it)
    mz, it = mz[order][:150], it[order][:150]
    car = np.asarray(frag._carriers(s.get('adduct'), s.get('mode')), np.float64)
    shifts = np.array([c + dh * H for c in car for dh in H_SHIFTS])
    dh_of = np.array([dh for c in car for dh in H_SHIFTS])
    return dict(mz=mz, it=it, lin=it / it.sum(), sq=np.sqrt(it) / np.sqrt(it).sum(), prec=prec if prec > 0 else float(mz.max()),
                shifts=shifts, dh=dh_of, adduct=str(s.get('adduct') or ''))


def _match(levels, p, tol):
    """Per peak: fewest cleavages explaining it (0/1/2, 9 = none), smallest |error| (Da) and its dh."""
    npk = len(p['mz'])
    q = p['mz'][None, :] - p['shifts'][:, None]                        # [n_shift, n_peak] neutral masses
    best_k = np.full(npk, 9, np.int8)
    best_err = np.full(npk, np.inf)
    best_dh = np.zeros(npk, np.int8)
    for k, arr in enumerate(levels[:3]):
        if arr is None or len(arr) == 0:
            continue
        idx = np.searchsorted(arr, q)
        lo = np.clip(idx - 1, 0, len(arr) - 1)
        hi = np.clip(idx, 0, len(arr) - 1)
        err = np.minimum(np.abs(arr[lo] - q), np.abs(arr[hi] - q))   # [n_shift, n_peak]
        jmin = err.argmin(axis=0)
        e = err[jmin, np.arange(npk)]
        hit = (e <= tol) & (best_k == 9)
        best_k[hit] = k
        best_err[hit] = e[hit]
        best_dh[hit] = p['dh'][jmin[hit]]
    return best_k, best_err, best_dh


def _spec_features(levels, p):
    out = {}
    for tol, tag in ((TOLS[1], ''), (TOLS[0], '_tight')):
        k, err, dh = _match(levels, p, tol)
        e1, e2 = k <= 1, k <= 2
        if tag == '':
            out['expl_k1_lin'] = float(p['lin'] @ e1)
            out['expl_k2_lin'] = float(p['lin'] @ e2)
            out['expl_k1_sqrt'] = float(p['sq'] @ e1)
            out['expl_k2_sqrt'] = float(p['sq'] @ e2)
            out['top5_k2'] = float(e2[:5].mean())
            out['top10_k2'] = float(e2[:10].mean())
            out['top10_k1'] = float(e1[:10].mean())
            out['top3_k1'] = float(e1[:3].mean())
            out['npeaks_k2'] = float(e2.mean())
            hi = p['mz'] >= 0.5 * p['prec']
            out['hi_mass_k2'] = float(p['lin'][hi] @ e2[hi] / max(p['lin'][hi].sum(), 1e-9)) if hi.any() else 0.0
            lo = ~hi
            out['lo_mass_k2'] = float(p['lin'][lo] @ e2[lo] / max(p['lin'][lo].sum(), 1e-9)) if lo.any() else 0.0
            w = p['lin'][e2]
            out['h0_share'] = float(w @ (np.abs(dh[e2]) <= 1) / max(w.sum(), 1e-9)) if e2.any() else 0.0
            out['mass_err_mda'] = float(1000 * (w @ err[e2]) / max(w.sum(), 1e-9)) if e2.any() else TOLS[1] * 1000
            out['k2_only_share'] = float(p['lin'] @ (k == 2))
        else:
            out['expl_k2_lin_tight'] = float(p['lin'] @ e2)
            out['expl_k2_sqrt_tight'] = float(p['sq'] @ e2)
    return out


def _observed_losses(p, tol=0.006):
    """{loss: intensity share of peaks at precursor - loss} for one spectrum (precursor m/z taken as given)."""
    out = {}
    for name, (m, _) in LOSSES.items():
        target = p['prec'] - m
        hit = np.abs(p['mz'] - target) <= tol
        out[name] = float(p['lin'][hit].sum())
    return out


def _cand_facts(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    pats = _patterns()
    groups = {k: int(m.HasSubstructMatch(pt)) if pt is not None else 0 for k, pt in pats.items()}
    ri = m.GetRingInfo()
    heavy = m.GetNumHeavyAtoms()
    arom = sum(a.GetIsAromatic() for a in m.GetAtoms())
    sym = [a.GetSymbol() for a in m.GetAtoms()]
    return dict(groups=groups, heavy_atoms=heavy, n_rings=ri.NumRings(), frac_aromatic=arom / max(heavy, 1),
                n_O=sym.count('O'), n_N=sym.count('N'))


def features_for_list(spectra, smiles_list, cache=None, cfg_breaks=2):
    """Feature matrix for candidates of one molecule (rows of NaN for unparsable SMILES)."""
    cache = {} if cache is None else cache
    preps = [p for p in (prep_spectrum(s) for s in (spectra or [])) if p is not None]
    obs = None
    if preps:
        ol = [_observed_losses(p) for p in preps]
        obs = {k: float(np.mean([o[k] for o in ol])) for k in LOSSES}
    X = np.full((len(smiles_list), N_FEATURES), np.nan)
    c = frag.resolve_cfg(dict(max_breaks=cfg_breaks))
    for i, smi in enumerate(smiles_list):
        key = ('lv', smi)
        if key not in cache:
            cache[key] = frag._levels_cached(smi, c, None)
        lv = cache[key]
        fkey = ('facts', smi)
        if fkey not in cache:
            try:
                cache[fkey] = _cand_facts(smi)
            except Exception:
                cache[fkey] = None
        facts = cache[fkey]
        if lv is None or facts is None:
            continue
        row = {}
        if preps:
            sf = [_spec_features(lv, p) for p in preps]
            for name in SPEC_FEATURES:
                row[name] = float(np.mean([d[name] for d in sf]))
        else:
            for name in SPEC_FEATURES:
                row[name] = 0.0
        row['log_n_k1'] = float(np.log1p(len(lv[1]) if len(lv) > 1 else 0))
        row['log_n_k2'] = float(np.log1p(len(lv[2]) if len(lv) > 2 else 0))
        for name in ('heavy_atoms', 'n_rings', 'frac_aromatic', 'n_O', 'n_N'):
            row[name] = float(facts[name])
        for name in LOSSES:
            seen = obs[name] if obs else 0.0
            has = facts['groups'][name]
            row[f'loss_{name}_agree'] = seen * has
            row[f'loss_{name}_clash'] = seen * (1 - has)
        X[i] = [row[n] for n in FEATURE_NAMES]
    return X


def centre_within(X):
    """Candidate-list-relative copy: each column minus its list mean (NaN-safe)."""
    with np.errstate(all='ignore'):
        mu = np.nanmean(X, axis=0, keepdims=True)
    return X - np.where(np.isfinite(mu), mu, 0.0)


def full_features(spectra, smiles_list, cache=None):
    X = features_for_list(spectra, smiles_list, cache)
    return np.hstack([X, centre_within(X)])


FULL_FEATURE_NAMES = FEATURE_NAMES + [f'{n}_c' for n in FEATURE_NAMES]
