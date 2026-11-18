"""072a: does per-spectrum disaggregation of the fragment evidence beat its mean?

The shipped RR8 isomer reranker (068a cell 23) calls frag_rescore.score_molecule(spectra, ...)
once per molecule, and that function returns the **mean over spectra**. A candidate whose
fragments are explained at one collision energy and contradicted at another is therefore
indistinguishable from a candidate explained moderately at every energy.

This probe keeps every other part of RR8 identical and adds only four cross-spectrum
consistency features derived by calling the same scorer once per spectrum:
    m1_min   min over spectra of the MetFrag-lite score
    m1_std   spread over spectra
    m1_mean  mean (redundant with the shipped column; kept as a control)
    m1_win   share of spectra where the candidate scores above the list leader
Raw m/z arrays are never averaged across spectra (that was 083a's defect: it fabricated
peaks). Each spectrum is scored on its own peaks, then the scalar scores are aggregated.

Arms, all trained on the panel's 60 'train' molecules and scored on its 30 calibration +
30 test molecules:
    A  shipped: 49 spectrum features (mean over spectra) + 4 fragment columns
    B  +4 cross-spectrum consistency features
    A2 control: same 49+4 columns recomputed per spectrum and re-meaned, to show that any
       gain comes from the consistency terms rather than from recomputation noise.

Guards copied from RR8: lists whose leader has library_max >= LIB_TAU are untouched;
training-set structures are frozen in their slot; only candidates sharing the leader's
formula move, and only into slots already occupied by such candidates.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'rr8mod'))

import frag_rescore as frag          # noqa: E402
import rr8_features as rf            # noqa: E402
from rdkit import Chem, RDLogger     # noqa: E402
from rdkit.Chem import Descriptors   # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

RDLogger.DisableLog('rdApp.*')

PANEL = Path('artifacts/ds/catalogue-experiment-saner/catalogue_replay_checkpoint/panel')
SCORES = Path('artifacts/ds/catalogue-experiment-saner/catalogue_experiment/labelled_candidate_scores.parquet')
TRAIN = Path('data/raw/train.parquet')
OUT = Path('artifacts/rr8x')
LIB_TAU = 0.90
ALPHA = 1.0


def formula_of(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return CalcMolFormula(m) if m is not None else None


def key_of(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m)[:14]
    except Exception:
        return None


def train_keys() -> set:
    cache = OUT / 'train_keys.txt'
    if cache.is_file():
        return set(cache.read_text().split())
    import pyarrow.parquet as pq
    keys: set = set()
    f = pq.ParquetFile(TRAIN)
    for i in range(f.metadata.num_row_groups):
        col = f.read_row_group(i, columns=['inchikey14']).column('inchikey14')
        keys.update(k for k in col.to_pylist() if k)
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text('\n'.join(sorted(keys)))
    return keys


def spec_dicts(g: pd.DataFrame, max_n: int = 12) -> list:
    """Same selection rule as the shipped select_spectra: <=2 per (adduct, CE/10) bucket."""
    buckets: dict = {}
    for r in g.itertuples(index=False):
        mz = np.asarray(r.ms2_mzs, float).ravel()
        it = np.asarray(r.ms2_normalized_intensities, float).ravel()
        n = min(len(mz), len(it))
        mz, it = mz[:n], it[:n]
        good = np.isfinite(mz) & np.isfinite(it) & (it > 0)
        if good.sum() < 5:
            continue
        ce = r.collision_energy_ev
        try:
            ce = float(np.mean(ce)) if ce is not None and len(ce) else 30.0
        except Exception:
            ce = 30.0
        if not np.isfinite(ce):
            ce = 30.0
        adduct = str(r.adduct)
        item = dict(mz=mz[good], it=it[good], prec=float(r.precursor_mz), adduct=adduct,
                    mode=1 if 'pos' in str(r.ionization_mode).lower() else -1)
        buckets.setdefault((adduct, round(ce / 10)), []).append((int(good.sum()), item))
    out = []
    for k in sorted(buckets):
        out += [v for _, v in sorted(buckets[k], key=lambda x: -x[0])[:2]]
    return out[:max_n]


def per_spectrum_matrix(specs, smiles, cache):
    """[n_spectra, n_candidates] of the MetFrag-lite score; one call per spectrum."""
    rows = []
    for sp in specs:
        vals = frag.score_molecule([sp], smiles, cache=cache)
        rows.append([np.nan if v is None else float(v) for v in vals])
    return np.asarray(rows, float)


def build_features(mid, cand, spectra, cache, frag_cache, train_set, arm):
    smiles = cand.smiles.tolist()
    lib = float(cand.library_max.iloc[0])
    X = rf.features_for_list(spectra, smiles, cache)
    m1 = np.asarray([np.nan if v is None else v for v in frag.score_molecule(spectra, smiles, cache=frag_cache)], float)
    m2 = np.asarray([np.nan if v is None else v for v in
                     frag.score_molecule(spectra, smiles, cfg=frag.BASELINE_CFG, cache=frag_cache)], float)
    M = np.stack([m1, m2], 1)
    Mc = M - np.nanmean(M, axis=0, keepdims=True)
    base = np.hstack([X, M, Mc])
    if arm == 'A':
        return base, None
    per = per_spectrum_matrix(spectra, smiles, frag_cache)
    if arm == 'A2':
        m1_re = np.nanmean(per, 0)
        extra = m1_re.reshape(-1, 1)
        return np.hstack([base, extra]), None
    lead = per[:, [0]]
    with np.errstate(invalid='ignore'):
        win = np.nanmean(per >= lead, axis=0)
    extra = np.column_stack([
        np.nanmin(per, axis=0),
        np.nanstd(per, axis=0),
        np.nanmean(per, axis=0),
        win,
    ])
    return np.hstack([base, extra]), dict(n_spec=per.shape[0])


def rerank(cand, scores, frozen, alpha=ALPHA):
    """RR8 rule: movable same-formula members reorder among their own slots."""
    smiles = cand.smiles.tolist()
    f0 = formula_of(smiles[0])
    if f0 is None:
        return smiles, False
    group = [s for s in smiles if formula_of(s) == f0]
    movable = [s for s in group if not frozen.get(s, False)]
    if len(movable) < 2:
        return smiles, False
    pos = {s: i for i, s in enumerate(smiles)}
    sc = dict(zip(smiles, scores))
    slots = sorted(pos[s] for s in movable)
    base = np.array([-np.log(slots[i] + 1) for i in range(len(movable))], float)
    rr = np.array([sc[s] for s in movable], float)
    z = lambda v: (v - v.mean()) / v.std() if v.std() > 1e-12 else np.zeros_like(v)
    fused = z(base) + alpha * z(rr)
    order = [movable[i] for i in np.argsort(-fused, kind='stable')]
    out = list(smiles)
    for slot, s in zip(slots, order):
        out[slot] = s
    return out, True


def mrr(order, truth_idx):
    if truth_idx is None:
        return 0.0
    try:
        return 1.0 / (order.index(truth_idx) + 1)
    except ValueError:
        return 0.0


def main() -> int:
    frag.warmup()
    OUT.mkdir(parents=True, exist_ok=True)
    te = pd.read_parquet(PANEL / 'test.parquet')
    sc = pd.read_parquet(SCORES)
    lab = pd.read_parquet(PANEL / 'labels.parquet')
    role = dict(zip(lab.molecule_id.astype(str), lab.role))
    keys = train_keys()
    print(f'panel molecules={sc.molecule_id.nunique()} spectra={len(te)} train_keys={len(keys)}', flush=True)

    rows = []
    t0 = time.time()
    for mi, (mid, cand) in enumerate(sc.groupby('molecule_id', sort=False)):
        mid = str(mid)
        cand = cand.sort_values('baseline_rank', kind='stable').reset_index(drop=True)
        g = te[te.molecule_id.astype(str) == mid]
        spectra = spec_dicts(g)
        if not spectra or len(cand) < 3:
            continue
        frozen = {s: (key_of(s) in keys) for s in cand.smiles}
        for arm in ('A', 'A2', 'B'):
            X, meta = build_features(mid, cand, spectra, {}, {}, keys, arm)
            rows.append(dict(molecule_id=mid, arm=arm, role=role.get(mid, '?'),
                             smiles=cand.smiles.tolist(),
                             y=cand.label.values, libmax=float(cand.library_max.iloc[0]),
                             frozen=[frozen[s] for s in cand.smiles], X=X,
                             n_spec=(meta or {}).get('n_spec', len(spectra))))
        if (mi + 1) % 20 == 0:
            print(f'  featurised {mi+1} molecules  {time.time()-t0:.0f}s', flush=True)
    print(f'feature build done {time.time()-t0:.0f}s  rows={len(rows)}', flush=True)
    pd.to_pickle(rows, OUT / 'features.pkl')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
