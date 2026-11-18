"""Build same-formula training triples from train.parquet (072a scale-up, stage 1).

For each train molecule with usable spectra:
  query = that molecule's own spectra (<=5, balanced by adduct/CE like the shipped selector)
  list  = the true structure + 24 decoy structures sharing its molecular formula, drawn from
          the panel's public structure catalogue (291k SMILES)
  label = the truth's position in the list

The true structure is never its own decoy, and every panel truth / visible truth key is
excluded from training so the later panel evaluation is a genuine hold-out.
"""
from __future__ import annotations

import collections
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, str(Path(__file__).resolve().parent / 'rr8mod'))

OUT = Path('artifacts/rr8x')
OUT.mkdir(parents=True, exist_ok=True)
TMP = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode')
N_MOL = 4000
MAX_SPEC = 5
N_DECOY = 24
RNG = random.Random(7)

COLS = ['inchikey14', 'normalized_smiles', 'adduct', 'precursor_mz', 'ms2_mzs',
        'ms2_normalized_intensities', 'ionization_mode', 'collision_energy_ev']


def spec_from_lists(mz, it, prec, adduct, mode, ce):
    mz = np.asarray(mz, float).ravel()
    it = np.asarray(it, float).ravel()
    n = min(len(mz), len(it))
    mz, it = mz[:n], it[:n]
    good = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    if good.sum() < 5:
        return None
    try:
        ce = float(np.mean(ce)) if ce is not None and len(ce) else 30.0
    except Exception:
        ce = 30.0
    if not np.isfinite(ce):
        ce = 30.0
    return dict(mz=mz[good], it=it[good], prec=float(prec), adduct=str(adduct),
                mode=1 if 'pos' in str(mode).lower() else -1, ce=ce)


def balanced(specs, max_n=MAX_SPEC):
    b = collections.defaultdict(list)
    for sp in specs:
        b[(sp['adduct'], round(sp['ce'] / 10))].append(sp)
    out = []
    for k in sorted(b):
        out += sorted(b[k], key=lambda s: -len(s['mz']))[:2]
    return out[:max_n]


def main() -> int:
    cat = pickle.load(open(TMP / 'cat_formula.pkl', 'rb'))
    byf = collections.defaultdict(list)
    for smi, (f, key) in cat.items():
        byf[f].append((smi, key))

    held = set(pd.read_parquet('artifacts/ds/catalogue-experiment-saner/catalogue_replay_checkpoint/panel/labels.parquet')
               .true_key.astype(str))
    held |= set(pd.read_parquet('artifacts/truth/visible_truth.parquet').inchikey14.astype(str))
    print('held-out keys', len(held), flush=True)

    f = pq.ParquetFile('data/raw/train.parquet')
    mols: dict = collections.OrderedDict()
    for gi in range(f.metadata.num_row_groups):
        t = f.read_row_group(gi, columns=COLS)
        cols = {c: t.column(c).to_pylist() for c in COLS}
        for i in range(t.num_rows):
            key = cols['inchikey14'][i]
            if not key or key in held:
                continue
            if key not in mols:
                if len(mols) >= N_MOL:
                    break
                smi = cols['normalized_smiles'][i]
                m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
                if m is None:
                    continue
                formula = CalcMolFormula(m)
                pool = byf.get(formula, [])
                if len(pool) < N_DECOY + 1:
                    continue
                mols[key] = dict(smiles=smi, formula=formula, spectra=[])
            if len(mols[key]['spectra']) < MAX_SPEC * 2:
                sp = spec_from_lists(cols['ms2_mzs'][i], cols['ms2_normalized_intensities'][i],
                                     cols['precursor_mz'][i], cols['adduct'][i],
                                     cols['ionization_mode'][i], cols['collision_energy_ev'][i])
                if sp is not None:
                    mols[key]['spectra'].append(sp)
        print(f'  rowgroup {gi}: molecules={len(mols)}', flush=True)
        if len(mols) >= N_MOL:
            break

    triples = []
    for key, m in mols.items():
        specs = balanced(m['spectra'])
        if len(specs) < 2:
            continue
        pool = [(s, k) for s, k in byf[m['formula']] if k != key and s != m['smiles']]
        if len(pool) < N_DECOY:
            continue
        decoys = RNG.sample(pool, N_DECOY)
        pos = RNG.randrange(N_DECOY + 1)
        smiles = [d[0] for d in decoys]
        smiles.insert(pos, m['smiles'])
        triples.append(dict(key=key, smiles=smiles, truth=pos, spectra=specs, formula=m['formula']))
    print(f'triples built {len(triples)} from {len(mols)} molecules', flush=True)
    pickle.dump(triples, open(OUT / 'triples.pkl', 'wb'))
    print('wrote', OUT / 'triples.pkl')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
