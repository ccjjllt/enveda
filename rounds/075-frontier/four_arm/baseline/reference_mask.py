"""Prevent held-out spectra and copies from entering the public notebook's reference libraries."""
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


PANEL = Path('/tmp/four_arm_comp')
GROUPS = json.loads((PANEL / 'mask.json').read_text())
KEYS = set(GROUPS['keys'])
SCAFFOLDS = set(GROUPS['scaffolds'])
SOURCES = tuple(GROUPS['sources'])
TAUT = rdMolStandardize.TautomerEnumerator()
COPIES = None


def digest(value):
    return sha256(value.encode()).hexdigest()


def blocked_source(value):
    name = re.sub(r'[^a-z0-9]+', '', str(value).lower())
    return any(src in name for src in SOURCES)


@lru_cache(maxsize=100000)
def blocked_scaffold(scaffold):
    if not isinstance(scaffold, str):
        return True
    if not scaffold:
        return GROUPS['block_acyclic']
    mol = Chem.MolFromSmiles(scaffold)
    if mol is None:
        return True
    generic = MurckoScaffold.MakeScaffoldGeneric(mol)
    return digest(Chem.MolToSmiles(generic)) in SCAFFOLDS


@lru_cache(maxsize=100000)
def excluded_smiles(smiles):
    if not isinstance(smiles, str) or not smiles:
        return True
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True
    Chem.RemoveStereochemistry(mol)
    mol = TAUT.Canonicalize(mol)
    key = Chem.MolToInchiKey(mol)[:14]
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return digest(key) in KEYS or (scaf.GetNumAtoms() == 0 and GROUPS['block_acyclic']) or (scaf.GetNumAtoms() > 0 and
                                  digest(Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaf))) in SCAFFOLDS)


def peaks(x, y, precursor):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (x < precursor - 0.5) & (y > 0)
    x, y = x[good], y[good]
    if len(x):
        keep = y >= 0.01 * y.max()
        x, y = x[keep], np.sqrt(y[keep])
        y = y / np.linalg.norm(y)
    return x, y


def copied_spectrum(precursor, mz, intensity):
    global COPIES
    if COPIES is None:
        query = pd.read_parquet(PANEL / 'test.parquet')
        COPIES = [(r.precursor_mz, *peaks(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz))
                  for r in query.itertuples(index=False)]
    nearby = [(x, y) for mass, x, y in COPIES if abs(mass - precursor) <= max(.01, mass * 20e-6)]
    if not nearby:
        return False
    rx, ry = peaks(mz, intensity, precursor)
    for x, y in nearby:
        if not len(rx) or not len(x):
            continue
        left, right = np.where(np.abs(x[:, None] - rx) <= .01)
        weights = y[left] * ry[right]
        used_left, used_right = set(), set()
        score = 0.
        for n in np.argsort(-weights):
            i, j = left[n], right[n]
            if i not in used_left and j not in used_right:
                used_left.add(i)
                used_right.add(j)
                score += weights[n]
        if len(used_left) >= min(4, len(x), len(rx)) and score >= .98:
            return True
    return False


def filter_train(raw, structure_index):
    structs = pd.read_parquet(structure_index, columns=['inchikey14', 'scorer_key', 'scaffold'])
    bad_scaffolds = {s for s in structs.scaffold.dropna().unique() if blocked_scaffold(s)}
    excluded = structs.loc[structs.scaffold.isna() | structs.scorer_key.isna()
                           | structs.scorer_key.map(lambda s: digest(str(s)) in KEYS)
                           | structs.scaffold.isin(bad_scaffolds)]
    blocked = set(excluded.inchikey14)
    (PANEL / 'blocked_scorer.json').write_text(json.dumps(sorted(set(excluded.scorer_key))))
    # No row escapes the reference lookup because only a subset of the source was mapped.
    known = pa.array(sorted(set(structs.inchikey14)), type=pa.string())
    blocked_keys = pa.array(sorted(blocked), type=pa.string())
    pf = pq.ParquetFile(raw)
    query = pd.read_parquet(PANEL / 'test.parquet', columns=['precursor_mz'])
    targets = np.unique(query.precursor_mz.to_numpy(float))
    counts = Counter()
    with pq.ParquetWriter(PANEL / 'train.parquet', pf.schema_arrow) as writer:
        for b in pf.iter_batches(batch_size=16384):
            k = b.column('inchikey14')
            m = np.asarray(b.column('precursor_mz'), dtype=float)
            valid_key = pc.is_in(k, value_set=known).to_numpy(zero_copy_only=False)
            blocked_key = pc.is_in(k, value_set=blocked_keys).to_numpy(zero_copy_only=False)
            src = b.column('ingest_lib')
            banned = [s for s in pc.unique(src).to_pylist() if blocked_source(s)]
            by_source = pc.is_in(src, value_set=pa.array(banned, type=pa.string())).to_numpy(zero_copy_only=False)
            keep = valid_key & ~blocked_key & ~by_source
            counts.update(unmapped=int((~valid_key).sum()), heldout_group=int(blocked_key.sum()),
                          source=int(by_source.sum()))
            i = np.searchsorted(targets, m)
            gap = np.minimum(abs(m-targets[np.clip(i, 0, len(targets)-1)]),
                             abs(m-targets[np.clip(i-1, 0, len(targets)-1)]))
            for j in np.flatnonzero(keep & (gap <= np.maximum(.01001, m * 20.01e-6))):
                if copied_spectrum(m[j], b.column('ms2_mzs')[int(j)].as_py(),
                                   b.column('ms2_normalized_intensities')[int(j)].as_py()):
                    keep[j] = False
                    counts['spectral_copy'] += 1
            writer.write_batch(b.filter(pa.array(keep)))
            counts.update(total=len(b), retained=int(keep.sum()))
    if not counts['retained']:
        raise ValueError('Empty training reference corpus')
    (PANEL / 'blocked_raw.json').write_text(json.dumps(sorted(blocked)))
    return dict(counts, excluded_raw_keys=len(blocked))
