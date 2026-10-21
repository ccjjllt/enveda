"""025: PubChem-present, test-distribution panel (mask P = R-style).
Queries: gnps/riken/pluskal_ms2 spectra with adduct [M+H]+, precursor 280-380, structure ik14 in PubChem tier.
250 molecules (seed 25), <=3 spectra each.  Mask: every train spectrum of these ik14 / metric keys removed; pools kept.
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_e006_local as E
kin, out = Path('/workspace/kin'), Path('/workspace/e021')
MK = os.environ.get('PANEL_MASK', 'P')  # 026: Q = popularity-weighted
comp = kin / 'competitions' / E.SLUG
cols = ['ingest_lib', 'inchikey14', 'normalized_smiles', 'adduct', 'precursor_mz']
tr = pd.read_parquet(comp / 'train.parquet', columns=cols)
sel = tr.ingest_lib.isin(['gnps', 'riken', 'pluskal_ms2']) & (tr.adduct == '[M+H]+') & tr.precursor_mz.between(280, 380)
ik = np.array(sorted(set(tr.inchikey14[sel])), dtype='S14')
pc = np.load(kin / 'datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior/pc_ik14.npy', mmap_mode='r')
pdir = kin / 'datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior'
ls_, lp_ = np.load(pdir / 'pc_lsid.npy', mmap_mode='r'), np.load(pdir / 'pc_lpmid.npy', mmap_mode='r')
inpc = np.zeros(len(ik), bool); pop = np.zeros(len(ik))
for a in range(0, len(pc), 10_000_000):
    blk = np.asarray(pc[a:a + 10_000_000])
    m = np.isin(blk, ik)
    if m.any():
        j = np.searchsorted(ik, blk[m]); pv = ls_[a:a + 10_000_000][m].astype(float) + lp_[a:a + 10_000_000][m].astype(float)
        inpc[j] = True; np.maximum.at(pop, j, pv)
ikp = [x.decode() for x in ik[inpc]]
print('candidate ik14', len(ik), 'in pubchem', len(ikp), flush=True)
rng = np.random.default_rng(25)
popp = pop[inpc]
print('pop quantiles 50/90/99', np.percentile(popp, [50, 90, 99]), flush=True)
w = np.exp(popp) / np.exp(popp).sum() if MK == 'Q' else None
pick = set(rng.choice(ikp, 250, replace=False, p=w))
print('picked pop median', float(np.median(popp[np.isin(ikp, list(pick))])), flush=True)
idx = tr.index[sel & tr.inchikey14.isin(pick)]
q = tr.loc[idx].copy(); q['row'] = idx
q = q.groupby('inchikey14', group_keys=False).apply(lambda g: g.sample(min(3, len(g)), random_state=0))
rows = set(q.row)
M = E._metric()
truth = sorted(set(q.normalized_smiles))
keyset = {M.canonicalize_smiles_to_ik14(s) for s in truth} - {None}
compset = {M.heavy_atom_composition(s) for s in truth} - {None}
us = pd.Series(tr.normalized_smiles.unique())
comps = E._pmap(E._comp_worker, us.tolist(), n=4)
cand = us[[c in compset for c in comps]].tolist()
ks = E._pmap(E._key_worker, cand, n=4)
hit = {s for s, k in zip(cand, ks) if k in keyset}
rm = (tr.normalized_smiles.isin(hit) | tr.inchikey14.isin(pick)).values
print('queries', len(rows), 'train removed', int(rm.sum()), flush=True)
del tr
d = out / f'in_{MK}' / 'competitions' / E.SLUG; d.mkdir(parents=True, exist_ok=True)
ds_ = out / f'in_{MK}' / 'datasets'
if not ds_.exists(): ds_.symlink_to(kin / 'datasets')
pf = pq.ParquetFile(comp / 'train.parquet'); w = None; pos = 0; qs = []
for b in pf.iter_batches(batch_size=50000):
    n = b.num_rows; t = pa.Table.from_batches([b])
    qm = np.array([i in rows for i in range(pos, pos + n)])
    if qm.any(): qs.append(t.filter(pa.array(qm)).to_pandas())
    t = t.filter(pa.array(~rm[pos:pos + n])); pos += n
    if w is None: w = pq.ParquetWriter(d / 'train.parquet', t.schema)
    w.write_table(t)
w.close()
qq = pd.concat(qs).reset_index(drop=True)
qq['molecule_id'] = 'pc_' + qq.inchikey14.astype(str)
qq = qq.sort_values('molecule_id', kind='stable').reset_index(drop=True)
qq['spectrum_id'] = [f'pcs_{i:05d}' for i in range(len(qq))]
qq[E.TEST_COLS].to_parquet(d / 'test.parquet', index=False)
pd.DataFrame({'molecule_id': sorted(qq.molecule_id.unique()), 'smiles': 'CCO'}).to_csv(d / 'sample_submission.csv', index=False)
sol = qq.groupby('molecule_id').normalized_smiles.agg(lambda s: ','.join(dict.fromkeys(s))).reset_index()
sol.columns = ['molecule_id', 'smiles']; sol.to_csv(out / f'solution_{MK}.csv', index=False)
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
json.dump(dict(keys=sorted(keyset), ik14=sorted(pick), mass=[ExactMolWt(Chem.MolFromSmiles(s)) for s in truth]), open(out / f'drop_{MK}.json', 'w'))
print('done', len(qq), qq.molecule_id.nunique(), flush=True)
