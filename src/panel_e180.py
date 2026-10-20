"""024: E180 panel = the 400 visible-test molecules (exact copies of enveda-180 train spectra; same distribution as hidden).
Mask H (hidden-like): remove every train spectrum whose structure has a truth identity (any library), so Engine2's
train-structure pool loses them automatically; Engine1 v2-pool drops truth entries only when they are train-derived
(src==1); COCONUT/ChEBI/PubChem keep them.  Output root /workspace/e021 layout under in_H / solution_H.csv / drop_H.json.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_e006_local as E
SLUG = E.SLUG
kin, out = Path('/workspace/kin'), Path('/workspace/e021')
comp = kin / 'competitions' / SLUG
te = pd.read_parquet(comp / 'test.parquet')
sol = pd.read_csv('/workspace/e024/visible_solution.csv')
M = E._metric()
truth = sorted(set(sol.smiles))
keyset = {M.canonicalize_smiles_to_ik14(s) for s in truth} - {None}
compset = {M.heavy_atom_composition(s) for s in truth} - {None}
tr = pd.read_parquet(comp / 'train.parquet', columns=['inchikey14', 'normalized_smiles'])
us = pd.Series(tr.normalized_smiles.unique())
comps = E._pmap(E._comp_worker, us.tolist(), n=4)
cand = us[[c in compset for c in comps]].tolist()
ks = E._pmap(E._key_worker, cand, n=4)
hit = {s for s, k in zip(cand, ks) if k in keyset}
rm = tr.normalized_smiles.isin(hit).values
print('train spectra removed', int(rm.sum()), flush=True)
del tr
d = out / 'in_H' / 'competitions' / SLUG; d.mkdir(parents=True, exist_ok=True)
te.to_parquet(d / 'test.parquet', index=False)
pd.DataFrame({'molecule_id': sorted(te.molecule_id.unique()), 'smiles': 'CCO'}).to_csv(d / 'sample_submission.csv', index=False)
ds_ = out / 'in_H' / 'datasets'
if not ds_.exists(): ds_.symlink_to(kin / 'datasets')
pf = pq.ParquetFile(comp / 'train.parquet'); w = None; pos = 0
for b in pf.iter_batches(batch_size=100000):
    mm = ~rm[pos:pos + b.num_rows]; pos += b.num_rows
    t = pa.Table.from_batches([b]).filter(pa.array(mm))
    if w is None: w = pq.ParquetWriter(d / 'train.parquet', t.schema)
    w.write_table(t)
w.close()
sol.to_csv(out / 'solution_H.csv', index=False)
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
json.dump(dict(keys=sorted(keyset), ik14=[], mass=[ExactMolWt(Chem.MolFromSmiles(s)) for s in truth]), open(out / 'drop_H.json', 'w'))
# Engine1 v2-pool: drop truth identity only for train-derived entries (src==1)
meta = pd.read_parquet(kin / 'datasets/ahmedberatozer/casmi26-v2-pool/pool_meta.parquet', columns=['key', 'smiles', 'mass', 'src'])
import os; os.environ['E021_DROP'] = str(out / 'drop_H.json')
dm = E.drop_mask(meta.mass.values, meta.smiles.values, None)
print('v2-pool truth entries', int(dm.sum()), 'by src', meta.src[dm].value_counts().to_dict(), flush=True)
np.save(out / 'drop1_H.npy', dm & (meta.src.values == 1))
print('done', flush=True)
