"""048 step1: rebuild 044 holdout isomer sets, score with model3 on CPU, dump per-candidate base scores."""
import sys, os, json, random, pickle, numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, '/workspace/e048/ds')
from feats import spectrum, meta, mol_fp
from model import OwnSpec
TRAIN = '/workspace/kin/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet'
NQ = int(os.environ.get('NQ', '1500')); NC = int(os.environ.get('NC', '31'))
pf = pq.ParquetFile(TRAIN)
info, form = {}, {}
for i in range(pf.num_row_groups):
    d = pf.read_row_group(i, columns=['inchikey14', 'normalized_smiles', 'molecular_formula']).to_pydict()
    for k, s, f in zip(d['inchikey14'], d['normalized_smiles'], d['molecular_formula']):
        if k not in info: info[k] = s; form[k] = f
ulist = sorted(info); U = len(ulist)
hold = np.random.default_rng(20261025).random(U) < 0.02
hk = {ulist[i] for i in np.nonzero(hold)[0]}
print('U', U, 'hold', len(hk), flush=True)
byf = {}
for k in ulist: byf.setdefault(form[k], []).append(k)
# collect one spectrum per holdout mol (N>=3), candidates>=5
spec = {}
cols = ['inchikey14', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities', 'collision_energy_ev', 'instrument_type', 'ionization_mode']
for i in range(pf.num_row_groups):
    d = pf.read_row_group(i, columns=cols).to_pydict()
    for j, k in enumerate(d['inchikey14']):
        if k in hk and (k not in spec or random.Random(k + str(i) + str(j)).random() < 0.3):
            if d['ms2_mzs'][j] and len(d['ms2_mzs'][j]) >= 3:
                spec[k] = {c: d[c][j] for c in cols}
qs = [k for k in sorted(spec) if len(byf[form[k]]) >= 6]
random.Random(int(os.environ.get('SEED','48'))).shuffle(qs); qs = qs[int(os.environ.get('OFF','0')):int(os.environ.get('OFF','0'))+NQ]
print('queries', len(qs), flush=True)
ck = torch.load('/workspace/e048/ds/model3.pt', map_location='cpu')
net = OwnSpec(**ck['cfg']); net.load_state_dict(ck['state_dict']); net.eval()
torch.set_num_threads(6)
rows = []
fpc = {}
def fp(k):
    if k not in fpc: fpc[k] = mol_fp(info[k])[1]
    return fpc[k]
with torch.no_grad():
    for qi, k in enumerate(qs):
        s = spec[k]
        c = [x for x in byf[form[k]] if x != k]; random.Random(k).shuffle(c); c = [k] + c[:NC]
        F = [fp(x) for x in c]; ok = [f is not None for f in F]
        c = [x for x, o in zip(c, ok) if o]; F = np.stack([f for f in F if f is not None])
        if c[0] != k: continue
        x, n = spectrum(s['ms2_mzs'], s['ms2_normalized_intensities'], s['precursor_mz'])
        m, a, ins = meta(s['precursor_mz'], s['adduct'], s['collision_energy_ev'], s['instrument_type'])
        z, fl = net.spec(torch.from_numpy(x)[None], torch.tensor([n]), torch.from_numpy(m)[None], torch.tensor([a]), torch.tensor([ins]))
        tf = torch.from_numpy(F); M = net.molecule(tf)
        emb = (z @ M.T * net.logit_scale.exp())[0].numpy()
        b = (tf > 0).float(); ll = (torch.nn.functional.logsigmoid(fl) @ b.T + torch.nn.functional.logsigmoid(-fl) @ (1 - b).T)[0].numpy() / b.shape[1] * 100
        p = torch.sigmoid(fl[0]).numpy()
        bb = b.numpy()
        tani = (bb * p).sum(1) / (bb.sum(1) + p.sum() - (bb * p).sum(1) + 1e-6)
        for j, ck_ in enumerate(c):
            rows.append(dict(q=qi, key=ck_, smiles=info[ck_], y=int(j == 0), emb=float(emb[j]), ll=float(ll[j]), tani=float(tani[j])))
        if qi % 200 == 0: print('q', qi, flush=True)
pickle.dump(dict(rows=rows, spec={k: spec[k] for k in qs}, qs=qs), open(os.environ.get('OUT','/workspace/e048/val.pkl'), 'wb'))
import pandas as pd
df = pd.DataFrame(rows); df['base'] = df.emb + df.ll
def mrr(df, col):
    r = df.groupby('q').apply(lambda g: 1.0 / (1 + (g[col] > g[g.y == 1][col].iloc[0]).sum()))
    return r.mean()
print('MRR base', mrr(df, 'base'), 'emb', mrr(df, 'emb'), 'll', mrr(df, 'll'), 'tani', mrr(df, 'tani'), 'n', df.q.nunique())
