"""048 step5: fit linear frag correction on val1 (emb score, fp_w=0 as Kaggle cell), test on independent val2 (63 cands)."""
import sys, json, pickle, numpy as np, pandas as pd, torch
from multiprocessing import Pool
sys.path.insert(0, '/workspace/enveda/src/fe048')
from frag_feats import features
RAW = ['fr_cov', 'fr_icov', 'fr_top10', 'fr_icov0', 'fr_sqi', 'nl_cov', 'nl_icov', 'fr_any', 'fr_n']
def load(p, cache):
    V = pickle.load(open(p, 'rb')); df = pd.DataFrame(V['rows']); S, qs = V['spec'], V['qs']
    try: F = pd.read_pickle(cache)
    except FileNotFoundError:
        with Pool(7) as pl: F = pd.DataFrame(pl.starmap(_job, [(S[qs[q]], s) for q, s in zip(df.q, df.smiles)], chunksize=64))
        F.to_pickle(cache)
    return pd.concat([df, F], axis=1).fillna(0)
def _job(s, smi):
    try: return features(smi, s['ms2_mzs'], s['ms2_normalized_intensities'], s['precursor_mz'], s['adduct'], s['ionization_mode'])
    except Exception: return {}
def mrr(d, s):
    out = []
    for _, g in pd.DataFrame(dict(q=d.q.values, s=s, y=d.y.values)).groupby('q'):
        t = g.s.values[g.y.values == 1][0]; out.append(1.0 / (1 + (g.s.values > t).sum()))
    return np.array(out)
def fit(d, base, l2=1e-2):
    X = torch.tensor(d[RAW].values, dtype=torch.float32); mu, sd = X.mean(0), X.std(0) + 1e-6; X = (X - mu) / sd
    b = torch.tensor(d[base].values, dtype=torch.float32) / 10; y = torch.tensor(d.y.values); q = torch.tensor(pd.factorize(d.q)[0]); nq = int(q.max()) + 1
    w = torch.zeros(len(RAW), requires_grad=True); a = torch.ones(1, requires_grad=True); opt = torch.optim.LBFGS([w, a], max_iter=300)
    def clo():
        opt.zero_grad(); s = a * b + X @ w
        mx = torch.zeros(nq).scatter_reduce(0, q, s, 'amax', include_self=False)
        lse = torch.log(torch.zeros(nq).index_add(0, q, torch.exp(s - mx[q]))) + mx
        L = -(s[y == 1].sum() - lse.sum()) / nq + l2 * (w ** 2).sum(); L.backward(); return L
    opt.step(clo)
    return dict(feats=RAW, mu=mu.tolist(), sd=sd.tolist(), w=w.detach().tolist(), a=float(a), base=base, base_div=10.0)
def apply(d, P):
    X = (d[P['feats']].values - np.array(P['mu'])) / np.array(P['sd'])
    return P['a'] * d[P['base']].values / P['base_div'] + X @ np.array(P['w'])
d1 = load('/workspace/e048/val.pkl', '/workspace/e048/feats.pkl'); d2 = load('/workspace/e048/val2.pkl', '/workspace/e048/feats2.pkl')
d1['b1'] = d1.emb + d1.ll; d2['b1'] = d2.emb + d2.ll
rng = np.random.default_rng(1)
for base in ['emb', 'b1']:
    P = fit(d1, base); r0 = mrr(d2, d2[base].values); r1 = mrr(d2, apply(d2, P)); dd = r1 - r0
    bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(1000)]
    print(f'holdout val2 base={base}: {r0.mean():.4f} -> {r1.mean():.4f} ({dd.mean():+.4f}, CI [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}], W/L {(dd>0).sum()}/{(dd<0).sum()}, n={len(dd)})', flush=True)
    top1 = lambda r: (r == 1).mean(); print('   top1', top1(r0).round(4), '->', top1(r1).round(4))
    if base == 'emb':
        Pall = fit(pd.concat([d1, d2.assign(q=d2.q + 100000)]), 'emb'); json.dump(Pall, open('/workspace/e048/frag_lin.json', 'w'), indent=1)
        print('weights', dict(zip(RAW, np.round(Pall['w'], 3))), 'a', round(Pall['a'], 3))
