"""048 step3: compute features, LightGBM lambdarank 5-fold GroupKFold vs model3 baseline (isomer MRR)."""
import sys, pickle, numpy as np, pandas as pd, lightgbm as lgb
from multiprocessing import Pool
sys.path.insert(0, '/workspace/enveda/src/fe048')
from frag_feats import features
V = pickle.load(open('/workspace/e048/val.pkl', 'rb')); qs = V['qs']; S = V['spec']
df = pd.DataFrame(V['rows'])
def job(r):
    s = S[qs[r[0]]]
    try: return features(r[1], s['ms2_mzs'], s['ms2_normalized_intensities'], s['precursor_mz'], s['adduct'], s['ionization_mode'])
    except Exception as e: return {}
import os
if os.path.exists('/workspace/e048/feats.pkl'): F = pd.read_pickle('/workspace/e048/feats.pkl')
else:
    with Pool(7) as p: F = pd.DataFrame(p.map(job, list(zip(df.q, df.smiles)), chunksize=64))
    F.to_pickle('/workspace/e048/feats.pkl')
df = pd.concat([df.reset_index(drop=True), F], axis=1)
df['base'] = df.emb + df.ll
g = df.groupby('q')
base_cols = ['emb', 'll', 'tani', 'base']
fcols = [c for c in F.columns]
for c in base_cols + [c for c in fcols if c.startswith(('fr_', 'nl_'))]:
    df[c + '_r'] = g[c].rank(ascending=False, pct=True)
    df[c + '_dm'] = df[c] - g[c].transform('max')
    df[c + '_z'] = (df[c] - g[c].transform('mean')) / (g[c].transform('std') + 1e-6)
df['nc'] = g['y'].transform('size')
def mrr(d, col):
    out = []
    for _, gg in d.groupby('q'):
        t = gg[col].values[gg.y.values == 1][0]; out.append(1.0 / (1 + (gg[col].values > t).sum()))
    return np.array(out)
from sklearn.model_selection import GroupKFold
def cv(cols, tag):
    pred = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(df, groups=df.q):
        a, b = df.iloc[tr], df.iloc[te]
        m = lgb.LGBMRanker(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        m.fit(a[cols], a.y, group=a.groupby('q').size().loc[a.q.unique()].values)
        pred[te] = m.predict(b[cols])
    df['p_' + tag] = pred; r = mrr(df, 'p_' + tag); return r, m
r0 = mrr(df, 'base'); print('model3 base MRR', r0.mean().round(4), 'queries', len(r0))
for c in ['fr_icov', 'fr_any', 'nl_icov', 'fr_top10']: print(' single', c, mrr(df, c).mean().round(4))
sets = {
  'lgb_base': [c for c in df.columns if c.split('_')[0] in ('emb', 'll', 'tani', 'base')] + ['nc'],
  'lgb_base+frag': [c for c in df.columns if c.split('_')[0] in ('emb', 'll', 'tani', 'base', 'fr', 'nl')] + ['nc'],
  'lgb_all': [c for c in df.columns if c not in ('q', 'key', 'smiles', 'y') and not c.startswith('p_')],
}
rng = np.random.default_rng(0)
for k, cols in sets.items():
    r, m = cv(cols, k); d = r - r0
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    print(f'{k:16s} MRR {r.mean():.4f} delta {d.mean():+.4f} CI [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] W/L {(d>0).sum()}/{(d<0).sum()}', flush=True)
    if k == 'lgb_all':
        imp = pd.Series(m.booster_.feature_importance('gain'), cols).sort_values(ascending=False); print(imp.head(20).round(0).to_string())
df.drop(columns=['smiles']).to_pickle('/workspace/e048/df_feat.pkl')
