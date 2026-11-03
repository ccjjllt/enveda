"""048 step4: residual fusion: base score + learned correction (linear softmax ranker / LGB with init_score)."""
import numpy as np, pandas as pd, lightgbm as lgb, torch
from sklearn.model_selection import GroupKFold
df = pd.read_pickle('/workspace/e048/df_feat.pkl').fillna(0)
qid = df.q.values
def mrr_vec(s):
    d = pd.DataFrame(dict(q=qid, s=s, y=df.y.values)); out = []
    for _, g in d.groupby('q'):
        t = g.s.values[g.y.values == 1][0]; out.append(1.0 / (1 + (g.s.values > t).sum()))
    return np.array(out)
r0 = mrr_vec(df.base.values); print('base', r0.mean().round(4))
frag = [c for c in df.columns if c.startswith(('fr_', 'nl_'))]
desc = [c for c in df.columns if c.startswith('d_')]
def lin(cols, l2=1e-2, T=None):
    """softmax listwise linear correction on top of base (base coefficient learned too)."""
    X = df[cols].values.astype(np.float32); mu, sd = X.mean(0), X.std(0) + 1e-6; X = (X - mu) / sd
    pred = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(df, groups=qid):
        Xt = torch.tensor(X[tr]); bt = torch.tensor(df.base.values[tr], dtype=torch.float32); yt = torch.tensor(df.y.values[tr])
        qt = torch.tensor(pd.factorize(qid[tr])[0]); nq = int(qt.max()) + 1
        w = torch.zeros(X.shape[1], requires_grad=True); a = torch.ones(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, a], max_iter=200)
        def clo():
            opt.zero_grad(); s = a * bt / 10 + Xt @ w
            mx = torch.zeros(nq).scatter_reduce(0, qt, s, 'amax', include_self=False)
            lse = torch.log(torch.zeros(nq).index_add(0, qt, torch.exp(s - mx[qt]))) + mx
            loss = -(s[yt == 1].sum() - lse.sum()) / nq + l2 * (w ** 2).sum(); loss.backward(); return loss
        opt.step(clo)
        pred[te] = (a * torch.tensor(df.base.values[te], dtype=torch.float32) / 10 + torch.tensor(X[te]) @ w).detach().numpy()
    return pred
def lgbres(cols, n=200):
    pred = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(df, groups=qid):
        a, b = df.iloc[tr], df.iloc[te]
        m = lgb.LGBMRanker(n_estimators=n, learning_rate=0.02, num_leaves=7, min_child_samples=50, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)
        m.fit(a[cols], a.y, group=a.groupby('q', sort=False).size().values, init_score=a.base.values / 10)
        pred[te] = b.base.values / 10 + m.predict(b[cols])
    return pred
rng = np.random.default_rng(0)
def rep(tag, p):
    r = mrr_vec(p); d = r - r0; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    print(f'{tag:28s} {r.mean():.4f} {d.mean():+.4f} CI[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] W/L {(d>0).sum()}/{(d<0).sum()}', flush=True)
rawf = ['fr_cov', 'fr_icov', 'fr_top10', 'fr_icov0', 'fr_sqi', 'nl_cov', 'nl_icov', 'fr_any', 'fr_n']
rep('lin frag raw', lin(rawf))
rep('lin frag z/r', lin([c for c in frag if c.endswith(('_z', '_r'))]))
rep('lin frag+desc', lin(rawf + desc))
rep('lin frag+desc+tani', lin(rawf + desc + ['tani', 'll', 'emb']))
for n in (100, 300):
    rep(f'lgb init frag n{n}', lgbres(frag, n))
    rep(f'lgb init frag+desc n{n}', lgbres(frag + desc, n))
