"""072a scale-up, stage 2: retrain RR8 on 3,657 same-formula triples, then score on the panel.

Arms (identical apart from the four cross-spectrum columns):
    A  53 shipped-style columns
    B  57 = A + m1_min, m1_std, m1_mean, m1_win

Training data are synthetic same-formula lists built by build_rr8_train_set.py.
Evaluation is the panel's 120 molecules with their real pipeline candidate lists and the
shipped RR8 fusion rule, freeze off (the hidden-test condition). A model only ships if it
beats the untouched baseline order, not merely the other arm.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'rr8mod'))

import frag_rescore as frag          # noqa: E402
import rr8_features as rf            # noqa: E402
from rr8_cross_spectrum_experiment import rerank  # noqa: E402

OUT = Path('artifacts/rr8x')
LIB_TAU = 0.90
ALPHA = 1.0
N_BOOT = 10000
RNG = np.random.default_rng(20261117)


def feats(smiles, spectra, cache, fcache, with_cross):
    X = rf.features_for_list(spectra, smiles, cache)
    m1 = np.array([np.nan if v is None else v for v in frag.score_molecule(spectra, smiles, cache=fcache)], float)
    m2 = np.array([np.nan if v is None else v for v in
                   frag.score_molecule(spectra, smiles, cfg=frag.BASELINE_CFG, cache=fcache)], float)
    M = np.stack([m1, m2], 1)
    base = np.hstack([X, M, M - np.nanmean(M, 0, keepdims=True)])
    if not with_cross:
        return base
    per = np.array([[np.nan if v is None else float(v) for v in frag.score_molecule([sp], smiles, cache=fcache)]
                    for sp in spectra])
    lead = per[:, [0]]
    with np.errstate(invalid='ignore'):
        win = np.nanmean(per >= lead, 0)
    extra = np.column_stack([np.nanmin(per, 0), np.nanstd(per, 0), np.nanmean(per, 0), win])
    return np.hstack([base, extra])


def boot(d):
    d = np.asarray(d, float)
    if not len(d):
        return dict(n=0)
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    m = d[idx].mean(1)
    return dict(n=int(len(d)), mean=float(d.mean()), lo=float(np.quantile(m, .025)),
                hi=float(np.quantile(m, .975)), p_gt0=float((m > 0).mean()))


def main() -> int:
    frag.warmup()
    triples = pickle.load(open(OUT / 'triples.pkl', 'rb'))
    print('triples', len(triples), flush=True)
    store = {}
    t0 = time.time()
    cache_file = OUT / 'triple_features.pkl'
    if cache_file.is_file():
        store = pickle.load(open(cache_file, 'rb'))
        print(f'loaded cached features {len(store)} with {len(next(iter(store.values()))["A"])} candidates', flush=True)
    for i, tr in enumerate([] if store else triples):
        cache, fcache = {}, {}
        store[str(tr['key'])] = dict(
            A=feats(tr['smiles'], tr['spectra'], cache, fcache, False),
            B=feats(tr['smiles'], tr['spectra'], cache, fcache, True),
            y=tr['truth'])
        if (i + 1) % 250 == 0:
            print(f'  featurised {i+1}/{len(triples)}  {time.time()-t0:.0f}s', flush=True)
    pickle.dump(store, open(OUT / 'triple_features.pkl', 'wb'))
    print(f'featurisation done {time.time()-t0:.0f}s', flush=True)

    from sklearn.ensemble import HistGradientBoostingClassifier
    panel = pd.read_pickle(OUT / 'features.pkl')
    ev = [r for r in panel if r['arm'] == 'B']
    report = {}
    for arm in ('A', 'B'):
        Xtr = np.vstack([store[k][arm] for k in store])
        ytr = np.concatenate([[1 if j == store[k]['y'] else 0 for j in range(25)] for k in store])
        m = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.06,
                                           l2_regularization=1.0, min_samples_leaf=40,
                                           max_bins=128, random_state=0)
        m.fit(Xtr, ytr)
        d, aucs, ups, downs, changed = [], [], 0, 0, 0
        for r in ev:
            smiles = r['smiles']
            y = np.asarray(r['y'])
            truth = int(np.flatnonzero(y == 1)[0]) if y.sum() else None
            base_order = list(smiles)
            b = 1.0 / (base_order.index(smiles[truth]) + 1) if truth is not None else 0.0
            s = m.predict_proba(r['X'] if arm == 'B' else r['X'][:, :53])[:, 1]
            if truth is not None and y.sum() and (~y.astype(bool)).sum():
                from sklearn.metrics import roc_auc_score
                aucs.append(roc_auc_score(y.astype(bool), s))
            if r['libmax'] >= LIB_TAU:
                new = base_order
            else:
                new, _ = rerank(pd.DataFrame({'smiles': smiles}), s, {x: False for x in smiles}, ALPHA)
            a = 1.0 / (new.index(smiles[truth]) + 1) if truth is not None else 0.0
            d.append(a - b)
            ups += int(a > b + 1e-12)
            downs += int(a < b - 1e-12)
            changed += int(new != base_order)
        report[arm] = dict(n=len(d), delta=float(np.mean(d)), up=ups, down=downs, changed=changed,
                           auc=float(np.mean(aucs)) if aucs else None, boot=boot(d))
        print(f"[scaleup/{arm}] delta={np.mean(d):+.4f} up/down={ups}/{downs} changed={changed} "
              f"AUC={report[arm]['auc']:.4f} boot[{report[arm]['boot']['lo']:+.4f},"
              f"{report[arm]['boot']['hi']:+.4f}] p>0={report[arm]['boot']['p_gt0']:.3f}", flush=True)
    (OUT / 'scaleup_report.json').write_text(json.dumps(report, indent=1, default=float))
    print('wrote', OUT / 'scaleup_report.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
