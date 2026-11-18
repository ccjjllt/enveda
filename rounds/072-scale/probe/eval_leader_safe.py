"""Is the RR8 fusion rule, not the feature set, what costs score?

`_RR8_OVERLAY.rerank_list` lets the leader be replaced. On the panel the truth sits at rank 1
for 87% of molecules, so every wrong promotion costs a full 1.0 while a correct one can earn
at most 0.5. The competition's other rerankers are leader-safe for exactly this reason.

This probe holds everything else fixed and only adds the rule "position 0 never changes",
then re-scores: shipped booster, and the two retrained arms from train_rr8_scaled.py.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'rr8mod'))

OUT = Path('artifacts/rr8x')
LIB_TAU = 0.90
ALPHA = 1.0
N_BOOT = 10000
RNG = np.random.default_rng(20261117)


def rerank(smiles, scores, alpha=ALPHA, protect_top1=True):
    pos = {s: i for i, s in enumerate(smiles)}
    movable = [s for s in smiles if (pos[s] >= 1 if protect_top1 else True)]
    if len(movable) < 2:
        return list(smiles)
    slots = sorted(pos[s] for s in movable)
    sc = dict(zip(smiles, scores))
    base = np.array([-np.log(pos[s] + 1) for s in movable], float)
    rr = np.array([sc[s] for s in movable], float)

    def z(v):
        return (v - v.mean()) / v.std() if v.std() > 1e-12 else np.zeros_like(v)
    order = [movable[i] for i in np.argsort(-(z(base) + alpha * z(rr)), kind='stable')]
    out = list(smiles)
    for slot, s in zip(slots, order):
        out[slot] = s
    return out


def boot(d):
    d = np.asarray(d, float)
    if not len(d):
        return dict(n=0)
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    m = d[idx].mean(1)
    return dict(n=int(len(d)), mean=float(d.mean()), lo=float(np.quantile(m, .025)),
                hi=float(np.quantile(m, .975)), p_gt0=float((m > 0).mean()))


def score(ev, preds, protect):
    d, ups, downs, changed = [], 0, 0, 0
    for r in ev:
        smiles = r['smiles']
        y = np.asarray(r['y'])
        truth = int(np.flatnonzero(y == 1)[0]) if y.sum() else None
        base_order = list(smiles)
        b = 1.0 / (base_order.index(smiles[truth]) + 1) if truth is not None else 0.0
        if r['libmax'] >= LIB_TAU:
            new = base_order
        else:
            new = rerank(smiles, preds[r['molecule_id']], protect_top1=protect)
        a = 1.0 / (new.index(smiles[truth]) + 1) if truth is not None else 0.0
        d.append(a - b)
        ups += int(a > b + 1e-12)
        downs += int(a < b - 1e-12)
        changed += int(new != base_order)
    return dict(delta=float(np.mean(d)), up=ups, down=downs, changed=changed, boot=boot(d))


def main() -> int:
    panel = pd.read_pickle(OUT / 'features.pkl')
    ev = [r for r in panel if r['arm'] == 'B']
    store = pickle.load(open(OUT / 'triple_features.pkl', 'rb'))
    recs = pd.read_pickle(OUT / 'shipped_recs.pkl')
    shipped = {r['molecule_id']: r['scores'] for r in recs}

    from sklearn.ensemble import HistGradientBoostingClassifier
    report = {}
    for arm in ('A', 'B'):
        Xtr = np.vstack([store[k][arm] for k in store])
        ytr = np.concatenate([[1 if j == store[k]['y'] else 0 for j in range(25)] for k in store])
        m = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.06,
                                           l2_regularization=1.0, min_samples_leaf=40,
                                           max_bins=128, random_state=0).fit(Xtr, ytr)
        preds = {r['molecule_id']: m.predict_proba(r['X'] if arm == 'B' else r['X'][:, :53])[:, 1]
                 for r in ev}
        for protect in (False, True):
            s = score(ev, preds, protect)
            report[f'{arm}_protect{int(protect)}'] = s
            print(f"[retrain/{arm}] protect_top1={protect!s:5s} delta={s['delta']:+.4f} "
                  f"up/down={s['up']}/{s['down']} changed={s['changed']} "
                  f"boot[{s['boot']['lo']:+.4f},{s['boot']['hi']:+.4f}] p>0={s['boot']['p_gt0']:.3f}",
                  flush=True)

    for protect in (False, True):
        s = score(ev, shipped, protect)
        report[f'shipped_protect{int(protect)}'] = s
        print(f"[shipped]     protect_top1={protect!s:5s} delta={s['delta']:+.4f} "
              f"up/down={s['up']}/{s['down']} changed={s['changed']} "
              f"boot[{s['boot']['lo']:+.4f},{s['boot']['hi']:+.4f}] p>0={s['boot']['p_gt0']:.3f}",
              flush=True)

    (OUT / 'leader_safe_report.json').write_text(json.dumps(report, indent=1, default=float))
    print('wrote', OUT / 'leader_safe_report.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
