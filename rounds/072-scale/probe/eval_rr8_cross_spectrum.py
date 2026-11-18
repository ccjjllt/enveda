"""Evaluate the 072a arms: shipped mean-over-spectra RR8 features vs +cross-spectrum consistency.

Protocol
  primary   fit on the panel's 60 'train' molecules, score the 30 calibration + 30 test
            molecules (the panel author's own split), report MRR@25 per molecule and the
            paired per-molecule delta against the untouched baseline order
  bootstrap 10,000 resamples of the molecules, paired, for a lower bound on mean delta
  cv        5-fold grouped by molecule over all 120, to see whether the primary split is luck

Arms: A = shipped 53 columns (49 spectrum features + 4 fragment columns)
      A2 = same, recomputed through an explicit per-spectrum mean (control)
      B  = A + 4 cross-spectrum consistency columns
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'rr8mod'))

from rr8_cross_spectrum_experiment import rerank, formula_of  # noqa: E402

OUT = Path('artifacts/rr8x')
LIB_TAU = 0.90
N_BOOT = 10000
RNG = np.random.default_rng(20261117)


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=200, learning_rate=0.06, l2_regularization=1.0,
        min_samples_leaf=20, max_bins=128, random_state=0)


def fit_predict(rows_train, rows_eval, arm):
    Xtr = np.vstack([r['X'] for r in rows_train if r['arm'] == arm])
    ytr = np.concatenate([r['y'] for r in rows_train if r['arm'] == arm])
    if ytr.sum() < 2:
        return None
    m = make_model().fit(Xtr, ytr)
    out = {}
    for r in rows_eval:
        if r['arm'] != arm:
            continue
        out[r['molecule_id']] = m.predict_proba(r['X'])[:, 1]
    return out


def evaluate(rows, arm, fit_ids, eval_ids, use_freeze=True):
    tr = [r for r in rows if r['molecule_id'] in fit_ids]
    ev = [r for r in rows if r['molecule_id'] in eval_ids]
    pred = fit_predict(tr, ev, arm)
    if pred is None:
        return {}
    res = {}
    for r in ev:
        if r['arm'] != arm:
            continue
        mid = r['molecule_id']
        y = np.asarray(r['y'])
        truth = int(np.flatnonzero(y == 1)[0]) if y.sum() else None
        smiles = r['smiles']
        base_order = list(smiles)
        base_mrr = 1.0 / (base_order.index(smiles[truth]) + 1) if truth is not None else 0.0
        if r['libmax'] >= LIB_TAU:
            new_order, moved = list(smiles), False
        else:
            frozen = dict(zip(smiles, r['frozen'])) if use_freeze else {s: False for s in smiles}
            new_order, moved = rerank(pd.DataFrame({'smiles': smiles}), pred[mid], frozen)
        new_mrr = 1.0 / (new_order.index(smiles[truth]) + 1) if truth is not None else 0.0
        # candidate-level discrimination: does the model score put the truth above the decoys?
        s = np.asarray(pred[mid], float)
        yv = y.astype(bool)
        auc = float('nan')
        if truth is not None and yv.sum() and (~yv).sum() and np.isfinite(s).all():
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(yv, s))
        pred_rank = int((s > s[truth]).sum()) + 1 if truth is not None else None
        res[mid] = dict(base=base_mrr, arm=new_mrr, moved=bool(moved),
                        top1_changed=bool(new_order[0] != base_order[0]),
                        libmax=float(r['libmax']), role=r['role'],
                        hard=bool(truth is not None and (truth + 1) >= 2),
                        auc=auc, pred_rank=pred_rank, base_rank=(truth + 1) if truth is not None else None)
    return res


def bootstrap(deltas):
    d = np.asarray(deltas, float)
    if len(d) == 0:
        return dict(n=0)
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(1)
    return dict(n=int(len(d)), mean=float(d.mean()),
                lo=float(np.quantile(means, 0.025)), hi=float(np.quantile(means, 0.975)),
                p_gt0=float((means > 0).mean()))


def summarise(res, name):
    if not res:
        return None
    mids = list(res)
    base = np.array([res[m]['base'] for m in mids])
    arm = np.array([res[m]['arm'] for m in mids])
    d = arm - base
    hard = [m for m in mids if res[m]['hard']]
    aucs = np.array([res[m]['auc'] for m in mids if np.isfinite(res[m]['auc'])])
    out = dict(name=name, n=len(mids), mrr_base=float(base.mean()), mrr_arm=float(arm.mean()),
               delta=float(d.mean()), moved=int(sum(res[m]['moved'] for m in mids)),
               top1_changed=int(sum(res[m]['top1_changed'] for m in mids)),
               n_guarded=int(sum(res[m]['libmax'] >= LIB_TAU for m in mids)),
               up=int((d > 1e-12).sum()), down=int((d < -1e-12).sum()),
               boot_delta=bootstrap(d),
               n_hard=len(hard),
               hard_delta=float(np.mean([res[m]['arm'] - res[m]['base'] for m in hard])) if hard else None,
               hard_top1=int(sum(res[m]['pred_rank'] == 1 for m in hard)) if hard else 0,
               hard_pred_rank=float(np.mean([res[m]['pred_rank'] for m in hard])) if hard else None,
               hard_base_rank=float(np.mean([res[m]['base_rank'] for m in hard])) if hard else None,
               auc_mean=float(aucs.mean()) if len(aucs) else None,
               auc_n=int(len(aucs)))
    return out


def main() -> int:
    rows = pd.read_pickle(OUT / 'features.pkl')
    all_ids = sorted({r['molecule_id'] for r in rows})
    role = {r['molecule_id']: r['role'] for r in rows}
    fit_ids = {m for m in all_ids if role[m] == 'train'}
    eval_ids = {m for m in all_ids if role[m] != 'train'}
    print(f'molecules total={len(all_ids)} fit={len(fit_ids)} eval={len(eval_ids)}')

    report = {'primary': {}, 'cv': {}}
    per_mol = {}
    # The panel's truths are all train structures, so the shipped FREEZE guard blocks every
    # move. The hidden test's molecules are unseen, so the informative condition is freeze off.
    for use_freeze in (False, True):
        tag = 'freeze' if use_freeze else 'nofreeze'
        for arm in ('A', 'A2', 'B'):
            res = evaluate(rows, arm, fit_ids, eval_ids, use_freeze=use_freeze)
            s = summarise(res, f'{arm}_{tag}')
            report['primary'][f'{arm}_{tag}'] = s
            per_mol[f'{arm}_{tag}'] = res
            if s:
                print(f"[primary/{tag}] {arm:2s} n={s['n']} MRR base={s['mrr_base']:.4f} arm={s['mrr_arm']:.4f} "
                      f"delta={s['delta']:+.4f} up/down={s['up']}/{s['down']} moved={s['moved']} "
                      f"top1={s['top1_changed']} guarded={s['n_guarded']} "
                      f"boot[{s['boot_delta']['lo']:+.4f},{s['boot_delta']['hi']:+.4f}]", flush=True)
                print(f"                  hard-n={s['n_hard']} hard base_rank={s['hard_base_rank']} "
                      f"pred_rank={s['hard_pred_rank']} hard_top1={s['hard_top1']} "
                      f"hard_delta={s['hard_delta']} AUC={s['auc_mean']}", flush=True)
        for a, b in (('B', 'A'), ('A2', 'A')):
            if per_mol[f'{a}_{tag}'] and per_mol[f'{b}_{tag}']:
                common = sorted(set(per_mol[f'{a}_{tag}']) & set(per_mol[f'{b}_{tag}']))
                d = np.array([per_mol[f'{a}_{tag}'][m]['arm'] - per_mol[f'{b}_{tag}'][m]['arm'] for m in common])
                bt = bootstrap(d)
                report['primary'][f'{a}_minus_{b}_{tag}'] = dict(n=len(common), mean=float(d.mean()),
                                                                 up=int((d > 1e-12).sum()),
                                                                 down=int((d < -1e-12).sum()), boot=bt)
                print(f"[primary/{tag}] {a}-{b} n={len(common)} mean={d.mean():+.4f} "
                      f"up/down={(d>1e-12).sum()}/{(d<-1e-12).sum()} "
                      f"boot[{bt['lo']:+.4f},{bt['hi']:+.4f}] p>0={bt['p_gt0']:.3f}", flush=True)

    # 5-fold grouped CV over all molecules, freeze off (the informative condition)
    from sklearn.model_selection import GroupKFold
    ids = np.array(all_ids)
    gkf = GroupKFold(n_splits=5)
    for arm in ('A', 'A2', 'B'):
        folds = []
        for tr_i, te_i in gkf.split(ids, groups=ids):
            tr = set(ids[tr_i])
            te = set(ids[te_i])
            folds.append(evaluate(rows, arm, tr, te, use_freeze=False))
        merged = {k: v for f in folds for k, v in f.items()}
        s = summarise(merged, f'{arm}_cv')
        report['cv'][arm] = s
        per_mol[f'{arm}_cv'] = merged
        print(f"[cv/nofreeze] {arm:2s} n={s['n']} MRR base={s['mrr_base']:.4f} arm={s['mrr_arm']:.4f} "
              f"delta={s['delta']:+.4f} up/down={s['up']}/{s['down']} "
              f"boot[{s['boot_delta']['lo']:+.4f},{s['boot_delta']['hi']:+.4f}]", flush=True)
        print(f"                hard-n={s['n_hard']} hard base_rank={s['hard_base_rank']} "
              f"pred_rank={s['hard_pred_rank']} hard_top1={s['hard_top1']} "
              f"hard_delta={s['hard_delta']} AUC={s['auc_mean']}", flush=True)
    for a, b in (('B', 'A'), ('A2', 'A')):
        common = sorted(set(per_mol[f'{a}_cv']) & set(per_mol[f'{b}_cv']))
        d = np.array([per_mol[f'{a}_cv'][m]['arm'] - per_mol[f'{b}_cv'][m]['arm'] for m in common])
        bt = bootstrap(d)
        report['cv'][f'{a}_minus_{b}'] = dict(n=len(common), mean=float(d.mean()),
                                              up=int((d > 1e-12).sum()),
                                              down=int((d < -1e-12).sum()), boot=bt)
        print(f"[cv/nofreeze] {a}-{b} n={len(common)} mean={d.mean():+.4f} "
              f"up/down={(d>1e-12).sum()}/{(d<-1e-12).sum()} "
              f"boot[{bt['lo']:+.4f},{bt['hi']:+.4f}] p>0={bt['p_gt0']:.3f}", flush=True)
    if report['cv'].get('A') and report['cv'].get('B'):
        d = []
        for m in sorted(set(report['cv']['A']) and all_ids):
            pass
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, default=float))
    pd.to_pickle(per_mol, OUT / 'per_molecule.pkl')
    print('\nwrote', OUT / 'report.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
