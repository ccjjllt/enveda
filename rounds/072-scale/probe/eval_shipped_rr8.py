"""Evaluate the SHIPPED RR8 reranker on the labelled panel, and test a cross-spectrum veto.

The shipped model (068a `_RR8_MODEL_Z`, LightGBM, 102 features) is loaded unchanged. Its
input is rebuilt exactly as `_RR8_OVERLAY.model_features` does: rf.full_features (49 raw +
49 list-centred) followed by the two fragment scores and their list-centred copies.

Two conditions:
  freeze on   what the notebook does on the panel; every panel truth is a train structure,
              so the FREEZE guard makes every move impossible and the delta is exactly 0
  freeze off  the hidden-test condition: the test structures are unseen, so nothing is frozen

The veto is the cheap, shippable idea: execute RR8's reorder for a molecule only when the
per-spectrum evidence agrees. Three thresholds are swept. A veto can only *suppress* moves,
never add them, so it cannot do worse than RR8 alone on a molecule it suppresses.
"""
from __future__ import annotations

import base64
import json
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'rr8mod'))

import frag_rescore as frag            # noqa: E402
import rr8_features as rf              # noqa: E402
from rr8_cross_spectrum_experiment import (spec_dicts, rerank, formula_of,  # noqa: E402
                                           train_keys, key_of)

OUT = Path('artifacts/rr8x')
B64 = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/rr8/rr8_model.b64')
LIB_TAU = 0.90
ALPHA = 1.0
N_BOOT = 10000
RNG = np.random.default_rng(20261117)


def load_booster():
    import lightgbm as lgb
    raw = zlib.decompress(base64.b64decode(B64.read_text())).decode()
    return lgb.Booster(model_str=raw)


def model_features(spectra, smiles, cache, fcache):
    X = rf.full_features(spectra, smiles, cache)
    m1 = frag.score_molecule(spectra, smiles, cfg=None, cache=fcache)
    m2 = frag.score_molecule(spectra, smiles, cfg=frag.BASELINE_CFG, cache=fcache)
    M = np.array([[np.nan if a is None else a, np.nan if b is None else b] for a, b in zip(m1, m2)], float)
    with np.errstate(all='ignore'):
        Mc = M - np.nanmean(M, axis=0, keepdims=True)
    return np.hstack([X, M, Mc])


def mrr_of(order, truth):
    if truth is None:
        return 0.0
    try:
        return 1.0 / (order.index(truth) + 1)
    except ValueError:
        return 0.0


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
    te = pd.read_parquet('artifacts/ds/catalogue-experiment-saner/catalogue_replay_checkpoint/panel/test.parquet')
    sc = pd.read_parquet('artifacts/ds/catalogue-experiment-saner/catalogue_experiment/labelled_candidate_scores.parquet')
    booster = load_booster()
    keys = train_keys()
    cache, fcache = {}, {}
    recs = []
    for mid, cand in sc.groupby('molecule_id', sort=False):
        mid = str(mid)
        cand = cand.sort_values('baseline_rank', kind='stable').reset_index(drop=True)
        g = te[te.molecule_id.astype(str) == mid]
        spectra = spec_dicts(g)
        if not spectra or len(cand) < 3:
            continue
        smiles = cand.smiles.tolist()
        y = cand.label.values
        truth_i = int(np.flatnonzero(y == 1)[0]) if y.sum() else None
        truth = smiles[truth_i] if truth_i is not None else None
        X = model_features(spectra, smiles, cache, fcache)
        s = booster.predict(X)
        per = np.array([[np.nan if v is None else float(v) for v in
                         frag.score_molecule([sp], smiles, cache=fcache)] for sp in spectra])
        lead = per[:, [0]]
        with np.errstate(invalid='ignore'):
            win = np.nanmean(per >= lead, axis=0)     # share of spectra beating the leader
            spread = np.nanstd(per, axis=0)
        recs.append(dict(molecule_id=mid, smiles=smiles, truth=truth, y=y, scores=s,
                         libmax=float(cand.library_max.iloc[0]),
                         frozen=[key_of(x) in keys for x in smiles],
                         win=win, spread=spread, base_rank=(truth_i + 1) if truth_i is not None else None))
        if len(recs) % 25 == 0:
            print(f'  scored {len(recs)}', flush=True)
    print(f'records {len(recs)}', flush=True)
    pd.to_pickle(recs, OUT / 'shipped_recs.pkl')

    # ---- shipped RR8, both freeze settings -------------------------------------------
    report = {}
    for freeze in (True, False):
        d, ups, downs, changed, guarded = [], 0, 0, 0, 0
        for r in recs:
            base_order = list(r['smiles'])
            b = mrr_of(base_order, r['truth'])
            if r['libmax'] >= LIB_TAU:
                guarded += 1
                new = base_order
            else:
                fr = dict(zip(r['smiles'], r['frozen'])) if freeze else {s: False for s in r['smiles']}
                new, _ = rerank(pd.DataFrame({'smiles': r['smiles']}), r['scores'], fr, ALPHA)
            a = mrr_of(new, r['truth'])
            d.append(a - b)
            ups += int(a > b + 1e-12)
            downs += int(a < b - 1e-12)
            changed += int(new != base_order)
        tag = 'freeze' if freeze else 'nofreeze'
        report[tag] = dict(n=len(d), delta=float(np.mean(d)), up=ups, down=downs,
                           changed=changed, guarded=guarded, boot=boot(d))
        print(f"[shipped/{tag}] delta={np.mean(d):+.4f} up/down={ups}/{downs} "
              f"changed={changed} guarded={guarded} boot[{report[tag]['boot']['lo']:+.4f},"
              f"{report[tag]['boot']['hi']:+.4f}]", flush=True)

    # ---- veto sweep on top of shipped RR8 (freeze off) -------------------------------
    print('\nveto: only keep RR8 moves when the promoted leader wins on >= WIN of its spectra')
    veto = {}
    for win_th in (0.0, 0.4, 0.5, 0.6, 0.75, 1.0):
        d, kept, blocked = [], 0, 0
        for r in recs:
            base_order = list(r['smiles'])
            b = mrr_of(base_order, r['truth'])
            if r['libmax'] >= LIB_TAU:
                a = b
                new = base_order
            else:
                fr = {s: False for s in r['smiles']}
                new, _ = rerank(pd.DataFrame({'smiles': r['smiles']}), r['scores'], fr, ALPHA)
                if new != base_order:
                    promoted = new[0]
                    j = r['smiles'].index(promoted)
                    w = r['win'][j]
                    if not (np.isfinite(w) and w >= win_th):
                        new = base_order
                        blocked += 1
                    else:
                        kept += 1
                a = mrr_of(new, r['truth'])
            d.append(a - b)
        veto[str(win_th)] = dict(delta=float(np.mean(d)), kept=kept, blocked=blocked, boot=boot(d))
        print(f"  win>={win_th:.2f}  delta={np.mean(d):+.4f}  kept={kept} blocked={blocked} "
              f"boot[{veto[str(win_th)]['boot']['lo']:+.4f},{veto[str(win_th)]['boot']['hi']:+.4f}]", flush=True)

    report['veto'] = veto
    (OUT / 'shipped_report.json').write_text(json.dumps(report, indent=1, default=float))
    print('\nwrote', OUT / 'shipped_report.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
