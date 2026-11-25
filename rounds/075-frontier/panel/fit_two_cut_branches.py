"""Estimate motif-class weights on hard same-formula pairs with scaffold holdout."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import monotonic

import numpy as np

import cleavage_rank
from fit_dual_energy_pairs import VIEWS, evaluate, fit
from probe_two_cut_fragments import spectral_views
from read_dual_training_scans import load_pairs


ARMS = ('single','uniform_two_cut','learned_two_cut','learned_when_single_ties')


def inputs(data, present, channels):
    active = [i for i in channels if present[i]]
    if not active:
        return None
    score = data.copy()
    score[:, :, 3] = data[:, :, 3] - data[:, :, 2] - data[:, :, 1] + data[:, :, 0]
    score[:, :, 1:3] -= data[:, :, :1]
    pooled = score[active].mean(axis=0)
    matched = np.zeros_like(score)
    matched[active] = score[active] / np.sqrt(len(active))
    stacked = np.concatenate((pooled, matched.transpose(1, 0, 2).reshape(2, -1)), axis=1)
    ordinary = data[active].mean(axis=0)
    return ordinary[:,0], ordinary[:,3], stacked


def summarize(evidence):
    values = {arm:[row[arm] for row in evidence] for arm in ARMS}
    result = {arm:evaluate(margins) for arm,margins in values.items()}
    classify = lambda margin: 'win' if margin[0]-margin[1] > 1e-10 else ('loss' if margin[0]-margin[1] < -1e-10 else 'tie')
    changes = {arm:dict(Counter(f'{classify(a)}_to_{classify(b)}'
                               for a,b in zip(values['single'],values[arm],strict=True))) for arm in ARMS[1:]}
    return dict(models=result, sign_changes=changes)


def cross_validate(samples, channels):
    rows = []
    for pair, score, present in samples:
        x = inputs(score, present, channels)
        if x is not None:
            rows.append((pair, *x))
    predictions = []
    for fold in sorted({pair['fold'] for pair, *_ in rows}):
        training = [(single, features) for pair,single,_,features in rows if pair['fold'] != fold]
        if not training:
            continue
        two_cut = fit([features for _,features in training])
        single_coef = fit([single[:, None] for single,_ in training])[0]
        for pair,single,uniform,features in rows:
            if pair['fold'] != fold:
                continue
            local = features @ two_cut
            preserved = np.where(np.abs(single)>1e-10, single * abs(single_coef), local)
            predictions.append(dict(fold=fold, similarity=pair['morgan_tanimoto'],
                                    **{name:array.tolist() for name,array in zip(ARMS,(single,uniform,local,preserved),strict=True)}))
    return dict(all=summarize(predictions), very_close=summarize([row for row in predictions if row['similarity']>=0.8]),
                folds={str(fold):summarize([r for r in predictions if r['fold']==fold])
                       for fold in sorted({row['fold'] for row in predictions})})


def run(train, pairs, receipt, out, per_fold):
    start = monotonic()
    cleavage_rank.load_chemistry()
    chosen, scans = load_pairs(pairs,receipt,train,per_fold=per_fold,minimum_similarity=0.6)
    samples = []
    for pair in chosen:
        data, present, _ = spectral_views(pair,scans)
        samples.append((pair,data,present))
    report = dict(scope='Group-held-out scaffold folds of competition training pairs. No submission scores.',
                  selected_pairs=len(chosen), unique_training_scans=len(scans), seconds=monotonic()-start,
                  views={name:cross_validate(samples,channels) for name,channels in VIEWS.items()})
    out.mkdir(parents=True,exist_ok=True)
    (out/'learned_two_cut_hard_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:value['all']['models'] for name,value in report['views'].items()}),flush=True)


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('train','pairs','receipt','out'):
        cli.add_argument('--'+name,type=Path,required=True)
    cli.add_argument('--per-fold',type=int,default=4096)
    run(**vars(cli.parse_args()))
