"""Fit a training-fold-only branch matcher with within-formula spectrum swaps."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import cleavage_rank
from read_dual_training_scans import load_pairs


ENERGIES = (20, 40, 60)
ADDUCTS = ('[M+H]+', '[M-H]-')
CHANNELS = tuple(f'{adduct}/{energy}eV' for adduct in ADDUCTS for energy in ENERGIES)
WIDTH = cleavage_rank.PER_ENERGY
VIEWS = {'positive': (0, 1, 2), 'negative': (3, 4, 5), 'dual': tuple(range(len(CHANNELS)))}


def observed(row):
    mz = np.asarray(row['ms2_mzs'], dtype=float)
    intensity = np.asarray(row['ms2_normalized_intensities'], dtype=float)
    valid = (np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0)
             & (mz > 20) & (mz < row['precursor_mz'] - 1.5))
    mz, intensity = mz[valid], intensity[valid]
    if len(mz) < 4:
        return None
    order = np.argsort(mz, kind='stable')
    weights = np.sqrt(intensity[order])
    return mz[order], weights / weights.sum()


def branch_support(peaks, smiles, adduct):
    masses, codes = cleavage_rank.fragments(smiles)
    if adduct == ADDUCTS[1]:
        masses = masses - 2 * cleavage_rank.PROTON
    supported = np.zeros((len(peaks), WIDTH), dtype=bool)
    tolerance = np.maximum(0.003, masses * 10e-6)
    right = np.searchsorted(peaks, masses)
    for index in (right - 1, right):
        valid = np.flatnonzero((index >= 0) & (index < len(peaks)))
        matched = valid[np.abs(peaks[index[valid]] - masses[valid]) <= tolerance[valid]]
        supported[index[matched, None], codes[matched]] = True
    return supported


def feature_difference(observation, left, right, adduct):
    mz, weights = observation
    a = branch_support(mz, left, adduct)
    b = branch_support(mz, right, adduct)
    ordinary = weights @ (a.astype(np.float32) - b)
    exclusive = weights @ ((a & ~b[:, -1:]).astype(np.float32) - (b & ~a[:, -1:]))
    return np.concatenate((ordinary, exclusive))


def paired_features(pair, scans):
    left, right = pair['left'], pair['right']
    data = np.zeros((2, len(CHANNELS), 2 * WIDTH), dtype=np.float32)
    available = np.zeros(len(CHANNELS), dtype=bool)
    energies = pair['matched_dual_energies']
    for index, channel in enumerate(CHANNELS):
        adduct, energy = channel.split('/')
        if int(energy.removesuffix('eV')) not in energies:
            continue
        records = [observed(scans[side['train_row_0based'][channel]]) for side in (left, right)]
        if any(record is None for record in records):
            continue
        available[index] = True
        for side, observation in enumerate(records):
            data[side, index] = feature_difference(observation, left['smiles'], right['smiles'], adduct)
    for energy in range(len(ENERGIES)):
        if not available[energy] or not available[energy + len(ENERGIES)]:
            available[[energy, energy + len(ENERGIES)]] = False
    return data, available


def view(data, available, channels, conditional):
    picked = [channel for channel in channels if available[channel]]
    if not picked:
        return None
    pooled = data[:, picked].mean(axis=1)
    if not conditional:
        return pooled
    per_channel = np.zeros_like(data)
    per_channel[:, picked] = data[:, picked] / np.sqrt(len(picked))
    return np.concatenate((pooled, per_channel.reshape(2, -1)), axis=1)


def fit(arrays):
    # Each sample is the difference between the correct pairing and the swapped pairing.
    # The reverse orientation contributes an equally weighted negative example.
    difference = np.stack([pair[0] - pair[1] for pair in arrays])
    scale = np.maximum(np.sqrt((difference ** 2).mean(axis=0)), 0.02)
    x = difference / scale
    model = LogisticRegression(C=0.6, fit_intercept=False, max_iter=750)
    model.fit(np.concatenate((x, -x)), np.repeat((1, 0), len(x)))
    return model.coef_[0] / scale


def evaluate(margins):
    values = np.asarray(margins, dtype=float).reshape(-1, 2)
    if not len(values):
        return {'pairs': 0}
    signed = values * [1, -1]
    assignment = signed.sum(axis=1)
    ties = np.abs(assignment) <= 1e-10
    wins = assignment > 1e-10
    losses = assignment < -1e-10
    rr = np.where(signed > 1e-10, 1, np.where(signed < -1e-10, 0.5, 0.75))
    return dict(pairs=len(values), wins=int(wins.sum()), ties=int(ties.sum()), losses=int(losses.sum()),
                assignment_accuracy_with_ties=float((wins.sum() + 0.5 * ties.sum()) / len(values)),
                both_spectra_correct=int((signed > 1e-10).all(axis=1).sum()), mean_mrr_at_two=float(rr.mean()))


def summarize(rows):
    pooled = {name: evaluate([row[name] for row in rows]) for name in ('uniform', 'pooled_trained', 'local_trained')}
    source_counts = Counter(tuple(row['matched_energies']) for row in rows)
    return dict(models=pooled, matched_energy_counts={','.join(map(str, k)):v for k,v in source_counts.items()},
                hard_subset={name:evaluate([row[name] for row in rows if row['tanimoto'] >= 0.6]) for name in pooled})


def cross_validate(samples, channels):
    names = ('uniform', 'pooled_trained', 'local_trained')
    examples = []
    for pair, data, available in samples:
        pooled = view(data, available, channels, False)
        if pooled is not None:
            examples.append((pair, pooled, view(data, available, channels, True), available))
    predictions = []
    for fold in sorted({pair['fold'] for pair, _, _, _ in examples}):
        train = [(pooled, conditional) for pair, pooled, conditional, _ in examples if pair['fold'] != fold]
        if not train:
            continue
        ordinary = fit([pooled[:, :WIDTH] for pooled, _ in train])
        local = fit([conditional for _, conditional in train])
        for pair, pooled, conditional, available in examples:
            if pair['fold'] != fold:
                continue
            scores = (pooled[:, WIDTH - 1], pooled[:, :WIDTH] @ ordinary, conditional @ local)
            predictions.append(dict(fold=fold, tanimoto=pair['morgan_tanimoto'],
                                    matched_energies=[e for i,e in enumerate(ENERGIES) if available[i]],
                                    **{name:value.tolist() for name,value in zip(names, scores, strict=True)}))
    return predictions


def run(pairs, receipt, train, out, pairs_per_fold):
    cleavage_rank.load_chemistry()
    selected, spectra = load_pairs(pairs, receipt, train, per_fold=pairs_per_fold)
    samples = [(pair, *paired_features(pair, spectra)) for pair in selected]
    results = {name:cross_validate(samples, channels) for name, channels in VIEWS.items()}
    out.mkdir(parents=True, exist_ok=True)
    with (out/'dual_branch_swap_predictions.jsonl').open('w') as file:
        for name, rows in results.items():
            for row in rows:
                file.write(json.dumps(dict(view=name, **row), separators=(',', ':'))+'\n')
    summary = dict(scope='Scaffold-group folds on competition training spectra. Only pair assignment; not the official leaderboard metric.',
                   selected_pairs=len(samples), observations_loaded=len(spectra), pairs_per_fold=pairs_per_fold,
                   selected_pairs_with_shared_dual_energy=sum(bool(available.any()) for _, _, available in samples),
                   comparison='Equal-weight single-cut coverage vs learned pooled branch features vs energy/polarity-conditioned exclusive branch features.',
                   views={name:summarize(rows) for name,rows in results.items()})
    (out/'dual_branch_swap_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('pairs','receipt','train','out'):
        cli.add_argument('--'+name, type=Path, required=True)
    cli.add_argument('--pairs-per-fold', type=int, default=400)
    run(**vars(cli.parse_args()))
