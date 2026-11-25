"""Compare frozen ICEBERG scores on real, same-formula energy-matched spectrum swaps."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import numpy as np

from forward_ce import DELTA_WEIGHT, clean_peaks, cosine, differences, entropy, load_iceberg, match_peaks
from fit_dual_energy_pairs import evaluate, fit
from read_dual_training_scans import load_pairs


ION = '[M+H]+'
FEATURES = ('observed_cosine', 'observed_entropy', 'fixed_40ev', 'reversed_energy',
            'exclusive_cosine', 'fragment_response')


def score_pair(pair, scans, ice, predictions):
    sides = (pair['left'], pair['right'])
    names = tuple(row['smiles'] for row in sides)
    energies = pair['matched_dual_energies']
    if any(not len(predictions[name][energy][0]) for name in names for energy in energies):
        return None
    features = []
    for side in sides:
        queries = {}
        for energy in energies:
            scan = scans[side['train_row_0based'][f'{ION}/{energy}eV']]
            q = clean_peaks(scan['ms2_mzs'], scan['ms2_normalized_intensities'], scan['precursor_mz'])
            if len(q[0]) < 4:
                return None
            queries[energy] = q
        channels = []
        for energy, reverse in zip(energies, reversed(energies), strict=True):
            q = queries[energy]
            a, b = [predictions[name][energy] for name in names]
            aligned_a, aligned_b = (match_peaks(q[0], p) for p in (a, b))
            exclusives = np.sqrt(aligned_a) * (aligned_b == 0) - np.sqrt(aligned_b) * (aligned_a == 0)
            channels.append((
                np.sqrt(q[1]) @ (np.sqrt(aligned_a) - np.sqrt(aligned_b)),
                entropy(ice, q, a) - entropy(ice, q, b),
                cosine(q, predictions[names[0]][40]) - cosine(q, predictions[names[1]][40]),
                cosine(q, predictions[names[0]][reverse]) - cosine(q, predictions[names[1]][reverse]),
                np.sqrt(q[1]) @ exclusives,
            ))
        rows = [SimpleNamespace(query_id=energy, collision_energy=energy) for energy in energies]
        response = differences(rows, queries, {name: predictions[name] for name in names})
        if any(len(response[name]) != len(energies) - 1 for name in names):
            return None
        features.append([*np.mean(channels, axis=0), float(np.mean(response[names[0]])-np.mean(response[names[1]]))])
    values = np.asarray(features)
    return values if np.isfinite(values).all() else None


def summarize(samples):
    outcomes = {name: [] for name in (*FEATURES, 'cosine_plus_response', 'held_out_learned')}
    very_close = {name: [] for name in outcomes}
    folds = {}
    for fold in sorted({pair['fold'] for pair, _ in samples}):
        train = [arr[:, (0, 1, 4, 5)] for pair, arr in samples if pair['fold'] != fold]
        held = [(pair, arr) for pair, arr in samples if pair['fold'] == fold]
        if not train:
            raise ValueError('A scaffold-held-out comparison requires multiple nonempty folds')
        weights = fit(train)
        rows = {name: [] for name in outcomes}
        for _, arr in held:
            for column, name in enumerate(FEATURES):
                rows[name].append(arr[:, column])
            rows['cosine_plus_response'].append(arr[:, 0] + DELTA_WEIGHT * arr[:, 5])
            rows['held_out_learned'].append(arr[:, (0, 1, 4, 5)] @ weights)
        for name, margins in rows.items():
            outcomes[name].extend(margins)
            very_close[name].extend(value for (pair, _), value in zip(held, margins, strict=True)
                                    if pair['morgan_tanimoto'] >= 0.8)
        folds[str(fold)] = dict(
            train_pairs=len(train), validation_pairs=len(held), learned_weights=weights.tolist(),
            models={name: evaluate(value) for name, value in rows.items()},
        )
    return dict(models={name: evaluate(value) for name, value in outcomes.items()},
                similarity_at_least_080={name: evaluate(value) for name, value in very_close.items()},
                by_fold=folds)


def run(train, pairs, receipt, out, per_fold):
    start = monotonic()
    chosen, scans = load_pairs(pairs, receipt, train, per_fold=per_fold, minimum_similarity=0.6)
    ice, predictor, device = load_iceberg(Path('/kaggle/input'), 'cpu', 24)
    if ION not in ice.COV:
        raise ValueError('The model does not support the frozen positive adduct')
    instrument = ice.instr_token({'instrument': 'timsTOF'})
    supported = {}
    planned, conditions = [], set()
    for pair in chosen:
        names = [pair[key]['smiles'] for key in ('left', 'right')]
        for name in names:
            if name not in supported:
                supported[name] = ice.smiles_ok(name)[0]
        energies = pair['matched_dual_energies']
        if len(energies) < 2 or not all(supported[name] for name in names):
            continue
        planned.append(pair)
        conditions.update((name, ION, instrument, float(energy)) for name in names for energy in energies)
    if not planned:
        raise ValueError('The frozen paired comparison has no supported structures')
    print('ICEBERG hard pairs', len(planned), 'unique model conditions', len(conditions), flush=True)
    predictor.predict(conditions, float('inf'))
    predictions = {}
    # Both candidate spectra for a same-formula pair use the same observed precursor exclusion.
    samples, exclusions = [], Counter()
    for pair in planned:
        energies = pair['matched_dual_energies']
        sides = (pair['left'], pair['right'])
        names = [side['smiles'] for side in sides]
        precursor = min(scans[side['train_row_0based'][f'{ION}/{en}eV']]['precursor_mz']
                        for side in sides for en in energies)
        if any(predictor.cache.get((name, ION, instrument, float(energy))) is None
               for name in names for energy in energies):
            exclusions['missing_model_prediction'] += 1
            continue
        for name in names:
            predictions[name] = {en: clean_peaks(*predictor.cache[(name, ION, instrument, float(en))], precursor)
                                 for en in energies}
        scores = score_pair(pair, scans, ice, predictions)
        if scores is None:
            exclusions['unusable_peak_series'] += 1
            continue
        samples.append((pair, scores))
    if not samples:
        raise ValueError('No paired spectra received a complete model comparison')
    result = dict(
        scope='Competition training data, same-formula hard pairs, positive ion only',
        shared_instrument=instrument, measured_energy_ev=[20, 40, 60], device=device,
        selected_pairs=len(chosen), planned_pairs=len(planned), scored_pairs=len(samples),
        model_conditions=len(conditions), missing_conditions=sum(predictor.cache.get(key) is None for key in conditions),
        pair_exclusions=dict(exclusions),
        paired_evaluation=summarize(samples),
        elapsed_seconds=monotonic()-start,
    )
    out.mkdir(parents=True, exist_ok=True)
    evidence = ''.join(json.dumps(dict(fold=pair['fold'], tanimoto=pair['morgan_tanimoto'],
                                      structures=[pair[side]['key'] for side in ('left', 'right')],
                                      measured_energies_ev=pair['matched_dual_energies'],
                                      oriented_scores=scores.tolist()), separators=(',', ':'), allow_nan=False)+'\n'
                       for pair, scores in samples).encode()
    (out/'ice_forward_swap_features.jsonl').write_bytes(evidence)
    result.update(features=FEATURES, evidence_sha256=sha256(evidence).hexdigest())
    (out/'ice_forward_swap_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['paired_evaluation']['models']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'pairs', 'receipt', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--per-fold', type=int, default=48)
    run(**vars(parser.parse_args()))
