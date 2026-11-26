"""Calibrate peak-level evidence on exact archived GLACIER hard-pair predictions."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from read_dual_training_scans import load_pairs


ADDUCT = '[M+H]+'
ENERGIES = (20, 40, 60)
GL_SUMMARY_SHA = '3f40b93bc00b3ebfab3d030f90ab0031096dbfdba85524205fbbbb48b222bb00'
PAIR_EXPORT_SHA = '4c4de6596c9a1e7becfd8071112e527028b50b16eb5ef560b2ac0638f1a2ed2f'
FILES = {'gl_forward_swap_selection.jsonl': '97785c43ec9935b1da9055f62eba528b807dd69d5a7de69fbd6f9a3519f9b619',
         'gl_forward_swap_prediction_peaks.parquet': '2854a4e7aaff4f158ca4776e1f79140203015bbcf140318c3d833ce0d5073870',
         'gl_forward_swap_features.jsonl': '43a8e85c34185275282e1ccedffcceb85709371915afd508c1ae53b2dbaec246'}
ARMS = ('native_cosine', 'predicted_peak_llr', 'bidirectional_peak_llr',
        'mismatched_native_cosine', 'mismatched_predicted_peak_llr', 'mismatched_bidirectional_peak_llr')
RELATIVE_MASS_EDGES = (0.25, 0.5, 0.75)
INTENSITY_EDGES = (0.003, 0.01, 0.03)
STRONG_MATCH = 0.02
CONTEXTS = 3 * 4 * 4
PRIOR_PAIR_WEIGHT = 1.0


def checked(path, digest):
    body = path.read_bytes()
    if sha256(body).hexdigest() != digest:
        raise ValueError(f'Frozen source changed: {path.name}')
    return body


def clean_peaks(mz, intensity, precursor):
    # Reproduce the archived GL hard-pair preprocessing, including precursor exclusion.
    mz, intensity = np.asarray(mz, dtype=float), np.asarray(intensity, dtype=float)
    keep = (np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0) & (mz < precursor - 0.5))
    mz, intensity = mz[keep], intensity[keep]
    if not len(mz):
        return mz, intensity
    keep = intensity >= 0.002 * intensity.max()
    mz, intensity = mz[keep], intensity[keep]
    if len(mz) > 256:
        strongest = np.argsort(intensity)[-256:]
        mz, intensity = mz[strongest], intensity[strongest]
    order = np.argsort(mz)
    return mz[order], intensity[order] / intensity.sum()


def aligned_indexes(query_mass, predicted_mass):
    """The archived nearest-mass matcher, keeping each prediction at most once."""
    hi = np.minimum(np.searchsorted(predicted_mass, query_mass), len(predicted_mass) - 1)
    lo = np.maximum(hi - 1, 0)
    best = np.where(np.abs(predicted_mass[hi] - query_mass) < np.abs(predicted_mass[lo] - query_mass), hi, lo)
    error = np.abs(predicted_mass[best] - query_mass)
    within = np.flatnonzero(error <= np.maximum(0.01, query_mass * 20e-6))
    nearest = within[np.argsort(error[within], kind='stable')]
    _, positions = np.unique(best[nearest], return_index=True)
    measured = nearest[positions]
    return measured, best[measured]


def event_context(mz, intensity, precursor, energy):
    mass = np.searchsorted(RELATIVE_MASS_EDGES, mz / precursor, side='right')
    level = np.searchsorted(INTENSITY_EDGES, intensity, side='right')
    return ((ENERGIES.index(energy) * 4 + mass) * 4 + level).astype(np.intp)


def evidence(query, prediction, precursor, energy):
    q_mz, q_int = query
    p_mz, p_int = prediction
    q_match, p_match = aligned_indexes(q_mz, p_mz)
    p_aligned, q_aligned = np.zeros(len(q_mz)), np.zeros(len(p_mz))
    p_aligned[q_match], q_aligned[p_match] = p_int[p_match], q_int[q_match]
    cosine = float(np.sqrt(q_int) @ np.sqrt(p_aligned))
    # Categories are absent, aligned below 2% relative intensity, aligned at >=2% intensity.
    outcomes = [(values > 0).astype(np.intp) + (values >= STRONG_MATCH) for values in (q_aligned, p_aligned)]
    codes = (event_context(p_mz, p_int, precursor, energy), event_context(q_mz, q_int, precursor, energy))
    return cosine, tuple(code * 3 + outcome for code, outcome in zip(codes, outcomes, strict=True))


def index_rows(raw, key):
    rows = [json.loads(line) for line in raw.splitlines()]
    result = {key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate frozen evidence rows')
    return result


def trial_key(row):
    return row['fold'], tuple(row['structures']), tuple(row['measured_energies_ev'])


def load(train, pairs, receipt, gl):
    raw_summary = checked(gl / 'gl_forward_swap_summary.json', GL_SUMMARY_SHA)
    manifest = json.loads(raw_summary)
    if (manifest['pair_export_sha256'], manifest['scored_pairs'], manifest['model_conditions'],
        manifest['shared_instrument'], manifest['adduct']) != (PAIR_EXPORT_SHA, 239, 1060, 'QTOF', ADDUCT):
        raise ValueError('Unexpected archived hard-pair population')
    chosen, scans = load_pairs(pairs, receipt, train, per_fold=48, minimum_similarity=0.6)
    if len(chosen) != 240 or {pair['fold'] for pair in chosen} != set(range(5)):
        raise ValueError('Expected the original selection of five scaffold folds')
    if sha256(pairs.read_bytes()).hexdigest() != PAIR_EXPORT_SHA:
        raise ValueError('Wrong hard-pair export')
    selection, peaks, features = (checked(gl / name, digest) for name, digest in FILES.items())
    selections = index_rows(selection, trial_key)
    originals = index_rows(features, trial_key)
    prediction_path = gl / 'gl_forward_swap_prediction_peaks.parquet'
    rows = pq.read_table(prediction_path).to_pylist()
    if len(rows) != 1060 or any(not row['available'] or row['adduct'] != ADDUCT or row['instrument'] != 'QTOF'
                                or row['energy_ev'] not in ENERGIES for row in rows):
        raise ValueError('Unexpected candidate–energy prediction archive')
    predictions = {(row['smiles'], row['energy_ev']): (row['mz'], row['intensity']) for row in rows}
    if len(predictions) != len(rows) or len(selections) != 240 or len(originals) != 239:
        raise ValueError('Frozen GL predictions or pair IDs were duplicated or removed')
    del peaks
    membership, cases, all_keys = {}, [], set()
    max_baseline_error = 0.0
    for pair in chosen:
        sides = pair['left'], pair['right']
        fold, energies = pair['fold'], tuple(pair['matched_dual_energies'])
        key = fold, tuple(side['key'] for side in sides), energies
        selected = selections[key]
        all_keys.add(key)
        smiles = [side['smiles'] for side in sides]
        row_ids = [[side['train_row_0based'][f'{ADDUCT}/{energy}eV'] for energy in energies] for side in sides]
        if (selected['positive_train_rows'] != row_ids or selected['smiles'] != smiles
                or selected['tanimoto'] != pair['morgan_tanimoto'] or sides[0]['formula'] != sides[1]['formula']):
            raise ValueError('Frozen selected structures differ from archived GL predictions')
        for side in sides:
            for name, ref in (('identity', side['key']), ('scaffold', side['scaffold_sha256'])):
                if membership.setdefault((name, ref), fold) != fold:
                    raise ValueError(f'{name} occurs across scaffold splits')
        if selected['status'] != 'scored':
            if key in originals or selected['status'] != 'unusable_peak_series':
                raise ValueError('Unexpected archived pair exclusion')
            continue
        precursor = min(scans[row]['precursor_mz'] for query in row_ids for row in query)
        observed = [[clean_peaks(scans[row]['ms2_mzs'], scans[row]['ms2_normalized_intensities'],
                                 scans[row]['precursor_mz']) for row in query] for query in row_ids]
        modeled = [[clean_peaks(*predictions[name, energy], precursor) for energy in energies] for name in smiles]
        if any(len(scan[0]) < 4 for query in observed for scan in query) or any(not len(scan[0]) for q in modeled for scan in q):
            raise ValueError('Cannot reproduce the archived accepted scan support')
        expected = np.asarray(originals[key]['oriented_scores'])[:, 0]
        same_input = pair_evidence(observed, modeled, precursor, energies)
        obtained = margins(same_input)['native_cosine']
        max_baseline_error = max(max_baseline_error, float(np.max(np.abs(expected - obtained))))
        cases.append(dict(key=key, fold=fold, energies=energies, precursor=precursor, observed=observed, modeled=modeled,
                          scaffold={side['scaffold_sha256'] for side in sides}, data=same_input,
                          similarity=selected['tanimoto'], row_ids=row_ids))
    if all_keys != set(selections) or len(cases) != len(originals) or max_baseline_error > 1e-12:
        raise ValueError(f'Historical cosine did not reproduce: max delta {max_baseline_error}')
    return cases, max_baseline_error


def pair_evidence(observed, modeled, precursor, energies):
    return [[[evidence(query[i], candidate[i], precursor, energy) for i, energy in enumerate(energies)]
             for candidate in modeled] for query in observed]


def fit(cases):
    counts = np.zeros((2, 2, CONTEXTS, 3))
    for case in cases:
        weight = 1 / (2 * len(case['energies']))
        for side, candidates in enumerate(case['data']):
            for index, scans in enumerate(candidates):
                label = int(side == index)
                for _, codes in scans:
                    for direction, code in enumerate(codes):
                        histogram = np.bincount(code, minlength=CONTEXTS * 3).reshape(CONTEXTS, 3)
                        counts[label, direction] += (weight / len(code)) * histogram
    background = counts.sum(axis=(0, 2)) + 0.5
    background /= background.sum(axis=1, keepdims=True)
    adjusted = counts + PRIOR_PAIR_WEIGHT * background[None, :, None, :]
    probability = adjusted / adjusted.sum(axis=-1, keepdims=True)
    return np.log(probability[1] / probability[0]).reshape(2, CONTEXTS * 3)


def margins(pair, model=None):
    cosine = np.asarray([[np.mean([v for v, _ in scans]) for scans in row] for row in pair])
    out = {'native_cosine': cosine[:, 0] - cosine[:, 1]}
    if model is None:
        return out
    likelihood = np.asarray([[[np.mean([model[direction][code[direction]].mean() for _, code in scans])
                               for direction in range(2)] for scans in row] for row in pair])
    delta = likelihood[:, 0] - likelihood[:, 1]
    out.update(predicted_peak_llr=delta[:, 0], bidirectional_peak_llr=delta.mean(axis=1))
    return out


def mismatch(cases):
    controls = {}
    for case in cases:
        possible = [other for other in cases if other['fold'] == case['fold']
                    and set(case['energies']) <= set(other['energies'])
                    and not set(case['key'][1]) & set(other['key'][1])
                    and not case['scaffold'] & other['scaffold']]
        if not possible:
            continue
        donor = min(possible, key=lambda item: (abs(item['precursor'] - case['precursor']), item['key']))
        identity = '/'.join((*case['key'][1], *donor['key'][1]))
        flip = sha256(identity.encode()).digest()[0] & 1
        observed = donor['observed'][::-1] if flip else donor['observed']
        order = [donor['energies'].index(energy) for energy in case['energies']]
        aligned = [[side[i] for i in order] for side in observed]
        controls[case['key']] = (pair_evidence(aligned, case['modeled'], case['precursor'], case['energies']), donor['key'],
                                 abs(donor['precursor'] - case['precursor']))
    return controls


def summarize(rows):
    values = np.asarray(rows, dtype=float)
    signed = values * [1, -1]
    assignment = signed.sum(axis=1)
    ties = np.abs(assignment) <= 1e-10
    wins = assignment > 1e-10
    rr = np.where(signed > 1e-10, 1, np.where(signed < -1e-10, 0.5, 0.75))
    return dict(pairs=len(values), wins=int(wins.sum()), ties=int(ties.sum()), losses=int((assignment < -1e-10).sum()),
                assignment_accuracy_with_ties=float((wins.sum() + 0.5 * ties.sum()) / len(values)),
                both_spectra_correct=int((signed > 1e-10).all(axis=1).sum()), mean_mrr_at_two=float(rr.mean()))


def run(train, pairs, receipt, gl, out):
    cases, error = load(train, pairs, receipt, gl)
    controls = mismatch(cases)
    compared = [case for case in cases if case['key'] in controls]
    scores, folds = {}, {}
    for fold in range(5):
        train_cases = [case for case in cases if case['fold'] != fold]
        evaluation = [case for case in compared if case['fold'] == fold]
        if not train_cases or not evaluation:
            raise ValueError('The matched control cannot cover all scaffold folds')
        fitted = fit(train_cases)
        for case in evaluation:
            control, donor, precursor_delta = controls[case['key']]
            measures = margins(case['data'], fitted)
            negative = {f'mismatched_{name}': val for name, val in margins(control, fitted).items()}
            values = measures | negative
            scores[case['key']] = dict(fold=fold, structures=case['key'][1], measured_energies_ev=case['energies'],
                                       positive_train_rows=case['row_ids'], donor_structures=donor[1],
                                       donor_precursor_delta_da=precursor_delta, tanimoto=case['similarity'],
                                       margins={name: values[name].tolist() for name in ARMS})
        folds[str(fold)] = dict(train_pairs=len(train_cases), held_pairs=len(evaluation),
                                model={arm: summarize([scores[case['key']]['margins'][arm]
                                                       for case in evaluation]) for arm in ARMS})
    models = {arm: summarize([row['margins'][arm] for row in scores.values()]) for arm in ARMS}
    close = [row for row in scores.values() if row['tanimoto'] >= 0.8]
    wrong_deltas = [row['donor_precursor_delta_da'] for row in scores.values()]
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'fragment_reliability_per_pair.jsonl'
    path.write_text(''.join(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n' for row in scores.values()))
    result = dict(status='complete', scope='GLACIER measured positive-ion Enveda-180 hard pairs; MRR@2 only',
                  reference_archive_sha256=GL_SUMMARY_SHA, pair_export_sha256=PAIR_EXPORT_SHA,
                  source_sha256=FILES, reference_pairs=len(cases), matched_control_pairs=len(scores),
                  reference_cosine_max_abs_error=error, new_predictions=0, primary='predicted_peak_llr',
                  wrong_scan_control='Closest precursor other pair in the same held fold with no shared identity or generic scaffold; same measured eV; all arms report the covered pairs',
                  wrong_scan_precursor_delta_da=[float(np.median(wrong_deltas)), max(wrong_deltas)],
                  fit='Per-peak outcome log likelihood ratio: paired authentic vs wrong candidate, other scaffold folds only; equal total weight per pair and class',
                  context=['measured collision energy', 'fraction precursor mass', 'own normalized peak intensity'],
                  outcome=['peak absent', 'aligned relative intensity below 2%', 'aligned relative intensity at least 2%'],
                  smoothing_pseudo_pairs=PRIOR_PAIR_WEIGHT, models=models, by_fold=folds,
                  by_energy_count=dict(Counter(len(row['measured_energies_ev']) for row in scores.values())),
                  similarity_at_least_080={arm: summarize([row['margins'][arm] for row in close]) for arm in ARMS} if close else {},
                  per_pair_sha256=sha256(path.read_bytes()).hexdigest())
    (out / 'fragment_reliability_receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(models), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'pairs', 'receipt', 'gl', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(**vars(parser.parse_args()))
