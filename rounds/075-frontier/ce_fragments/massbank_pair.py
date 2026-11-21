"""Compare observed-energy cosine and differential fragments on fixed MassBank leaders."""

from collections import Counter
import json
from math import isclose
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')
PANEL = 'enveda-075b-external-massbank-202611'
PIPELINE = 'enveda-075b-massbank-pipeline'
ARMS = ('original', 'observed_ce_cosine', 'differential_fragments')


def selected_groups(queries):
    measured = measured_queries(queries, ('[M+H]+',), require_reported_ev=True)
    grouping = measurement_groups(measured)
    if not {'source_library', 'instrument_type'} <= set(grouping):
        raise ValueError('Explicit source and instrument are required for energy grouping')
    for name in grouping:
        values = measured[name].astype('string').str.strip().str.lower()
        measured = measured.loc[values.notna() & ~values.isin(('', 'unknown'))]
    by_query = {}
    for (mid, adduct, raw_instrument, *_), rows in measured.groupby(['molecule_id', *grouping], sort=False):
        if rows['collision_energy'].nunique() >= 2:
            by_query.setdefault(mid, []).append((adduct, raw_instrument, list(rows.itertuples(index=False))))
    return by_query


def score_group(pair, rows, adduct, instrument, predictor):
    query_peaks = {row.query_id: clean_peaks(row.mz, row.intensity, float(row.precursor_mz)) for row in rows}
    energies = {float(row.collision_energy) for row in rows}
    precursor = min(float(row.precursor_mz) for row in rows)
    if not all(len(peaks[0]) for peaks in query_peaks.values()):
        return None
    predicted = {}
    for candidate in pair:
        peaks = {}
        for energy in energies:
            spectrum = predictor.cache.get((candidate.smiles, adduct, instrument, energy))
            if spectrum is None:
                return None
            peaks[energy] = clean_peaks(*spectrum, precursor)
            if not len(peaks[energy][0]):
                return None
        predicted[candidate.candidate_id] = peaks
    delta = differences(rows, query_peaks, predicted)
    if not all(delta.values()):
        return None
    cosine_scores = {cid: [float(cosine(query_peaks[row.query_id], peaks[float(row.collision_energy)]))
                          for row in rows] for cid, peaks in predicted.items()}
    return cosine_scores, delta


def main():
    queries = pd.read_parquet(artifact(PANEL, 'queries.parquet'))
    candidates = pd.read_parquet(artifact(PIPELINE, 'candidates.parquet'),
                                 columns=['molecule_id', 'candidate_id', 'smiles', 'formula', 'rank'])
    evidence = pd.read_parquet(artifact(PIPELINE, 'frozen_candidate_evidence.parquet'),
                              columns=['molecule_id', 'library_max'])
    candidates = candidates.sort_values('rank', kind='stable')
    ranking = {mid: list(rows.itertuples(index=False)) for mid, rows in candidates.groupby('molecule_id')}
    library = evidence.groupby('molecule_id')['library_max']
    if len(queries) != 170 or len(ranking) != 110 or not queries['query_id'].is_unique or (
        set(ranking) != set(queries['molecule_id']) or set(ranking) != set(library.groups)
        or not library.nunique().eq(1).all()
    ):
        raise ValueError('The 110 frozen candidate lists and measured MassBank spectra do not align')
    library = library.first().to_dict()
    groups = selected_groups(queries)
    available = {mid: ranking[mid][:2] for mid in groups if library[mid] < 0.90}
    pairs = {mid: pair for mid, pair in available.items() if len(pair) == 2 and pair[0].rank == 1
             and pair[1].rank == 2 and pair[0].candidate_id != pair[1].candidate_id
             and all(pd.notna(candidate.formula) for candidate in pair) and pair[0].formula == pair[1].formula}
    started = time.monotonic()
    prediction_seconds, scores, coverage = 0.0, {}, Counter(multi_ev=len(groups), eligible=len(pairs))
    jobs = set()
    if pairs:
        ice, predictor, device = load_iceberg(INPUT, 'cpu', 24)
        before = time.monotonic()
        work = {}
        for mid, pair in pairs.items():
            if not all(ice.smiles_ok(candidate.smiles)[0] for candidate in pair):
                continue
            work[mid] = [(adduct, ice.instr_token({'instrument': raw_instrument}), rows)
                         for adduct, raw_instrument, rows in groups[mid]]
            jobs.update((c.smiles, adduct, instrument, float(row.collision_energy))
                        for adduct, instrument, rows in work[mid] for row in rows for c in pair)
        coverage['supported'] = len(work)
        print(f'Fixed panel with {len(groups)} multi-CE, {len(pairs)} eligible, {len(jobs)} model conditions', flush=True)
        complete = predictor.predict(jobs, time.time() + 720)
        prediction_seconds = time.monotonic() - before
        print(f'ICEBERG {device} predicted {predictor.n_pred}/{len(jobs)} conditions in {prediction_seconds:.1f}s '
              f'complete={complete}', flush=True)
        coverage.update(predicted_conditions=predictor.n_pred, prediction_failures=predictor.n_fail,
                        completed_prediction=int(complete))
        for mid, group in work.items():
            pair = pairs[mid]
            results = [score_group(pair, rows, adduct, instrument, predictor) for adduct, instrument, rows in group]
            if any(result is None for result in results):
                continue
            scores[mid] = {arm: [float(np.mean([value for score in results for value in score[arm_index][candidate.candidate_id]]))
                                 for candidate in pair]
                           for arm_index, arm in enumerate(ARMS[1:])}
            coverage['spectra_compared'] += sum(len(rows) for _, _, rows in group)
            coverage['adjacent_energy_pairs'] += sum(len({row.collision_energy for row in rows}) - 1
                                                    for _, _, rows in group)

    labels = pd.read_parquet(artifact(PANEL, 'labels.parquet'), columns=['molecule_id', 'candidate_id'])
    if set(labels['molecule_id']) != set(ranking) or not labels['molecule_id'].is_unique:
        raise ValueError('Labels do not match the fixed 110 questions')
    rows = []
    for mid, truth in labels.itertuples(index=False, name=None):
        entries = ranking[mid]
        original = next((row.rank for row in entries if row.candidate_id == truth), 0)
        row = dict(molecule_id=mid, original=original, has_multi_ev=mid in groups, eligible=mid in pairs,
                   comparable=mid in scores, library_max=library[mid],
                   first=entries[0].candidate_id, second=entries[1].candidate_id if len(entries) > 1 else None,
                   measurement_groups=len(groups.get(mid, ())))
        for arm in ARMS[1:]:
            left, right = scores.get(mid, {}).get(arm, (None, None))
            swap = left is not None and right > left
            row.update({arm: 3 - original if swap and original in (1, 2) else original,
                        arm + '_first': left, arm + '_second': right, arm + '_swap': swap})
        rows.append(row)
    ranks = pd.DataFrame(rows)
    scopes = {'all': ranks, 'eligible': ranks[ranks['eligible']]}
    methods = {name: summaries(frame, ARMS, 'original') if len(frame) else {} for name, frame in scopes.items()}
    old = methods['all']['original']
    if (old['rank1'], old['found_at_25']) != (33, 58) or not isclose(old['mrr_at_25'], 0.3368516738288762):
        raise ValueError(f'Unexpected original MassBank control {old}')
    coverage.update(comparable=len(scores), requested_conditions=len(jobs))
    report = dict(coverage=dict(coverage), prediction_seconds=round(prediction_seconds, 2),
                  seconds=round(time.monotonic() - started, 2),
                  results={name: {'molecules': len(scopes[name]), 'methods': by_method}
                           for name, by_method in methods.items()},
                  swaps={arm: ranks.loc[ranks[arm + '_swap'], ['molecule_id', 'original', arm]].to_dict('records')
                         for arm in ARMS[1:]})
    ranks.to_csv(OUTPUT / 'ce_pair_ranks.csv', index=False)
    (OUTPUT / 'ce_pair_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
