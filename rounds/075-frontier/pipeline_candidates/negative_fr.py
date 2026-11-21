"""Score native [M-H]- spectra against the held-out pipeline's final candidates."""

from collections import Counter, defaultdict
import json
from math import isclose, isfinite
from pathlib import Path
import time

import pandas as pd


INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')
PANEL = 'enveda-075b-external-massbank-adducts'
PIPELINE = 'enveda-075b-massbank-adducts-pipeline'
ARMS = ('original', 'author_ranks_3_to_25', 'same_formula_fr')


def author_order(logits):
    base = list(range(len(logits)))
    proposed = _f3_blend(base, logits, _F3_W)
    positions = [i for i in base[_F3_KEEP:] if isfinite(logits[i])]
    ordered = [i for i in proposed[_F3_KEEP:] if isfinite(logits[i])]
    for position, candidate in zip(positions, ordered, strict=True):
        base[position] = candidate
    return base


def formula_order(formulas, logits):
    groups = defaultdict(list)
    for i, (formula, logit) in enumerate(zip(formulas, logits, strict=True)):
        if pd.notna(formula) and formula and isfinite(logit):
            groups[formula].append(i)
    order = list(range(len(logits)))
    for positions in groups.values():
        for position, candidate in zip(positions, sorted(positions, key=lambda i: -logits[i]), strict=True):
            order[position] = candidate
    return order


def main():
    import torch

    started = time.monotonic()
    torch.set_num_threads(2)
    fields = ['molecule_id', 'ms2_mzs', 'ms2_normalized_intensities', 'adduct', 'ionization_mode', 'precursor_mz']
    query = pd.read_parquet(artifact(PANEL, 'pipeline_queries.parquet'), columns=fields)
    query = query.loc[query['adduct'].eq('[M-H]-') & query['ionization_mode'].eq('negative')]
    if len(query) != 47 or not query['molecule_id'].is_unique:
        raise ValueError('Expected one measured [M-H]- spectrum for each of the fixed 47 molecules')
    spectra = _f3_spectra(query)
    candidates = pd.read_parquet(artifact(PIPELINE, 'candidates.parquet'),
                                 columns=['molecule_id', 'candidate_id', 'smiles', 'formula', 'rank'])
    candidates = candidates.loc[candidates['molecule_id'].isin(spectra)].sort_values('rank', kind='stable')
    if (set(candidates['molecule_id']) != spectra.keys()
            or not candidates['rank'].between(1, 25).all()
            or candidates.duplicated(['molecule_id', 'rank']).any()):
        raise ValueError('Final ranking does not match the fixed negative-ion panel')

    ff, nets, model = _f3_load(_f3_find_assets(INPUT), 'cpu')
    scoring_started = time.monotonic()
    coverage = Counter()
    predictions, score_rows = {}, []
    for mid, entries in candidates.groupby('molecule_id', sort=True):
        keys = entries['candidate_id'].tolist()
        (logits,), npeaks, errors = _f3_score_list(ff, nets, spectra[mid], entries['smiles'].tolist())
        scored = sum(isfinite(x) for x in logits)
        coverage.update(molecules_attempted=1, molecules_scored=int(scored > 0),
                        candidates_attempted=len(keys), candidates_scored=scored, feature_errors=errors,
                        too_few_peaks=int(npeaks < _F3_MIN_PEAKS))
        orders = (list(range(len(keys))), author_order(logits), formula_order(entries['formula'], logits))
        predictions[mid] = {name: [keys[i] for i in order] for name, order in zip(ARMS, orders, strict=True)}
        score_rows.extend(dict(molecule_id=mid, candidate_id=row.candidate_id, original_rank=row.rank,
                               formula=row.formula, fr_logit=logit, merged_peaks=npeaks)
                          for row, logit in zip(entries.itertuples(index=False), logits, strict=True))
    scoring_seconds = time.monotonic() - scoring_started

    labels = pd.read_parquet(artifact(PANEL, 'labels.parquet'), columns=['molecule_id', 'candidate_id'])
    labels = labels.loc[labels['molecule_id'].isin(spectra)].copy()
    if len(labels) != len(spectra) or not labels['molecule_id'].is_unique:
        raise ValueError('Labels do not match the fixed negative-ion panel')
    for method in ARMS:
        labels[method] = [next((i for i, key in enumerate(predictions[mid][method], 1) if key == truth), 0)
                          for mid, truth in labels[['molecule_id', 'candidate_id']].itertuples(index=False, name=None)]
    metrics = summaries(labels, ARMS, 'original')
    original = metrics['original']
    if original['rank1'] != 17 or original['found_at_25'] != 28 or not isclose(
        original['mrr_at_25'], 0.41061589120099756, abs_tol=1e-10
    ):
        raise ValueError(f'Original ranking does not reproduce the approved control: {original}')
    for name in ARMS:
        metrics[name]['mrr_delta'] = metrics[name]['mrr_at_25'] - original['mrr_at_25']
        metrics[name]['lists_changed'] = sum(row[name] != row['original'] for row in predictions.values())
    result = dict(model=model, coverage=dict(coverage), methods=metrics,
                  scoring_seconds=round(scoring_seconds, 2), seconds=round(time.monotonic() - started, 2))
    pd.DataFrame(score_rows).to_parquet(OUTPUT / 'negative_fr_scores.parquet', index=False)
    labels.to_csv(OUTPUT / 'negative_fr_ranks.csv', index=False)
    (OUTPUT / 'negative_fr_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
