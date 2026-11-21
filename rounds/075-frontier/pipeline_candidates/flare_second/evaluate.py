"""Evaluate a fixed FLARE swap of the first two same-formula candidates."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')
PIPELINE = 'enveda-075b-massbank-pipeline'
PANEL = 'enveda-075b-external-massbank-202611'
FLARE = 'enveda-075b-massbank-flare-mist'
CROSS_FORMULA = 'enveda-075b-massbank-pubchem-cross-formula'
SAME_FORMULA = 'enveda-075b-massbank-pubchem-dual-support'


def artifact(notebook, filename):
    direct = INPUT / 'notebooks' / 'jiyujia' / notebook / filename
    if direct.is_file():
        return direct
    paths = [path for path in INPUT.rglob(filename) if notebook in path.parts]
    if len(paths) != 1:
        raise FileNotFoundError(f'Expected one {notebook}/{filename}, found {paths}')
    return paths[0]


def apply_rule(candidates, scores, queries, library):
    columns = ['molecule_id', 'candidate_id', 'formula']
    first = candidates.loc[candidates['rank'].eq(1), columns]
    second = candidates.loc[candidates['rank'].eq(2), columns]
    pairs = first.merge(second, on='molecule_id', suffixes=('_first', '_second'), validate='one_to_one')
    pairs['library_max'] = pairs['molecule_id'].map(library)
    pairs['same_formula'] = (pairs['formula_first'].notna() & pairs['formula_first'].eq(pairs['formula_second'])
                             & pairs['candidate_id_first'].ne(pairs['candidate_id_second']))
    pairs['weak_library'] = np.isfinite(pairs['library_max']) & pairs['library_max'].lt(0.90)
    scores = scores.loc[scores['model'].eq('flare') & np.isfinite(scores['score'])]
    scores = scores.merge(queries, on=['molecule_id', 'query_id'], validate='many_to_one')
    paired = []
    for position in ('first', 'second'):
        candidate = pairs[['molecule_id', f'candidate_id_{position}']].rename(
            columns={f'candidate_id_{position}': 'candidate_id'})
        match = candidate.merge(scores, on=['molecule_id', 'candidate_id'], validate='one_to_many')
        paired.append(match[['molecule_id', 'query_id', 'score']].rename(columns={'score': 'flare_' + position}))
    common = paired[0].merge(paired[1], on=['molecule_id', 'query_id'], validate='one_to_one')
    aggregate = common.groupby('molecule_id').agg(paired_spectra=('query_id', 'size'),
                                                   paired_query_ids=('query_id', list),
                                                   flare_first=('flare_first', 'mean'),
                                                   flare_second=('flare_second', 'mean'))
    pairs = pairs.join(aggregate, on='molecule_id')
    pairs['paired_spectra'] = pairs['paired_spectra'].fillna(0).astype(int)
    pairs['eligible'] = pairs['same_formula'] & pairs['weak_library'] & pairs['paired_spectra'].gt(0)
    pairs['triggered'] = pairs['eligible'] & pairs['flare_second'].gt(pairs['flare_first'])
    return pairs


def metrics(result):
    before, after = result['rank_before'], result['rank_after']
    return dict(molecules=len(result), same_formula=int(result['same_formula'].sum()),
                same_formula_weak_library=int((result['same_formula'] & result['weak_library']).sum()),
                eligible=int(result['eligible'].sum()), triggered=int(result['triggered'].sum()),
                multi_spectrum_eligible=int((result['eligible'] & result['paired_spectra'].gt(1)).sum()),
                baseline_rank1=int(before.eq(1).sum()), updated_rank1=int(after.eq(1).sum()),
                moved_to_first=int((before.eq(2) & after.eq(1)).sum()),
                lost_first=int((before.eq(1) & after.eq(2)).sum()),
                recall_count=int(before.gt(0).sum()), mrr_before=float(result['rr_before'].mean()),
                mrr_after=float(result['rr_after'].mean()), delta_mrr=float(result['delta_rr'].mean()))


def existing_overlap(result):
    cross = json.loads(artifact(CROSS_FORMULA, 'pubchem_cross_formula_summary.json').read_text())
    triggers = cross['all']['triggered_cases']
    cross_cases = pd.DataFrame(triggers)
    within = pd.read_parquet(artifact(SAME_FORMULA, 'pubchem_dual_support_ranks.parquet'),
                             columns=['molecule_id', 'pubchem_first', 'triggered', 'original_rank', 'promoted_rank'])
    within = within.loc[within['triggered']].rename(columns={'pubchem_first': 'challenger',
                                                            'promoted_rank': 'new_rank'})
    flare = result.loc[result['triggered']]
    report = {}
    for name, control in (('dual_cross_formula', cross_cases), ('dual_same_formula', within)):
        baseline = result.set_index('molecule_id')['rank_before']
        if not baseline.reindex(control['molecule_id']).eq(control['original_rank'].to_numpy()).all():
            raise ValueError(f'Frozen baseline differs from {name}')
        common = flare.merge(control, on='molecule_id', validate='one_to_one')
        report[name] = dict(control_triggers=len(control), overlap_triggers=len(common),
                            shared_molecule_ids=common['molecule_id'].tolist(),
                            same_challenger=int(common['candidate_id_second'].eq(common['challenger']).sum()),
                            overlap_gains=int((common['delta_rr'].gt(0) & common['new_rank'].eq(1)).sum()))
    return report


def main():
    candidates = pd.read_parquet(artifact(PIPELINE, 'candidates.parquet'),
                                 columns=['molecule_id', 'candidate_id', 'formula', 'rank'])
    library = pd.read_parquet(artifact(PIPELINE, 'frozen_candidate_evidence.parquet'),
                              columns=['molecule_id', 'library_max']).groupby('molecule_id')['library_max']
    if library.nunique().gt(1).any():
        raise ValueError('Inconsistent Engine 1 library values')
    queries = pd.read_parquet(artifact(PANEL, 'queries.parquet'), columns=['molecule_id', 'query_id'])
    scores = pd.read_parquet(artifact(FLARE, 'scores.parquet'))
    pairs = apply_rule(candidates, scores, queries, library.first())

    labels = pd.read_parquet(artifact(PANEL, 'labels.parquet'),
                             columns=['molecule_id', 'candidate_id', 'scaffold_tier', 'msg_train_val_match'])
    truth = labels.merge(candidates, on=['molecule_id', 'candidate_id'], validate='one_to_one', how='left')
    truth['rank_before'] = truth['rank'].fillna(0).astype(int)
    result = truth.drop(columns=['rank', 'formula']).merge(pairs, on='molecule_id', validate='one_to_one')
    if len(result) != len(truth) or len(result) != 110:
        raise ValueError('Expected the complete 110-molecule external MassBank panel')
    before = result['rank_before']
    swap = result['triggered'] & before.isin([1, 2])
    result['rank_after'] = before.where(~swap, 3 - before)
    for column in ('before', 'after'):
        rank = result['rank_' + column]
        result['rr_' + column] = (1 / rank.where(rank.gt(0))).fillna(0.0)
    result['delta_rr'] = result['rr_after'] - result['rr_before']

    report = dict(all=metrics(result),
                  enveda_generic_scaffold_absent=metrics(result[result['scaffold_tier'].eq('generic_absent_in_enveda')]),
                  msg_train_val_identity_absent=metrics(result[~result['msg_train_val_match']]),
                  paired_dual_model_controls=existing_overlap(result))
    truth_changes = result.loc[result['delta_rr'].ne(0), ['molecule_id', 'rank_before', 'rank_after', 'delta_rr']]
    report['truth_rank_changes'] = truth_changes.to_dict(orient='records')
    baseline = json.loads(artifact(PIPELINE, 'candidate_export.json').read_text())
    if abs(report['all']['mrr_before'] - baseline['mrr_at_25']) > 1e-12:
        raise ValueError('Final candidate ranks do not reproduce the baseline MRR')
    result.to_parquet(OUTPUT / 'flare_second_ranks.parquet', index=False)
    (OUTPUT / 'flare_second_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
