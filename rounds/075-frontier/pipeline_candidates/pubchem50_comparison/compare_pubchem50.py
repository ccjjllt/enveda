"""Compare the fixed MassBank panel before and after PubChem depth 50."""

import json
from pathlib import Path

import pandas as pd


INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')
PANEL = 'enveda-075b-external-massbank-202611'
MEMBERSHIP = 'enveda-075b-external-membership'
ORIGINAL = 'enveda-075b-massbank-pipeline'
DEPTH50 = 'enveda-075b-massbank-pubchem50'
FORWARD = 'enveda-075b-massbank-pubchem-forward'
RANK_FIELDS = ['molecule_id', 'candidate_id', 'rank']


def artifact(notebook, filename):
    direct = INPUT / 'notebooks' / 'jiyujia' / notebook / filename
    if direct.is_file():
        return direct
    matches = [path for path in INPUT.rglob(filename) if notebook in path.parts]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one {notebook}/{filename}, found {matches}')
    return matches[0]


def truth_rank(candidates, truth):
    hits = candidates.merge(truth, on=['molecule_id', 'candidate_id'], validate='many_to_one')
    return hits.groupby('molecule_id')['rank'].min().astype('Int64')


def read_run(notebook, run, truth):
    final = pd.read_parquet(artifact(notebook, 'candidates.parquet'), columns=RANK_FIELDS)
    if set(final['molecule_id']) != set(truth['molecule_id']) or not final['rank'].between(1, 25).all():
        raise ValueError(f'Incomplete or invalid final top 25 in {notebook}')
    sources = pd.read_parquet(artifact(notebook, 'source_candidates.parquet'),
                              columns=[*RANK_FIELDS, 'source'])
    frozen = pd.read_parquet(artifact(notebook, 'frozen_candidate_evidence.parquet'),
                             columns=['molecule_id', 'metric_key', 'baseline_rank']).rename(
                                 columns={'metric_key': 'candidate_id', 'baseline_rank': 'rank'})
    if not {'pubchem', 'prefusion'} <= set(sources['source']):
        raise ValueError(f'Missing PubChem or prefusion export in {notebook}')
    frames = {stage: group[RANK_FIELDS] for stage, group in sources.groupby('source')}
    frames.update(final=final, frozen=frozen)
    return pd.DataFrame({f'{run}_{stage}_rank': truth_rank(frame, truth)
                         for stage, frame in frames.items()})


def metrics(series):
    in25 = series.where(series.between(1, 25))
    return dict(mrr_at_25=float((1.0 / in25).fillna(0.0).mean()),
                rank1=int(in25.eq(1).sum()), found_at_25=int(in25.notna().sum()),
                recall_at_25=float(in25.notna().mean()))


def annotate(frame):
    first = frame['original_final_rank']
    new = frame['depth50_final_rank']
    frame['new_final_hit'] = first.isna() & new.notna()
    frame['lost_final_hit'] = first.notna() & new.isna()
    frame['moved_to_first'] = first.ne(1).fillna(True) & new.eq(1).fillna(False)
    frame['lost_first'] = first.eq(1).fillna(False) & new.ne(1).fillna(True)
    frame['new_pubchem_hit'] = frame['original_pubchem_rank'].isna() & frame['depth50_pubchem_rank'].notna()
    frame['fusion_trace'] = pd.Series(pd.NA, index=frame.index, dtype='string')
    pc = frame['new_pubchem_hit']
    prefusion = frame['depth50_prefusion_rank'].notna()
    frozen = frame['depth50_frozen_rank'].notna()
    frame.loc[pc & new.notna(), 'fusion_trace'] = 'present_in_final25'
    frame.loc[pc & new.isna() & frozen, 'fusion_trace'] = 'present_in_frozen25_missing_from_final25'
    frame.loc[pc & new.isna() & ~frozen & prefusion, 'fusion_trace'] = 'present_in_prefusion_missing_from_frozen25'
    frame.loc[pc & new.isna() & ~frozen & ~prefusion, 'fusion_trace'] = 'missing_from_prefusion_and_later_exports'
    return frame


def rows(frame, marked, columns):
    return json.loads(frame.loc[marked, columns].reset_index().to_json(orient='records'))


def summarize(frame):
    ranks = ['original_final_rank', 'depth50_final_rank']
    stages = ['original_pubchem_rank', 'depth50_pubchem_rank', 'depth50_prefusion_rank', 'depth50_frozen_rank',
              *ranks]
    original, depth50 = (metrics(frame[column]) for column in ranks)
    forward = frame['standalone_rescue']
    new_pc = frame['new_pubchem_hit']
    return dict(
        molecules=len(frame),
        original=original,
        depth50=depth50,
        mrr_delta=depth50['mrr_at_25'] - original['mrr_at_25'],
        moved_to_first=int(frame['moved_to_first'].sum()),
        lost_first=int(frame['lost_first'].sum()),
        newly_found=rows(frame, frame['new_final_hit'], ['candidate_id', *stages]),
        lost_hits=rows(frame, frame['lost_final_hit'], ['candidate_id', *stages]),
        pubchem_channel={run: metrics(frame[f'{run}_pubchem_rank']) for run in ('original', 'depth50')},
        pubchem_rescues=rows(frame, new_pc, ['candidate_id', 'fusion_trace', 'standalone_rescue', *stages]),
        pubchem_rescues_by_fusion_trace=frame.loc[new_pc, 'fusion_trace'].value_counts().to_dict(),
        standalone_rescues=rows(frame, forward, ['candidate_id', 'standalone_depth50_rank', 'fusion_trace', *stages]),
    )


def main():
    labels = pd.read_parquet(artifact(PANEL, 'labels.parquet'), columns=['molecule_id', 'candidate_id'])
    cohort = pd.read_parquet(artifact(MEMBERSHIP, 'structure_membership.parquet'),
                             columns=['molecule_id', 'candidate_id', 'membership', 'pubchem_tier_observed_10ppm'])
    truth = labels.merge(cohort, on=['molecule_id', 'candidate_id'], validate='one_to_one')
    truth['pubchem_only'] = truth['membership'].eq('pubchem_store_found') & truth['pubchem_tier_observed_10ppm']
    if len(labels) != 110 or len(truth) != 110 or truth['pubchem_only'].sum() != 99:
        raise ValueError('Expected the fixed 110 MassBank identities and 99 PubChem-only identities')
    forward = pd.read_csv(artifact(FORWARD, 'pc_forward_ranks.csv'),
                          usecols=['molecule_id', 'stored_25', 'depth_50']).set_index('molecule_id')
    output = truth.set_index('molecule_id')
    if not forward.index.is_unique or set(forward.index) != set(output.index):
        raise ValueError('Standalone physical-score comparison used a different panel')
    output = output.join(read_run(ORIGINAL, 'original', labels)).join(read_run(DEPTH50, 'depth50', labels))
    output['standalone_depth50_rank'] = forward['depth_50'].where(forward['depth_50'].gt(0)).astype('Int64')
    output['standalone_rescue'] = forward['stored_25'].eq(0) & forward['depth_50'].gt(0)
    output = annotate(output)
    report = {'all': summarize(output), 'pubchem_only': summarize(output.loc[output['pubchem_only']])}
    output.reset_index().to_parquet(OUTPUT / 'pubchem50_comparison_molecules.parquet', index=False)
    (OUTPUT / 'pubchem50_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    for name, group in report.items():
        print(name, {key: group[key] for key in ('original', 'depth50', 'mrr_delta', 'moved_to_first',
                                                'lost_first', 'pubchem_rescues_by_fusion_trace')}, flush=True)


if __name__ == '__main__':
    main()
