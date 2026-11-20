"""Replay a fixed PubChem leader promotion from original ICE and GL scores."""

from collections import Counter
import json
from math import isfinite
from pathlib import Path

import pandas as pd


INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')
PANEL = 'enveda-075b-external-massbank-202611'
PIPELINE = 'enveda-075b-massbank-pipeline'
MEMBERSHIP = 'enveda-075b-external-membership'
FIELDS = ['molecule_id', 'candidate_id', 'smiles', 'formula', 'rank']


def artifact(notebook, filename, optional=False):
    direct = INPUT / 'notebooks' / 'jiyujia' / notebook / filename
    if direct.is_file():
        return direct
    matches = [path for path in INPUT.rglob(Path(filename).name) if notebook in path.parts and path.match(filename)]
    if not matches and optional:
        return None
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one {notebook}/{filename}, found {matches}')
    return matches[0]


def model_outputs(kind):
    directory = f'{kind}_work'
    files = [artifact(PIPELINE, f'{directory}/{kind}_{suffix}', optional=True)
             for suffix in ('in.json', 'out.json', 'out.json.meta.json')]
    if not all(files):
        return {}, {}, {'status': 'missing_inputs_outputs_or_metadata'}
    items, scores, meta = [json.loads(path.read_text()) for path in files]
    if meta['status'] != 'ok':
        return {}, {}, {'status': meta['status']}
    contexts = {item['mid']: item for item in items}
    if len(contexts) != len(items):
        raise ValueError(f'Duplicate molecule input to {kind}')
    return contexts, scores, {'status': 'ok', 'query_contexts': len(contexts), 'queries_with_output': len(scores)}


def compared_scores(samples, scores, mid, first, leader):
    sample = samples.get(mid)
    if not sample or not {first, leader}.issubset(sample['cands']):
        return None, None
    values = scores.get(mid, {})
    return tuple(float(score) if score is not None and isfinite(score) else None
                 for score in (values.get(first), values.get(leader)))


def apply_rule(final, pubchem, models):
    groups = {mid: group.sort_values('rank', kind='stable') for mid, group in final.groupby('molecule_id')}
    if len(groups) != 110 or groups.keys() != set(pubchem.index) or not pubchem.index.is_unique:
        raise ValueError('Expected one PubChem leader for each of the fixed 110 final lists')
    (ice_inputs, ice_scores, _), (gl_inputs, gl_scores, _) = models
    predictions, diagnostics = {}, []
    for mid, group in groups.items():
        keys = group['candidate_id'].tolist()
        if len(keys) > 25 or len(keys) != len(set(keys)) or group['rank'].tolist() != list(range(1, len(keys) + 1)):
            raise ValueError(f'Duplicate identities or invalid final ranks for {mid}')
        first = group.iloc[0]
        leader = pubchem.loc[mid]
        row = dict(molecule_id=mid, original_first=first.candidate_id, pubchem_first=leader.candidate_id,
                   original_first_smiles=first.smiles, pubchem_first_smiles=leader.smiles,
                   original_formula=first.formula, pubchem_formula=leader.formula,
                   pubchem_leader_original_rank=(keys.index(leader.candidate_id) + 1
                                                 if leader.candidate_id in keys else None),
                   ice_original=None, ice_pubchem=None, gl_original=None, gl_pubchem=None,
                   ice_complete=False, gl_complete=False, inputs_match=False,
                   triggered=False, inserted=False)
        new = keys
        if first.candidate_id == leader.candidate_id:
            row['status'] = 'same_structure'
        elif first.formula != leader.formula:
            row['status'] = 'different_formula'
        else:
            ice = compared_scores(ice_inputs, ice_scores, mid, first.smiles, leader.smiles)
            gl = compared_scores(gl_inputs, gl_scores, mid, first.smiles, leader.smiles)
            same_spectra = (mid in ice_inputs and mid in gl_inputs
                            and ice_inputs[mid]['spectra'] == gl_inputs[mid]['spectra'])
            row.update(ice_original=ice[0], ice_pubchem=ice[1], gl_original=gl[0], gl_pubchem=gl[1],
                       ice_complete=None not in ice, gl_complete=None not in gl, inputs_match=same_spectra)
            if None in ice or None in gl:
                row['status'] = 'missing_scores'
            elif not same_spectra:
                row['status'] = 'different_observed_spectra'
            elif ice[1] > ice[0] and gl[1] > gl[0]:
                row.update(status='triggered', triggered=True, inserted=leader.candidate_id not in keys)
                new = ([leader.candidate_id] + [key for key in keys if key != leader.candidate_id])[:25]
            else:
                row['status'] = 'no_strict_dual_win'
        predictions[mid] = keys, new
        diagnostics.append(row)
    return predictions, pd.DataFrame(diagnostics)


def metrics(frame):
    def run(rank):
        hits = rank.between(1, 25)
        return dict(mrr_at_25=float((1 / rank.where(hits)).fillna(0.0).mean()),
                    found_at_25=int(hits.sum()), recall_at_25=float(hits.mean()), rank1=int(rank.eq(1).sum()))

    statuses = Counter(frame['status'])
    eligible = frame.loc[~frame['status'].isin(['same_structure', 'different_formula'])]
    original, changed = frame['original_rank'], frame['promoted_rank']
    return dict(molecules=len(frame), original=run(original), promoted=run(changed),
                same_formula_distinct=len(eligible), ice_pair_scores=int(eligible['ice_complete'].sum()),
                gl_pair_scores=int(eligible['gl_complete'].sum()),
                both_models_pair_scores=int((eligible['ice_complete'] & eligible['gl_complete']).sum()),
                comparable=statuses['triggered'] + statuses['no_strict_dual_win'], statuses=dict(statuses),
                triggers=int(frame['triggered'].sum()), inserted_from_outside_top25=int(frame['inserted'].sum()),
                correct_inserts=int((frame['inserted'] & changed.eq(1)).sum()),
                moved_to_first=int((original.ne(1) & changed.eq(1)).sum()),
                lost_first=int((original.eq(1) & changed.ne(1)).sum()),
                newly_found=int((original.eq(0) & changed.gt(0)).sum()),
                lost_hit=int((original.gt(0) & changed.eq(0)).sum()))


def main():
    final = pd.read_parquet(artifact(PIPELINE, 'candidates.parquet'), columns=FIELDS)
    sources = pd.read_parquet(artifact(PIPELINE, 'source_candidates.parquet'), columns=[*FIELDS, 'source'])
    pubchem = sources.loc[sources['source'].eq('pubchem') & sources['rank'].eq(1)].set_index('molecule_id')
    models = [model_outputs(kind) for kind in ('ice', 'gl')]
    predictions, diagnostics = apply_rule(final, pubchem, models)

    labels = pd.read_parquet(artifact(PANEL, 'labels.parquet'), columns=['molecule_id', 'candidate_id'])
    membership = pd.read_parquet(artifact(MEMBERSHIP, 'structure_membership.parquet'),
                                 columns=['molecule_id', 'candidate_id', 'membership', 'pubchem_tier_observed_10ppm'])
    truth = labels.merge(membership, on=['molecule_id', 'candidate_id'], validate='one_to_one')
    results = diagnostics.merge(truth, on='molecule_id', validate='one_to_one')
    results['pubchem_only'] = results['membership'].eq('pubchem_store_found') & results['pubchem_tier_observed_10ppm']
    if len(results) != 110 or results['pubchem_only'].sum() != 99:
        raise ValueError('Expected the fixed 110 MassBank molecules and 99 PubChem-only molecules')
    for i, name in enumerate(('original_rank', 'promoted_rank')):
        results[name] = [next((rank for rank, key in enumerate(predictions[mid][i], 1) if key == candidate), 0)
                         for mid, candidate in results[['molecule_id', 'candidate_id']].itertuples(index=False, name=None)]
    report = dict(model_outputs={kind: m[2] for kind, m in zip(('ice', 'gl'), models, strict=True)},
                  all=metrics(results), pubchem_only=metrics(results.loc[results['pubchem_only']]))
    baseline = json.loads(artifact(PIPELINE, 'candidate_export.json').read_text())
    if abs(report['all']['original']['mrr_at_25'] - baseline['mrr_at_25']) > 1e-12:
        raise ValueError('Original final ranks did not replay the pipeline audit')
    results.to_parquet(OUTPUT / 'pubchem_dual_support_ranks.parquet', index=False)
    (OUTPUT / 'pubchem_dual_support_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
