"""Read the two published experiment arms and attribute newly recovered structures."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')
PANEL = 'enveda-075b-external-massbank-adducts'
ORIGINAL = 'enveda-075b-massbank-adducts-pipeline'
CONTROL = 'enveda-075-negative-window-control'
FUSION = 'enveda-075-negative-window-fusion'
WINDOW = 'enveda-075-negative-full-window'


def artifact(slug, name):
    path, = (p for p in INPUT.rglob(name) if slug in p.parts)
    return path


def load(slug, name):
    return json.loads(artifact(slug, name).read_text())


def parq(slug, name):
    return pd.read_parquet(artifact(slug, name))


def population(slug):
    final = parq(slug, 'candidates.parquet')
    candidates = parq(slug, 'source_candidates.parquet')
    union = set(zip(final.molecule_id, final.candidate_id)) | set(zip(candidates.molecule_id, candidates.candidate_id))
    final_rank = (final.sort_values('rank').drop_duplicates(['molecule_id', 'candidate_id'])
                  .set_index(['molecule_id', 'candidate_id'])['rank'].to_dict())
    return final, union, final_rank, candidates


def ranking_metrics(series):
    raw = series.to_numpy(dtype=float)
    rank = np.where(np.isfinite(raw), raw, 0.)
    valid = (rank > 0) & (rank <= 25)
    return dict(queries=len(rank), mrr25=float(np.divide(1., rank, out=np.zeros(len(rank)), where=valid).mean()),
                top1=int((rank == 1).sum()), top5=int((valid & (rank <= 5)).sum()), top25=int(valid.sum()))


def main():
    spectra = parq(PANEL, 'pipeline_queries.parquet')
    panel = spectra.loc[spectra.adduct.eq('[M-H]-') & spectra.ionization_mode.eq('negative')].set_index('molecule_id')
    negative = set(panel.index)
    contributors = panel.spectrum_id.str.split('-').str[1].to_dict()
    labels = parq(PANEL, 'labels.parquet').set_index('molecule_id').loc[sorted(negative)]
    if len(labels) != 47 or not labels.index.is_unique:
        raise ValueError('Expected the complete fixed negative panel.')
    archival, old_union, old_ranks, _ = population(ORIGINAL)
    reference, control_union, control_ranks, _ = population(CONTROL)
    expanded, expanded_union, expanded_ranks, expanded_source = population(FUSION)
    old_miss_count = sum((m, k) not in old_union for m, k in labels.candidate_id.items())
    if old_miss_count != 18:
        raise ValueError('Could not reproduce the archived 18 misses across all sources.')
    ordered = [frame.loc[frame.molecule_id.isin(negative)].sort_values('rank')
               .groupby('molecule_id').candidate_id.agg(tuple).reindex(sorted(negative))
               for frame in (archival, reference, expanded)]
    archive_order, replay_order, expanded_order = ordered
    receipt = load(WINDOW, 'negative_window_receipt.json')
    full = parq(WINDOW, 'negative_window_ranks.parquet')
    proposals = load(WINDOW, 'negative_window_proposals.json')
    if set(proposals) != negative or receipt['queries'] != 47:
        raise ValueError('Missing fixed-window retrieval queries.')
    baseline_pc = load(CONTROL, 'e004_candidate_inputs.json')['pubchem']
    expanded_pc = load(FUSION, 'e004_candidate_inputs.json')['pubchem']
    pc_rank = (expanded_source[expanded_source.source.eq('pubchem')].sort_values('rank')
               .drop_duplicates(['molecule_id', 'candidate_id'])
               .set_index(['molecule_id', 'candidate_id'])['rank'].to_dict())
    for slug in (CONTROL, FUSION):
        audit = load(slug, 'candidate_export.json')
        if audit['query_molecules'] != 48 or audit['query_spectra'] != 48:
            raise ValueError(f'{slug} ran a different fusion batch.')
    full_keys = full.set_index(['molecule_id', 'candidate_id'])['rank'].to_dict()
    rows = []
    for row in labels.itertuples():
        mid, key = row.Index, row.candidate_id
        base, patch = baseline_pc.get(mid, {}), expanded_pc.get(mid, {})
        size = len(base.get('pc', []))
        if size != len(patch.get('pc', [])) or size > 25:
            raise ValueError(f'Candidate budget differs for {mid}')
        applied = patch.get('negative_window') or {}
        inserted = applied.get('inserted', [])
        if len(inserted) > 5:
            raise ValueError(f'Supplemental slate exceeded its budget for {mid}')
        index = (mid, key)
        proposed = [item['key'] for item in proposals[mid]['entries']]
        if not set(inserted).issubset(proposed) or any((mid, k) in old_union for k in inserted):
            raise ValueError(f'An insertion did not come from the frozen outside-union slate for {mid}')
        mass = proposals[mid]['target']
        contributor = contributors[mid]
        family = (contributor == 'BAFG' and row.truth_smiles.count('C#N') >= 2
                  and 'S(=O)(=O)O' in row.truth_smiles)
        rows.append(dict(molecule_id=mid, original_union_has_truth=index in old_union,
                         original_final_rank=old_ranks.get(index), control_union_has_truth=index in control_union,
                         control_final_rank=control_ranks.get(index), full_window_rank=full_keys.get(index),
                         proposal_rank=proposed.index(key) + 1 if key in proposed else None,
                         inserted=key in inserted, pc_rank_after_physics=pc_rank.get(index),
                         expanded_union_has_truth=index in expanded_union, expanded_final_rank=expanded_ranks.get(index),
                         truth_mass_error_ppm=(row.truth_neutral_mass - mass) / mass * 1e6,
                         massbank_contributor=contributor,
                         validation_group='BAFG PAN-Sulfonate' if family else contributor,
                         scaffold_tier=row.scaffold_tier,
                         window_candidates=receipt['by_query'][mid]['normalized_candidates'],
                         pc_slots=size, pc_insertions=len(inserted),
                         pc_insertions_in_final=sum((mid, k) in expanded_ranks for k in inserted)))
    result = pd.DataFrame(rows)
    misses = result.loc[~result.original_union_has_truth]
    result['delta_rr25'] = (1 / result.expanded_final_rank).fillna(0) - (1 / result.control_final_rank).fillna(0)
    changes = result.delta_rr25
    changed = result.loc[changes.ne(0), ['molecule_id', 'control_final_rank', 'expanded_final_rank', 'delta_rr25']]
    new_top5 = result.expanded_final_rank.le(5) & ~result.control_final_rank.le(5)
    lost_top5 = result.control_final_rank.le(5) & ~result.expanded_final_rank.le(5)
    by_group = {name: dict(control=ranking_metrics(group.control_final_rank),
                           expanded=ranking_metrics(group.expanded_final_rank),
                           new_top5=int(new_top5.loc[group.index].sum()),
                           lost_top5=int(lost_top5.loc[group.index].sum()))
                for name, group in result.groupby('validation_group')}
    payload = dict(original=ranking_metrics(result.original_final_rank),
                   control=ranking_metrics(result.control_final_rank), expanded=ranking_metrics(result.expanded_final_rank),
                   delta_mrr25=float(changes.mean()), improved=int(changes.gt(0).sum()), worsened=int(changes.lt(0).sum()),
                   exact_control_reproductions=int((archive_order == replay_order).sum()),
                   changed_final_lists=int((replay_order != expanded_order).sum()),
                   new_final_top5=result.loc[new_top5, 'molecule_id'].tolist(),
                   lost_final_top5=result.loc[lost_top5, 'molecule_id'].tolist(),
                   validation_groups=by_group,
                   old_union_misses=len(misses), old_union_misses_in_window=int(misses.full_window_rank.notna().sum()),
                   old_union_misses_in_proposals=int(misses.proposal_rank.notna().sum()),
                   old_union_misses_inserted=int(misses.inserted.sum()),
                   old_union_misses_in_expanded_union=int(misses.expanded_union_has_truth.sum()),
                   old_union_misses_in_expanded_top25=int(misses.expanded_final_rank.notna().sum()),
                   old_union_misses_in_expanded_top5=int(misses.expanded_final_rank.le(5).sum()),
                   old_union_misses_inserted_and_in_final_top5=int((misses.inserted & misses.expanded_final_rank.le(5)).sum()),
                   newly_hit_vs_both_controls=result.loc[(~result.original_union_has_truth) &
                     (~result.control_union_has_truth) & result.expanded_final_rank.notna(), 'molecule_id'].tolist(),
                   measured_mass_window=receipt['ppm'], total_window_candidates=len(full),
                   total_equal_pc_slots=int(result.pc_slots.sum()), total_replacements=int(result.pc_insertions.sum()),
                   inserted_candidates_in_final=int(result.pc_insertions_in_final.sum()),
                   expanded_queries=int(result.pc_insertions.gt(0).sum()), retrieval_seconds=receipt['seconds'],
                   misses=misses.astype(object).where(misses.notna(), None).to_dict('records'),
                   changes=changed.astype(object).where(changed.notna(), None).to_dict('records'))
    result.to_csv(OUTPUT / 'negative_window_per_query.csv', index=False)
    (OUTPUT / 'negative_window_comparison.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in payload.items() if k != 'misses'}), flush=True)


if __name__ == '__main__':
    main()
