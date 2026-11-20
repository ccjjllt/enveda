"""Rank eligible PubChem depth-50 candidates with the published FR logits."""

from collections import Counter, defaultdict
from math import isfinite


FORWARD = 'enveda-075b-massbank-pubchem-forward'
MEMBERSHIP = 'enveda-075b-external-membership'
METHODS = ('fr_only', 'fz_fr_equal')


def rank_formula_slots(entries, logits, head_size, combine):
    groups = defaultdict(list)
    for index, candidate in enumerate(entries):
        groups[candidate.formula].append(index)

    ranked = list(range(head_size))
    for indices in groups.values():
        # A missing score leaves incumbents in place and excludes an unscored tail entry.
        scored = [i for i in indices if isfinite(logits[i]) and (not combine or isfinite(entries[i].score))]
        scores = [logits[i] for i in scored]
        if combine:
            scores = _f3_z(scores) + _f3_z([entries[i].score for i in scored])
        order = [i for _, i in sorted(zip(scores, scored), key=lambda pair: -pair[0])]
        head_positions = [i for i in scored if i < head_size]
        for position, candidate in zip(head_positions, order):
            ranked[position] = candidate

    if len(set(ranked)) != head_size or any(entries[i].formula != entries[j].formula for i, j in enumerate(ranked)):
        raise RuntimeError('FR changed the fixed formula slots or duplicated positions')
    return ranked


def score_lists(heads, lists, spectra):
    ff, nets, model = _f3_load(_f3_find_assets(INPUT), 'cpu')
    started = time.monotonic()
    rows = []
    ranks = {name: {} for name in METHODS}
    coverage = Counter()
    total = sum(map(len, lists.values()))
    for n, (mid, entries) in enumerate(lists.items(), 1):
        (logits,), merged_peaks, errors = _f3_score_list(ff, nets, spectra[mid], [c.smiles for c in entries])
        count = sum(isfinite(logit) for logit in logits)
        coverage.update(attempted=len(entries), scored=count, scored_molecules=int(count > 0), feature_errors=errors)
        head_size = len(heads[mid])
        positions = {}
        for name in METHODS:
            order = rank_formula_slots(entries, logits, head_size, name == 'fz_fr_equal')
            ranks[name][mid] = [entries[i].metric_key for i in order]
            positions[name] = {i: j for j, i in enumerate(order, 1)}
        for i, (candidate, logit) in enumerate(zip(entries, logits, strict=True)):
            rows.append(dict(molecule_id=mid, candidate_id=candidate.metric_key, smiles=candidate.smiles,
                             formula=candidate.formula, pc_fz=candidate.score, fr_logit=logit,
                             eligible_rank=i + 1, merged_peaks=merged_peaks,
                             **{name + '_rank': pos.get(i) for name, pos in positions.items()}))
        if n % 10 == 0:
            elapsed = time.monotonic() - started
            projected = elapsed * total / coverage['attempted']
            print(f'FR {n}/{len(lists)} molecules, {coverage["scored"]}/{coverage["attempted"]} scores, '
                  f'{elapsed:.1f}s elapsed, projected {projected:.1f}s', flush=True)
            if (n == 20 and projected > 45 * 60) or elapsed > 43 * 60:
                timing = dict(status='time_budget', projected_seconds=projected, scored_molecules=n,
                              total_molecules=len(lists), elapsed_seconds=elapsed, coverage=dict(coverage))
                (OUTPUT / 'pc_fr_timing.json').write_text(json.dumps(timing, indent=2) + '\n')
                print(json.dumps(timing), flush=True)
                return None
    pd.DataFrame(rows).to_parquet(OUTPUT / 'pc_fr_scores.parquet', index=False)
    return ranks, dict(model=model, **coverage, scoring_seconds=time.monotonic() - started)


def result_ranks(labels, lists):
    return [next((rank for rank, key in enumerate(lists[mid], 1) if key == truth), 0)
            for mid, truth in labels[['molecule_id', 'candidate_id']].itertuples(index=False, name=None)]


def main():
    import torch
    from rdkit import RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    started = time.monotonic()
    global TAUTOMERS
    torch.set_num_threads(2)
    RDLogger.DisableLog('rdApp.*')
    TAUTOMERS = rdMolStandardize.TautomerEnumerator()

    raw = read_json(artifact(PIPELINE, 'pc_lists.json'))
    deep = read_json(artifact(DEPTH, 'pc_depth.json'))
    heads = {mid: unpack(record, 25) for mid, record in raw.items()}
    spectra = _f3_spectra(pd.read_parquet(artifact(PANEL, 'pipeline_queries.parquet')))
    if len(heads) != 110 or heads.keys() != deep.keys() or heads.keys() != spectra.keys():
        raise ValueError('Expected aligned candidates and spectra for the fixed 110-molecule panel')
    if any(len(head) != 25 for head in heads.values()):
        raise ValueError('Expected the original 25 PubChem formula slots for every molecule')
    lists = {mid: head + [candidate for _, candidate in eligible_tail(head, unpack(deep[mid], 50))]
             for mid, head in heads.items()}
    reference = read_json(artifact(FORWARD, 'pc_forward_summary.json'))
    n_tail = sum(map(len, lists.values())) - sum(map(len, heads.values()))
    if n_tail != reference['total_same_formula_tail']['50']:
        raise ValueError('Eligible tail size differs from the ICE+GL depth-50 comparison')
    print(f'Fixed {len(heads)} molecules, {n_tail} eligible tail candidates, '
          f'{sum(map(len, lists.values()))} candidate entries', flush=True)

    scored = score_lists(heads, lists, spectra)
    if scored is None:
        return
    ranks, model_info = scored
    labels = pd.read_parquet(artifact(PANEL, 'labels.parquet'), columns=['molecule_id', 'candidate_id'])
    membership = pd.read_parquet(artifact(MEMBERSHIP, 'structure_membership.parquet'),
                                 columns=['molecule_id', 'candidate_id', 'membership', 'pubchem_tier_observed_10ppm'])
    results = labels.merge(membership, on=['molecule_id', 'candidate_id'], validate='one_to_one')
    results['pubchem_only'] = results['membership'].eq('pubchem_store_found') & results['pubchem_tier_observed_10ppm']
    if len(results) != 110 or set(results['molecule_id']) != heads.keys() or results['pubchem_only'].sum() != 99:
        raise ValueError('Expected 110 molecules and the fixed 99 PubChem-only subgroup')

    results['raw_25'] = result_ranks(results, {mid: [c.metric_key for c in head] for mid, head in heads.items()})
    results['ice_gl_25'] = result_ranks(results, {mid: [describe(smi)[1] for smi in smis]
                                                 for mid, smis in recorded_ranks().items()})
    forward = pd.read_csv(artifact(FORWARD, 'pc_forward_ranks.csv'),
                          usecols=['molecule_id', 'raw_25', 'stored_25', 'depth_50']).set_index('molecule_id')
    if not forward.index.is_unique or set(forward.index) != heads.keys():
        raise ValueError('Original physical-score comparison used a different panel')
    forward = forward.loc[results['molecule_id']]
    if (results['raw_25'].to_list() != forward['raw_25'].to_list()
            or results['ice_gl_25'].to_list() != forward['stored_25'].to_list()):
        raise ValueError('Recomputed raw and ICE+GL top 25 truth ranks differ from the pinned comparison')
    results['ice_gl_50'] = forward['depth_50'].to_list()
    for name, rank_lists in ranks.items():
        results[name] = result_ranks(results, rank_lists)

    arms = ('raw_25', 'ice_gl_25', 'ice_gl_50', *METHODS)
    report = dict(molecules=len(results), eligible_tail=n_tail, **model_info)
    for name, frame in (('all', results), ('pubchem_only', results.loc[results['pubchem_only']])):
        report[name] = dict(molecules=len(frame), relative_to_ice_gl_25=summaries(frame, arms, 'ice_gl_25'),
                            relative_to_ice_gl_50=summaries(frame, METHODS, 'ice_gl_50'),
                            relative_to_raw_25=summaries(frame, METHODS, 'raw_25'))
    report['seconds'] = round(time.monotonic() - started, 1)
    results.to_parquet(OUTPUT / 'pc_fr_ranks.parquet', index=False)
    (OUTPUT / 'pc_fr_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
