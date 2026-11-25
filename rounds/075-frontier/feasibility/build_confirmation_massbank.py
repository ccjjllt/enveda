"""Confirm the frozen absolute rule on the 068a MassBank 2026.11 archive and its non-Eawag stratum."""
import argparse
from collections import Counter
import json
from pathlib import Path


SLUG = 'enveda-075-absolute-confirm-massbank'
PREDICTION = 'enveda-075b-massbank-pipeline'
BLIND = 'enveda-075b-external-massbank-202611'
SCORED = 'enveda-075b-massbank-ce-pair'
SOURCE_SIZES = {'BAFG': 2, 'Eawag': 87, 'MSSJ': 1, 'MUI': 20}
PINS = {
    (PREDICTION, 'candidates.parquet'): 'a9a63eb9a5c8f55d8965f499b447842ee679ef4d71ef0a0c00677eb97d6c95d7',
    (PREDICTION, 'submission.csv'): 'a2277d569ccf838c11cc2b8fd36cf6bcb1b6565ffbe3c4312017480eaac7c5a1',
    (PREDICTION, 'frozen_candidate_evidence.parquet'): '0dcc671e581b5632d971c4422d8a748533dc688eab2d53ef5a41b923589d29f8',
    (PREDICTION, 'e004_candidate_inputs.json'): '124c25cb20d039baf975ac7d62f1205838ec51e1ca878f9b5c58f678e17f04a4',
    (BLIND, 'pipeline_queries.parquet'): '74929665fcceed3eb50e38f1bba500f7d208e21f7874c397b94df9071b6e75d5',
    (SCORED, 'ce_pair_ranks.csv'): 'c9bae469ccc41467c693ab917334052a9a3ec11f259e75eacb45c1488c269475',
}


def path(inputs, slug, name):
    from e180_confirmation import mounted
    return mounted(inputs, slug, name, PINS)


def rank_blind(inputs, out):
    import pandas as pd
    from rdkit import rdBase
    from e180_confirmation import source_heads
    from rank import answers, digest, protects_first, rank_absolute, read_scans

    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Expected the competition metric RDKit')
    base_path = path(inputs, PREDICTION, 'submission.csv')
    base = answers(base_path)
    candidates = pd.read_parquet(path(inputs, PREDICTION, 'candidates.parquet'),
                                 columns=['molecule_id', 'rank', 'smiles', 'candidate_id'])
    if (len(base) != 110 or len(candidates) != 2750 or set(candidates.molecule_id) != set(base)
            or candidates.duplicated(['molecule_id','candidate_id']).any()):
        raise ValueError('Incomplete or duplicated archived candidate population')
    for mid, group in candidates.sort_values('rank').groupby('molecule_id', sort=False):
        if (group['rank'].tolist() != list(range(1, 26)) or group.smiles.tolist() != base[mid]
                or len(set(base[mid])) != 25):
            raise ValueError(f'The archived 068a submission and scored candidate list differ for {mid}')
    queries = path(inputs, BLIND, 'pipeline_queries.parquet')
    deposited = pd.read_parquet(queries, columns=['molecule_id', 'spectrum_id', 'adduct'])
    if (len(deposited) != 170 or not deposited.spectrum_id.is_unique
            or set(deposited.molecule_id) != set(base) or set(deposited.adduct) != {'[M+H]+'}):
        raise ValueError('The measured acquisition archive differs from the archived evaluation')
    institutions = deposited.spectrum_id.str.extract(r'^MSBNK-([^-]+)-', expand=False)
    groups = deposited.assign(source=institutions)[['molecule_id','source']].drop_duplicates()
    if (groups.source.isna().any() or groups.molecule_id.duplicated().any()
            or groups.source.value_counts().to_dict() != SOURCE_SIZES):
        raise ValueError('The acquisition sources differ from the prespecified split')
    heads = source_heads(inputs, set(base), PREDICTION, PINS)
    if any(not row[src] for row in heads.values() for src in ('main','engine2','pubchem')):
        raise ValueError('Missing a source head from the original 068a three-channel guard')
    scans = read_scans(queries)
    ranked, swaps = rank_absolute(base, heads, scans)
    if set(base) != set(ranked) or any(Counter(base[mid]) != Counter(ranked[mid]) for mid in base):
        raise ValueError('Query or candidate inventory changed')
    absolute = out / 'absolute.csv'
    pd.DataFrame([(mid, ';'.join(ranked[mid])) for mid in base], columns=['molecule_id','smiles']).to_csv(absolute, index=False)
    eligible = {s.mid for s in scans.values()}
    protected = {mid for mid in base if protects_first(base[mid][0], heads[mid])}
    seal = dict(panel='MassBank 2026.11', baseline='068a N-1c LAM=0.35', candidate_budget=25,
                molecules=len(base), candidate_slots=len(candidates), measured_spectra=len(deposited),
                eligible_spectra=len(scans), eligible_molecules=len(eligible), protected_leaders=len(protected),
                eligible_with_unprotected_leader=len(eligible-protected), swaps=swaps,
                swaps_by_anchor=dict(Counter(s['anchor_rank'] for s in swaps)),
                baseline_sha256=digest(base_path), query_sha256=digest(queries), absolute_sha256=digest(absolute),
                candidate_ids_sha256=PINS[PREDICTION,'candidates.parquet'],
                source_inputs_sha256=PINS[PREDICTION,'e004_candidate_inputs.json'],
                source_ranks_sha256=PINS[PREDICTION,'frozen_candidate_evidence.parquet'],
                source_by_query=groups.set_index('molecule_id').source.to_dict(),
                frozen_rule_sha256=digest(Path(__file__).with_name('rank.py')))
    (out / 'ranking_receipt.json').write_text(json.dumps(seal,indent=2))
    print('Sealed', seal['absolute_sha256'], 'swaps', seal['swaps_by_anchor'], flush=True)


def metrics(rows):
    from retrieval_pilot_score import reciprocal
    def counts(field):
        values = [r[field] for r in rows]
        return dict(top1=values.count(1), top5=sum(0 < v <= 5 for v in values), top25=sum(v > 0 for v in values),
                    mrr25=sum(map(reciprocal, values))/len(values))
    original, absolute = counts('baseline_rank'), counts('absolute_rank')
    result = dict(molecules=len(rows), baseline=original, absolute=absolute, delta_mrr25=absolute['mrr25']-original['mrr25'])
    for cutoff in (1,5):
        changes = [(r['molecule_id'], 0 < r['baseline_rank'] <= cutoff, 0 < r['absolute_rank'] <= cutoff) for r in rows]
        result[f'new_top{cutoff}'] = [mid for mid, before, after in changes if after and not before]
        result[f'lost_top{cutoff}'] = [mid for mid, before, after in changes if before and not after]
    return result


def score_sealed(inputs, out):
    import pandas as pd
    from rank import answers, digest
    from retrieval_pilot_score import reciprocal

    seal = json.loads((out / 'ranking_receipt.json').read_text())
    if digest(out / 'absolute.csv') != seal['absolute_sha256']:
        raise ValueError('Sealed prediction changed')
    ranked = answers(out / 'absolute.csv')
    base = answers(path(inputs, PREDICTION, 'submission.csv'))
    old = pd.read_csv(path(inputs, SCORED, 'ce_pair_ranks.csv'), usecols=['molecule_id','original'],
                      dtype={'molecule_id':str,'original':int})
    if (not old.molecule_id.is_unique or set(old.molecule_id) != set(base) or not old.original.between(0,25).all()):
        raise ValueError('Archived score is missing, duplicated or out of range')
    rows = []
    for mid,before in old.itertuples(index=False):
        after = ranked[mid].index(base[mid][before-1])+1 if before else 0
        rows.append(dict(molecule_id=mid,source=seal['source_by_query'][mid],baseline_rank=before,absolute_rank=after,
                         delta_rr=reciprocal(after)-reciprocal(before)))
    result = metrics(rows)
    original = result['baseline']
    if ([original[k] for k in ('top1','top5','top25')] != [33,41,58]
            or abs(original['mrr25'] - 0.3368516738288762) > 1e-12):
        raise ValueError('Historical 068a baseline failed to reproduce')
    result.update(non_eawag=metrics([r for r in rows if r['source'] != 'Eawag']),
                  by_source={s:metrics([r for r in rows if r['source']==s]) for s in sorted(SOURCE_SIZES)},
                  scored_rank_sha256=PINS[SCORED,'ce_pair_ranks.csv'], ranking_sha256=seal['absolute_sha256'])
    pd.DataFrame(rows).to_csv(out / 'per_query.csv', index=False)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    from build_confirmation_e180 import build
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.out, study=Path(__file__), slug=SLUG,
          kernel_sources=[f'jiyujia/{s}/1' for s in (PREDICTION, BLIND, SCORED)])
