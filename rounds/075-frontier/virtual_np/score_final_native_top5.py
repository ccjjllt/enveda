"""Score a sealed final-list permutation on the already authorized private panel."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import RDLogger, rdBase

from evaluate import ORIGINAL_LISTS, baseline
from retrieval_pilot_score import CAP, identified_rank, metric_identity, paired_scaffold_interval, reciprocal, summarize


BASELINE_SHA = '05509b32cfb9bcbe8ab563244f91c97e89f878eba3cc40263e3f48fd4a1f372d'
QUERY_SHA = '2a8bf747f50e6920fcd4f24e04302b721281f8794c947cc58fe545264b12d0d4'
ARMS = ('historical', 'protected', 'open')
FILES = (*[name+'_final.json' for name in ARMS], 'final_native_top5_cases.json')
SEAL = 'final_native_top5_receipt.json'


def validated_seal(raw, query_sha):
    seal = json.loads(raw)
    if (query_sha != QUERY_SHA or seal['status'] != 'sealed' or seal['arm_names'] != list(ARMS)
            or seal['molecules'] != 60 or seal['queries_sha256'] != query_sha
            or seal['baseline_archive_sha256'] != BASELINE_SHA
            or seal['candidate_budget'] != CAP or seal['actual_slots'] != 1476
            or seal['new_predictor_requests'] != 0 or seal['source_library_threshold'] != .9
            or set(seal['files']) != set(FILES)):
        raise ValueError('Expected the sealed 60-query no-new-prediction top-five comparison')
    return seal


def ranked_lists(blind, seal, historical):
    rankings = {}
    for name in FILES:
        body = (blind/name).read_bytes()
        if sha256(body).hexdigest() != seal['files'][name]:
            raise ValueError(f'Blind output drifted: {name}')
        if name.endswith('_final.json'):
            rankings[name.removesuffix('_final.json')] = json.loads(body)
    if rankings['historical'] != historical:
        raise ValueError('The original final lists do not match the frozen baseline')
    lengths = dict(Counter(str(len(candidates)) for candidates in historical.values()))
    if seal['original_lengths'] != lengths:
        raise ValueError('The sealed original candidate-count histogram differs')
    for arm in ARMS[1:]:
        updated = rankings[arm]
        if set(updated) != set(historical):
            raise ValueError(f'Missing query in {arm}')
        changed, changed_first = [], []
        for mid, before in historical.items():
            after = updated[mid]
            if (len(before) != len(after) or before[5:] != after[5:]
                    or sorted(before[:5]) != sorted(after[:5])
                    or (arm == 'protected' and before[:1] != after[:1])):
                raise ValueError(f'The original candidate budget or protected ranks changed in {arm}/{mid}')
            if before != after:
                changed.append(mid)
            if before[:1] != after[:1]:
                changed_first.append(mid)
        receipt = seal['arms'][arm]
        if (sorted(changed) != sorted(receipt['changed']) or sorted(changed_first) != sorted(receipt['first_changed'])
                or receipt['top5_members_changed'] or receipt['final_members_changed']):
            raise ValueError(f'The ranking change ledger differs from {arm}')
    return rankings


def score(args):
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Expected the official scoring RDKit')
    RDLogger.DisableLog('rdApp.*')
    raw = (args.blind/SEAL).read_bytes()
    if (sha256(raw).hexdigest() != args.seal_sha256
            or sha256(args.baseline.read_bytes()).hexdigest() != BASELINE_SHA):
        raise ValueError('The blind or original sealed files have changed')
    seal = validated_seal(raw, args.query_sha256)
    base_receipt, historical, sources_by_query = baseline(args.baseline)
    if base_receipt['queries_sha256'] != QUERY_SHA:
        raise ValueError('Unexpected original frozen panel digest')
    rankings = ranked_lists(args.blind, seal, historical)
    if sha256(args.labels.read_bytes()).hexdigest() != args.label_sha256:
        raise ValueError('The 60 authorized answers have changed')
    labels = pq.read_table(args.labels,columns=['molecule_id','truth_smiles','truth_key','source','scaffold']).to_pylist()
    if len(labels)!=60 or {row['molecule_id'] for row in labels}!=set(historical):
        raise ValueError('Expected exactly the previously authorized evaluation labels')
    rows = []
    for item in sorted(labels, key=lambda row:row['molecule_id']):
        mid = item['molecule_id']
        identity = metric_identity(item['truth_smiles'])
        if identity is None or identity[0]!=item['truth_key']:
            raise ValueError('The official identity normalization differs from the panel')
        ranks = {arm:identified_rank(lists[mid],identity) for arm,lists in rankings.items()}
        before = ranks.pop('historical')
        if any((0 < rank <= 5) != (0 < before <= 5) or bool(rank) != bool(before) for rank in ranks.values()):
            raise ValueError(f'Candidate-set drift on {mid}')
        missed = not (before or any(identified_rank(sources_by_query[mid][branch],identity) for branch in ORIGINAL_LISTS))
        rows.append(dict(molecule_id=mid, source=item['source'], scaffold=item['scaffold'], baseline_rank=before,
                         baseline_sources_missed=missed,
                         **{f'{arm}_rank':rank for arm,rank in ranks.items()},
                         **{f'{arm}_delta_rr':reciprocal(rank)-reciprocal(before) for arm,rank in ranks.items()}))
    sources = sorted({row['source'] for row in rows})
    summary = dict(scope='Permutation of the original final top five on the frozen 60-query development panel',
                   query_sha256=QUERY_SHA, seal_sha256=args.seal_sha256, new_predictor_requests=0, actual_slots=seal['actual_slots'],
                   models={arm:summarize(rows,arm) for arm in ('baseline',*ARMS[1:])},
                   interval={arm:paired_scaffold_interval(rows,arm) for arm in ARMS[1:]},
                   by_source={source:{arm:summarize([r for r in rows if r['source']==source],arm)
                                      for arm in ('baseline',*ARMS[1:])} for source in sources})
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'final_native_top5_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (args.out/'final_native_top5_per_query.csv').open('w',newline='') as stream:
        fields = [name for name in rows[0] if name not in ('scaffold','baseline_sources_missed')]
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary['models']),flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for option in ('blind','baseline','labels','out'):
        cli.add_argument('--'+option, type=Path, required=True)
    for option in ('query-sha256','label-sha256','seal-sha256'):
        cli.add_argument('--'+option, required=True)
    score(cli.parse_args())
