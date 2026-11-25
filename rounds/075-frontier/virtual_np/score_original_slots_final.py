"""Score the final fusion that reuses archived predictor scores."""

import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pyarrow.parquet as pq
from rdkit import RDLogger, rdBase

from evaluate import ORIGINAL_LISTS, baseline
from retrieval_pilot_score import CAP, SOURCE, digest, identified_rank, metric_identity, paired_scaffold_interval, reciprocal, summarize


BASELINE_SHA = '05509b32cfb9bcbe8ab563244f91c97e89f878eba3cc40263e3f48fd4a1f372d'
ARMS = ('original_slots', 'native_only')


def verified_fusion(path, seal_sha, query_sha, arm='original_slots'):
    with ZipFile(path) as source:
        seal_name = 'c41_full_receipt.json'
        raw = source.read(seal_name)
        if sha256(raw).hexdigest() != seal_sha:
            raise ValueError('The scored T4 input differs from its official blind seal')
        seal = json.loads(raw)
        expected = dict(arm=arm, source=SOURCE, status='complete', molecules=60, queries_sha256=query_sha,
                        baseline_archive_sha256=BASELINE_SHA, candidate_slots=CAP, new_predictor_requests=0)
        if any(seal[key] != value for key,value in expected.items()):
            raise ValueError('The fixed-budget fusion did not match the frozen holdout and control')
        required = ('submission.csv','four_source_lists.json','pilot_final_slots.json','c41_routed_c43.json')
        members = source.namelist()
        if not set(required) <= set(seal['hashes']) or set(members) != set(seal['hashes']) | {seal_name} or len(members)!=len(set(members)):
            raise ValueError('The published fusion files do not match the seal')
        kept = {}
        for name,h in seal['hashes'].items():
            value = source.read(name)
            if sha256(value).hexdigest() != h:
                raise ValueError(f'Fusion file changed: {name}')
            if name in required:
                kept[name] = value
    reader = csv.DictReader(io.StringIO(kept['submission.csv'].decode()))
    if reader.fieldnames != ['molecule_id','smiles']:
        raise ValueError('Invalid final submission columns')
    rows = list(reader)
    final = {r['molecule_id']:r['smiles'].split(';') if r['smiles'] else [] for r in rows}
    if len(rows) != len(final):
        raise ValueError('Repeated final molecule rows')
    return seal, final, *(json.loads(kept[key]) for key in required[1:])


def score(args):
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Expected the official scoring RDKit')
    RDLogger.DisableLog('rdApp.*')
    arm = args.arm
    if digest(args.baseline) != BASELINE_SHA or digest(args.labels) != args.label_sha256:
        raise ValueError('The historical ranking or answers changed')
    base_receipt, original, old_sources = baseline(args.baseline)
    if base_receipt['queries_sha256'] != args.query_sha256:
        raise ValueError('The baseline and authorized panel have different query fingerprints')
    receipt, final, sources, slots, gate = verified_fusion(args.fusion, args.seal_sha256, args.query_sha256, arm)
    if (set(final) != set(original) or set(sources) != set(original) or gate != {mid:item['prefusion'] for mid,item in sources.items()}
            or slots['candidate_slots'] != CAP or slots['final_sha256'] != receipt['hashes']['submission.csv']):
        raise ValueError('Original/full lists or C43 provenance disagree')
    vacancies = {mid:CAP-len(candidates) for mid,candidates in original.items() if len(candidates)<CAP}
    if slots['vacant_slots_by_query'] != vacancies:
        raise ValueError('The final slot budget differs from the historical run')
    for mid, predictions in final.items():
        keys = {metric_identity(smiles) for smiles in predictions}
        if None in keys or len(predictions)!=len(keys) or len(predictions)!=len(original[mid]):
            raise ValueError(f'Invalid final candidate list: {mid}')
    labels = pq.read_table(args.labels,columns=['molecule_id','truth_smiles','truth_key','source','scaffold']).to_pylist()
    if len(labels) != 60 or {r['molecule_id'] for r in labels} != set(original):
        raise ValueError('The same held-out 60 molecules must be scored')
    rows = []
    for label in sorted(labels,key=lambda r:r['molecule_id']):
        mid = label['molecule_id']
        truth = metric_identity(label['truth_smiles'])
        if truth is None or truth[0] != label['truth_key']:
            raise ValueError(f'Contest identity mismatch for {mid}')
        prior_rank = identified_rank(original[mid],truth)
        new_rank = identified_rank(final[mid],truth)
        missed = not (prior_rank or any(identified_rank(old_sources[mid][branch],truth) for branch in ORIGINAL_LISTS))
        rows.append(dict(molecule_id=mid, source=label['source'], scaffold=label['scaffold'],
                         baseline_rank=prior_rank, baseline_sources_missed=missed,
                         **{f'{arm}_rank':new_rank,f'{arm}_delta_rr':reciprocal(new_rank)-reciprocal(prior_rank)},
                         old_c43_rank=identified_rank(old_sources[mid]['prefusion'],truth),
                         replayed_c43_rank=identified_rank(gate[mid],truth)))
    names = sorted({r['source'] for r in rows})
    report = dict(scope='Final full-fusion ranking on the sealed 60-query panel',
                  blind_seal_sha256=args.seal_sha256, query_sha256=args.query_sha256, new_predictor_requests=0,
                  original_request_counts={lane:sum(value[lane] for value in base_receipt['predictor_budget'].values())
                                           for lane in next(iter(base_receipt['predictor_budget'].values()))},
                  candidate_slots=CAP, actual_slots=sum(map(len,final.values())), vacancies=vacancies,
                  historical=summarize(rows,'baseline'), grouped_interval=paired_scaffold_interval(rows,arm),
                  by_source={name:summarize([r for r in rows if r['source']==name],arm) for name in names},
                  **{arm:summarize(rows,arm)})
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/f'{arm}_final_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.out/f'{arm}_final_per_query.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=[name for name in rows[0] if name != 'scaffold'],extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report[arm]),flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('fusion','baseline','labels','out'):
        cli.add_argument('--'+name,type=Path,required=True)
    for name in ('seal-sha256','label-sha256','query-sha256'):
        cli.add_argument('--'+name,required=True)
    cli.add_argument('--arm',choices=ARMS,default='original_slots')
    score(cli.parse_args())
