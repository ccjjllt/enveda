"""Check coverage and optional FPNet A ranks of sealed blind virtual product lists."""

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pyarrow.parquet as pq
from rdkit import RDLogger, rdBase

from retrieval_pilot_score import CAP, SOURCE, digest, identified_rank, metric_identity


ORIGINAL_LISTS = ('main', 'engine2', 'pubchem', 'prefusion', 'fused_overflow')
BRANCHES = ('proposals', 'known_main')
FP_ARMS = ('generated', 'native_c34')
FP_CAP = 24
FP_COLUMNS = ('molecule_id', 'arm', 'native_position', 'smiles', 'fpnet_a_fz', 'score_key')


def baseline(path):
    with ZipFile(path) as archive:
        receipt = json.loads(archive.read('pilot_arm_receipt.json'))
        if (receipt['arm'] != 'baseline' or receipt['source'] != SOURCE or
                receipt['molecules'] != 60 or receipt['candidate_slots'] != CAP):
            raise ValueError('Expected the sealed original for the 60-query comparison')
        from hashlib import sha256
        def checked(name):
            contents = archive.read(name)
            if sha256(contents).hexdigest() != receipt['hashes'][name]:
                raise ValueError(f'Baseline artifact changed: {name}')
            return contents
        rows = list(csv.DictReader(io.StringIO(checked('submission.csv').decode())))
        sources = json.loads(checked('four_source_lists.json'))
    ranking = {row['molecule_id']: row['smiles'].split(';') if row['smiles'] else [] for row in rows}
    if len(ranking) != len(rows) or set(ranking) != set(sources):
        raise ValueError('Different final and source lists')
    return receipt, ranking, sources


def candidate_matches(path, target_identities):
    columns = ('molecule_id', 'smiles', 'score_key', 'in_author_window')
    index_hits = {}
    counts = Counter()
    narrow_counts = Counter()
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=16384):
        for row in batch.to_pylist():
            mid = row['molecule_id']
            truth = target_identities[mid]
            counts[mid] += 1
            narrow_counts[mid] += bool(row['in_author_window'])
            if row['score_key'] == truth[0] and metric_identity(row['smiles']) == truth:
                if mid in index_hits:
                    raise ValueError(f'The generator returned a duplicate normalized truth identity: {mid}')
                index_hits[mid] = bool(row['in_author_window'])
    return index_hits, counts, narrow_counts


def score_order(arm, row):
    return (-row['fpnet_a_fz'], row['smiles']) if arm == 'generated' else (row['native_position'],)


def sealed_fpnet(path, query_sha, prediction, proposals, candidate_source='proposals'):
    receipt_file = path / 'fpnet_a_score_receipt.json'
    receipt = json.loads(receipt_file.read_text())
    inputs = receipt['inputs']
    artifact = f'{candidate_source}.parquet'
    if (inputs['query_sha256'] != query_sha
            or inputs['original_c34_archive_sha256'] != prediction['archived_pc_lists_sha256']
            or proposals.name != artifact
            or receipt.get('proposal_artifact', 'proposals.parquet') != artifact
            or receipt['proposals_sha256'] != prediction['output_sha256'][artifact]
            or receipt['cap'] != FP_CAP
            or not inputs['logits'].startswith('A-only under Pablo C34')):
        raise ValueError('FPNet A results do not describe the sealed blind panel and proposal set')
    raw, top = (path / name for name in ('fpnet_a_scores.parquet', 'fpnet_a_ranked25.parquet'))
    for artifact in (raw, top):
        if digest(artifact) != receipt['output_sha256'][artifact.name]:
            raise ValueError(f'FPNet A scoring artifact changed: {artifact.name}')

    actual, missing, best = Counter(), Counter(), defaultdict(list)
    cases = receipt['cases']
    proposal_columns = ['molecule_id', 'smiles', 'score_key']
    proposals = (row for batch in pq.ParquetFile(proposals).iter_batches(
        batch_size=16384, columns=proposal_columns) for row in batch.to_pylist())
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=16384, columns=list(FP_COLUMNS)):
        updated = set()
        for row in batch.to_pylist():
            mid, arm, position = (row[name] for name in ('molecule_id', 'arm', 'native_position'))
            if mid not in cases or arm not in FP_ARMS:
                raise ValueError('Unknown query or arm in the sealed full scores')
            key = (mid, arm)
            actual[key] += 1
            expected_position = actual[key] if arm == 'native_c34' else None
            if position != expected_position or (position is not None and position > CAP):
                raise ValueError(f'Invalid archived C34 direct position or generated position: {mid}')
            if arm == 'generated' and {name: row[name] for name in proposal_columns} != next(proposals, None):
                raise ValueError(f'FPNet A generated row does not match the sealed original proposal: {mid}')
            value = row['fpnet_a_fz']
            if value is None or not row['score_key']:
                missing[key] += 1
            elif not math.isfinite(value) or not row['smiles']:
                raise ValueError(f'Nonfinite score or invalid scored candidate: {mid}')
            else:
                best[key].append(row)
                updated.add(key)
        for mid, arm in updated:
            key = (mid, arm)
            best[key] = sorted(best[key], key=lambda row: score_order(arm, row))[:FP_CAP]
    if next(proposals, None) is not None:
        raise ValueError('FPNet A omitted one or more sealed original proposals')

    ranked = defaultdict(list)
    for row in pq.read_table(top, columns=[*FP_COLUMNS, 'rank']).to_pylist():
        mid, arm = row['molecule_id'], row['arm']
        if mid not in cases or arm not in FP_ARMS:
            raise ValueError('Unknown query or arm in the sealed FPNet A shortlist')
        ranked[mid, arm].append(row)
    budgets, lists = {}, {}
    for mid, case in cases.items():
        for arm, prefix in (('generated', 'generated'), ('native_c34', 'native')):
            key = (mid, arm)
            if actual[key] != case[f'{prefix}_total'] or missing[key] != case[f'{prefix}_invalid']:
                raise ValueError(f'FPNet A scored-list counts do not match the receipt: {mid} {arm}')
        budget = min(FP_CAP, *(actual[mid, arm] - missing[mid, arm] for arm in FP_ARMS))
        if budget != case['candidate_budget']:
            raise ValueError(f'Equal per-query budget does not match the candidate ledger: {mid}')
        budgets[mid] = budget
        lists[mid] = {}
        for arm in FP_ARMS:
            key = (mid, arm)
            expected = [dict(row, rank=rank) for rank, row in enumerate(best[key][:budget], 1)]
            if sorted(ranked[key], key=lambda row: row['rank']) != expected:
                raise ValueError(f'Shortlist ranks or candidates differ from the sealed scores: {mid} {arm}')
            lists[mid][arm] = [row['smiles'] for row in expected]
    return budgets, lists, dict(receipt_sha256=digest(receipt_file),
                               shortlist_sha256=receipt['output_sha256'][top.name],
                               scores_sha256=receipt['output_sha256'][raw.name])


def ranked_shortlist_stats(rows):
    def summarize(arm):
        name = f'fpnet_a_{arm}_rank'
        hits = {str(k): sum(1 <= row[name] <= k for row in rows) for k in (1, 5, FP_CAP)}
        new = {str(k): [row['molecule_id'] for row in rows
                         if row['original_lists_missed'] and 1 <= row[name] <= k]
               for k in (1, 5, FP_CAP)}
        return dict(top_hits=hits, new_all_source_hits={k: len(ids) for k, ids in new.items()},
                    new_all_source_ids=new)

    return dict(molecules=len(rows), molecules_with_paired_slots=sum(bool(row['fpnet_a_budget']) for row in rows),
                slots_per_arm=sum(row['fpnet_a_budget'] for row in rows),
                **{arm: summarize(arm) for arm in FP_ARMS})


def evaluate(args):
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Coverage uses the same RDKit release as the competition')
    RDLogger.DisableLog('rdApp.*')
    receipt, ranking, sources = baseline(args.baseline)
    prediction = json.loads((args.run / 'blind_virtual_np_receipt.json').read_text())
    if prediction['query_count'] != len(ranking) or prediction['archived_c34'] != 'jiyujia/enveda-075-np-recall-base/2':
        raise ValueError('The raw proposals were not generated from the pinned native C34 masses')
    for artifact in ('proposals.parquet', 'known_main.parquet', 'stats.parquet'):
        if digest(args.run / artifact) != prediction['output_sha256'][artifact]:
            raise ValueError(f'Blind generation artifact changed: {artifact}')
    stats = pq.read_table(args.run / 'stats.parquet', columns=['molecule_id']).column(0).to_pylist()
    if len(stats) != len(ranking) or set(stats) != set(ranking):
        raise ValueError('Blind enumeration did not account for all query IDs exactly once')
    fpnet = None
    if args.fpnet_run is not None:
        fpnet = sealed_fpnet(args.fpnet_run, receipt['queries_sha256'], prediction,
                             args.run / f'{args.candidate_source}.parquet', args.candidate_source)
        if set(fpnet[0]) != set(ranking):
            raise ValueError('Sealed FPNet A scores do not account for all panel queries')
    elif args.candidate_source != 'proposals':
        raise ValueError('The known-main comparison requires sealed FPNet A scores')

    if digest(args.labels) != args.labels_sha256:
        raise ValueError('The private validation manifest no longer matches the labels')
    labels = pq.read_table(args.labels, columns=['molecule_id', 'source', 'truth_smiles',
                                                 'truth_key', 'scaffold']).to_pylist()
    if len(labels) != len(ranking) or {row['molecule_id'] for row in labels} != set(ranking):
        raise ValueError('Validation labels do not match the sealed query IDs')
    target_keys = {row['molecule_id']: metric_identity(row['truth_smiles']) for row in labels}
    if any(not target_keys[row['molecule_id']] or target_keys[row['molecule_id']][0] != row['truth_key']
           for row in labels):
        raise ValueError('Competition-normalized truth IDs changed')

    studies = {name: candidate_matches(args.run / f'{name}.parquet', target_keys) for name in BRANCHES}
    records = []
    for row in sorted(labels, key=lambda x: x['molecule_id']):
        mid = row['molecule_id']
        truth = target_keys[mid]
        rank = identified_rank(ranking[mid], truth)
        source_ranks = {name: identified_rank(sources[mid][name], truth) for name in ORIGINAL_LISTS}
        entry = dict(molecule_id=mid, source=row['source'], scaffold=row['scaffold'], baseline_rank=rank,
                     original_lists_missed=not (rank or any(source_ranks.values())),
                     **{f'{name}_rank': position for name, position in source_ranks.items()})
        for name, (hits, counts, narrow) in studies.items():
            entry.update({f'{name}_count': counts[mid], f'{name}_author_count': narrow[mid],
                          f'{name}_correct_identity': mid in hits,
                          f'{name}_correct_in_author_window': hits.get(mid, False)})
        if fpnet is not None:
            budgets, shortlists, _ = fpnet
            if entry[f'{args.candidate_source}_count'] < budgets[mid]:
                raise ValueError(f'FPNet A shortlist exceeds the available generated candidates: {mid}')
            entry['fpnet_a_budget'] = budgets[mid]
            for arm in FP_ARMS:
                entry[f'fpnet_a_{arm}_rank'] = identified_rank(shortlists[mid][arm], truth)
        records.append(entry)

    def stats(rows):
        return dict(molecules=len(rows), baseline_top25=sum(bool(row['baseline_rank']) for row in rows),
                    absent_all_original=sum(row['original_lists_missed'] for row in rows),
                    raw_virtual_hit=sum(row['proposals_correct_identity'] for row in rows),
                    raw_virtual_hit_in_author_window=sum(row['proposals_correct_in_author_window'] for row in rows),
                    known_main_virtual_hit=sum(row['known_main_correct_identity'] for row in rows),
                    known_main_hit_absent_original_lists=sum(
                        row['original_lists_missed'] and row['known_main_correct_identity'] for row in rows),
                    raw_hits_absent_original_lists=sum(
                        row['original_lists_missed'] and (row['proposals_correct_identity'] or
                                                          row['known_main_correct_identity']) for row in rows),
                    entirely_new_virtual_hit=sum(row['original_lists_missed'] and row['proposals_correct_identity']
                                                 for row in rows),
                    entirely_new_virtual_hit_in_author_window=sum(
                        row['original_lists_missed'] and row['proposals_correct_in_author_window'] for row in rows))

    total = stats(records)
    priority = [row['molecule_id'] for row in records
                if row['original_lists_missed'] and row['proposals_correct_identity']]
    known_priority = [row['molecule_id'] for row in records
                      if row['original_lists_missed'] and row['known_main_correct_identity']]
    args.out.mkdir(parents=True, exist_ok=True)
    output = dict(query_sha256=receipt['queries_sha256'],
                  baseline_sha256=receipt['hashes']['submission.csv'],
                  generation_sha256=prediction['generator_sha256'], coverage=total,
                  max_delta_rr_if_all_new_correct_answer_rank1=len(priority) / len(records),
                  delta_rr_if_all_new_correct_answer_rank5=len(priority) / (5 * len(records)),
                  newly_reachable_ids=priority, known_main_reachable_ids=known_priority,
                  sources={source: stats([row for row in records if row['source'] == source])
                           for source in sorted({row['source'] for row in records})})
    if fpnet is not None:
        output['fpnet_a_shortlist'] = dict(
            scope='Candidate-list stage before C41/C43 and final fusion',
            cap=FP_CAP, native_control='Original C34 direct order at the same per-query budget',
            original_sources=['final', *ORIGINAL_LISTS], artifacts=fpnet[2],
            paired_equal_slots=True, total=ranked_shortlist_stats(records),
            sources={source: ranked_shortlist_stats([row for row in records if row['source'] == source])
                     for source in output['sources']})
        if args.candidate_source == 'known_main':
            output['fpnet_a_shortlist'].update(candidate_source='known_main',
                                              candidate_artifact='known_main.parquet',
                                              candidate_artifact_sha256=prediction['output_sha256']['known_main.parquet'])
    pd.DataFrame(records).to_csv(args.out / 'virtual_window_per_query.csv', index=False)
    (args.out / 'virtual_window_summary.json').write_text(json.dumps(output, indent=2) + '\n')
    print('VIRTUAL_WINDOW_COVERAGE', total, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('baseline', 'run', 'labels', 'out'):
        parser.add_argument('--' + arg, required=True, type=Path)
    parser.add_argument('--labels-sha256', required=True)
    parser.add_argument('--fpnet-run', type=Path)
    parser.add_argument('--candidate-source', choices=BRANCHES, default='proposals')
    evaluate(parser.parse_args())
