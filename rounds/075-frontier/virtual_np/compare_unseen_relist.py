"""Evaluate an already sealed paired shortlist using private labels."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import RDLogger, rdBase

from evaluate import ORIGINAL_LISTS, baseline
from relist_unseen import ARMS, FP_CAP, RECEIPT, SHORTLIST, original_identities
from retrieval_pilot_score import digest, metric_identity


def ranked_lists(folder, original, receipt):
    path = folder / SHORTLIST
    if digest(path) != receipt['shortlist_sha256']:
        raise ValueError('The blind shortlist differs from its seal')
    lists = defaultdict(list)
    for row in pq.read_table(path).to_pylist():
        mid, arm = row['molecule_id'], row['arm']
        if mid not in original or arm not in ARMS:
            raise ValueError('Unknown query or arm in the shortlist')
        identity = metric_identity(row['smiles'])
        if (identity is None or identity[0] != row['score_key']
                or (identity in original[mid]) != row['originally_seen']
                or (arm == 'unseen' and row['originally_seen'])):
            raise ValueError(f'Unexpected shortlist identity: {mid}')
        lists[mid, arm].append((row['rank'], identity, (-row['fpnet_a_fz'], row['smiles'])))
    for mid, case in receipt['cases'].items():
        budget = case['budget']
        if (budget != min(FP_CAP, case['finite_distinct'], case['finite_unseen_distinct'])
                or case['finite_unseen_distinct'] > case['finite_distinct']
                or case['original_identities'] != len(original[mid])):
            raise ValueError(f'Shortlist budget does not agree with the receipt: {mid}')
        for arm in ARMS:
            rows = sorted(lists[mid, arm])
            if ([rank for rank, _, _ in rows] != list(range(1, budget + 1))
                    or len({identity for _, identity, _ in rows}) != budget
                    or [order for _, _, order in rows] != sorted(order for _, _, order in rows)):
                raise ValueError(f'Invalid rank order in paired shortlist: {mid}')
            lists[mid, arm] = [identity for _, identity, _ in rows]
        plain, unseen = (lists[mid, arm] for arm in ARMS)
        if (sum(identity in original[mid] for identity in plain) != case['plain_seen_slots']
                or len(set(plain) & set(unseen)) != case['same_shortlist_slots']):
            raise ValueError(f'Shortlist overlap counts differ from the blind receipt: {mid}')
    return lists


def summarize(rows):
    thresholds = {}
    for k in (1, 5, FP_CAP):
        plain = {r['molecule_id'] for r in rows if 1 <= r['plain_rank'] <= k}
        unseen = {r['molecule_id'] for r in rows if 1 <= r['unseen_rank'] <= k}
        missed = {r['molecule_id'] for r in rows if not r['truth_in_original_lists']}
        thresholds[str(k)] = dict(plain=len(plain), unseen=len(unseen),
                                  plain_absent_all_original=len(plain & missed),
                                  unseen_absent_all_original=len(unseen & missed),
                                  added_absent_all_original=sorted((unseen - plain) & missed),
                                  lost_absent_all_original=sorted((plain - unseen) & missed))
    return dict(molecules=len(rows), queries_with_slots=sum(row['budget'] > 0 for row in rows),
                slots_per_arm=sum(row['budget'] for row in rows),
                occupied_plain_slots=sum(row['plain_seen_slots'] for row in rows), top_k=thresholds)


def compare(baseline_zip, relist, labels, labels_sha256, out):
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Expected official competition RDKit')
    RDLogger.DisableLog('rdApp.*')
    base_report, final, sources = baseline(baseline_zip)
    original = original_identities(final, sources)
    report_file = relist / RECEIPT
    report = json.loads(report_file.read_text())
    if (report['query_count'] != 60 or set(report['cases']) != set(original) or report['cap'] != FP_CAP
            or report['excluded_sources'] != ['final', *ORIGINAL_LISTS]
            or report['query_sha256'] != base_report['queries_sha256']
            or report['baseline_submission_sha256'] != base_report['hashes']['submission.csv']
            or report['baseline_source_lists_sha256'] != base_report['hashes']['four_source_lists.json']):
        raise ValueError('The blind relist does not match the baseline comparison')
    paired = ranked_lists(relist, original, report)
    if digest(labels) != labels_sha256:
        raise ValueError('The private panel labels differ from their manifest')
    truth = pq.read_table(labels, columns=['molecule_id', 'truth_key', 'truth_smiles', 'source']).to_pylist()
    if len(truth) != 60 or {r['molecule_id'] for r in truth} != set(original):
        raise ValueError('The labels do not cover the exact panel')

    rows = []
    for row in sorted(truth, key=lambda r: r['molecule_id']):
        mid = row['molecule_id']
        identity = metric_identity(row['truth_smiles'])
        if identity is None or identity[0] != row['truth_key']:
            raise ValueError(f'Label does not match the official normalization: {mid}')
        rank = {f'{arm}_rank': next((i for i, candidate in enumerate(paired[mid, arm], 1)
                                    if candidate == identity), 0) for arm in ARMS}
        case = report['cases'][mid]
        rows.append(dict(molecule_id=mid, source=row['source'], truth_in_original_lists=identity in original[mid],
                         budget=case['budget'], plain_seen_slots=case['plain_seen_slots'], **rank))

    summary = dict(scope='FPNet A shortlist before any fusion', generation_scope=report['generation_scope'],
                   blind_receipt_sha256=digest(report_file), cap=FP_CAP, total=summarize(rows),
                   sources={source: summarize([row for row in rows if row['source'] == source])
                            for source in sorted({row['source'] for row in rows})})
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'unseen_relist_per_query.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'unseen_relist_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('UNSEEN_SHORTLIST_COMPARE', summary['total'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-zip', 'relist', 'labels', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--labels-sha256', required=True)
    compare(**vars(parser.parse_args()))
