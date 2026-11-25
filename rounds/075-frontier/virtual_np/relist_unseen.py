"""Blind paired FPNet-A shortlist with and without original-list exclusion."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import RDLogger, rdBase

from evaluate import FP_CAP, FP_COLUMNS, ORIGINAL_LISTS, baseline
from retrieval_pilot_score import digest, metric_identity


ARMS = ('plain', 'unseen')
FULL_SCORES = 'fpnet_a_scores.parquet'
SCORE_RECEIPT = 'fpnet_a_score_receipt.json'
GENERATION_RECEIPT = 'blind_virtual_np_receipt.json'
SHORTLIST = 'unseen_shortlists.parquet'
RECEIPT = 'unseen_relist_receipt.json'
SCHEMA = pa.schema([('molecule_id', pa.string()), ('arm', pa.string()), ('rank', pa.int32()),
                    ('smiles', pa.string()), ('score_key', pa.string()), ('fpnet_a_fz', pa.float64()),
                    ('originally_seen', pa.bool_())])


def original_identities(final, sources):
    result = {}
    for mid, values in final.items():
        seen = {metric_identity(text) for group in [values, *(sources[mid][key] for key in ORIGINAL_LISTS)]
                for text in group}
        result[mid] = seen - {None}
    return result


def read_scores(raw, report, original):
    all_candidates = {mid: {} for mid in original}
    actual, invalid = Counter(), Counter()
    for batch in pq.ParquetFile(raw).iter_batches(columns=list(FP_COLUMNS), batch_size=16384):
        for row in batch.to_pylist():
            mid, arm, value = row['molecule_id'], row['arm'], row['fpnet_a_fz']
            if mid not in original or arm not in ('generated', 'native_c34'):
                raise ValueError('Full score artifact contains an unknown query or arm')
            counter_key = (mid, arm)
            actual[counter_key] += 1
            if value is None or not row['score_key']:
                invalid[counter_key] += 1
                continue
            if not math.isfinite(value):
                raise ValueError('The sealed artifact contains a nonfinite FPNet A score')
            if arm != 'generated':
                continue
            identity = metric_identity(row['smiles'])
            if identity is None or identity[0] != row['score_key']:
                raise ValueError(f'Sealed FPNet identity cannot be reproduced: {mid}')
            candidates = all_candidates[mid]
            order = (-value, row['smiles'])
            if identity not in candidates or order < candidates[identity][0]:
                candidates[identity] = (order, row)
    for mid, case in report['cases'].items():
        for arm, prefix in (('generated', 'generated'), ('native_c34', 'native')):
            key = (mid, arm)
            if actual[key] != case[f'{prefix}_total'] or invalid[key] != case[f'{prefix}_invalid']:
                raise ValueError(f'Full FPNet score accounting differs from the seal: {mid}, {arm}')
    return all_candidates


def relist(baseline_zip, generation, scores, out):
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Expected official competition RDKit')
    RDLogger.DisableLog('rdApp.*')
    baseline_report, final, sources = baseline(baseline_zip)
    original = original_identities(final, sources)
    generation_file = generation / GENERATION_RECEIPT
    score_file = scores / SCORE_RECEIPT
    raw = scores / FULL_SCORES
    origin = json.loads(generation_file.read_text())
    report = json.loads(score_file.read_text())
    source_hash = origin['output_sha256']['known_main.parquet']
    raw_hash = digest(raw)
    if (len(original) != origin['query_count'] or set(report['cases']) != set(original)
            or report['cap'] != FP_CAP or report['proposal_artifact'] != 'known_main.parquet'
            or report['proposals_sha256'] != source_hash or raw_hash != report['output_sha256'][FULL_SCORES]
            or report['inputs']['query_sha256'] != baseline_report['queries_sha256']
            or report['inputs']['original_c34_archive_sha256'] != origin['archived_pc_lists_sha256']
            or not report['inputs']['logits'].startswith('A-only under Pablo C34')):
        raise ValueError('Expected sealed known-main FPNet A scores for these exact baseline queries')
    all_candidates = read_scores(raw, report, original)
    cases, rows = {}, []
    for mid in sorted(original):
        candidates, seen = all_candidates[mid], original[mid]
        order = sorted(candidates, key=lambda identity: candidates[identity][0])
        unseen = [identity for identity in order if identity not in seen]
        budget = min(FP_CAP, len(order), len(unseen))
        plain = order[:budget]
        selected = unseen[:budget]
        cases[mid] = dict(budget=budget, finite_distinct=len(order), finite_unseen_distinct=len(unseen),
                          original_identities=len(seen), plain_seen_slots=sum(i in seen for i in plain),
                          same_shortlist_slots=len(set(plain) & set(selected)))
        for arm, keys in zip(ARMS, (plain, selected), strict=True):
            for rank, identity in enumerate(keys, 1):
                row = candidates[identity][1]
                rows.append(dict(molecule_id=mid, arm=arm, rank=rank, smiles=row['smiles'],
                                 score_key=identity[0], fpnet_a_fz=row['fpnet_a_fz'],
                                 originally_seen=identity in seen))
    out.mkdir(parents=True, exist_ok=True)
    shortlist = out / SHORTLIST
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), shortlist, compression='zstd')
    receipt = dict(query_count=len(original), query_sha256=baseline_report['queries_sha256'],
                   baseline_submission_sha256=baseline_report['hashes']['submission.csv'],
                   baseline_source_lists_sha256=baseline_report['hashes']['four_source_lists.json'],
                   generation_scope=origin.get('generation_scope', 'virtual-rules'),
                   generation_receipt_sha256=digest(generation_file), known_main_sha256=source_hash,
                   fpnet_receipt_sha256=digest(score_file), full_scores_sha256=raw_hash,
                   fpnet_inputs=report['inputs'], shortlist_sha256=digest(shortlist),
                   excluded_sources=['final', *ORIGINAL_LISTS], cap=FP_CAP, cases=cases)
    (out / RECEIPT).write_text(json.dumps(receipt, indent=2) + '\n')
    print('BLIND_UNSEEN_RELIST', len(cases), 'queries', sum(item['budget'] for item in cases.values()),
          'slots per arm', sum(item['plain_seen_slots'] for item in cases.values()), 'plain seen slots',
          flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-zip', 'generation', 'scores', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    relist(**vars(parser.parse_args()))
