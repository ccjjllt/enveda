"""Score sealed MS-GPT shortlists against the private independent panel."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolDescriptors

from evaluate import ORIGINAL_LISTS, baseline
from retrieval_pilot_score import digest, metric_identity


ARMS = ('top1', 'top5')
CAP = 25
SAMPLES = 100


def ranks(candidates, mid):
    identities = [metric_identity(row['smiles']) for row in candidates]
    encoded = [(row['score_key'], row['heavy_atoms']) for row in candidates]
    if len(identities) > CAP or len(set(identities)) != len(identities):
        raise ValueError(f'Unsealed or duplicate MS-GPT candidates for {mid}')
    for identity, (key, heavy) in zip(identities, encoded, strict=True):
        if identity is None or identity != (key, tuple((e, int(n)) for e, n in
                                                       (item.split(':') for item in heavy.split(',')))):
            raise ValueError(f'Candidate differs from the contest identity for {mid}')
    return {identity: rank for rank, identity in enumerate(identities, 1)}


def summarize(rows):
    return {
        'molecules': len(rows),
        'status': dict(Counter(row['status'] for row in rows)),
        'available_c34_formulas_when_supported': dict(Counter(row['formulas'] for row in rows
                                                              if row['status'] != 'no_supported_spectrum')),
        'original_source_misses': sum(row['source_miss'] for row in rows),
        'equal_slots_per_arm': sum(row['matched_slots'] for row in rows),
        'by_arm': {
            arm: {
                'draws': SAMPLES * sum(row['status'] == 'complete' for row in rows),
                'valid_slots_total': sum(row[f'{arm}_slots'] for row in rows),
                'true_formula_available': sum(row[f'{arm}_has_truth_formula'] for row in rows),
                'coverage': {
                    str(t): {
                        'hits': sum(0 < row[f'{arm}_rank'] <= t for row in rows),
                        'matched_slot_hits': sum(0 < row[f'{arm}_rank'] <= min(t, row['matched_slots']) for row in rows),
                        'original_source_miss_ids': [row['molecule_id'] for row in rows
                                                     if row['source_miss'] and 0 < row[f'{arm}_rank'] <= t],
                        'matched_slot_original_source_miss_ids': [row['molecule_id'] for row in rows
                            if row['source_miss'] and 0 < row[f'{arm}_rank'] <= min(t, row['matched_slots'])],
                    }
                    for t in (1, 5, CAP)
                },
            }
            for arm in ARMS
        },
    }


def compare(baseline_zip, generated, labels, labels_sha256, receipt_sha256, out):
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Expected the official competition RDKit')
    RDLogger.DisableLog('rdApp.*')
    receipt_path = generated / 'ms_gpt_blind_receipt.json'
    if digest(receipt_path) != receipt_sha256:
        raise ValueError('Blind receipt changed in transit')
    receipt = json.loads(receipt_path.read_text())
    original, final, sources = baseline(baseline_zip)
    prediction = generated / 'ms_gpt_blind_predictions.jsonl'
    if (digest(prediction) != receipt['output_sha256'] or receipt['query_sha256'] != original['queries_sha256']
            or (receipt['all_queries'], receipt['candidate_cap'], receipt['per_active_query_draws']) != (60, CAP, SAMPLES)):
        raise ValueError('Blind generated structures differ from their seal or the frozen queries')
    records = [json.loads(line) for line in prediction.read_text().splitlines()]
    by_query = {record['molecule_id']: record for record in records}
    if len(records) != len(by_query) or set(by_query) != set(final):
        raise ValueError('Blind generated structures differ from the sealed population')

    if digest(labels) != labels_sha256:
        raise ValueError('Private labels changed from the panel manifest')
    truth = pq.read_table(labels, columns=['molecule_id', 'truth_key', 'truth_smiles', 'source']).to_pylist()
    if len(truth) != len(final) or {row['molecule_id'] for row in truth} != set(final):
        raise ValueError('Private labels are from a different population')

    rows = []
    for label in sorted(truth, key=lambda row: row['molecule_id']):
        mid = label['molecule_id']
        identity = metric_identity(label['truth_smiles'])
        if identity is None or identity[0] != label['truth_key']:
            raise ValueError(f'Private structure differs from the official metric identity for {mid}')
        source_candidates = [candidate for branch in ORIGINAL_LISTS for candidate in sources[mid][branch]]
        source_miss = all(metric_identity(candidate) != identity for candidate in [*final[mid], *source_candidates])
        record = by_query[mid]
        status = record['status']
        forms = [record.get(f'{arm}_formulas', []) for arm in ARMS]
        totals = [sum(formula['draws'] for formula in arm_forms) for arm_forms in forms]
        if status == 'complete':
            if totals != [SAMPLES, SAMPLES] or len(forms[0]) != 1 or not 1 <= len(forms[1]) <= 5 or forms[0][0]['formula'] != forms[1][0]['formula']:
                raise ValueError(f'Decoding or formula budgets differ from the frozen protocol for {mid}')
        elif status not in ('no_eligible_c34_formula', 'no_supported_spectrum') or any(totals) or any(record[arm] for arm in ARMS):
            raise ValueError(f'Unexpected unscored measurement for {mid}')
        rank = {arm: ranks(record[arm], mid) for arm in ARMS}
        truth_formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(label['truth_smiles']))
        rows.append(dict(molecule_id=mid, source=label['source'], status=status, formulas=len(forms[1]),
                         source_miss=source_miss, matched_slots=min(len(r) for r in rank.values()),
                         **{f'{arm}_rank': rank[arm].get(identity, 0) for arm in ARMS},
                         **{f'{arm}_slots': len(rank[arm]) for arm in ARMS},
                         **{f'{arm}_has_truth_formula': any(entry['formula'] == truth_formula for entry in arm_forms)
                            for arm, arm_forms in zip(ARMS, forms, strict=True)}))

    active = sum(row['status'] == 'complete' for row in rows)
    if active != receipt['active_queries'] or any(receipt['counts'][f'{arm}_draws'] != active * SAMPLES
            or receipt['counts'][f'{arm}_candidates'] != sum(row[f'{arm}_slots'] for row in rows) for arm in ARMS):
        raise ValueError('The paired population or decoding counts differ from the seal')
    report = {'scope': 'Standalone generated shortlist before spectral forward scoring and fusion',
              'query_sha256': receipt['query_sha256'], 'blind_receipt_sha256': receipt_sha256,
              'all_queries': summarize(rows),
              'supported_positive': summarize([row for row in rows if row['status'] != 'no_supported_spectrum']),
              'sources': {source: summarize([row for row in rows if row['source'] == source])
                          for source in sorted({row['source'] for row in rows})}}
    (out / 'ms_gpt_shortlist_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    with (out / 'ms_gpt_shortlist_per_query.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-zip', 'generated', 'labels', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--labels-sha256', required=True)
    parser.add_argument('--receipt-sha256', required=True)
    compare(**vars(parser.parse_args()))
