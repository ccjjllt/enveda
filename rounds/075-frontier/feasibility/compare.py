"""Score sealed feasibility arms with authorized labels after blind ranking."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd
from rdkit import rdBase

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'four_arm'))

from retrieval_pilot_score import answers, digest, identified_rank, metric_identity, paired_scaffold_interval, reciprocal


ARMS = ('baseline', 'absolute', 'relative', 'matched_real', 'matched_donor',
        'tight_real', 'tight_donor')


def compare(rows, control, intervention):
    if not rows:
        return {'molecules': 0}
    changes = [(row[f'{control}_rank'], row[f'{intervention}_rank']) for row in rows]
    paired = [dict(scaffold=row['scaffold'], delta_delta_rr=reciprocal(after) - reciprocal(before))
              for row, (before, after) in zip(rows, changes)]
    count = len(rows)
    top5 = set(range(1, 6))
    return dict(molecules=count, control_mrr25=sum(reciprocal(before) for before, _ in changes) / count,
                mrr25=sum(reciprocal(after) for _, after in changes) / count,
                delta_mrr25=sum(row['delta_delta_rr'] for row in paired) / count,
                new_top1=[row['molecule_id'] for row, (before, after) in zip(rows, changes)
                          if before != 1 and after == 1],
                lost_top1=[row['molecule_id'] for row, (before, after) in zip(rows, changes)
                           if before == 1 and after != 1],
                new_top5=[row['molecule_id'] for row, (before, after) in zip(rows, changes)
                          if before not in top5 and after in top5],
                lost_top5=[row['molecule_id'] for row, (before, after) in zip(rows, changes)
                           if before in top5 and after not in top5],
                paired_ci=paired_scaffold_interval(paired, 'delta'))


def correlations(path):
    data = pd.read_csv(path)
    if data.empty:
        return {}
    real = data[data.arm == 'absolute']
    results = {'absolute': [], 'relative': []}
    for _, group in real.groupby(['molecule_id', 'anchor_rank'], sort=False):
        native = group.native_coverage.rank()
        for scheme in results:
            rho = native.corr(group[scheme].rank()) if native.nunique() > 1 and group[scheme].nunique() > 1 else None
            if rho is not None and pd.notna(rho):
                results[scheme].append(rho)
    return {scheme: dict(events=len(values), mean_within_event_spearman=sum(values) / len(values))
            for scheme, values in results.items() if values}


def score(args):
    receipt = json.loads((args.rank / 'receipt.json').read_text())
    required = {'audit.json', 'candidate_evidence.csv', *(f'{arm}.csv' for arm in ARMS)}
    if not required <= set(receipt['hashes']) or any(digest(args.rank / name) != receipt['hashes'][name]
                                                    for name in required):
        raise ValueError('Blind outputs differ from the ranking seal')
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Expected the official metric RDKit version')
    audit = json.loads((args.rank / 'audit.json').read_text())
    lists = {arm: answers(args.rank / f'{arm}.csv') for arm in ARMS}
    baseline = lists['baseline']
    if any(set(lists[arm]) != set(baseline) or any(Counter(values) != Counter(baseline[mid])
                                                 for mid, values in lists[arm].items()) for arm in ARMS):
        raise ValueError('A sealed arm changes the candidates')
    labels = pd.read_parquet(args.labels, columns=['molecule_id', 'truth_key', 'truth_smiles', 'source', 'scaffold'])
    if set(labels.molecule_id) != set(baseline) or not labels.molecule_id.is_unique or not labels.truth_key.is_unique:
        raise ValueError('Validation labels differ from the sealed query population')
    rows = []
    for label in labels.sort_values('molecule_id').itertuples(index=False):
        truth = metric_identity(label.truth_smiles)
        if truth is None or truth[0] != label.truth_key:
            raise ValueError('Label identity differs from the official metric')
        mid = label.molecule_id
        ranks = {f'{arm}_rank': identified_rank(lists[arm][mid], truth) for arm in ARMS}
        rows.append(dict(molecule_id=mid, source=label.source, scaffold=label.scaffold, **ranks))

    results = {f'{arm}_versus_baseline': compare(rows, 'baseline', arm) for arm in ('absolute', 'relative')}
    results['relative_versus_absolute'] = compare(rows, 'absolute', 'relative')
    for name in ('matched', 'tight'):
        events = [event for event in audit['events'] if event['arm'] == f'{name}_real' and not event['protected']]
        matched_ids = {event['molecule_id'] for event in events}
        matched = [row for row in rows if row['molecule_id'] in matched_ids]
        results[f'{name}_real_versus_donor'] = compare(matched, f'{name}_donor', f'{name}_real')
        results[f'{name}_real_versus_baseline'] = compare(matched, 'baseline', f'{name}_real')
        results[f'{name}_donor_versus_baseline'] = compare(matched, 'baseline', f'{name}_donor')
    results['source_strata'] = {source: compare([row for row in rows if row['source'] == source], 'absolute', 'relative')
                                for source in sorted({row['source'] for row in rows})}
    results['native_coverage_correlation'] = correlations(args.rank / 'candidate_evidence.csv')
    results['blind_receipt_sha256'] = digest(args.rank / 'receipt.json')
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / 'per_query.csv', index=False)
    (args.out / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name: details['delta_mrr25'] for name, details in results.items()
                      if isinstance(details, dict) and 'delta_mrr25' in details}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rank', 'labels', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    score(parser.parse_args())
