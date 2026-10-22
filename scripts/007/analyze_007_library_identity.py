"""Manual read-only audit of R1 raw-key/canonical library lookup.

Reads identity tables, existing feature JSON and train's inchikey14 column only.
No spectra/peaks, model calls, training, or feature mutations.
Counts are POTENTIAL reference rows BEFORE query-hash exclusion and deduplication.
"""
import argparse
import collections
import json
from pathlib import Path
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument('--output', default='artifacts/runs/007/review-r1/library-identity-audit.json')
    args = ap.parse_args()
    root = Path(args.repo)
    identity_path = root / 'artifacts/runs/005/structure_identity_train.parquet'
    train_path = root / 'data/raw/train.parquet'
    ident = pq.read_table(identity_path, columns=['raw_key', 'canonical_group_key']).to_pydict()
    mapping = dict(zip(ident['raw_key'], ident['canonical_group_key']))
    aliases = collections.defaultdict(set)
    for raw, canon in mapping.items():
        aliases[canon].add(raw)
    counts = collections.Counter()
    for batch in pq.ParquetFile(train_path).iter_batches(columns=['inchikey14'], batch_size=131072):
        counts.update(batch.column(0).to_pylist())
    report = {
        'scope': 'Potential reference rows before query-hash exclusion/deduplication; no peak columns read.',
        'inputs': {'identity': str(identity_path), 'train_key_column': str(train_path) + ':inchikey14'},
        'raw_rows_counted': sum(counts.values()),
        'code_issue': 'build_lib.py:74 indexes canonical candidate keys, while :130-131 looks up raw inchikey14 without mapping.',
        'datasets': {},
    }
    for ds, filename in [('np', 'feature_table_np250.json'), ('g1', 'feature_table_g1.json')]:
        feature_path = root / 'artifacts/runs/007-r1' / filename
        features = json.loads(feature_path.read_text())
        candidates = {r['c'] for m in features.values() for r in m['rows']}
        truths = {m['truth'] for m in features.values()}
        omitted = {r: n for r, n in counts.items()
                   if mapping.get(r, r) in candidates and mapping.get(r, r) != r}
        truth_omitted = {r: n for r, n in omitted.items() if mapping.get(r) in truths}
        affected_groups = set()
        affected_pairs = 0
        truth_details = []
        for key, mol in features.items():
            for row in mol['rows']:
                canon = row['c']
                other = sorted(aliases[canon] - {canon})
                if not other:
                    continue
                affected_groups.add(canon)
                affected_pairs += 1
                if canon == mol['truth']:
                    truth_details.append({
                        'query_key': key, 'canonical_truth': canon,
                        'recorded_has_lib': row['has_lib'],
                        'recorded_n_library_spectra': row['lib_n_spectra'],
                        'omitted_aliases': [{'raw_key': raw, 'potential_rows_before_exclusion': counts[raw]} for raw in other],
                    })
        # Also check whether a raw key that is itself requested as a canonical
        # candidate maps to a DIFFERENT canonical structure in the identity table.
        # These rows would be attached to the wrong candidate, not merely omitted.
        misattached = [{'raw_key_used_as_candidate': raw, 'actual_canonical_group': mapping[raw],
                        'potential_rows_before_exclusion': counts[raw]}
                       for raw in sorted(candidates & mapping.keys())
                       if mapping[raw] != raw and counts[raw] > 0]
        report['datasets'][ds] = {
            'feature_table': str(feature_path),
            'query_count': len(features),
            'affected_candidate_groups': len(affected_groups),
            'affected_query_candidate_rows': affected_pairs,
            'omitted_raw_alias_keys': len(omitted),
            'potential_omitted_reference_rows_before_exclusion': sum(omitted.values()),
            'affected_truth_groups_in_shortlist': len(truth_details),
            'omitted_truth_raw_alias_keys': len(truth_omitted),
            'potential_omitted_truth_rows_before_exclusion': sum(truth_omitted.values()),
            'truth_alias_details': truth_details,
            'zero_library_truth_examples_with_omitted_aliases': [r for r in truth_details if r['recorded_has_lib'] == 0],
            'raw_keys_attached_to_wrong_canonical_candidates': misattached,
        }
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({ds: {k: v for k, v in d.items() if not isinstance(v, list) and k != 'feature_table'}
                      for ds, d in report['datasets'].items()}, ensure_ascii=False))
    print('wrote', args.output)


if __name__ == '__main__':
    main()
