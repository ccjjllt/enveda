"""Stage the frozen absolute rerank for a completed Pablo v29 run."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import pyarrow.parquet as pq
from rdkit import rdBase

from rank import rank_absolute, read_scans
from retrieval_pilot_score import CAP, answers, digest


HEAD_NAMES = {'main', 'engine2', 'pubchem'}


def stage(work, comp, out):
    started = time.monotonic()
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError(f'Expected competition RDKit, got {rdBase.rdkitVersion}')
    submission = work / 'submission.csv'
    source_path = work / 'absolute_source_heads.json'
    query_path = comp / 'test.parquet'
    before = digest(submission)
    baseline = answers(submission)
    with (comp / 'sample_submission.csv').open(newline='') as stream:
        sample_ids = [row['molecule_id'] for row in csv.DictReader(stream)]
    if list(baseline) != sample_ids or any(not 1 <= len(values) <= CAP or not all(values)
                                           for values in baseline.values()):
        raise ValueError('The Pablo submission disagrees with the competition sample format')

    inner = json.loads((work / 'v43_execution_receipt.json').read_text())
    source_lists = json.loads(source_path.read_text())
    query_ids = set(pq.read_table(query_path, columns=['molecule_id'])['molecule_id'].to_pylist())
    if None in query_ids or not query_ids <= set(sample_ids) or len(query_ids) != inner['input_molecules']:
        raise ValueError('Competition query IDs disagree with the native execution receipt')
    expected = query_ids if inner['scoring_rerun'] else set(sorted(query_ids)[:inner['smoke_n']])
    active = set(inner['processed_ids'])
    if (active != expected or set(source_lists) != active or len(active) != inner['processed_molecules']
            or any(set(heads) != HEAD_NAMES or any(not isinstance(values, list) or len(values) > 1
                                                  or any(not isinstance(s, str) or not s for s in values)
                                                  for values in heads.values())
                   for heads in source_lists.values())):
        raise ValueError('The source heads disagree with the native full/smoke execution receipt')
    subset = {mid: values for mid, values in baseline.items() if mid in active}
    scans = {sid: scan for sid, scan in read_scans(query_path).items() if scan.mid in active}
    ranked, swaps = rank_absolute(subset, source_lists, scans)

    expected_swaps = {mid: list(values) for mid, values in subset.items()}
    for swap in swaps:
        mid, anchor, challenger = (swap[key] for key in ('molecule_id', 'anchor_rank', 'chosen_rank'))
        values = expected_swaps[mid]
        if anchor not in (1, 5) or not anchor < challenger <= min(anchor + 2, len(values)):
            raise ValueError('The frozen absolute rule returned an invalid swap')
        values[anchor - 1], values[challenger - 1] = values[challenger - 1], values[anchor - 1]
    if (ranked != expected_swaps
            or any(Counter(values) != Counter(subset[mid]) for mid, values in ranked.items())):
        raise ValueError('The absolute result does not preserve the final candidate multisets')

    changed = sum(ranked[mid] != subset[mid] for mid in subset)
    after = before
    if changed:
        staged = out / 'absolute_submission.csv'
        with staged.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('molecule_id', 'smiles'))
            writer.writerows((mid, ';'.join(ranked.get(mid, values))) for mid, values in baseline.items())
        after = digest(staged)
    audit = dict(status='applied' if changed else 'unchanged', method='frozen_absolute',
                 scoring_rerun=inner['scoring_rerun'], candidate_slots=CAP,
                 sample_rows=len(baseline), processed_molecules=len(active),
                 eligible_molecules=len({scan.mid for scan in scans.values()}),
                 eligible_spectra=len(scans), unchanged_placeholder_rows=len(baseline) - len(active),
                 changed_molecules=changed, swaps=swaps, submission_before_sha256=before,
                 submission_after_sha256=after, source_heads_sha256=digest(source_path),
                 queries_sha256=digest(query_path), elapsed_seconds=round(time.monotonic() - started, 2))
    (out / 'absolute_audit.json').write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work', 'comp', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    stage(**vars(parser.parse_args()))
