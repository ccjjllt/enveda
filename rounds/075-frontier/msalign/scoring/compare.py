"""Read sealed MSAlign predictions and calculate paired benchmark deltas."""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import random
import statistics

import pyarrow.parquet as pq

from retrieval_pilot_score import identified_rank, metric_identity, reciprocal
from score import digest, frozen_baseline, write_csv


CHANNEL = 'bins_fingerprint'
WEIGHT = 0.60
REPLICATES = 32
ARMS = ('baseline', 'rank_fusion', 'same_formula_top5', 'pure_model_diagnostic')
EXPOSURE = Path(__file__).resolve().parents[1] / 'exposure.json'


def ordered(indices, scores, mode):
    if not indices:
        return []
    model = sorted(indices, key=lambda i: (-scores[i], i))
    if mode == 'pure':
        return model
    model_ranks = {i: rank for rank, i in enumerate(model, 1)}
    baseline_ranks = {i: rank for rank, i in enumerate(indices, 1)}
    return sorted(indices,
                  key=lambda i: (-(1 / baseline_ranks[i] + WEIGHT / model_ranks[i]), i))


def formula_permutation(scores, formulas, mid, replicate=None):
    if not formulas:
        return [], []
    eligible = [i for i, candidate in enumerate(formulas[:5]) if candidate == formulas[0]]
    if replicate is not None:
        seed = sha256(f'msalign-fixed-null-2026-11-25/{mid}/{replicate}'.encode()).digest()
        rng = random.Random(seed)
        scrambled = [scores[i] for i in eligible]
        rng.shuffle(scrambled)
        scores = scores.copy()
        for i, value in zip(eligible, scrambled):
            scores[i] = value
    order = list(range(len(scores)))
    for original, new in zip(eligible, ordered(eligible, scores, 'fusion')):
        order[original] = new
    return order, eligible


def report(rows, arm):
    if not rows:
        return None
    ranks = [(item['molecule_id'], item['baseline_rank'], item[f'{arm}_rank']) for item in rows]
    return dict(
        molecules=len(ranks),
        baseline_mrr25=statistics.mean(reciprocal(old) for _, old, _ in ranks),
        mrr25=statistics.mean(reciprocal(new) for _, _, new in ranks),
        delta_mrr=statistics.mean(reciprocal(new) - reciprocal(old) for _, old, new in ranks),
        new_top1=[mid for mid, old, new in ranks if new == 1 and old != 1],
        lost_top1=[mid for mid, old, new in ranks if old == 1 and new != 1],
        new_top5=[mid for mid, old, new in ranks if 1 <= new <= 5 and not 1 <= old <= 5],
        lost_top5=[mid for mid, old, new in ranks if 1 <= old <= 5 and not 1 <= new <= 5],
        improved=[mid for mid, old, new in ranks if reciprocal(new) > reciprocal(old)],
        harmed=[mid for mid, old, new in ranks if reciprocal(new) < reciprocal(old)],
        old_top1=sum(old == 1 for _, old, _ in ranks),
        top1=sum(new == 1 for _, _, new in ranks),
        old_top5=sum(1 <= old <= 5 for _, old, _ in ranks),
        top5=sum(1 <= new <= 5 for _, _, new in ranks),
        top25=sum(new > 0 for _, _, new in ranks),
    )


def compare(args):
    baseline, candidates = frozen_baseline(args.baseline, args.queries)
    receipt = json.loads((args.predictions / 'msalign_receipt.json').read_text())
    candidate_scores = args.predictions / 'msalign_scores.csv'
    spectrum_scores = args.predictions / 'msalign_spectrum_scores.csv'
    if (receipt['baseline_sha256'] != baseline['hashes']['submission.csv']
            or receipt['queries_sha256'] != baseline['queries_sha256']
            or receipt['scores_sha256'] != digest(candidate_scores)
            or receipt['spectra_scores_sha256'] != digest(spectrum_scores)
            or CHANNEL not in receipt['channels'] or receipt['uploaded_truth'] is not False):
        raise ValueError('Predictions are not sealed against these blind inputs')

    adducts = defaultdict(set)
    scan_ids = defaultdict(set)
    spectra = pq.read_table(args.queries, columns=['molecule_id', 'spectrum_id', 'adduct']).to_pylist()
    for scan in spectra:
        adducts[scan['molecule_id']].add(scan['adduct'])
        scan_ids[scan['molecule_id']].add(scan['spectrum_id'])
    usable = receipt['scored_scans_by_query'][CHANNEL]
    skipped = receipt['skipped_scans_by_channel'][CHANNEL]
    if set(usable) != set(candidates) or not set(skipped) <= {row['spectrum_id'] for row in spectra}:
        raise ValueError('MSAlign coverage refers to different queries')
    for mid, ids in scan_ids.items():
        missed = ids & skipped.keys()
        if not (usable[mid] + len(missed) == len(ids) and
                all(skipped[sid]['molecule_id'] == mid for sid in missed)):
            raise ValueError(f'Missing or contradictory MSAlign coverage for {mid}')
    scores, formulas = defaultdict(list), defaultdict(list)
    with candidate_scores.open(newline='') as stream:
        for row in csv.DictReader(stream):
            mid = row['molecule_id']
            idx = len(scores[mid])
            if idx >= len(candidates[mid]) or idx + 1 != int(row['baseline_rank']) \
                    or candidates[mid][idx] != row['smiles']:
                raise ValueError(f'Candidate slots differ from Pablo baseline: {mid}')
            value = float(row[CHANNEL]) if row[CHANNEL] else None
            if ((value is None) != (usable[mid] == 0)
                    or value is not None and not float('-inf') < value < float('inf')
                    or not row['formula']):
                raise ValueError(f'Prediction differs from coverage or has no candidate formula: {mid}')
            scores[mid].append(value)
            formulas[mid].append(row['formula'])
    if any(len(scores[mid]) != len(guesses) for mid, guesses in candidates.items()):
        raise ValueError('Predictions skipped a candidate on the fixed list')

    # The comparison reads labels only after authenticating all score artifacts.
    if digest(args.labels) != args.label_sha256:
        raise ValueError('Label digest differs from the panel manifest')
    columns = ['molecule_id', 'truth_key', 'truth_smiles', 'source', 'scaffold']
    truth = pq.read_table(args.labels, columns=columns).to_pylist()
    if len(truth) != len(candidates) or {row['molecule_id'] for row in truth} != set(candidates):
        raise ValueError('Private label IDs do not match the blind benchmark')
    exposure = None
    scaffold_detected = set()
    possible_audit = args.exposure or EXPOSURE
    if possible_audit.is_file():
        audited = json.loads(possible_audit.read_text())
        if (audited['queries_sha256'] == receipt['queries_sha256']
                and audited['labels_sha256'] == args.label_sha256):
            exposure = audited
            scaffold_detected = set(audited['training_scaffold_detected_query_ids'])
            if (audited['panel_molecules'] != len(truth) or not scaffold_detected <= set(candidates)
                    or len(scaffold_detected) != audited['generic_scaffold_matches_by_fold']['train']):
                raise ValueError('External training exposure summary differs from the panel')

    rows, null = [], [0.] * REPLICATES
    for label in sorted(truth, key=lambda row: row['molecule_id']):
        mid = label['molecule_id']
        identity = metric_identity(label['truth_smiles'])
        if not identity or identity[0] != label['truth_key']:
            raise ValueError(f'The metric cannot resolve recorded truth {mid}')
        guesses, value, formula = candidates[mid], scores[mid], formulas[mid]
        indices = list(range(len(guesses)))
        positions = {name: indices for name in ARMS}
        eligible = []
        if usable[mid]:
            positions['same_formula_top5'], eligible = formula_permutation(value, formula, mid)
            positions['rank_fusion'] = ordered(indices, value, 'fusion')
            positions['pure_model_diagnostic'] = ordered(indices, value, 'pure')
        rank = {name: identified_rank([guesses[i] for i in order], identity)
                for name, order in positions.items()}
        if len({bool(i) for i in rank.values()}) != 1:
            raise ValueError(f'Reranking changed the candidate coverage: {mid}')
        baseline_rank = rank['baseline']
        if usable[mid]:
            for replicate in range(REPLICATES):
                permutation, _ = formula_permutation(value, formula, mid, replicate)
                new_rank = identified_rank([guesses[i] for i in permutation], identity)
                null[replicate] += reciprocal(new_rank) - reciprocal(baseline_rank)
        ion_stratum = 'msg_adduct' if adducts[mid] <= {'[M+H]+', '[M+Na]+'} else 'outside_msg_adduct'
        item = dict(molecule_id=mid, source=label['source'], scaffold=label['scaffold'],
                    ion_stratum=ion_stratum, slots=len(guesses), eligible_top5_slots=len(eligible),
                    usable_scans=usable[mid], deposited_scans=len(scan_ids[mid]))
        if exposure is not None:
            item['msg_training_scaffold_detected'] = mid in scaffold_detected
        for name in ARMS:
            item[f'{name}_rank'] = rank[name]
            item[f'{name}_delta_rr'] = reciprocal(rank[name]) - reciprocal(baseline_rank)
        rows.append(item)

    groups = {'all': rows, 'same_formula_top5_has_competitor': [
        row for row in rows if row['eligible_top5_slots'] > 1]}
    if exposure is not None:
        groups['msg_training_scaffold_detected'] = [
            row for row in rows if row['msg_training_scaffold_detected']]
        groups['msg_training_scaffold_not_detected'] = [
            row for row in rows if not row['msg_training_scaffold_detected']]
    for column in ('source', 'ion_stratum'):
        for name in sorted({row[column] for row in rows}):
            groups[f'{column}/{name}'] = [row for row in rows if row[column] == name]
    null = [value / len(rows) for value in null]
    summaries = {name: {arm: report(items, arm) for arm in ARMS} for name, items in groups.items()}
    rollup = dict(molecules=len(rows), rank_fusion_weight=WEIGHT, channel=CHANNEL,
                  scored_scans=sum(usable.values()), skipped_scans=len(skipped),
                  baseline_fallback_ids=sorted(mid for mid in usable if not usable[mid]),
                  baseline_sha256=receipt['baseline_sha256'], score_sha256=receipt['scores_sha256'],
                  private_label_sha256=args.label_sha256, groups=summaries,
                  within_query_score_shuffle=dict(replicates=REPLICATES, mean_delta_mrr=statistics.mean(null),
                                                  min_delta_mrr=min(null), max_delta_mrr=max(null)),
                  population_note='Source/scaffold holdout from Pablo. Refer to the MSAlign '
                                  'exposure audit for exact identities and generic scaffold sharing.',
                  msalign_training_exposure=(
                      {key: exposure[key] for key in ('split', 'exact_identity_matches_by_fold',
                                                     'generic_scaffold_matches_by_fold', 'scaffold_method')}
                      if exposure is not None else None))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'msalign_paired_summary.json').write_text(json.dumps(rollup, indent=2) + '\n')
    write_csv(args.out / 'msalign_paired_per_query.csv', rows, tuple(rows[0]))
    print(json.dumps(summaries['all'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'queries', 'predictions', 'labels', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--label-sha256', required=True)
    parser.add_argument('--exposure', type=Path)
    compare(parser.parse_args())
