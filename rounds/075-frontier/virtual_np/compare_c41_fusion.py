"""Compare the sealed historical baseline and two complete C41 fusion arms."""

import argparse
from collections import Counter
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
ARMS = ('baseline', 'original_slots', 'replacement')
CHANNELS = ('ice_native', 'gl_native', 'gl_proxy', 'ice_shadow', 'gl_shadow')
COMMON = ('c41_study_receipt.json', 'c41_delta_receipt.json', 'c41_budget_plan.json',
          'c41_forward_observed.json', 'c41_slot_fallbacks.json')
FILES = ('submission.csv', 'submission_unpadded.csv', 'pilot_final_slots.json', 'four_source_lists.json',
         'c41_historical_c43.json', 'c41_routed_c43.json', *COMMON)


def read_arm(path, arm, original, original_catalog, lengths, key):
    with ZipFile(path) as package:
        manifest = 'c41_full_receipt.json'
        body = package.read(manifest)
        report = json.loads(body)
        names = package.namelist()
        if (len(names) != len(set(names)) or set(names) != set(report['hashes']) | {manifest}
                or any(Path(name).name != name for name in names) or not set(FILES) <= set(report['hashes'])):
            raise ValueError(f'{arm}: the complete fusion archive differs from its receipt')
        if (sha256(body).hexdigest() != key or report['status'] != 'complete' or report['arm'] != arm
                or report['candidate_slots'] != CAP or report['source'] != SOURCE
                or report['baseline_archive_sha256'] != BASELINE_SHA or report['queries_sha256'] != original['queries_sha256']):
            raise ValueError(f'{arm}: comparison mismatches the sealed historical inputs')
        contents = {}
        for name, expected in report['hashes'].items():
            raw = package.read(name)
            if sha256(raw).hexdigest() != expected:
                raise ValueError(f'{arm}: {name} does not match its final digest')
            if name in FILES:
                contents[name] = raw

    reader = csv.DictReader(io.StringIO(contents['submission.csv'].decode()))
    if reader.fieldnames != ['molecule_id', 'smiles']:
        raise ValueError(f'{arm}: invalid prediction columns')
    rows = list(reader)
    ranking = {row['molecule_id']: row['smiles'].split(';') if row['smiles'] else [] for row in rows}
    cases = json.loads(contents['four_source_lists.json'])
    if (len(rows) != len(ranking) or set(ranking) != set(cases)
            or {mid:len(values) for mid, values in ranking.items()} != lengths):
        raise ValueError(f'{arm}: final list lengths differ from the historical candidate budget')
    slots = json.loads(contents['pilot_final_slots.json'])
    expected_vacancies = {mid: CAP - count for mid, count in lengths.items() if count < CAP}
    if (slots['candidate_slots'] != CAP or slots['vacant_slots_by_query'] != expected_vacancies
            or slots['raw_sha256'] != report['hashes']['submission_unpadded.csv']
            or slots['final_sha256'] != report['hashes']['submission.csv']):
        raise ValueError(f'{arm}: final-slot evidence disagrees with the sealed prediction rows')
    for mid, values in ranking.items():
        normalized = {metric_identity(value) for value in values}
        if not set(ORIGINAL_LISTS) <= cases[mid].keys() or None in normalized or len(normalized) != len(values):
            raise ValueError(f'{arm}: missing upstream provenance or duplicate/invalid final structures for {mid}')
    historic_entry = json.loads(contents['c41_historical_c43.json'])
    routed_entry = json.loads(contents['c41_routed_c43.json'])
    if (historic_entry != {mid:row['prefusion'] for mid,row in original_catalog.items()}
            or routed_entry != {mid:row['prefusion'] for mid,row in cases.items()}):
        raise ValueError(f'{arm}: final source ledger disagrees with historical or routed C43')
    return report, ranking, cases, {name:contents[name] for name in COMMON}


def measure_budgets(original, old_budget, common):
    study = json.loads(common['c41_study_receipt.json'])
    plan = json.loads(common['c41_budget_plan.json'])
    actual = json.loads(common['c41_forward_observed.json'])
    ids = set(original['predictor_budget'])
    if study['native_no_shadow_controls_exact'] != len(ids) or set(plan) != ids or set(actual) != ids:
        raise ValueError('No full historical C41/C43 recovery for the measured budget')
    changed, unequal_finite, unequal_effective = [], [], []
    total, control_finite, treatment_finite = Counter(), Counter(), Counter()
    effective_finite, delta = Counter(), Counter()
    for mid in sorted(ids):
        if set(plan[mid]) != set(CHANNELS) or set(actual[mid]) != set(CHANNELS):
            raise ValueError(f'Incomplete forward channel ledger for {mid}')
        distinct, attempted_differences, effective_differences = set(), set(), set()
        for channel in CHANNELS:
            expected = original['predictor_budget'][mid][f'{channel}_candidate_spectra']
            p, observed = plan[mid][channel], actual[mid][channel]
            before, after = p['original_request_sha256'], p['treatment_request_sha256']
            key = 'gl_native_and_proxy' if channel == 'gl_native' else channel
            expected_finite = old_budget[mid]['finite_counts'].get(key, 0)
            if (before != original['predictor_requests_sha256'][mid][channel] or len(after) != 64
                    or p['candidate_spectra'] != expected or observed['candidate_spectra'] != expected
                    or p['delta_candidate_spectra'] != observed['delta_candidate_spectra']
                    or not 0 <= observed['delta_candidate_spectra'] <= expected
                    or p['finite_control'] != expected_finite or observed['finite_control'] != expected_finite
                    or not p['finite_known'] <= observed['finite_treatment'] <= p['finite_max']
                    or not 0 <= observed['finite_effective'] <= expected):
                raise ValueError(f'Changed or incomplete paired prediction budget for {mid} {channel}')
            if before != after:
                distinct.add(channel)
            if observed['finite_treatment'] != expected_finite:
                attempted_differences.add(channel)
            if observed['finite_effective'] != expected_finite:
                effective_differences.add(channel)
            total[channel] += observed['candidate_spectra']
            control_finite[channel] += expected_finite
            treatment_finite[channel] += observed['finite_treatment']
            effective_finite[channel] += observed['finite_effective']
            delta[channel] += observed['delta_candidate_spectra']
        if ((distinct and distinct != {'ice_shadow', 'gl_shadow'})
                or (attempted_differences | effective_differences) - distinct):
            raise ValueError(f'Forward-request changes outside paired E1 shadow channels for {mid}')
        if distinct:
            changed.append(mid)
        if attempted_differences:
            unequal_finite.append(mid)
        if effective_differences:
            unequal_effective.append(mid)
    return dict(same_request_counts=True, same_finite_counts=not unequal_finite, totals=dict(total),
                control_finite=dict(control_finite), treatment_finite=dict(treatment_finite),
                effective_treatment_finite=dict(effective_finite), unequal_effective_query_ids=unequal_effective,
                unequal_finite_query_ids=unequal_finite, changed_request_query_ids=changed,
                additional_computations_within_existing_slots=dict(delta))


def rank_rows(labels, predictions, catalogs):
    required = ['molecule_id', 'truth_key', 'truth_smiles', 'source', 'scaffold']
    entries = pq.read_table(labels, columns=required).to_pylist()
    original = predictions['baseline']
    if len(entries) != len(original) or {row['molecule_id'] for row in entries} != set(original):
        raise ValueError('The held-out panel differs from the sealed prediction rows')
    rows = []
    for row in sorted(entries, key=lambda item:item['molecule_id']):
        mid = row['molecule_id']
        truth = metric_identity(row['truth_smiles'])
        if truth is None or truth[0] != row['truth_key']:
            raise ValueError(f'The held-out identity changed under the official metric for {mid}')
        output = {field:row[field] for field in ('molecule_id', 'source', 'scaffold')}
        for arm in ARMS:
            rank = identified_rank(predictions[arm][mid], truth)
            output[f'{arm}_rank'] = rank
            output.update({f'{arm}_{name}_rank':identified_rank(catalogs[arm][mid][name], truth) for name in ORIGINAL_LISTS})
        output['baseline_sources_missed'] = not (output['baseline_rank'] or any(output[f'baseline_{name}_rank'] for name in ORIGINAL_LISTS))
        for arm in ARMS[1:]:
            output[f'{arm}_delta_rr'] = reciprocal(output[f'{arm}_rank']) - reciprocal(output['baseline_rank'])
        rows.append(output)
    return rows


def contrasts(rows):
    measured = {arm:summarize(rows, arm) for arm in ARMS}
    paired = [dict(row, baseline_rank=row['original_slots_rank'], replacement_delta_rr=
                   reciprocal(row['replacement_rank']) - reciprocal(row['original_slots_rank'])) for row in rows]
    intervals = {arm:paired_scaffold_interval(rows, arm) for arm in ARMS[1:]}
    intervals['replacement_vs_original_slots'] = paired_scaffold_interval(paired, 'replacement')
    return dict(versus_historical=measured, replacement_vs_original_slots=summarize(paired, 'replacement'),
                delta_mrr_scaffold_bootstrap=intervals)


def compare(baseline_zip, original_slots, replacement, original_slots_receipt_sha, replacement_receipt_sha,
            labels, labels_sha, out):
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Expected the official competition RDKit')
    RDLogger.DisableLog('rdApp.*')
    if digest(baseline_zip) != BASELINE_SHA or digest(labels) != labels_sha:
        raise ValueError('Historical baseline or held-out labels differ from the frozen panel')
    old_report, old_final, old_sources = baseline(baseline_zip)
    with ZipFile(baseline_zip) as archive:
        raw = archive.read('pilot_forward_budget.json')
        if sha256(raw).hexdigest() != old_report['hashes']['pilot_forward_budget.json']:
            raise ValueError('The historical forward budget changed after sealing')
        old_budget = json.loads(raw)['cases']
    lengths = {mid:len(values) for mid, values in old_final.items()}
    experiments = dict(original_slots=(original_slots, original_slots_receipt_sha),
                       replacement=(replacement, replacement_receipt_sha))
    reports, ranks, catalogs = {}, {'baseline':old_final}, {'baseline':old_sources}
    control_artifacts = None
    for arm, (path, receipt) in experiments.items():
        reports[arm], ranks[arm], catalogs[arm], shared = read_arm(path, arm, old_report, old_sources, lengths, receipt)
        if control_artifacts is not None and control_artifacts != shared:
            raise ValueError('The two arms did not execute against the same sealed comparison and forward ledger')
        control_artifacts = shared
    delta = json.loads(control_artifacts['c41_delta_receipt.json'])
    if any(report['plan_sha256'] != delta['plan_sha256'] for report in reports.values()):
        raise ValueError('The scored arms disagree with the attempted request plan')
    budget = measure_budgets(old_report, old_budget, control_artifacts)
    rows = rank_rows(labels, ranks, catalogs)
    source_names = sorted({row['source'] for row in rows})
    unequal = set(budget['unequal_finite_query_ids'])
    unequal_effective = set(budget['unequal_effective_query_ids'])
    matched = [row for row in rows if row['molecule_id'] not in unequal]
    matched_effective = [row for row in rows if row['molecule_id'] not in unequal_effective]
    study = json.loads(control_artifacts['c41_study_receipt.json'])
    replacements = {key:study[key] for key in ('allocated', 'attempted', 'effective', 'calibration_fallbacks', 'prediction_fallbacks')}
    report = dict(queries_sha256=old_report['queries_sha256'], molecules=len(rows), candidate_slots=CAP,
                  actual_slots=sum(lengths.values()), original_vacancies={k:CAP-v for k,v in lengths.items() if v<CAP},
                  predecessor=BASELINE_SHA, forward_budget=budget,
                  slot_replacements=replacements,
                  slot_fallbacks_sha256=sha256(control_artifacts['c41_slot_fallbacks.json']).hexdigest(),
                  arms={arm:{'receipt_sha256':p[1], 'c41_main_sha256':reports[arm]['c41_main_sha256'],
                             'c43_sha256':reports[arm]['c43_sha256']} for arm,p in experiments.items()},
                  total=contrasts(rows),
                  equal_finite_subset=contrasts(matched) if unequal and matched else None,
                  equal_effective_subset=contrasts(matched_effective) if unequal_effective and matched_effective else None,
                  by_source={name:contrasts([row for row in rows if row['source'] == name]) for name in source_names})
    out.mkdir(parents=True, exist_ok=True)
    (out/'c41_fusion_metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (out/'c41_fusion_per_query.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[key for key in rows[0] if key != 'scaffold'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-zip', 'original-slots', 'replacement', 'labels', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('original-slots-receipt-sha', 'replacement-receipt-sha', 'labels-sha'):
        parser.add_argument('--'+name, required=True)
    compare(**vars(parser.parse_args()))
