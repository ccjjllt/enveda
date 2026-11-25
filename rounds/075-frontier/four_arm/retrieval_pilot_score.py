"""Seal complete Pablo replay arms before opening labels, then compare paired MRR."""

import argparse
from collections import Counter
import csv
from functools import lru_cache
from hashlib import file_digest
import json
from pathlib import Path
import shutil

import pandas as pd
from rdkit import Chem, rdBase, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


ARMS = ('baseline', 'early_score', 'late_dedupe', 'joint')
CAP = 25
SOURCE = 'pablodiegoo/enveda-casmi26-starter-notebook/29'
APPLIED = ('n1c_omni', 'n1c_dl1', 'dual_sync', 'p5tail', 'fr3to25', 'aprior', 'uniq', 'npx', 'e180x')
SNAPSHOTS = ('submission_pre_n1c.csv', 'submission_pre_dualsync.csv', 'submission_pre_apex_extlib.csv')
STAGES = ('execution_receipt.json', 'v43_execution_receipt.json', 'e004_verification.json',
          'four_reference_audit.json', 'four_inner_reference.json', 'spectral_parallax_report.json',
          'merge_report.json', 'v8_report.json', 'apex_extlib_audit.json', 'pilot_provenance_audit.json',
          *(f'{name}_audit.json' for name in APPLIED))
FROZEN = ('submission.csv', 'submission_unpadded.csv', 'pilot_final_slots.json',
          'four_source_lists.json', 'pilot_forward_budget.json', *SNAPSHOTS, *STAGES)
SOURCE_LISTS = ('prefusion', 'engine2', 'main', 'pubchem')
FILL_ORDER = ('fused_overflow', *SOURCE_LISTS)
TAUTOMER = rdMolStandardize.TautomerEnumerator()


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


@lru_cache(maxsize=25000)
def metric_identity(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        elements = tuple(sorted(Counter(atom.GetSymbol() for atom in mol.GetAtoms()
                                        if atom.GetSymbol() != 'H').items()))
        Chem.RemoveStereochemistry(mol)
        key = Chem.MolToInchiKey(TAUTOMER.Canonicalize(mol))[:14]
        return (key, elements) if len(key) == 14 else None
    except (ValueError, RuntimeError, TypeError):
        return None


def answers(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['molecule_id', 'smiles']:
            raise ValueError(f'Unexpected output columns in {path}')
        rows = [(row['molecule_id'], row['smiles'].split(';') if row['smiles'] else []) for row in reader]
    result = dict(rows)
    if len(result) != len(rows):
        raise ValueError(f'Duplicate queries in {path}')
    return result


def fill_sparse_lists(work):
    """Keep 25 slots per query; leave unused slots vacant after evaluated sources run out."""
    path = work / 'submission.csv'
    raw_path = work / 'submission_unpadded.csv'
    # Resealing a workspace starts from the same raw output rather than from the previous fill.
    if not raw_path.exists():
        shutil.copyfile(path, raw_path)
    current = answers(raw_path)
    upstream = read_json(work / 'four_source_lists.json')
    if set(current) != set(upstream):
        raise ValueError('Cannot close final slots with a different set of query IDs')
    rows, changes, vacancies = {}, {}, {}
    for mid, guesses in current.items():
        if len(guesses) > CAP:
            raise ValueError(f'{mid}: the original scoring chain exceeded the final slot cap')
        kept, seen = [], set()
        for candidate in guesses:
            key = metric_identity(candidate)
            if key is not None and key not in seen:
                seen.add(key)
                kept.append(candidate)
        retained = len(kept)
        added = []
        for source in FILL_ORDER:
            if len(kept) == CAP:
                break
            for candidate in upstream[mid].get(source, []):
                key = metric_identity(candidate)
                if key is None or key in seen:
                    continue
                seen.add(key)
                kept.append(candidate)
                added.append({'source': source, 'final_rank': len(kept), 'metric_key': key[0]})
                if len(kept) == CAP:
                    break
        if len(kept) < CAP:
            vacancies[mid] = CAP - len(kept)
        if kept != guesses:
            changes[mid] = {'original_length': len(guesses), 'original_identities_retained': retained,
                            'filled': added}
        rows[mid] = kept
    if changes:
        tmp = path.with_suffix('.fill.tmp')
        with tmp.open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['molecule_id', 'smiles'])
            writer.writerows((mid, ';'.join(guesses)) for mid, guesses in rows.items())
        tmp.replace(path)
    elif digest(path) != digest(raw_path):
        shutil.copyfile(raw_path, path)
    write_json(work / 'pilot_final_slots.json', dict(fill_order=FILL_ORDER, queries=len(rows),
        candidate_slots=CAP, vacant_slots_by_query=vacancies,
        raw_sha256=digest(raw_path), final_sha256=digest(path), changed_queries=len(changes), changes=changes))


def ensure_complete(work, queries, arm):
    query = pd.read_parquet(queries, columns=['molecule_id', 'spectrum_id'])
    ids = set(query.molecule_id)
    if not 50 <= len(ids) <= 100 or not query.spectrum_id.is_unique:
        raise ValueError('The blinded panel must contain 50–100 identities and unique spectra')
    hashes = {name: digest(work / name) for name in FROZEN}
    read = lambda name: read_json(work / name)
    outer = read('execution_receipt.json')
    inner = read('v43_execution_receipt.json')
    device = read('e004_verification.json')
    if not (outer['is_full_run'] and outer['processed_molecules'] == len(ids)
            and inner['scoring_rerun'] and inner['main_errors'] == inner['placeholder_rows'] == 0
            and inner['main_lists'] == device['processed_molecules'] == len(ids)
            and set(inner['processed_ids']) == ids):
        raise ValueError(f'{arm}: the original pipeline did not process the entire panel')
    if device['main_rdkit'] != '2026.03.3' or set(device['gpu'].splitlines()) != {'Tesla T4'}:
        raise ValueError(f'{arm}: the pipeline did not run with the pinned scorer on Kaggle T4')
    if not outer.get('candidate_promotion_fix'):
        raise ValueError(f'{arm}: the original candidate-specific evidence path was lost')
    query_hash = digest(queries)
    if inner['test']['sha256'] != query_hash or inner['processed_test']['sha256'] != query_hash:
        raise ValueError(f'{arm}: original pipeline did not use the sealed query spectra')
    reference = read('four_inner_reference.json')
    if reference['error'] is not None:
        raise ValueError(f'{arm}: failed to mask the original external reference extension')
    parallax = read('spectral_parallax_report.json')
    if parallax.get('time_budget_exceeded') is not False or parallax.get('rows') != len(ids):
        raise ValueError(f'{arm}: physical Parallax did not complete on all queries')
    provenance = read('pilot_provenance_audit.json')
    if not provenance.get('status', '').startswith('disabled'):
        raise ValueError(f'{arm}: optional provenance stage differs from the scored original')
    if read('merge_report.json').get('status') != 'applied':
        raise ValueError(f'{arm}: variant-M fusion did not run')
    mass = read('v8_report.json')
    if mass['mass']['status'] != 'on' or mass['mass_lead']['status'] != 'ran':
        raise ValueError(f'{arm}: v8 mass reranking did not run')
    audits = {name: read(f'{name}_audit.json') for name in (*APPLIED, 'apex_extlib')}
    for name in APPLIED:
        status = audits[name]['status']
        if not status.startswith('applied') and not (name == 'npx' and status.startswith('gate failed')):
            raise ValueError(f'{arm}: {name} failed or skipped: {status}')
    for name in ('n1c_omni', 'dual_sync', 'apex_extlib'):
        if audits[name]['molecules'] != len(ids):
            raise ValueError(f'{arm}: {name} did not process all queries')
    if audits['n1c_dl1'].get('changed_lists', 0):
        snapshot = 'submission_pre_n1c_dl1.csv'
        hashes[snapshot] = digest(work / snapshot)
    apex = audits['apex_extlib']
    if not (apex.get('reference_rows', 0) > 0 or
            (apex.get('source_rows_removed', 0) > 0 and
             apex['status'] == 'applied: no reference survives source holdout')):
        raise ValueError(f'{arm}: missing blinded APEX search coverage')
    p5 = audits['p5tail']['rerun']
    if p5['molecules'] != len(ids) or p5['n_err'] != 0:
        raise ValueError(f'{arm}: incomplete P5 recovery sweep: {p5}')
    final = answers(work / 'submission.csv')
    if set(final) != ids:
        raise ValueError(f'{arm}: final input and output IDs differ')
    slot_audit = read('pilot_final_slots.json')
    vacancies = {mid: CAP - len(candidates) for mid, candidates in final.items() if len(candidates) < CAP}
    if slot_audit['candidate_slots'] != CAP or slot_audit['vacant_slots_by_query'] != vacancies:
        raise ValueError(f'{arm}: vacant slot ledger differs from the final lists')
    for mid, candidates in final.items():
        keys = {metric_identity(x) for x in candidates}
        if len(candidates) > CAP or len(keys) != len(candidates) or None in keys:
            raise ValueError(f'{arm}: {mid} exceeds the slot budget or contains invalid identities')
    upstream = read('four_source_lists.json')
    if set(upstream) != ids or any(not set(SOURCE_LISTS) <= value.keys() for value in upstream.values()):
        raise ValueError(f'{arm}: missing upstream candidate source ledger')
    budget = read('pilot_forward_budget.json')
    if budget['arm'] != arm or set(budget['cases']) != ids:
        raise ValueError(f'{arm}: incomplete prediction budget receipt')
    ledger = {mid: entry['predictor_budget'] for mid, entry in budget['cases'].items()}
    requests = {mid: entry['request_sha256'] for mid, entry in budget['cases'].items()}
    distinct_columns = {tuple(sorted(entry)) for entry in ledger.values()}
    if (any(not counts or any(type(n) is not int or n < 0 for n in counts.values())
            for counts in ledger.values()) or len(distinct_columns) != 1
            or not any(n > 0 for counts in ledger.values() for n in counts.values())):
        raise ValueError(f'{arm}: missing non-empty measured predictor budgets')
    if any(not values or any(not isinstance(value, str) or len(value) != 64 for value in values.values())
           for values in requests.values()):
        raise ValueError(f'{arm}: one or more predictors did not log a valid request digest')
    return dict(arm=arm, source=SOURCE, molecules=len(ids), spectra=len(query), candidate_slots=CAP,
                queries_sha256=query_hash, metric_rdkit=rdBase.rdkitVersion, gpu=device['gpu'].splitlines(),
                hashes=hashes, stage_status={name: audits[name]['status'] for name in APPLIED},
                coverage={name: details.get('counts', details.get('scored', details.get('molecules')))
                          for name, details in audits.items()}, predictor_budget=ledger,
                predictor_requests_sha256=requests, predictor_parameters=budget['parameters'],
                finite_scores_by_model=dict(sum((Counter(entry.get('finite_counts', {}))
                    for entry in budget['cases'].values()), Counter())),
                reference_mask_sha256=hashes['four_reference_audit.json'],
                extension_counts={key: value for key, value in reference.items() if key != 'path'},
                final_slot_fill_queries=slot_audit['changed_queries'], vacant_slots_by_query=vacancies,
                core_elapsed_seconds=inner['elapsed_seconds'])


def seal(args):
    fill_sparse_lists(args.work)
    receipt = ensure_complete(args.work, args.queries, args.arm)
    output = args.work / 'pilot_arm_receipt.json'
    write_json(output, receipt)
    print('PILOT_ARM_COMPLETE', args.arm, receipt['molecules'], receipt['core_elapsed_seconds'], flush=True)


def identified_rank(candidates, truth):
    for rank, candidate in enumerate(candidates, 1):
        if metric_identity(candidate) == truth:
            return rank
    return 0


def reciprocal(rank):
    return 1.0 / rank if rank else 0.0


@lru_cache(maxsize=25000)
def formula(smiles):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcMolFormula(mol) if mol is not None else None


def baseline_opportunity(candidates, truth_smiles, rank, sources_missed):
    truth_formula = formula(truth_smiles)
    candidate_formulas = [formula(candidate) for candidate in candidates]
    leader_formula = candidate_formulas[0] if candidate_formulas else None
    matched_formula = candidate_formulas[rank - 1] if rank else None
    formula_rank = rank
    if not rank:
        group = 'missing_from_reported_sources' if sources_missed else 'lost_after_retrieval'
    elif rank == 1:
        group = 'correct_leader'
    elif leader_formula == matched_formula:
        group = 'same_formula'
    else:
        group = 'different_formula'
        formula_rank = candidate_formulas[:rank].count(matched_formula)
    return dict(baseline_opportunity=group, baseline_truth_formula=truth_formula,
                baseline_leader_formula=leader_formula, baseline_matched_formula=matched_formula,
                formula_oracle_rank=formula_rank,
                formula_oracle_delta_rr=reciprocal(formula_rank) - reciprocal(rank))


def summarize(rows, arm):
    ranks = [(r['baseline_rank'], r[f'{arm}_rank'], r['baseline_sources_missed']) for r in rows]
    count = len(ranks)
    return dict(molecules=count,
                mrr25=sum(reciprocal(new) for _, new, _ in ranks) / count,
                delta_mrr=sum(reciprocal(new) - reciprocal(old) for old, new, _ in ranks) / count,
                top1=sum(new == 1 for _, new, _ in ranks),
                new_top1=sum(new == 1 and old != 1 for old, new, _ in ranks),
                lost_top1=sum(old == 1 and new != 1 for old, new, _ in ranks),
                top5=sum(1 <= new <= 5 for _, new, _ in ranks),
                new_top5=sum(1 <= new <= 5 and not 1 <= old <= 5 for old, new, _ in ranks),
                lost_top5=sum(1 <= old <= 5 and not 1 <= new <= 5 for old, new, _ in ranks),
                top25=sum(new > 0 for _, new, _ in ranks),
                new_top25=sum(new > 0 and old == 0 for old, new, _ in ranks),
                lost_top25=sum(old > 0 and new == 0 for old, new, _ in ranks),
                recovered_top25_all_original_sources_missed=sum(new > 0 and old == 0 and no_source
                                                                for old, new, no_source in ranks),
                recovered_top5_all_original_sources_missed=sum(1 <= new <= 5 and old == 0 and no_source
                                                               for old, new, no_source in ranks))


def score(args):
    work = {arm: args.runs / arm for arm in ARMS}
    receipts = {arm: read_json(path / 'pilot_arm_receipt.json') for arm, path in work.items()}
    baseline = receipts['baseline']
    for arm, receipt in receipts.items():
        if receipt['arm'] != arm or receipt['candidate_slots'] != CAP:
            raise ValueError(f'{arm}: arm or candidate limit differs')
        for key in ('source', 'gpu', 'queries_sha256', 'spectra', 'molecules', 'metric_rdkit',
                    'predictor_budget', 'predictor_requests_sha256', 'predictor_parameters',
                    'reference_mask_sha256', 'extension_counts'):
            if receipt[key] != baseline[key]:
                raise ValueError(f'{arm}: differing frozen {key}')
        if receipt['stage_status']['npx'] != baseline['stage_status']['npx']:
            raise ValueError(f'{arm}: NPX applied a different validation decision')
        for name, expected in receipt['hashes'].items():
            if digest(work[arm] / name) != expected:
                raise ValueError(f'{arm}: {name} changed after rankings were sealed')
    lists = {arm: answers(path / 'submission.csv') for arm, path in work.items()}
    catalogs = {arm: read_json(path / 'four_source_lists.json') for arm, path in work.items()}
    labels = pd.read_parquet(args.labels, columns=['molecule_id', 'truth_key', 'truth_smiles', 'source', 'scaffold'])
    if (set(labels.molecule_id) != set(lists['baseline']) or not labels.molecule_id.is_unique
            or not labels.truth_key.is_unique or labels.source.isna().any()):
        raise ValueError('Truth rows do not match the sealed IDs and unique structures')
    rows = []
    for label in labels.sort_values('molecule_id').itertuples(index=False):
        mid = label.molecule_id
        truth = metric_identity(label.truth_smiles)
        if truth is None or label.truth_key != truth[0]:
            raise ValueError(f'{mid} truth key differs from the official metric')
        old_rank = identified_rank(lists['baseline'][mid], truth)
        item = dict(molecule_id=mid, source=label.source, scaffold=label.scaffold, truth_key=truth[0])
        for arm in ARMS:
            rank = identified_rank(lists[arm][mid], truth)
            source_ranks = {name: identified_rank(catalogs[arm][mid][name], truth) for name in SOURCE_LISTS}
            item.update({f'{arm}_rank': rank, **{f'{arm}_{name}_rank': n for name, n in source_ranks.items()},
                         f'{arm}_vacant_slots': receipts[arm]['vacant_slots_by_query'].get(mid, 0),
                         f'{arm}_sources_missed': not any(source_ranks.values()),
                         f'{arm}_reciprocal_rank': reciprocal(rank), f'{arm}_delta_rr': reciprocal(rank) - reciprocal(old_rank),
                         f'{arm}_new_top1': rank == 1 and old_rank != 1, f'{arm}_lost_top1': old_rank == 1 and rank != 1,
                         f'{arm}_new_top5': 1 <= rank <= 5 and not 1 <= old_rank <= 5,
                         f'{arm}_lost_top5': 1 <= old_rank <= 5 and not 1 <= rank <= 5})
        item.update(baseline_opportunity(lists['baseline'][mid], label.truth_smiles, old_rank,
                                         item['baseline_sources_missed']))
        rows.append(item)
    full_lists = [row for row in rows if all(row[f'{arm}_vacant_slots'] == 0 for arm in ARMS)]
    data = dict(source=SOURCE, molecules=len(rows), candidate_slots_per_query=CAP,
                labels_sha256=digest(args.labels), queries_sha256=baseline['queries_sha256'],
                equal_prediction_budget=True, equal_prediction_requests=True,
                predictor_budget_totals={key: sum(case[key] for case in baseline['predictor_budget'].values())
                                         for key in next(iter(baseline['predictor_budget'].values()))},
                finite_prediction_scores={arm: value['finite_scores_by_model'] for arm, value in receipts.items()},
                final_slot_fill_queries={arm: value['final_slot_fill_queries'] for arm, value in receipts.items()},
                vacant_slots_by_query={arm: value['vacant_slots_by_query'] for arm, value in receipts.items()},
                baseline_opportunities=dict(Counter(row['baseline_opportunity'] for row in rows)),
                formula_oracle_delta_mrr=sum(row['formula_oracle_delta_rr'] for row in rows) / len(rows),
                stage_status={arm: value['stage_status'] for arm, value in receipts.items()},
                summaries={arm: summarize(rows, arm) for arm in ARMS},
                all_arms_full_lists={arm: summarize(full_lists, arm) for arm in ARMS} if full_lists else {},
                by_source={source: {arm: summarize([r for r in rows if r['source'] == source], arm) for arm in ARMS}
                           for source in sorted({r['source'] for r in rows})})
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / 'per_query.csv', index=False)
    write_json(args.out / 'metrics.json', data)
    print('RETRIEVAL_PILOT_COMPLETE', json.dumps(data['summaries'], ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    lock = commands.add_parser('seal')
    lock.add_argument('--arm', choices=ARMS, required=True)
    for name in ('work', 'queries'):
        lock.add_argument('--' + name, type=Path, required=True)
    compare = commands.add_parser('score')
    for name in ('runs', 'labels', 'out'):
        compare.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if rdBase.rdkitVersion != '2026.03.3':
        raise RuntimeError(f'Required competition RDKit 2026.03.3, got {rdBase.rdkitVersion}')
    RDLogger.DisableLog('rdApp.*')
    (seal if args.command == 'seal' else score)(args)


if __name__ == '__main__':
    main()
