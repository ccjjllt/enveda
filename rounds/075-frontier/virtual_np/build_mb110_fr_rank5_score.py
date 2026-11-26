"""Package independent scoring of sealed MassBank 110 C34 and FR rank-5 runs."""

import argparse
from collections import Counter
import csv
from inspect import getsource
import json
from math import fsum, isfinite
from pathlib import Path
import re

from build import IMAGE, METRIC_BOOTSTRAP, WHEEL, pack
from build_final_native_outsider_mb110_score import (
    BASE, BASELINE_MRR, LABELS, LABELS_SHA, PINS, artifact, checksum, pinned, rank, read_baseline,
)
from c39_request_cache import identity_function


RUNNER = 'enveda-075-mb110-fr-rank5-run'
SLUG = 'enveda-075-mb110-fr-rank5-score'
SEAL = 'mb110_fr_rank5_run.json'
ACTIONS = 'mb110_fr_rank5_actions.json'
SUMMARY = 'mb110_fr_rank5_score.json'
PER_QUERY = 'mb110_fr_rank5_score_per_query.csv'
HISTORICAL_SHA = '8568ed5abecb88ef43243b122fa726959c85893ff7ae32db22f07d72de0fa44d'
ARMS = ('historical', 'c34', 'fr')
PAIRS = (('c34', 'historical'), ('fr', 'historical'), ('fr', 'c34'))


def load_seal(path):
    seal = json.loads(path.read_bytes())
    expected = dict(status='sealed', candidate_budget=25, molecules=110, actual_slots=2750,
                    slot=5, native_device='cpu')
    if any(seal.get(key) != value for key, value in expected.items()):
        raise ValueError('Blind runner did not seal the fixed 110 × 25 inventory')
    artifacts = seal['files']
    if (not {ACTIONS, *(f'{arm}_final.json' for arm in ARMS)} <= artifacts.keys()
            or artifacts['historical_final.json'] != HISTORICAL_SHA
            or any(Path(name).name != name or name in ('.', '..', SEAL) or not re.fullmatch('[0-9a-f]{64}', digest)
                   for name, digest in artifacts.items())):
        raise ValueError('Blind artifact seals or the historical SHA differ')
    budget = seal['logical_requests_per_arm_and_model']
    if set(budget) != set(ARMS[1:]) or budget['c34'] != budget['fr']:
        raise ValueError('Sealed native prediction budgets differ')
    return seal


def verify_lists(inputs, receipt, seal, identity, formula):
    import pyarrow.parquet as pq

    artifacts = {name: pinned(receipt.with_name(name), digest) for name, digest in seal['files'].items()}
    rosters = {arm: json.loads(artifacts[f'{arm}_final.json'].read_bytes()) for arm in ARMS}
    actions = json.loads(artifacts[ACTIONS].read_bytes())
    original = read_baseline(pinned(artifact(inputs, BASE, 'submission.csv'), PINS['submission.csv']))
    if rosters['historical'] != original or any(listing.keys() != original.keys() for listing in [actions, *rosters.values()]):
        raise ValueError('Sealed lists differ from the archived historical roster or query population')
    known = {mid: {identity(smiles) for smiles in listing} for mid, listing in original.items()}
    source_path = pinned(artifact(inputs, BASE, 'source_candidates.parquet'), PINS['source_candidates.parquet'])
    source_rows = pq.read_table(source_path, columns=['molecule_id', 'candidate_id', 'source', 'smiles']).to_pylist()
    raw_path = pinned(artifact(inputs, BASE, 'pc_lists.json'), PINS['pc_lists.json'])
    pc = json.loads(raw_path.read_bytes())
    if pc.keys() != original.keys() or any(len(row['pc']) != 25 for row in pc.values()):
        raise ValueError('Archived original PubChem heads changed')
    for row in source_rows:
        mid, key = row['molecule_id'], identity(row['smiles'])
        if mid not in known or key is None or key[0] != row['candidate_id']:
            raise ValueError('Archived original source inventory failed competition normalization')
        known[mid].add(key)
    swaps = dict.fromkeys(ARMS[1:], 0)
    paired = 0
    for mid, before in original.items():
        known[mid].update(identity(text) for text in pc[mid]['pc'])
        if None in known[mid]:
            raise ValueError(f'An original candidate had no competition identity: {mid}')
        ledger = actions[mid]
        if set(ledger) != set(ARMS[1:]) or not 0 <= ledger['c34']['logical_nominees'] == ledger['fr']['logical_nominees'] <= 4:
            raise ValueError(f'Unequal prediction candidates for {mid}')
        paired += bool(ledger['c34']['logical_nominees'])
        for arm in ARMS[1:]:
            after = rosters[arm][mid]
            evidence = ledger[arm]
            action = evidence['action']
            if len(after) != 25 or after[:4] != before[:4] or after[5:] != before[5:]:
                raise ValueError(f'Sealed roster changed outside rank 5: {mid}/{arm}')
            if bool(action) != (after[4] != before[4]) or not 0 <= evidence['valid_nominees'] <= evidence['logical_nominees']:
                raise ValueError(f'Sealed rank-5 action differs from its final roster: {mid}/{arm}')
            if action:
                key = identity(after[4])
                deltas = action['difference']
                if (key is None or key in known[mid] or formula(after[4]) != formula(before[4])
                        or action['slot'] != 5 or action['challenger'] != after[4] or action['incumbent'] != before[4]
                        or action['formula'] != formula(before[4]) or not 1 <= action['shortlist_rank'] <= evidence['logical_nominees']
                        or not evidence['valid_nominees'] or set(deltas) != {'ice', 'gl'}
                        or not all(isfinite(value) and value > 0 for value in deltas.values())):
                    raise ValueError(f'Sealed action is not a novel, dual-supported same-formula entrant: {mid}/{arm}')
                swaps[arm] += 1
    if seal['swaps'] != swaps or seal['paired_queries'] != paired:
        raise ValueError('Sealed treatment counts differ from the per-query ledger')
    return rosters, known, Counter(row['source'] for row in source_rows), actions


def reciprocal(position):
    return 1 / position if position else 0.


def score_summary(rows):
    metrics, comparisons = {}, {}
    for arm in ARMS:
        positions = [row[f'{arm}_rank'] for row in rows]
        metrics[arm] = dict(top1=positions.count(1), top5=sum(0 < v <= 5 for v in positions),
                            top25=sum(v > 0 for v in positions), mrr25=fsum(map(reciprocal, positions)) / len(rows))
    for arm, control in PAIRS:
        pair = f'{arm}_vs_{control}'
        comparison = dict(delta_mrr25=fsum(row[f'{pair}_delta_rr'] for row in rows) / len(rows),
                          improved=[r['molecule_id'] for r in rows if r[f'{pair}_delta_rr'] > 0],
                          worsened=[r['molecule_id'] for r in rows if r[f'{pair}_delta_rr'] < 0],
                          tied=sum(r[f'{pair}_delta_rr'] == 0 for r in rows),
                          newly_recalled_all_source_misses=[r['molecule_id'] for r in rows if
                              not r['seen_in_original_union'] and not r[f'{control}_rank'] and r[f'{arm}_rank']])
        for cutoff in (1, 5, 25):
            comparison[f'new_top{cutoff}'] = [r['molecule_id'] for r in rows if
                not 0 < r[f'{control}_rank'] <= cutoff and 0 < r[f'{arm}_rank'] <= cutoff]
            comparison[f'lost_top{cutoff}'] = [r['molecule_id'] for r in rows if
                0 < r[f'{control}_rank'] <= cutoff and not 0 < r[f'{arm}_rank'] <= cutoff]
        comparisons[pair] = comparison
    return metrics, comparisons


def score(inputs, output, reference, digest):
    import pyarrow.parquet as pq
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    RDLogger.DisableLog('rdApp.*')
    identity = identity_function()

    def formula(text):
        return CalcMolFormula(Chem.MolFromSmiles(text))

    receipt = pinned(artifact(inputs, reference, SEAL), digest)
    seal = load_seal(receipt)
    rosters, known, source_counts, actions = verify_lists(inputs, receipt, seal, identity, formula)
    # The already archived labels are opened only after verifying all blind outputs.
    label_path = pinned(artifact(inputs, LABELS, 'labels.parquet'), LABELS_SHA)
    labels = pq.read_table(label_path, columns=['molecule_id', 'candidate_id', 'truth_smiles']).to_pylist()
    population = {r['molecule_id'] for r in labels}
    if len(labels) != 110 or population != known.keys():
        raise ValueError('MassBank truth table differs from the archived 110-query population')
    rows = []
    for record in sorted(labels, key=lambda row: row['molecule_id']):
        mid = record['molecule_id']
        truth = identity(record['truth_smiles'])
        if truth is None or truth[0] != record['candidate_id']:
            raise ValueError(f'Archived truth differs from competition normalization: {mid}')
        ranks = {arm: rank(rosters[arm][mid], truth, identity) for arm in ARMS}
        row = dict(molecule_id=mid, seen_in_original_union=truth in known[mid],
                   historical_rank5_smiles=rosters['historical'][mid][4])
        for arm in ARMS:
            row[f'{arm}_rank'] = ranks[arm]
            if arm != 'historical':
                smiles = rosters[arm][mid][4]
                row[f'{arm}_rank5_smiles'] = smiles
                row[f'{arm}_replaced'] = smiles != row['historical_rank5_smiles']
                for name in ('logical_nominees', 'valid_nominees', 'reason'):
                    row[f'{arm}_{name}'] = actions[mid][arm][name]
        for arm, control in PAIRS:
            pair = f'{arm}_vs_{control}'
            row[f'{pair}_delta_rr'] = reciprocal(ranks[arm]) - reciprocal(ranks[control])
            row[f'{pair}_rank_gain'] = (ranks[control] or 26) - (ranks[arm] or 26)
        rows.append(row)
    metrics, comparisons = score_summary(rows)
    history = metrics['historical']
    misses = sum(not row['seen_in_original_union'] for row in rows)
    if ([history[key] for key in ('top1', 'top5', 'top25')] != [33, 41, 58]
            or abs(history['mrr25'] - BASELINE_MRR) > 1e-12 or misses != 45):
        raise ValueError('Independent score differs from the archived baseline or original union')
    output.mkdir(exist_ok=True, parents=True)
    per_query = output / PER_QUERY
    with per_query.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(status='scored', run=reference, run_receipt_sha256=digest, labels_sha256=LABELS_SHA,
                  historical_final_sha256=HISTORICAL_SHA, original_source_counts=source_counts,
                  molecules=len(rows), candidate_slots=2750, original_source_misses=misses,
                  paired_queries=seal['paired_queries'], swaps=seal['swaps'],
                  logical_requests_per_arm_and_model=seal['logical_requests_per_arm_and_model'],
                  shared_actual_requests_per_model=seal['shared_actual_requests_per_model'],
                  arms=metrics, comparisons=comparisons, per_query_sha256=checksum(per_query))
    (output / SUMMARY).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False), flush=True)


def build(seal, out, runner_version=1):
    load_seal(seal)
    reference = f'jiyujia/{RUNNER}/{runner_version}'
    constants = ('ARMS', 'PAIRS', 'BASE', 'BASELINE_MRR', 'LABELS', 'LABELS_SHA', 'PINS',
                 'SEAL', 'ACTIONS', 'SUMMARY', 'PER_QUERY', 'HISTORICAL_SHA')
    runtime = ('from collections import Counter\nimport csv\nfrom functools import lru_cache\n'
               'from hashlib import file_digest\nimport json\nfrom math import fsum, isfinite\nfrom pathlib import Path\nimport re\n\n'
               + '\n'.join(f'{key} = {globals()[key]!r}' for key in constants)
               + '\n\n' + '\n'.join(map(getsource, (identity_function, artifact, checksum, pinned, read_baseline, rank,
                                           load_seal, verify_lists, reciprocal, score_summary, score))))
    compile(runtime, 'scorer.py', 'exec')
    payload, digest = pack(runtime.encode())
    code = f'''
import base64, subprocess, sys, types, zlib
from hashlib import sha256
from pathlib import Path
inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}
body = zlib.decompress(base64.b64decode({payload!r}))
if sha256(body).hexdigest() != {digest!r}:
    raise ValueError('Embedded scoring source checksum differs')
scope = types.ModuleType('scorer')
exec(compile(body, 'scorer.py', 'exec'), scope.__dict__)
scope.score(inputs, Path('/kaggle/working'), {reference!r}, {checksum(seal)!r})
'''
    compile(code, 'score.py', 'exec')
    metadata = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='score.py', language='python', kernel_type='script',
                    is_private=True, enable_gpu=False, enable_internet=False, dataset_sources=[WHEEL],
                    kernel_sources=[reference, BASE, LABELS], competition_sources=[], docker_image=IMAGE)
    out.mkdir(exist_ok=True, parents=True)
    (out / 'score.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return checksum(out / 'score.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runner-version', type=int, default=1)
    print(build(**vars(parser.parse_args())))
