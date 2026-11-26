"""Package independent scoring of sealed MassBank 110 rank-5 request budgets."""

import argparse
import ast
from collections import Counter, namedtuple
import csv
from hashlib import sha256
from inspect import getsource
import json
from math import fsum, isclose, isfinite
from pathlib import Path
import re

from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack
from build_final_native_outsider_mb110_depth50 import SUPPLEMENT, SUPPLEMENT_PINS, spectral_array_bytes
from build_mb110_rank5_budget_replay import (
    CAPS, MODELS, PER_QUERY as REPLAY_ROWS, REFERENCE, REFERENCE_SHA, SEAL as REFERENCE_FILE,
    SLUG as REPLAY_SLUG, SUMMARY as REPLAY_RECEIPT, admission, budget,
)
from build_final_native_outsider_mb110_score import (
    BASE, BASELINE_MRR, DEPTH, LABELS, LABELS_SHA, PINS, artifact, checksum, pinned, rank, read_baseline,
)


SLUG = 'enveda-075-mb110-rank5-budget-score'
REPLAY = f'jiyujia/{REPLAY_SLUG}/1'
SUMMARY = 'mb110_rank5_budget_score.json'
PER_QUERY = 'mb110_rank5_budget_score_per_query.csv'
PHYSICAL_IN_SHA = '03d101f53cdd58b4aa8894409fad731c7cb5f78001a8486c6c92a6ad52dbbe4b'
LEDGER_SHA = '0dcc671e581b5632d971c4422d8a748533dc688eab2d53ef5a41b923589d29f8'
EXPECTED_RANK5 = 0.33866985564705804
ARMS = ('historical', *CAPS)
Candidate = namedtuple('Candidate', ('rank', 'smiles'))


def load_seal(path):
    seal = json.loads(path.read_bytes())
    expected = dict(status='sealed', reference=REFERENCE, reference_receipt_sha256=REFERENCE_SHA,
                    baseline=BASE, candidate_archive=DEPTH, physical_archive=SUPPLEMENT,
                    molecules=110, candidate_budget=25, actual_slots=2750, slot=5,
                    candidate_ranks=[26, 50], predictor_requests_run=0)
    if any(seal[key] != value for key, value in expected.items()) or set(seal['arms']) != set(CAPS):
        raise ValueError('Unexpected replay provenance or fixed roster constraints')
    names = {f'{arm}_final.json' for arm in ARMS} | {REPLAY_ROWS}
    if set(seal['files']) != names or any(not re.fullmatch('[0-9a-f]{64}', h) for h in seal['files'].values()):
        raise ValueError('Incomplete blind replay file seals')
    for arm, cap in CAPS.items():
        if seal['arms'][arm]['cap'] != cap or set(seal['arms'][arm]['requests_by_model']) != set(MODELS):
            raise ValueError(f'Unexpected candidate cap or models: {arm}')
    return seal


def physical_requests(inputs, original, seal):
    requests, byte_spans = {}, {}
    for reference in (BASE, SUPPLEMENT):
        requests[reference], byte_spans[reference] = {}, {}
        for model in MODELS:
            name = f'{model}_in.json'
            digest = PHYSICAL_IN_SHA if reference == BASE else SUPPLEMENT_PINS[f'{model}_work/{name}']
            slug = reference.split('/')[1]
            path, = (p for p in inputs.rglob(name) if slug in p.parts and p.parent.name == f'{model}_work')
            path = pinned(path, digest)
            records = json.loads(path.read_bytes())
            keyed = {row['mid']: row for row in records}
            if len(records) != len(original) or keyed.keys() != original.keys() or any(
                    not row['spectra'] or len(row['cands']) != len(set(row['cands'])) for row in records):
                raise ValueError(f'Invalid pinned physical request population: {reference}/{model}')
            requests[reference][model] = keyed
            byte_spans[reference][model] = spectral_array_bytes(path)
    for mid in original:
        if len({byte_spans[reference][model][mid] for reference in requests for model in MODELS}) != 1:
            raise ValueError(f'The four accepted spectra arrays differ: {mid}')
    for model in MODELS:
        old, new = requests[BASE][model], requests[SUPPLEMENT][model]
        cost = seal['historical_cost'][model]
        stock = sum(len(row['cands']) * len(row['spectra']) for row in old.values())
        expanded = sum(len(row['cands']) * len(row['spectra']) for row in new.values())
        added = sum(len(set(new[mid]['cands']) - set(row['cands'])) * len(row['spectra']) for mid, row in old.items())
        removed = sum(len(set(row['cands']) - set(new[mid]['cands'])) * len(row['spectra']) for mid, row in old.items())
        if (stock, expanded, added, removed) != (12855, 16718, 3863, 0) or cost != dict(
                historical_stock_candidate_spectra=stock, historical_depth50_candidate_spectra=expanded,
                historical_added_literal_candidate_spectra=added, historical_removed_literal_candidate_spectra=removed,
                added_with_unchanged_accepted_spectra=added, removed_with_unchanged_accepted_spectra=removed):
            raise ValueError(f'Published historical cost disagrees with pinned requests: {model}')
    return requests[BASE], requests[SUPPLEMENT]


def verify_replay(inputs, original, seal, rankings, rows, identity, formula):
    import pyarrow.parquet as pq

    ref_path = pinned(artifact(inputs, REFERENCE, REFERENCE_FILE), REFERENCE_SHA)
    ref = json.loads(ref_path.read_bytes())
    reference = json.loads(pinned(ref_path.with_name('rank5_final.json'), ref['files']['rank5_final.json']).read_bytes())
    if (rankings['historical'] != original or reference.keys() != original.keys() or len(rows) != 110
            or {row['molecule_id'] for row in rows} != original.keys()
            or any(lists.keys() != original.keys() for lists in rankings.values())):
        raise ValueError('Sealed arms, rows and reference must match the original historical population')
    stock, expanded = physical_requests(inputs, original, seal)
    sources = {mid: [] for mid in original}
    path = pinned(artifact(inputs, BASE, 'source_candidates.parquet'), PINS['source_candidates.parquet'])
    raw = pq.read_table(path, columns=['molecule_id', 'source', 'smiles']).to_pylist()
    if Counter(r['source'] for r in raw) != ref['source_channels']:
        raise ValueError('Original candidate-source counts changed')
    for row in raw:
        sources[row['molecule_id']].append(row['smiles'])
    raw_pc = json.loads(pinned(artifact(inputs, BASE, 'pc_lists.json'), PINS['pc_lists.json']).read_bytes())
    tail = json.loads(pinned(artifact(inputs, DEPTH, 'pc_depth.json'), PINS['pc_depth.json']).read_bytes())
    if raw_pc.keys() != original.keys() or tail.keys() != original.keys():
        raise ValueError('Pinned C34 query populations changed')
    ledger_path = pinned(artifact(inputs, BASE, 'frozen_candidate_evidence.parquet'), LEDGER_SHA)
    ledger = {}
    for item in pq.read_table(ledger_path, columns=['molecule_id', 'library_max', 'proxy_gl', 'processed']).to_pylist():
        mid = item['molecule_id']
        value = item['library_max'], item['proxy_gl']
        if (not item['processed'] or not isfinite(value[0]) or type(value[1]) is not bool
                or ledger.setdefault(mid, value) != value):
            raise ValueError(f'Conflicting historical prediction eligibility: {mid}')
    if ledger.keys() != original.keys():
        raise ValueError('Historical eligibility ledger population changed')

    changed, mismatches = {arm: [] for arm in CAPS}, {arm: [] for arm in CAPS}
    requests = {arm: {model: Counter() for model in MODELS} for arm in CAPS}
    admitted, active = Counter(), Counter()
    source_union, before_labels = {}, {}
    for row in rows:
        mid = row['molecule_id']
        before = original[mid]
        direct, deep = raw_pc[mid]['pc'], tail[mid]['pc']
        if (len(direct) != 25 or len(deep) != 100 or {identity(v) for v in direct} != {identity(v) for v in deep[:25]}):
            raise ValueError(f'Original and C34 tail do not describe the same head: {mid}')
        known = {identity(text) for text in [*before, *direct, *sources[mid]]}
        if None in known:
            raise ValueError(f'Unusable original source identity: {mid}')
        source_union[mid] = known
        seen, pool = set(known), []
        for candidate_rank, text in enumerate(deep[25:50], 26):
            key = identity(text)
            if key is not None and key not in seen:
                pool.append(Candidate(candidate_rank, text))
                seen.add(key)
        incumbent = before[4]
        strength, proxy = ledger[mid]
        eligible = admission(pool, incumbent, formula(incumbent), strength, proxy, True,
                             {model: expanded[model][mid] for model in MODELS},
                             ref['source_library_threshold'], formula)
        if [item.rank for item in eligible] != row['admitted_c34_ranks'] or set(row['budgets']) != set(CAPS):
            raise ValueError(f'Pre-score admission differs from pinned request membership: {mid}')
        checked = {}
        for arm, cap in CAPS.items():
            after, saved = rankings[arm][mid], row['budgets'][arm]
            selected = eligible if cap is None else eligible[:cap]
            winner = saved['winner_c34_rank']
            expected = list(before)
            if winner is not None:
                selected_by_rank = {item.rank: item.smiles for item in selected}
                if winner not in selected_by_rank:
                    raise ValueError(f'Winner is not within sealed admission prefix: {mid}/{arm}')
                expected[4] = selected_by_rank[winner]
                changed[arm].append(mid)
            if after != expected or saved['reproduces_published_rank5'] != (after == reference[mid]):
                raise ValueError(f'Sealed fixed-place final roster does not match archived placement: {mid}/{arm}')
            if after != reference[mid]:
                mismatches[arm].append(mid)
            counts = {model: budget(selected, incumbent, stock[model][mid], expanded[model][mid]) for model in MODELS}
            if counts != saved['requests']:
                raise ValueError(f'Sealed per-query costs differ from archived request arithmetic: {mid}/{arm}')
            for model in MODELS:
                requests[arm][model].update(counts[model])
            admitted[arm] += len(selected)
            active[arm] += bool(selected)
            checked[arm] = (winner, counts)
        before_labels[mid] = checked
    for arm, by_model in requests.items():
        metadata = seal['arms'][arm]
        if (metadata['admitted_query_candidates'] != admitted[arm] or metadata['active_queries'] != active[arm]
                or sorted(metadata['triggered_queries']) != sorted(changed[arm])
                or sorted(metadata['differs_from_published_queries']) != sorted(mismatches[arm])
                or metadata['reproduces_published_rank5'] != (not mismatches[arm])):
            raise ValueError(f'Sealed arm statistics differ from independent roster check: {arm}')
        for model, values in by_model.items():
            expected = dict(values, archived_depth50_added_requests=3863,
                            savings_if_stock_reused=3863 - values['incremental_if_stock_reused'],
                            savings_with_incumbent_rescore=3863 - values['incremental_with_incumbent_rescore'])
            if metadata['requests_by_model'][model] != expected:
                raise ValueError(f'Sealed totals differ from independently verified per-query costs: {arm}/{model}')
    return source_union, before_labels


def summaries(rows):
    metrics, comparisons = {}, {}
    for arm in ARMS:
        column = arm + '_rank'
        values = [row[column] for row in rows]
        metrics[arm] = dict(molecules=len(rows), top1=values.count(1), top5=sum(0 < v <= 5 for v in values),
                            top25=sum(v > 0 for v in values), mrr25=fsum(1 / v for v in values if v) / len(values))
        if arm != 'historical':
            result = dict(delta_mrr25=metrics[arm]['mrr25'] - metrics['historical']['mrr25'],
                          newly_recalled_all_source_misses=[r['molecule_id'] for r in rows
                                                            if not r['seen_in_original_union'] and r[column] > 0])
            for cutoff in (1, 5, 25):
                result[f'new_top{cutoff}'] = [r['molecule_id'] for r in rows
                                             if not 0 < r['historical_rank'] <= cutoff and 0 < r[column] <= cutoff]
                result[f'lost_top{cutoff}'] = [r['molecule_id'] for r in rows
                                              if 0 < r['historical_rank'] <= cutoff and not 0 < r[column] <= cutoff]
            comparisons[arm] = result
    return metrics, comparisons


def score(inputs, out, digest):
    import pyarrow.parquet as pq
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from budget_metric import identity_function

    RDLogger.DisableLog('rdApp.*')
    identity = identity_function()
    receipt = pinned(artifact(inputs, REPLAY, REPLAY_RECEIPT), digest)
    seal = load_seal(receipt)
    files = {name: pinned(receipt.with_name(name), expected) for name, expected in seal['files'].items()}
    rankings = {arm: json.loads(files[f'{arm}_final.json'].read_bytes()) for arm in ARMS}
    rows = json.loads(files[REPLAY_ROWS].read_bytes())
    original = read_baseline(pinned(artifact(inputs, BASE, 'submission.csv'), PINS['submission.csv']))

    def formula(text):
        mol = Chem.MolFromSmiles(text)
        return CalcMolFormula(mol) if mol is not None else None

    known, checked = verify_replay(inputs, original, seal, rankings, rows, identity, formula)
    # Labels are mounted only in this scorer and first read after all fixed-slot and budget checks.
    labels_path = pinned(artifact(inputs, LABELS, 'labels.parquet'), LABELS_SHA)
    raw = pq.read_table(labels_path, columns=['molecule_id', 'candidate_id', 'truth_smiles']).to_pylist()
    labels = {r['molecule_id']: r for r in raw}
    if len(raw) != 110 or labels.keys() != original.keys():
        raise ValueError('Unexpected private MassBank 110 audit labels')
    scored = []
    for mid, record in sorted(labels.items()):
        truth = identity(record['truth_smiles'])
        if truth is None or truth[0] != record['candidate_id']:
            raise ValueError(f'Archived label does not reproduce the metric identity: {mid}')
        row = dict(molecule_id=mid, seen_in_original_union=truth in known[mid],
                   **{arm + '_rank': rank(rankings[arm][mid], truth, identity) for arm in ARMS})
        for arm, (winner, counts) in checked[mid].items():
            row[f'{arm}_winner_c34_rank'] = winner
            for model in MODELS:
                for scenario in ('incremental_if_stock_reused', 'incremental_with_incumbent_rescore'):
                    row[f'{arm}_{model}_{scenario}'] = counts[model][scenario]
        scored.append(row)
    metrics, comparisons = summaries(scored)
    history = metrics['historical']
    replay_equal = seal['arms']['all']['reproduces_published_rank5']
    if ([history[key] for key in ('top1', 'top5', 'top25')] != [33, 41, 58]
            or not isclose(history['mrr25'], BASELINE_MRR, rel_tol=0, abs_tol=1e-12)
            or replay_equal and not isclose(metrics['all']['mrr25'], EXPECTED_RANK5, rel_tol=0, abs_tol=1e-12)):
        raise ValueError('Historical outcome, or a claimed reproduced official rank-5 arm, did not match')
    out.mkdir(parents=True, exist_ok=True)
    per_query = out / PER_QUERY
    with per_query.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    report = dict(status='scored', replay=REPLAY, replay_receipt_sha256=digest, reference=REFERENCE,
                  reference_receipt_sha256=REFERENCE_SHA, historical_mrr_validation=BASELINE_MRR,
                  official_rank5_mrr_validation=EXPECTED_RANK5 if replay_equal else None,
                  original_source_misses=sum(not r['seen_in_original_union'] for r in scored),
                  candidate_slots=2750, predictor_requests_run=0, arms=metrics, comparisons_vs_historical=comparisons,
                  verified_budgets=seal['arms'], historical_cost=seal['historical_cost'], budget_semantics=seal['budget_semantics'],
                  labels_sha256=LABELS_SHA, per_query_sha256=checksum(per_query))
    (out / SUMMARY).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False), flush=True)


def build(seal, out):
    load_seal(seal)
    body = (HERE / 'c39_request_cache.py').read_text()
    node, = (node for node in ast.parse(body).body if isinstance(node, ast.FunctionDef) and node.name == 'identity_function')
    metric = 'from collections import Counter\nfrom functools import lru_cache\n\n' + ast.get_source_segment(body, node)
    constants = ('SLUG', 'REPLAY', 'REFERENCE', 'REFERENCE_SHA', 'REFERENCE_FILE', 'REPLAY_RECEIPT', 'REPLAY_ROWS',
                 'SUMMARY', 'PER_QUERY', 'PHYSICAL_IN_SHA', 'LEDGER_SHA', 'EXPECTED_RANK5', 'ARMS', 'CAPS', 'MODELS',
                 'BASE', 'BASELINE_MRR', 'DEPTH', 'LABELS', 'LABELS_SHA', 'PINS', 'SUPPLEMENT', 'SUPPLEMENT_PINS')
    runtime = ('from collections import Counter, namedtuple\nimport csv\nfrom hashlib import file_digest\n'
               'import json\nfrom math import fsum, isclose, isfinite\nimport re\n\n'
               + '\n'.join(f'{key} = {globals()[key]!r}' for key in constants)
               + "\nCandidate = namedtuple('Candidate', ('rank', 'smiles'))\n\n"
               + '\n'.join(map(getsource, (artifact, checksum, pinned, read_baseline, rank, spectral_array_bytes,
                                            admission, budget, load_seal, physical_requests, verify_replay, summaries, score))))
    modules = {'budget_metric.py': metric, 'score_budget.py': runtime}
    payload = {}
    for name, source in modules.items():
        compile(source, name, 'exec')
        payload[name] = pack(source.encode())
    code = f'''
import base64, subprocess, sys, tempfile, zlib
from hashlib import sha256
from pathlib import Path
inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='massbank110-rank5-budget-score-') as scratch:
    scratch_path = Path(scratch)
    for name, (blob, expected) in {payload!r}.items():
        source = zlib.decompress(base64.b64decode(blob))
        if sha256(source).hexdigest() != expected:
            raise ValueError(f'Changed embedded scorer: {{name}}')
        (scratch_path / name).write_bytes(source)
    sys.path.insert(0, scratch)
    from score_budget import score
    score(inputs, Path('/kaggle/working'), {checksum(seal)!r})
'''
    compile(code, SLUG, 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='score.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, dataset_sources=[WHEEL],
                kernel_sources=[REPLAY, REFERENCE, BASE, DEPTH, SUPPLEMENT, LABELS],
                competition_sources=[], docker_image=IMAGE)
    out.mkdir(exist_ok=True, parents=True)
    (out / 'score.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return sha256(code.encode()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    print(build(**vars(parser.parse_args())))
