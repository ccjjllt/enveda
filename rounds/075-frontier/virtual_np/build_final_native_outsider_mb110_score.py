"""Package a private, post-seal score of the fixed MassBank 110 outsider arms."""

import argparse
import ast
from collections import Counter
import csv
from hashlib import file_digest, sha256
import json
from math import fsum, isfinite
from pathlib import Path
import re


BASE = 'jiyujia/enveda-075b-massbank-pipeline/1'
DEPTH = 'jiyujia/enveda-075b-massbank-pubchem-depth/2'
LABELS = 'jiyujia/enveda-075b-external-massbank-202611/1'
LABELS_SHA = '47a176798e7f145b2d537518f98739974f5a736dcf527af623e5d73df3201419'
SUPPLEMENT = 'jiyujia/enveda-075b-massbank-pubchem50/1'
PINS = {
    'submission.csv': 'a2277d569ccf838c11cc2b8fd36cf6bcb1b6565ffbe3c4312017480eaac7c5a1',
    'source_candidates.parquet': '805a7072a85aa53a0901387bd6b4799b98dc919e83577032749f4db19dfc09ac',
    'pc_lists.json': '58092dc1f9db483f2681a7f90efb81d8f18fb6254c7982d90ca042d80c6d8f6c',
    'pc_depth.json': 'ca1042629166b8935e1f08f726586246c763ee31f64de02865c41a19d5363f24',
}
TARGETS = {'rank5': 5, 'rank10': 10}
ARMS = ('historical', *TARGETS)
BASELINE_MRR = 0.3368516738288762


def names(mode):
    suffix = '' if mode == 'stock' else '-depth50'
    return 'enveda-075-final-native-outsider-mb110' + suffix, 'final_native_outsider_mb110' + suffix.replace('-', '_')


def checksum(path):
    with path.open('rb') as reader:
        return file_digest(reader, 'sha256').hexdigest()


def pinned(path, expected):
    if checksum(path) != expected:
        raise ValueError(f'SHA mismatch for {path.name}')
    return path


def artifact(inputs, reference, name):
    slug = reference.split('/')[1]
    path, = (path for path in inputs.rglob(name) if slug in path.parts)
    return path


def read_json(path):
    return json.loads(path.read_bytes())


def validate_seal(path, mode):
    seal = read_json(path)
    _, stem = names(mode)
    expected = dict(status='sealed', arm_names=list(ARMS), molecules=110, candidate_budget=25,
                    actual_slots=2750, new_predictor_requests=0, baseline=BASE, depth=DEPTH,
                    baseline_submission_sha256=PINS['submission.csv'], tail_ranks=[26, 100],
                    targets=TARGETS, original_lengths={'25': 110}, source_library_threshold=0.90)
    for key, value in expected.items():
        if seal.get(key) != value:
            raise ValueError(f'Sealed inventory differs at {key}')
    if (any(seal['input_file_sha256'].get(name) != digest for name, digest in PINS.items())
            or seal.get('scores_reference') != (SUPPLEMENT if mode == 'depth50' else None)):
        raise ValueError('Unexpected candidate archives or physical cache')
    files = {f'{arm}_final.json' for arm in ARMS} | {stem + '_cases.json'}
    if set(seal['files']) != files or any(not re.fullmatch('[0-9a-f]{64}', h) for h in seal['files'].values()):
        raise ValueError('Incomplete sealed rank/action file manifest')
    return seal


def read_baseline(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['molecule_id', 'smiles']:
            raise ValueError('Unexpected historical submission columns')
        rows = list(reader)
    result = {r['molecule_id']: r['smiles'].split(';') for r in rows}
    if len(rows) != 110 or len(result) != 110 or any(len(v) != 25 or len(set(v)) != 25 for v in result.values()):
        raise ValueError('The archived candidate inventory is not 110 × 25 distinct strings')
    return result


def validate_placements(ranks, cases, seal, original, sources, raw_pc, tail, identity, formula):
    if ranks['historical'] != original or any(set(arm) != set(original) for arm in ranks.values()):
        raise ValueError('Sealed arms differ from the historical query population or order')
    if any(population.keys() != original.keys() for population in (cases, sources, raw_pc, tail)):
        raise ValueError('Sealed/source/PubChem population differs from original')
    changed = {arm: [] for arm in TARGETS}
    known = {}
    for mid, before in original.items():
        native_pc = raw_pc[mid]['pc']
        pool = tail[mid]['pc']
        if len(native_pc) != 25 or len(pool) != 100:
            raise ValueError(f'Unexpected historical PubChem slot width: {mid}')
        source_keys = {identity(text) for text in [*before, *native_pc, *sources[mid]]}
        if None in source_keys:
            raise ValueError(f'Unusable historical candidate identity: {mid}')
        known[mid] = source_keys
        for arm, slot in TARGETS.items():
            after, row = ranks[arm][mid], cases[mid]
            action = row['actions'][arm]
            differences = [i for i, (a, b) in enumerate(zip(before, after, strict=True), 1) if a != b]
            if differences != ([slot] if action is not None else []):
                raise ValueError(f'{mid}/{arm} changed something outside its sealed replacement slot')
            if action is None:
                continue
            entrant, incumbent = after[slot - 1], before[slot - 1]
            rank = action['shortlist_rank']
            deltas = action['difference']
            key = identity(entrant)
            if (action['slot'] != slot or action['challenger'] != entrant or action['incumbent'] != incumbent
                    or not 26 <= rank <= 100 or entrant != pool[rank - 1]
                    or key is None or key in source_keys or formula(entrant) != formula(incumbent)
                    or formula(incumbent) != action['formula'] or not row['same_nonempty_native_spectra']
                    or row['proxy_gl'] or not row['library_max'] < seal['source_library_threshold']
                    or set(deltas) != {'ice', 'gl'} or not all(isfinite(x) and x > 0 for x in deltas.values())):
                raise ValueError(f'Invalid sealed outsider action: {mid}/{arm}')
            changed[arm].append(mid)
    if any(sorted(mids) != sorted(seal['arms'][arm]) for arm, mids in changed.items()):
        raise ValueError('Sealed action IDs disagree with the final lists')
    return known, changed


def rank(texts, truth, identity):
    return next((i for i, text in enumerate(texts, 1) if identity(text) == truth), 0)


def score_summary(rows):
    def rr(position):
        return 1 / position if position else 0.

    metrics, comparisons = {}, {}
    for arm in ARMS:
        values = [row[arm + '_rank'] for row in rows]
        metrics[arm] = dict(molecules=len(rows), top1=values.count(1), top5=sum(0 < i <= 5 for i in values),
                            top25=sum(i > 0 for i in values), mrr25=fsum(map(rr, values)) / len(rows))
        if arm == 'historical':
            continue
        compare = dict(delta_mrr25=metrics[arm]['mrr25'] - metrics['historical']['mrr25'],
                       newly_recalled_all_source_misses=[r['molecule_id'] for r in rows
                                                         if not r['seen_in_original_union'] and r[arm + '_rank'] > 0])
        for cut in (1, 5, 25):
            compare[f'new_top{cut}'] = [r['molecule_id'] for r in rows if
                                      not 0 < r['historical_rank'] <= cut and 0 < r[arm + '_rank'] <= cut]
            compare[f'lost_top{cut}'] = [r['molecule_id'] for r in rows if
                                       0 < r['historical_rank'] <= cut and not 0 < r[arm + '_rank'] <= cut]
        comparisons[arm] = compare
    return metrics, comparisons


def score(inputs, out, mode, seal_sha):
    import pyarrow.parquet as pq
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from outsider_metric import identity_function

    RDLogger.DisableLog('rdApp.*')
    identity = identity_function()
    slug, stem = names(mode)
    blind_ref = f'jiyujia/{slug}/1'
    receipt = pinned(artifact(inputs, blind_ref, stem + '_receipt.json'), seal_sha)
    seal = validate_seal(receipt, mode)
    files = {name: pinned(receipt.parent / name, digest) for name, digest in seal['files'].items()}
    ranks = {arm: read_json(files[f'{arm}_final.json']) for arm in ARMS}
    cases = read_json(files[stem + '_cases.json'])
    paths = {name: pinned(artifact(inputs, DEPTH if name == 'pc_depth.json' else BASE, name), digest)
             for name, digest in PINS.items()}
    original = read_baseline(paths['submission.csv'])
    source_rows = pq.read_table(paths['source_candidates.parquet'], columns=['molecule_id', 'source', 'smiles']).to_pylist()
    counts = Counter(row['source'] for row in source_rows)
    if counts != seal['source_channels']:
        raise ValueError('Source export counts disagree with the blind seal')
    sources = {mid: [] for mid in original}
    for row in source_rows:
        sources[row['molecule_id']].append(row['smiles'])

    def formula(text):
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            raise ValueError('A sealed action has unusable SMILES')
        return CalcMolFormula(mol)

    known, changed = validate_placements(ranks, cases, seal, original, sources, read_json(paths['pc_lists.json']),
                                        read_json(paths['pc_depth.json']), identity, formula)

    # Only the independent scorer reads the already archived evaluation labels, after roster checks.
    label_path = pinned(artifact(inputs, LABELS, 'labels.parquet'), LABELS_SHA)
    label_rows = pq.read_table(label_path, columns=['molecule_id', 'candidate_id', 'truth_smiles']).to_pylist()
    labels = {record['molecule_id']: record for record in label_rows}
    if len(label_rows) != 110 or labels.keys() != original.keys():
        raise ValueError('Archived labels do not describe the 110 frozen queries')
    rows = []
    for mid, record in sorted(labels.items()):
        truth = identity(record['truth_smiles'])
        if truth is None or truth[0] != record['candidate_id']:
            raise ValueError(f'Historical label fails competition normalization: {mid}')
        rows.append(dict(molecule_id=mid, seen_in_original_union=truth in known[mid],
                         **{arm + '_rank': rank(ranks[arm][mid], truth, identity) for arm in ARMS}))
    metrics, comparisons = score_summary(rows)
    original_score = metrics['historical']
    if ([original_score[key] for key in ('top1', 'top5', 'top25')] != [33, 41, 58]
            or abs(original_score['mrr25'] - BASELINE_MRR) > 1e-12):
        raise ValueError('Independent score did not reproduce the archived historical control')
    for arm, comparison in comparisons.items():
        comparison['trigger_ids'] = changed[arm]
    out.mkdir(parents=True, exist_ok=True)
    per_query = out / (stem + '_per_query.csv')
    with per_query.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(mode=mode, panel=LABELS, candidate_slots=2750, new_predictor_requests=0, arms=metrics,
                  comparisons_vs_historical=comparisons, original_source_misses=sum(not row['seen_in_original_union'] for row in rows),
                  blind_receipt_sha256=seal_sha, labels_sha256=LABELS_SHA, per_query_sha256=checksum(per_query))
    (out / (stem + '_metrics.json')).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False), flush=True)


def build(seal, out, mode):
    from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack

    validate_seal(seal, mode)
    slug, _ = names(mode)
    runner = 'enveda-075-mb110-depth50-score' if mode == 'depth50' else slug + '-score'
    identity_body = (HERE / 'c39_request_cache.py').read_text()
    node, = (item for item in ast.parse(identity_body).body if isinstance(item, ast.FunctionDef) and item.name == 'identity_function')
    metric = 'from collections import Counter\nfrom functools import lru_cache\n\n' + ast.get_source_segment(identity_body, node)
    payload = {name: pack(raw) for name, raw in {
        'outsider_metric.py': metric.encode(),
        'score_outsider.py': Path(__file__).read_bytes(),
    }.items()}
    code = f'''
import base64, subprocess, sys, tempfile, zlib
from hashlib import sha256
from pathlib import Path
inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='massbank110-outsider-score-') as tmp:
    root = Path(tmp)
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Changed embedded scoring source: {{name}}')
        (root / name).write_bytes(raw)
    sys.path.insert(0, str(root))
    from score_outsider import score
    score(inputs, Path('/kaggle/working'), {mode!r}, {checksum(seal)!r})
'''
    compile(code, runner, 'exec')
    meta = dict(id=f'jiyujia/{runner}', title=runner, code_file='run_score.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, dataset_sources=[WHEEL],
                kernel_sources=[f'jiyujia/{slug}/1', BASE, DEPTH, LABELS], competition_sources=[], docker_image=IMAGE)
    out.mkdir(exist_ok=True, parents=True)
    (out / 'run_score.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return sha256(code.encode()).hexdigest()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--mode', choices=('stock', 'depth50'), required=True)
    cli.add_argument('--seal', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    print(build(**vars(cli.parse_args())))
