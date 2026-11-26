"""Build a private scorer for sealed cross-formula swaps using legacy rank indices."""

import argparse
import ast
import csv
from hashlib import sha256
import json
from math import fsum
from pathlib import Path

from build_final_native_top5_e180_score import BASE, BASE_SHA, OLD_RANKS_SHA, SCORED, mounted, rr, totals, verified


BLIND = 'enveda-075-final-native-formula-e180'
SEAL = 'final_native_formula_e180_receipt.json'
ARMS = ('top5', 'dual_single', 'dual_replicated')
TOP5_SHA = '0d2b52c1f14e1ed060ad8eaea44a9e3cd7eef54c6cf2cf3389732e9b59e54339'
QUERIES_SHA = '6fe782c57481b6fea95d1197da05ce4968c393881e3d962bcf8cdaed09b3bf7b'


def read_seal(path):
    seal = json.loads(path.read_bytes())
    expected = dict(status='sealed', molecules=121, candidate_budget=25, actual_slots=3025,
                    new_predictor_requests=0, base_submission_sha256=BASE_SHA, queries_sha256=QUERIES_SHA,
                    arm_names=list(ARMS), first_prefix=10)
    if (any(seal.get(field) != value for field, value in expected.items())
            or seal['files']['top5_final.json'] != TOP5_SHA):
        raise ValueError('Unexpected formula study or changed reference top-five ordering')
    return seal


def check_lists(original, rankings):
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    top5 = rankings['top5']
    if (len(original) != 121 or any(len(candidates) != 25 for candidates in original.values())
            or any(set(arm) != set(original) for arm in rankings.values())):
        raise ValueError('Candidate budget or query population changed')
    for mid, before in top5.items():
        if sorted(before) != sorted(original[mid]):
            raise ValueError(f'Sealed top-five list has different structures from its original: {mid}')
        for arm in ARMS[1:]:
            after = rankings[arm][mid]
            if len(after) != len(before):
                raise ValueError(f'Candidate length changed: {arm}/{mid}')
            changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
            if not changed:
                continue
            if (len(changed) != 2 or changed[0] != 0 or changed[1] >= 10
                    or before[0] != after[changed[1]] or before[changed[1]] != after[0]):
                raise ValueError(f'Expected one head swap within the first 10 candidates: {arm}/{mid}')
            formulas = [CalcMolFormula(Chem.MolFromSmiles(text)) for text in (before[0], after[0])]
            if formulas[0] == formulas[1]:
                raise ValueError(f'Sealed swap did not change the top formula: {arm}/{mid}')


def score(inputs, out, seal_sha):
    import pandas as pd
    from rdkit import RDLogger
    from original_e180_metric import candidate_key_function

    RDLogger.DisableLog('rdApp.*')
    seal_path = verified(mounted(inputs, BLIND, SEAL), seal_sha)
    seal = read_seal(seal_path)
    rankings = {}
    for arm in ARMS:
        name = arm + '_final.json'
        rankings[arm] = json.loads(verified(seal_path.parent / name, seal['files'][name]).read_bytes())
    baseline = verified(mounted(inputs, BASE, 'submission.csv'), BASE_SHA)
    with baseline.open(newline='') as stream:
        original_rows = list(csv.DictReader(stream))
    original = {row['molecule_id']: row['smiles'].split(';') for row in original_rows}
    if len(original_rows) != 121:
        raise ValueError('Unexpected original submission row count')
    check_lists(original, rankings)

    # Existing rank indices stay outside the blind notebook and enter after its files are verified.
    past = verified(mounted(inputs, SCORED, 'e180_pair_ranks.csv'), OLD_RANKS_SHA)
    archived = pd.read_csv(past, usecols=['molecule_id', 'control'], dtype={'molecule_id': str, 'control': 'Int64'})
    if not archived.molecule_id.is_unique or set(archived.molecule_id) != set(original):
        raise ValueError('Different population in historical scoring archive')
    key = candidate_key_function()
    rows = []
    for mid, old in archived.sort_values('molecule_id').itertuples(index=False):
        before = 0 if pd.isna(old) else int(old)
        ranks = dict.fromkeys(ARMS, 0)
        if before:
            if not 1 <= before <= 25 or (hit := key(original[mid][before - 1].strip())) is None:
                raise ValueError(f'Invalid original match for {mid}')
            if any(key(text.strip()) == hit for text in original[mid][:before - 1]):
                raise ValueError(f'Expected the historical first match: {mid}')
            for arm in ARMS:
                ranks[arm] = next((i for i, text in enumerate(rankings[arm][mid], 1) if key(text.strip()) == hit), 0)
            if not all(ranks.values()):
                raise ValueError(f'Candidate membership changed for {mid}')
        rows.append(dict(molecule_id=mid, historical_rank=before, **{arm + '_rank': rank for arm, rank in ranks.items()},
                         **{arm + '_delta_rr': rr(ranks[arm]) - rr(ranks['top5']) for arm in ARMS[1:]}))

    metrics = {arm: totals(rows, arm) for arm in ('historical', *ARMS)}
    if (abs(metrics['historical']['mrr25'] - 0.5761142047602913) > 1e-12
            or abs(metrics['top5']['mrr25'] - 0.6342409265233767) > 1e-12):
        raise ValueError('Original and reference top-five metrics did not reproduce')
    comparisons = {}
    for arm in ARMS[1:]:
        cases = [(row['molecule_id'], row['top5_rank'], row[arm + '_rank']) for row in rows]
        result = dict(delta_mrr25=fsum(rr(b) - rr(a) for _, a, b in cases) / len(cases),
                      changed_first=[mid for mid in original if rankings['top5'][mid][0] != rankings[arm][mid][0]])
        for k in (1, 5):
            result[f'new_top{k}'] = [mid for mid, a, b in cases if not 0 < a <= k and 0 < b <= k]
            result[f'lost_top{k}'] = [mid for mid, a, b in cases if 0 < a <= k and not 0 < b <= k]
        comparisons[arm] = result

    out.mkdir(exist_ok=True, parents=True)
    per = out / 'final_native_formula_e180_per_query.csv'
    with per.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(arms=metrics, comparisons_vs_top5=comparisons, candidate_slots=3025, new_predictor_requests=0,
                  blind_receipt_sha256=seal_sha, archived_ranks_sha256=OLD_RANKS_SHA,
                  per_query_sha256=sha256(per.read_bytes()).hexdigest())
    (out / 'final_native_formula_e180_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def build(seal, out):
    from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack

    read_seal(seal)
    seal_sha = sha256(seal.read_bytes()).hexdigest()
    program = (HERE.parent / 'pipeline_candidates/first_formula/score_blind_pair.py').read_text()
    node, = (n for n in ast.parse(program).body if isinstance(n, ast.FunctionDef) and n.name == 'candidate_key_function')
    payload = {name: pack(body) for name, body in {
        'original_e180_metric.py': ('from functools import lru_cache\n' + ast.get_source_segment(program, node) + '\n').encode(),
        'build_final_native_top5_e180_score.py': (HERE / 'build_final_native_top5_e180_score.py').read_bytes(),
        'score_formula.py': Path(__file__).read_bytes(),
    }.items()}
    code = f'''
import base64, subprocess, sys, tempfile, zlib
from hashlib import sha256
from pathlib import Path
inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='e180-final-formula-score-') as tmp:
    root = Path(tmp)
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Changed embedded scoring source: {{name}}')
        (root / name).write_bytes(raw)
    sys.path.insert(0, str(root))
    from score_formula import score
    score(inputs, Path('/kaggle/working'), {seal_sha!r})
'''
    slug = BLIND + '-score'
    compile(code, slug, 'exec')
    meta = dict(id='jiyujia/' + slug, title=slug, code_file='score.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, dataset_sources=[WHEEL],
                kernel_sources=[f'jiyujia/{BLIND}/1', f'jiyujia/{BASE}/2', f'jiyujia/{SCORED}/1'],
                competition_sources=[], docker_image=IMAGE)
    out.mkdir(exist_ok=True, parents=True)
    (out / 'score.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    build(**vars(parser.parse_args()))
