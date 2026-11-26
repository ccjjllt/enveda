"""Score sealed deeper same-formula permutations with archived E180 rank indices."""

import argparse
import ast
from collections import Counter
import csv
from hashlib import sha256
import json
from math import fsum
from pathlib import Path

from build_final_native_top5_e180_score import BASE, BASE_SHA, OLD_RANKS_SHA, SCORED, mounted, rr, totals, verified


BLIND = 'enveda-075-final-native-depth-e180'
SEAL = 'final_native_depth_e180_receipt.json'
DEPTHS = {'historical': 0, 'top5': 5, 'top10': 10, 'top25': 25}
TOP5_SHA = '0d2b52c1f14e1ed060ad8eaea44a9e3cd7eef54c6cf2cf3389732e9b59e54339'
QUERIES_SHA = '6fe782c57481b6fea95d1197da05ce4968c393881e3d962bcf8cdaed09b3bf7b'


def read_seal(path):
    seal = json.loads(path.read_bytes())
    expected = dict(status='sealed', molecules=121, candidate_budget=25, actual_slots=3025,
                    new_predictor_requests=0, base_submission_sha256=BASE_SHA, queries_sha256=QUERIES_SHA,
                    arm_names=list(DEPTHS))
    if (any(seal.get(field) != value for field, value in expected.items())
            or seal['files']['top5_final.json'] != TOP5_SHA):
        raise ValueError('Unexpected depth seal or changed reference top-five ordering')
    return seal


def score(inputs, out, seal_sha):
    import pandas as pd
    from rdkit import RDLogger
    from original_e180_metric import candidate_key_function

    RDLogger.DisableLog('rdApp.*')
    seal_path = verified(mounted(inputs, BLIND, SEAL), seal_sha)
    seal = read_seal(seal_path)
    baseline_path = verified(mounted(inputs, BASE, 'submission.csv'), BASE_SHA)
    with baseline_path.open(newline='') as stream:
        records = list(csv.DictReader(stream))
    original = {row['molecule_id']: row['smiles'].split(';') for row in records}
    if len(records) != 121 or len(original) != 121 or any(len(values) != 25 for values in original.values()):
        raise ValueError('Original 121-query, 25-slot archive is incomplete')
    rankings = {}
    for arm, depth in DEPTHS.items():
        name = arm + '_final.json'
        values = json.loads(verified(seal_path.parent / name, seal['files'][name]).read_bytes())
        if set(values) != set(original) or any(values[mid][depth:] != old[depth:]
                                                or Counter(values[mid][:depth]) != Counter(old[:depth])
                                                for mid, old in original.items()):
            raise ValueError(f'{arm}: original candidates or ranks outside the tested depth changed')
        rankings[arm] = values

    past = verified(mounted(inputs, SCORED, 'e180_pair_ranks.csv'), OLD_RANKS_SHA)
    archived = pd.read_csv(past, usecols=['molecule_id', 'control'], dtype={'molecule_id': str, 'control': 'Int64'})
    if not archived.molecule_id.is_unique or set(archived.molecule_id) != set(original):
        raise ValueError('Different population in historical scoring archive')
    key = candidate_key_function()
    rows = []
    for mid, old in archived.sort_values('molecule_id').itertuples(index=False):
        before = 0 if pd.isna(old) else int(old)
        ranks = dict.fromkeys(DEPTHS, 0)
        if before:
            if not 1 <= before <= 25 or (hit := key(original[mid][before - 1])) is None:
                raise ValueError(f'Invalid historical matching identity for {mid}')
            for arm in DEPTHS:
                ranks[arm] = next((i for i, text in enumerate(rankings[arm][mid], 1) if key(text) == hit), 0)
            if ranks['historical'] != before or not all(ranks.values()):
                raise ValueError(f'Changed earliest match or candidate membership for {mid}')
        rows.append(dict(molecule_id=mid, **{arm + '_rank': rank for arm, rank in ranks.items()},
                         **{arm + '_vs_top5_delta_rr': rr(ranks[arm]) - rr(ranks['top5']) for arm in ('top10', 'top25')}))

    arms = {arm: totals(rows, arm) for arm in DEPTHS}
    if (abs(arms['historical']['mrr25'] - 0.5761142047602913) > 1e-12
            or abs(arms['top5']['mrr25'] - 0.6342409265233767) > 1e-12):
        raise ValueError('Reference rankings did not reproduce their measured MRR')
    comparisons = {}
    for control, arm in [('historical', 'top5'), ('historical', 'top10'), ('historical', 'top25'),
                         ('top5', 'top10'), ('top5', 'top25')]:
        ranks = [(row['molecule_id'], row[control + '_rank'], row[arm + '_rank']) for row in rows]
        comparison = dict(delta_mrr25=fsum(rr(b) - rr(a) for _, a, b in ranks) / len(rows))
        for k in (1, 5):
            comparison[f'new_top{k}'] = [mid for mid, a, b in ranks if not 0 < a <= k and 0 < b <= k]
            comparison[f'lost_top{k}'] = [mid for mid, a, b in ranks if 0 < a <= k and not 0 < b <= k]
        comparisons[f'{arm}_vs_{control}'] = comparison

    out.mkdir(exist_ok=True, parents=True)
    per = out / 'final_native_depth_e180_per_query.csv'
    with per.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = dict(arms=arms, comparisons=comparisons, candidate_slots=3025, new_predictor_requests=0,
                   blind_receipt_sha256=seal_sha, archived_ranks_sha256=OLD_RANKS_SHA,
                   per_query_sha256=sha256(per.read_bytes()).hexdigest())
    (out / 'final_native_depth_e180_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(arms), flush=True)


def build(seal, out):
    from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack

    read_seal(seal)
    seal_sha = sha256(seal.read_bytes()).hexdigest()
    metric_program = (HERE.parent / 'pipeline_candidates/first_formula/score_blind_pair.py').read_text()
    node, = (n for n in ast.parse(metric_program).body if isinstance(n, ast.FunctionDef)
             and n.name == 'candidate_key_function')
    metric = 'from functools import lru_cache\n' + ast.get_source_segment(metric_program, node) + '\n'
    payload = {name: pack(data) for name, data in {
        'original_e180_metric.py': metric.encode(),
        'build_final_native_top5_e180_score.py': (HERE / 'build_final_native_top5_e180_score.py').read_bytes(),
        'score_depth.py': Path(__file__).read_bytes(),
    }.items()}
    code = f'''
import base64, subprocess, sys, tempfile, zlib
from hashlib import sha256
from pathlib import Path
inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='e180-final-depth-score-') as tmp:
    root = Path(tmp)
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Changed embedded scoring source: {{name}}')
        (root / name).write_bytes(raw)
    sys.path.insert(0, str(root))
    from score_depth import score
    score(inputs, Path('/kaggle/working'), {seal_sha!r})
'''
    slug = BLIND + '-score'
    compile(code, slug, 'exec')
    cell = dict(cell_type='code', id='score-depth', execution_count=None, metadata={}, outputs=[], source=code.splitlines(True))
    notebook = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}},
                    cells=[cell])
    meta = dict(id='jiyujia/' + slug, title=slug, code_file='score.ipynb', language='python', kernel_type='notebook',
                is_private=True, enable_gpu=False, enable_internet=False, dataset_sources=[WHEEL],
                kernel_sources=[f'jiyujia/{BLIND}/1', f'jiyujia/{BASE}/2', f'jiyujia/{SCORED}/1'],
                competition_sources=[], docker_image=IMAGE)
    out.mkdir(exist_ok=True, parents=True)
    (out / 'score.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    build(**vars(parser.parse_args()))
