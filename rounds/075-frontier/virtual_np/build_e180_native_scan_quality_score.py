"""Score the sealed C41 aggregation diagnostic against archived first-hit ranks."""

import argparse
import ast
from collections import Counter
import csv
from hashlib import sha256
import json
from math import fsum
from pathlib import Path

from build_e180_native_scan_quality import ARCHIVE, ARCHIVE_ARM, ARMS, ORDERS, PINS, RECEIPT, REFERENCE, SLUG
from build_final_native_top5_e180_score import mounted, rr, totals, verified


def read_seal(path):
    seal = json.loads(path.read_bytes())
    expected = dict(status='sealed', arms=list(ARMS), molecules=60, observed_queries=58, observed_scans=197,
                    candidate_budget=25, new_predictor_requests=0, fixed_formula_slots=True,
                    archived_order=ARCHIVE_ARM, archived_sha256=PINS[REFERENCE, ARCHIVE])
    if any(seal.get(field) != value for field, value in expected.items()):
        raise ValueError('Unexpected frozen aggregation comparison')
    return seal


def score(inputs, out, seal_sha):
    from original_connectivity import candidate_key_function

    seal_path = verified(mounted(inputs, SLUG, RECEIPT), seal_sha)
    seal = read_seal(seal_path)
    blind = verified(seal_path.parent / ORDERS, seal['orders_sha256'])
    raw = [json.loads(line) for line in blind.read_bytes().splitlines()]
    ranks = {row['query']: row for row in raw}
    if (len(raw) != len(ranks) or len(ranks) != 60
            or sum(len(row['roster']) for row in raw) != seal['actual_slots']):
        raise ValueError('Wrong frozen comparison rows or candidate budget')
    base = verified(mounted(inputs, REFERENCE, ARCHIVE), PINS[REFERENCE, ARCHIVE])
    original = [json.loads(line) for line in base.read_bytes().splitlines()]
    if len(original) != 60 or len({row['query'] for row in original}) != 60:
        raise ValueError('Wrong historical query population')
    key = candidate_key_function()
    rows = []
    changes, first_changes = Counter(), Counter()
    for item in original:
        mid = item['query']
        con = item['orders'][ARCHIVE_ARM]
        study = ranks[mid]
        roster = [{k: row[k] for k in ('smiles', 'key', 'formula')} for row in item['candidates']]
        orders = study['orders']
        if study['roster'] != roster or con != orders['stock'] or len(roster) > 25 or set(orders) != set(ARMS):
            raise ValueError(f'Historical rankings or roster changed: {mid}')
        target_slots = [roster[i]['formula'] for i in con]
        expected_order = list(range(len(roster)))
        for arm, order in orders.items():
            if sorted(order) != expected_order or [roster[i]['formula'] for i in order] != target_slots:
                raise ValueError(f'Candidate or formula slots changed: {mid}/{arm}')
            changes[arm] += int(order != con)
            first_changes[arm] += int(order[:1] != con[:1])
        # The archived evaluation uses normalized InChIKey14 and the first matching occurrence.
        rank = item['ranks'][ARCHIVE_ARM] or 0
        identities = [key(row['smiles']) for row in roster]
        observed = {arm: next((i for i, slot in enumerate(order, 1) if identities[slot] == mid), 0)
                    for arm, order in orders.items()}
        if observed['stock'] != rank or any(bool(value) != bool(rank) for value in observed.values()):
            raise ValueError(f'Historical first hit or candidate recall changed: {mid}')
        rows.append(dict(molecule_id=mid, accepted_scans=study['native_scans'], eligible_formula_groups=study['eligible_groups'],
                         **{arm + '_rank': val for arm, val in observed.items()}))
    metrics = {arm: totals(rows, arm) for arm in ARMS}
    stock = metrics['stock']
    if (stock['top1'] != 31 or stock['top25'] != 45 or abs(stock['mrr25'] - 0.5916534391534392) > 1e-12
            or dict(changes) != seal['changed'] or dict(first_changes) != seal['first_changed']):
        raise ValueError('Stock C41 totals or frozen comparisons did not reproduce')
    comparisons = {}
    for arm in ARMS[1:]:
        pairs = [(row['molecule_id'], row['stock_rank'], row[arm + '_rank']) for row in rows]
        result = dict(delta_mrr25=fsum(rr(b) - rr(a) for _, a, b in pairs) / len(rows),
                      improved=[mid for mid, a, b in pairs if rr(b) > rr(a)],
                      worsened=[mid for mid, a, b in pairs if rr(b) < rr(a)])
        for k in (1, 5):
            result[f'new_top{k}'] = [mid for mid, a, b in pairs if not 0 < a <= k and 0 < b <= k]
            result[f'lost_top{k}'] = [mid for mid, a, b in pairs if 0 < a <= k and not 0 < b <= k]
        comparisons[arm] = result
    out.mkdir(exist_ok=True, parents=True)
    per = out / 'e180_native_scan_quality_per_query.csv'
    with per.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(scored_stage=seal['scored_stage'], new_predictor_requests=0, actual_slots=seal['actual_slots'],
                   observed_scans=seal['observed_scans'], arms=metrics, comparisons=comparisons,
                   blind_receipt_sha256=seal_sha, per_query_sha256=sha256(per.read_bytes()).hexdigest())
    (out / 'e180_native_scan_quality_metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


def build(seal, out):
    from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack

    read_seal(seal)
    seal_sha = sha256(seal.read_bytes()).hexdigest()
    metric_program = (HERE.parent / 'pipeline_candidates/first_formula/score_blind_pair.py').read_text()
    node, = (n for n in ast.parse(metric_program).body if isinstance(n, ast.FunctionDef)
             and n.name == 'candidate_key_function')
    payload = {name: pack(body) for name, body in {
        'original_connectivity.py': ('from functools import lru_cache\n' + ast.get_source_segment(metric_program, node) + '\n').encode(),
        'build_e180_native_scan_quality.py': (HERE / 'build_e180_native_scan_quality.py').read_bytes(),
        'build.py': (HERE / 'build.py').read_bytes(),
        'build_final_native_top5_e180_score.py': (HERE / 'build_final_native_top5_e180_score.py').read_bytes(),
        'score_quality.py': Path(__file__).read_bytes(),
    }.items()}
    code = f'''
import base64, subprocess, sys, tempfile, zlib
from hashlib import sha256
from pathlib import Path
inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='e180-native-scan-quality-score-') as tmp:
    root = Path(tmp)
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Changed embedded scoring source: {{name}}')
        (root / name).write_bytes(raw)
    sys.path.insert(0, str(root))
    from score_quality import score
    score(inputs, Path('/kaggle/working'), {seal_sha!r})
'''
    slug = SLUG + '-score'
    compile(code, slug, 'exec')
    meta = dict(id=f'jiyujia/{slug}', title=slug, code_file='score.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, dataset_sources=[WHEEL],
                kernel_sources=[f'jiyujia/{SLUG}/1', f'jiyujia/{REFERENCE}/1'], competition_sources=[], docker_image=IMAGE)
    out.mkdir(exist_ok=True, parents=True)
    (out / 'score.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    build(**vars(parser.parse_args()))
