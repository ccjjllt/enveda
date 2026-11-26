"""Score a sealed outsider insertion without transferring validation answers."""

import argparse
import ast
import csv
from hashlib import sha256
import json
from math import fsum
from pathlib import Path

from build_final_native_top5_e180_score import mounted, rr, totals, verified


SLUG = 'enveda-075-final-native-outsider-np60'
STEM = 'final_native_outsider_np60'
BASE = 'enveda-075-final-native-top5-compare'
SHORTLIST = 'enveda-075-full-main-unseen'
OLD_SCORE = SHORTLIST + '-compare'
BASE_SHA = '0d67baf0e5d4d61071a0a089dccfa88e2c180f9db1f6bc3ad074f4bfce9c3c44'
UNSEEN_RANK_SHA = '80e37357861d470e90533040bd48b227affc83f24e0226bede151824aeae3c0c'
SHORTLIST_SHA = 'eb91e2be682d8a472dc6335eb13cc65b55e8e07975cf28b540fc59f38828e584'
SHORTLIST_RECEIPT_SHA = '62a857d612fb7d04bafd8eb3ee32403f1d7bbcc95da93a42bd0f22823ffbe1de'
TOP5_SHA = '9c3003a666e8f3d8cb6301bd8b8a96d691329e33b41b44fa99a616335ba49d2e'
SLOTS = {'rank5': 5, 'rank10': 10}
ARMS = ('top5', *SLOTS)


def read_seal(path):
    seal = json.loads(path.read_bytes())
    expected = dict(status='sealed', molecules=60, actual_slots=1476, candidate_budget=25,
                    new_predictor_requests=0, arm_names=list(ARMS), targets=SLOTS,
                    shortlist_sha256=SHORTLIST_SHA, shortlist_receipt_sha256=SHORTLIST_RECEIPT_SHA,
                    queries_sha256='2a8bf747f50e6920fcd4f24e04302b721281f8794c947cc58fe545264b12d0d4',
                    baseline_archive_sha256='05509b32cfb9bcbe8ab563244f91c97e89f878eba3cc40263e3f48fd4a1f372d')
    if (any(seal.get(key) != value for key, value in expected.items())
            or seal['files']['top5_final.json'] != TOP5_SHA):
        raise ValueError('Unexpected outsider seal or changed frozen control')
    return seal


def csv_archive(inputs, slug, file, digest):
    with verified(mounted(inputs, slug, file), digest).open(newline='') as stream:
        records = list(csv.DictReader(stream))
    by_id = {row['molecule_id']: row for row in records}
    if len(records) != 60 or len(by_id) != 60:
        raise ValueError('Historical score archive has a different number of queries')
    return by_id


def score(inputs, out, seal_sha):
    import pyarrow.parquet as pq
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from outsider_identity import metric_identity

    RDLogger.DisableLog('rdApp.*')
    seal_path = verified(mounted(inputs, SLUG, STEM + '_receipt.json'), seal_sha)
    seal = read_seal(seal_path)
    ranks = {}
    for arm in ARMS:
        name = arm + '_final.json'
        ranks[arm] = json.loads(verified(seal_path.parent / name, seal['files'][name]).read_bytes())
    cases_file = STEM + '_cases.json'
    cases = json.loads(verified(seal_path.parent / cases_file, seal['files'][cases_file]).read_bytes())
    base = ranks['top5']
    if (len(base) != 60 or sum(map(len, base.values())) != 1476 or set(cases) != set(base)
            or any(set(arm) != set(base) for arm in ranks.values())):
        raise ValueError('Wrong frozen query population or candidate budget')
    path = verified(mounted(inputs, SHORTLIST, 'unseen_shortlists.parquet'), SHORTLIST_SHA)
    pool = pq.read_table(path, columns=['molecule_id', 'arm', 'rank', 'smiles', 'originally_seen']).to_pylist()
    unseen = [row for row in pool if row['arm'] == 'unseen']
    shortlist = {(row['molecule_id'], row['rank']): row['smiles'] for row in unseen}
    if (len(unseen) != len(shortlist) or len(unseen) != 670
            or any(row['originally_seen'] or row['molecule_id'] not in base for row in unseen)):
        raise ValueError('Wrong frozen unseen shortlist')

    triggered = {arm: [] for arm in SLOTS}
    for mid, initial in base.items():
        for arm, slot in SLOTS.items():
            after = ranks[arm][mid]
            action = cases[mid]['actions'][arm]
            changed = [i for i, (a, b) in enumerate(zip(initial, after, strict=True), 1) if a != b]
            if changed != ([slot] if action else []):
                raise ValueError('Final candidate structure changed outside the selected slot')
            if action:
                newcomer, incumbent = after[slot - 1], initial[slot - 1]
                if (action['slot'] != slot or action['challenger'] != newcomer or action['incumbent'] != incumbent
                        or shortlist[mid, action['shortlist_rank']] != newcomer
                        or CalcMolFormula(Chem.MolFromSmiles(newcomer)) != CalcMolFormula(Chem.MolFromSmiles(incumbent))
                        or metric_identity(newcomer) in {metric_identity(text) for text in initial}):
                    raise ValueError('Wrong replacement or shortlist rank coordinate')
                triggered[arm].append(mid)
    if any(sorted(mids) != sorted(seal['arms'][arm]) for arm, mids in triggered.items()):
        raise ValueError('Sealed action ledger disagrees with the submitted ranks')

    previous = csv_archive(inputs, BASE, 'final_native_top5_per_query.csv', BASE_SHA)
    coverage = csv_archive(inputs, OLD_SCORE, 'unseen_relist_per_query.csv', UNSEEN_RANK_SHA)
    if set(previous) != set(base) or set(coverage) != set(base):
        raise ValueError('Archived scores refer to a different panel')
    rows = []
    for mid, initial in sorted(base.items()):
        past, aux = previous[mid], coverage[mid]
        original, short = int(past['open_rank'] or 0), int(aux['unseen_rank'] or 0)
        originally_seen = aux['truth_in_original_lists'] == 'True'
        if ((original and not originally_seen) or (short and originally_seen) or past['source'] != aux['source']
                or not 0 <= original <= len(initial) or not 0 <= short <= int(aux['budget'])):
            raise ValueError('Inconsistent archived control and unseen ranks')
        observed = {'top5': original}
        identity = metric_identity(initial[original - 1]) if original else None
        if original and (identity is None or any(metric_identity(s) == identity for s in initial[:original - 1])):
            raise ValueError('Archived original earliest hit does not match the frozen control')
        for arm, slot in SLOTS.items():
            action = cases[mid]['actions'][arm]
            if original and action and original == slot:
                observed[arm] = next((i for i, s in enumerate(ranks[arm][mid], 1) if metric_identity(s) == identity), 0)
            elif not original and short and action and short == action['shortlist_rank']:
                observed[arm] = slot
            else:
                observed[arm] = original
        rows.append(dict(molecule_id=mid, source=past['source'], original_sources_missed=not originally_seen,
                         unseen_true_rank=short, **{arm + '_rank': value for arm, value in observed.items()}))

    metrics = {arm: totals(rows, arm) for arm in ARMS}
    stock = metrics['top5']
    if ((stock['top1'], stock['top5'], stock['top25']) != (20, 37, 48)
            or abs(stock['mrr25'] - 0.4596523615273615) > 1e-12):
        raise ValueError('The original top-five MRR did not reproduce')
    summaries = {}
    for arm in SLOTS:
        changes = [(row['molecule_id'], row['top5_rank'], row[arm + '_rank']) for row in rows]
        result = dict(delta_mrr25=fsum(rr(b) - rr(a) for _, a, b in changes) / len(rows),
                      trigger_ids=triggered[arm])
        for cut in (1, 5, 25):
            result[f'new_top{cut}'] = [mid for mid, a, b in changes if not 0 < a <= cut and 0 < b <= cut]
            result[f'lost_top{cut}'] = [mid for mid, a, b in changes if 0 < a <= cut and not 0 < b <= cut]
        summaries[arm] = result
    out.mkdir(exist_ok=True, parents=True)
    per = out / (STEM + '_per_query.csv')
    with per.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(arms=metrics, comparisons_vs_top5=summaries, candidate_slots=1476,
                  new_predictor_requests=0, blind_receipt_sha256=seal_sha,
                  baseline_score_sha256=BASE_SHA, unseen_score_sha256=UNSEEN_RANK_SHA,
                  per_query_sha256=sha256(per.read_bytes()).hexdigest())
    (out / (STEM + '_metrics.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def build(seal, out):
    from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack

    read_seal(seal)
    seal_sha = sha256(seal.read_bytes()).hexdigest()
    body = (HERE.parent / 'four_arm/retrieval_pilot_score.py').read_text()
    node, = (item for item in ast.parse(body).body if isinstance(item, ast.FunctionDef) and item.name == 'metric_identity')
    metric = ('from collections import Counter\nfrom functools import lru_cache\nfrom rdkit import Chem\n'
              'from rdkit.Chem.MolStandardize import rdMolStandardize\nTAUTOMER = rdMolStandardize.TautomerEnumerator()\n'
              + '\n'.join(body.splitlines()[node.decorator_list[0].lineno - 1:node.end_lineno]))
    payload = {name: pack(raw) for name, raw in {
        'outsider_identity.py': metric.encode(),
        'build_final_native_top5_e180_score.py': (HERE / 'build_final_native_top5_e180_score.py').read_bytes(),
        'score_outsider.py': Path(__file__).read_bytes(),
    }.items()}
    code = f'''
import base64, subprocess, sys, tempfile, zlib
from hashlib import sha256
from pathlib import Path
inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='final-outsider-score-') as tmp:
    root = Path(tmp)
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Embedded scoring source changed: {{name}}')
        (root / name).write_bytes(raw)
    sys.path.insert(0, str(root))
    from score_outsider import score
    score(inputs, Path('/kaggle/working'), {seal_sha!r})
'''
    slug = SLUG + '-score'
    compile(code, slug, 'exec')
    meta = dict(id=f'jiyujia/{slug}', title=slug, code_file='score.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, dataset_sources=[WHEEL],
                kernel_sources=[f'jiyujia/{SLUG}/1', f'jiyujia/{SHORTLIST}/1', f'jiyujia/{BASE}/2', f'jiyujia/{OLD_SCORE}/1'],
                competition_sources=[], docker_image=IMAGE)
    out.mkdir(exist_ok=True, parents=True)
    (out / 'score.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    build(**vars(parser.parse_args()))
