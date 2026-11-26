"""Score fixed cross-formula transfers using previously assessed top-five rank indices."""

import argparse
import ast
import csv
from hashlib import sha256
import json
from math import fsum
from pathlib import Path

from build_final_native_top5_e180_score import mounted, rr, totals, verified


ARMS = ('top5', 'dual_single', 'dual_replicated')
COHORTS = {
    'np60': dict(molecules=60, actual_slots=1476, historical='enveda-075-final-native-top5-compare/2',
                 historical_file='final_native_top5_per_query.csv',
                 rank_sha='0d67baf0e5d4d61071a0a089dccfa88e2c180f9db1f6bc3ad074f4bfce9c3c44',
                 top5_sha='9c3003a666e8f3d8cb6301bd8b8a96d691329e33b41b44fa99a616335ba49d2e',
                 reference_stats=(20, 37, 48, 0.4596523615273615),
                 chemistry='metric_identity', provenance={
                     'baseline_archive_sha256': '05509b32cfb9bcbe8ab563244f91c97e89f878eba3cc40263e3f48fd4a1f372d',
                     'queries_sha256': '2a8bf747f50e6920fcd4f24e04302b721281f8794c947cc58fe545264b12d0d4'}),
    'mb101': dict(molecules=101, actual_slots=2454, historical='enveda-075-final-native-top5-mb101-score/1',
                  historical_file='final_native_top5_mb101_per_query.csv',
                  rank_sha='2d3504e4a9af67d4ada69581e6e1412497f85e449e1b18f1c2fd6cdc3679db78',
                  top5_sha='193914c5d2aebd8ad3cd56494eebaff4d452d58f324b5bf092fdabef25cab1b4',
                  reference_stats=(71, 100, 101, 0.8237623762376237),
                  chemistry='candidate_key_function', provenance={
                      'base_submission_sha256': '864cf9f414e4ac2d8f634aaf50c6e76ef018c337739957c9fae56b2cc2f48bc4'}),
}


def stem(cohort):
    return f'final_native_formula_{cohort}'


def read_seal(path, config):
    seal = json.loads(path.read_bytes())
    expected = dict(status='sealed', molecules=config['molecules'], candidate_budget=25,
                    actual_slots=config['actual_slots'], new_predictor_requests=0,
                    arm_names=list(ARMS), first_prefix=10, **config['provenance'])
    if (any(seal.get(field) != value for field, value in expected.items())
            or seal['files']['top5_final.json'] != config['top5_sha']):
        raise ValueError('Unexpected transfer seal or changed reference top-five ordering')
    return seal


def validate_lists(rankings, seal):
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    base = rankings['top5']
    if (len(base) != seal['molecules'] or sum(map(len, base.values())) != seal['actual_slots']
            or any(not 1 <= len(rows) <= 25 for rows in base.values())
            or any(set(arm) != set(base) for arm in rankings.values())):
        raise ValueError('Different blind population or total candidate count')
    observed = {}
    for arm in ARMS[1:]:
        observed[arm] = []
        for mid, before in base.items():
            after = rankings[arm][mid]
            if len(after) != len(before):
                raise ValueError(f'Sealed candidate count changed: {arm}/{mid}')
            changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
            if not changed:
                continue
            if (len(changed) != 2 or changed[0] != 0 or changed[1] >= 10
                    or before[0] != after[changed[1]] or before[changed[1]] != after[0]):
                raise ValueError(f'Only a single head swap within first ten is allowed: {arm}/{mid}')
            formulas = [CalcMolFormula(Chem.MolFromSmiles(text)) for text in (before[0], after[0])]
            if formulas[0] == formulas[1]:
                raise ValueError(f'Final first formula did not change: {arm}/{mid}')
            observed[arm].append(mid)
        if sorted(observed[arm]) != sorted(seal['arms'][arm]):
            raise ValueError(f'Sealed trigger count differs from actual changes: {arm}')
    return observed


def score(inputs, out, seal_sha, cohort):
    from rdkit import RDLogger
    from formula_metric import key

    RDLogger.DisableLog('rdApp.*')
    c = COHORTS[cohort]
    slug = 'enveda-075-' + stem(cohort).replace('_', '-')
    seal_path = verified(mounted(inputs, slug, stem(cohort) + '_receipt.json'), seal_sha)
    seal = read_seal(seal_path, c)
    rankings = {}
    for arm in ARMS:
        name = arm + '_final.json'
        rankings[arm] = json.loads(verified(seal_path.parent / name, seal['files'][name]).read_bytes())
    changed = validate_lists(rankings, seal)

    past = verified(mounted(inputs, c['historical'].split('/')[0], c['historical_file']), c['rank_sha'])
    with past.open(newline='') as stream:
        records = list(csv.DictReader(stream))
    base = rankings['top5']
    if len(records) != len(base) or {record['molecule_id'] for record in records} != set(base):
        raise ValueError('Archived ranks have a different population')
    rows = []
    for record in sorted(records, key=lambda row: row['molecule_id']):
        mid = record['molecule_id']
        rank = int(record['open_rank'] or 0)
        if not 0 <= rank <= len(base[mid]):
            raise ValueError(f'Invalid archived rank: {mid}')
        observed = dict.fromkeys(ARMS, 0)
        if rank:
            identity = key(base[mid][rank - 1])
            if identity is None:
                raise ValueError(f'Previously scored candidate lacks identity: {mid}')
            for arm in ARMS:
                observed[arm] = next((i for i, text in enumerate(rankings[arm][mid], 1) if key(text) == identity), 0)
            if observed['top5'] != rank or not all(observed.values()):
                raise ValueError(f'Archived earliest match or candidate set disagrees: {mid}')
        rows.append(dict(molecule_id=mid, source=record.get('source', 'MassBank Eawag'),
                         **{arm + '_rank': val for arm, val in observed.items()},
                         **{arm + '_delta_rr': rr(observed[arm]) - rr(rank) for arm in ARMS[1:]}))

    metrics = {arm: totals(rows, arm) for arm in ARMS}
    expected = c['reference_stats']
    actual = tuple(metrics['top5'][field] for field in ('top1', 'top5', 'top25', 'mrr25'))
    if actual[:3] != expected[:3] or abs(actual[3] - expected[3]) > 1e-12:
        raise ValueError('Historical top-five metrics did not reproduce')
    comparisons = {}
    for arm in ARMS[1:]:
        cases = [(row['molecule_id'], row['top5_rank'], row[arm + '_rank']) for row in rows]
        result = dict(delta_mrr25=fsum(rr(b) - rr(a) for _, a, b in cases) / len(cases), changed_first=changed[arm])
        for k in (1, 5):
            result[f'new_top{k}'] = [mid for mid, a, b in cases if not 0 < a <= k and 0 < b <= k]
            result[f'lost_top{k}'] = [mid for mid, a, b in cases if 0 < a <= k and not 0 < b <= k]
        comparisons[arm] = result
    out.mkdir(exist_ok=True, parents=True)
    per = out / (stem(cohort) + '_per_query.csv')
    with per.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(cohort=cohort, arms=metrics, comparisons_vs_top5=comparisons, candidate_slots=seal['actual_slots'],
                  new_predictor_requests=0, blind_receipt_sha256=seal_sha, archived_ranks_sha256=c['rank_sha'],
                  per_query_sha256=sha256(per.read_bytes()).hexdigest())
    (out / (stem(cohort) + '_metrics.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def build(seal, out, cohort):
    from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack

    c = COHORTS[cohort]
    read_seal(seal, c)
    seal_sha = sha256(seal.read_bytes()).hexdigest()
    function = c['chemistry']
    if cohort == 'np60':
        program = HERE.parent / 'four_arm/retrieval_pilot_score.py'
        imports = ('from collections import Counter\nfrom functools import lru_cache\nfrom rdkit import Chem\n'
                   'from rdkit.Chem.MolStandardize import rdMolStandardize\nTAUTOMER = rdMolStandardize.TautomerEnumerator()\n')
        binding = 'key = metric_identity'
    else:
        program = HERE.parent / 'pipeline_candidates/first_formula/score_blind_pair.py'
        imports = 'from functools import lru_cache\n'
        binding = 'key = candidate_key_function()'
    code_source = program.read_text()
    node, = (item for item in ast.parse(code_source).body if isinstance(item, ast.FunctionDef) and item.name == function)
    first = min([node.lineno, *[decorator.lineno for decorator in node.decorator_list]])
    metric = imports + '\n'.join(code_source.splitlines()[first - 1:node.end_lineno]) + '\n' + binding + '\n'
    compile(metric, 'formula_metric', 'exec')
    payload = {name: pack(body) for name, body in {
        'formula_metric.py': metric.encode(),
        'build_final_native_top5_e180_score.py': (HERE / 'build_final_native_top5_e180_score.py').read_bytes(),
        'score_transfer.py': Path(__file__).read_bytes(),
    }.items()}
    code = f'''
import base64, subprocess, sys, tempfile, zlib
from hashlib import sha256
from pathlib import Path
inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='final-native-formula-score-') as tmp:
    root = Path(tmp)
    for name, (blob, expected) in {payload!r}.items():
        data = zlib.decompress(base64.b64decode(blob))
        if sha256(data).hexdigest() != expected:
            raise ValueError(f'Embedded scoring source changed: {{name}}')
        (root / name).write_bytes(data)
    sys.path.insert(0, str(root))
    from score_transfer import score
    score(inputs, Path('/kaggle/working'), {seal_sha!r}, {cohort!r})
'''
    blind = 'enveda-075-' + stem(cohort).replace('_', '-')
    slug = blind + '-score'
    compile(code, slug, 'exec')
    meta = dict(id=f'jiyujia/{slug}', title=slug, code_file='score.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, dataset_sources=[WHEEL],
                kernel_sources=[f'jiyujia/{blind}/1', 'jiyujia/' + c['historical']], competition_sources=[], docker_image=IMAGE)
    out.mkdir(exist_ok=True, parents=True)
    (out / 'score.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cohort', choices=COHORTS, required=True)
    build(**vars(parser.parse_args()))
