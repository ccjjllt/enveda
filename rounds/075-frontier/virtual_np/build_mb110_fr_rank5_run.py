"""Run both sealed MassBank rank-5 reserves with fresh native CPU predictions."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path

from build import HERE, METRIC_BOOTSTRAP, WHEEL, pack


def function(code, name):
    node, = (n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(code, node)


SLUG = 'enveda-075-mb110-fr-rank5-run'
PLANNER = 'enveda-075-mb110-fr-rank5-plan'
PLAN = 'mb110_fr_rank5_plan.json'
OUTPUT = 'mb110_fr_rank5_run.json'
BASELINE_SHA = '8568ed5abecb88ef43243b122fa726959c85893ff7ae32db22f07d72de0fa44d'
RUNTIME_SHA = '45992cc273789b27025445f42a20ff98ba1c1b6e9633e02ad0fd7f54d444a29e'
RUNTIME_IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
ARMS = ('c34', 'fr')
MODELS = ('ice', 'gl')


def verified_plan(path):
    plan = json.loads(path.read_bytes())
    hashes = plan.get('files', {})
    required = {'historical_final.json', *(f'{arm}_requests.json' for arm in ARMS)}
    if (plan.get('status') != 'sealed' or not required <= hashes.keys()
            or hashes['historical_final.json'] != BASELINE_SHA
            or any(Path(n).name != n or not isinstance(hashes[n], str) or len(hashes[n]) != 64 for n in required)):
        raise ValueError('Expected the sealed equal-budget plan with the official historical roster')
    return plan


def load_requests(plan_path, expected, formula):
    from c39_request_cache import population

    if file_hash(plan_path) != expected:
        raise ValueError('Mounted plan differs from its independently recorded seal')
    plan = verified_plan(plan_path)

    def artifact(name):
        path = plan_path.with_name(name)
        if file_hash(path) != plan['files'][name]:
            raise ValueError(f'Changed plan artifact: {name}')
        return json.loads(path.read_bytes())

    historical = artifact('historical_final.json')
    requests = {arm: population(artifact(f'{arm}_requests.json')) for arm in ARMS}
    if (len(historical) != 110 or any(len(row) != 25 for row in historical.values())
            or requests['c34'].keys() != requests['fr'].keys() or not set(requests['c34']) <= historical.keys()):
        raise ValueError('Unexpected frozen query population or candidate count')
    joint = []
    for mid, roster in historical.items():
        if mid not in requests['c34']:
            continue
        native, fr = [requests[arm][mid] for arm in ARMS]
        head, spectra = native['cands'][0], native['spectra']
        candidates = [native['cands'], fr['cands']]
        if (head != roster[4] or head != fr['cands'][0] or spectra != fr['spectra'] or not spectra
                or len(candidates[0]) != len(candidates[1]) or not 2 <= len(candidates[0]) <= 5
                or any(len(set(arm)) != len(arm) or set(arm[1:]).intersection(roster) for arm in candidates)
                or formula(head) is None or any(formula(s) != formula(head) for arm in candidates for s in arm)):
            raise ValueError(f'Unequal or invalid prediction request at query {mid}')
        union = list(dict.fromkeys(s for arm in candidates for s in arm))
        joint.append(dict(mid=mid, cands=union, spectra=spectra))
    return historical, requests, joint


def file_hash(path):
    from hashlib import file_digest

    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def original_helper(package, model):
    import importlib.util
    import sys

    fuse = 'fuse.py' if model == 'ice' else 'gl_fuse.py'
    spec = importlib.util.spec_from_file_location(f'c41_archived_{model}_fuse', package / fuse)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(package))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def predict(model, package, ice_dir, items, out, site):
    from c39_request_cache import checked_scores

    helper = original_helper(package, model)
    work = out / f'{model}_work'
    work.mkdir()
    options = dict(workdir=str(work), site=str(site), device='cpu', budget_s=5500, grace_s=45,
                   extra_args=['--any-rdkit'])
    if model == 'ice':
        scores = helper.run_ice(str(package), items, **options)
    else:
        scores = helper.run_gl(str(package), items, wheels=str(ice_dir / 'wheels'), **options)
    submitted = json.loads((work / f'{model}_in.json').read_bytes())
    raw = json.loads((work / f'{model}_out.json').read_bytes())
    meta = json.loads((work / f'{model}_out.json.meta.json').read_bytes())
    finite = sum(checked_scores(row) for row in scores.values())
    expected = {row['mid']: set(row['cands']) for row in items}
    if (submitted != items or scores != raw or meta.get('device') != 'cpu' or meta.get('status') != 'ok'
            or meta.get('n_cands') != sum(map(len, expected.values())) or meta.get('n_scored') != finite
            or set(scores) - expected.keys() or any(set(row) - expected[mid] for mid, row in scores.items())):
        raise ValueError(f'{model} did not complete the agreed CPU request')
    return scores, {key: meta.get(key) for key in ('status', 'device', 'seconds', 'n_cands', 'n_scored', 'rdkit', 'torch')}


def apply_rank5(historical, requests, scores, formula, replace_at_rank):
    from math import isfinite
    from types import SimpleNamespace

    final = {arm: {} for arm in ARMS}
    ledger = {}
    for mid, before in historical.items():
        ice, gl = [scores[model].get(mid, {}) for model in MODELS]
        joint = {s: (i, gl[s]) for s, i in ice.items() if i is not None and gl.get(s) is not None
                 and isfinite(i) and isfinite(gl[s])}
        ledger[mid] = {}
        for arm in ARMS:
            scheduled = requests[arm].get(mid, {}).get('cands', [])
            pool = [SimpleNamespace(smiles=s, rank=i) for i, s in enumerate(scheduled[1:], 1)]
            replacement, action, reason = replace_at_rank(before, pool, joint, 5, formula) if pool else (
                before, None, 'not_scheduled')
            final[arm][mid] = replacement
            ledger[mid][arm] = dict(logical_nominees=len(pool), valid_nominees=sum(c.smiles in joint for c in pool),
                                    action=action, reason=reason)
    return final, ledger


def run(inputs, out, plan_hash, preseed, formula, replace_at_rank):
    from c39_request_cache import write_json
    from rdkit import RDLogger
    from tempfile import TemporaryDirectory

    RDLogger.DisableLog('rdApp.*')
    plan_path, = (p for p in inputs.rglob(PLAN) if PLANNER in p.parts)
    historical, requests, items = load_requests(plan_path, plan_hash, formula)
    out.mkdir(parents=True, exist_ok=True)
    packages = {}
    for model, name, runner in (('ice', 'iceberg', 'ice_runner.py'), ('gl', 'glacier', 'gl_runner.py')):
        path, = (p for p in inputs.rglob(runner) if f'casmi26-{name}' in p.parts)
        packages[model] = path.parent
    scores, detail = {model: {} for model in MODELS}, {}
    if items:
        with TemporaryDirectory(prefix='mb110-fr-rank5-') as tmp:
            site = Path(tmp) / 'site'
            preseed(str(packages['ice']), str(site))
            for model in MODELS:
                scores[model], detail[model] = predict(model, packages[model], packages['ice'], items, out, site)
    final, ledger = apply_rank5(historical, requests, scores, formula, replace_at_rank)
    files = {f'{arm}_final.json': write_json(out / f'{arm}_final.json', roster)
             for arm, roster in {'historical': historical, **final}.items()}
    if files['historical_final.json'] != BASELINE_SHA:
        raise ValueError('Original roster serialization changed')
    files['mb110_fr_rank5_actions.json'] = write_json(out / 'mb110_fr_rank5_actions.json', ledger)
    logical = {arm: sum(len(r['cands']) * len(r['spectra']) for r in rows.values()) for arm, rows in requests.items()}
    if logical['c34'] != logical['fr']:
        raise ValueError('Logical candidate–spectrum budgets differ')
    summary = dict(status='sealed', plan_sha256=plan_hash, candidate_budget=25, molecules=110, actual_slots=2750,
                   slot=5, files=files, paired_queries=len(items), logical_requests_per_arm_and_model=logical,
                   shared_actual_requests_per_model=sum(len(r['cands']) * len(r['spectra']) for r in items),
                   native_device='cpu', predictors=detail,
                   swaps={arm: sum(row[arm]['action'] is not None for row in ledger.values()) for arm in ARMS})
    write_json(out / OUTPUT, summary)
    print(summary, flush=True)


def build(seal, runtime, out, planner_version):
    verified_plan(seal)
    seal_sha = file_hash(seal)
    if planner_version < 1 or file_hash(runtime) != RUNTIME_SHA:
        raise ValueError('Expected a versioned plan and the verified archived native CPU runtime')
    old = ''.join(json.loads(runtime.read_bytes())['cells'][1]['source'])
    site = ('import glob, os, sys\n' + function(old, 'simulator_rdkit_wheel') + '\n' + function(old, '_claw_preseed_site'))
    rules = ('from functools import lru_cache\nfrom rdkit import Chem\n'
             'from rdkit.Chem.rdMolDescriptors import CalcMolFormula\n\n@lru_cache(maxsize=32768)\n'
             + function((HERE / 'final_native_top5.py').read_text(), 'formula') + '\n\n'
             + function((HERE / 'build_final_native_outsider_np60.py').read_text(), 'replace_at_rank'))
    this_source = Path(__file__).read_text()
    local = this_source[this_source.index('SLUG = '):this_source.index('\ndef build(')]
    cache = HERE / 'c39_request_cache.py'
    modules = {cache.name: pack(cache.read_bytes())}
    modules.update(native_run=pack(('from hashlib import sha256\nimport json\nfrom pathlib import Path\n' + local).encode()),
                   native_site=pack(site.encode()), native_rule=pack(rules.encode()))
    program = f'''
import base64, hashlib, sys, zlib
from pathlib import Path
from tempfile import TemporaryDirectory
inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}
with TemporaryDirectory(prefix='mb110-rank5-run-code-') as tmp:
    scratch = Path(tmp)
    for name, (blob, expected) in {modules!r}.items():
        data = zlib.decompress(base64.b64decode(blob))
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Changed embedded code: {{name}}')
        (scratch / (name if name.endswith('.py') else name + '.py')).write_bytes(data)
    sys.path.insert(0, str(scratch))
    from native_rule import formula, replace_at_rank
    from native_site import _claw_preseed_site
    from native_run import run
    run(inputs, Path('/kaggle/working'), {seal_sha!r}, _claw_preseed_site, formula, replace_at_rank)
'''
    compile(program, SLUG, 'exec')
    out.mkdir(exist_ok=True, parents=True)
    (out / 'run_rank5.py').write_text(program)
    metadata = dict(id='jiyujia/' + SLUG, title=SLUG, code_file='run_rank5.py', language='python', kernel_type='script',
                    is_private=True, enable_gpu=False, enable_internet=False, competition_sources=[],
                    kernel_sources=[f'jiyujia/{PLANNER}/{planner_version}'],
                    dataset_sources=['dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets', WHEEL], docker_image=RUNTIME_IMAGE)
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return sha256(program.encode()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('seal', 'runtime', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--planner-version', type=int, default=1)
    print(build(**vars(parser.parse_args())))
