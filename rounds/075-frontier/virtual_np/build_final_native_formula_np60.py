"""Transfer the frozen E180 cross-formula action to the archived Pablo NP60 slate."""

import argparse
from collections import Counter
from hashlib import sha256
from inspect import getsource
import json
from math import isfinite
from pathlib import Path

from build_final_native_formula_e180 import ARMS, PREFIX, identity, swap_choices
from build_final_native_top5_e180 import COMMON, HERE, RULE_SHA, frozen_module


SLUG = 'enveda-075-final-native-formula-np60'
SEAL = 'final_native_formula_np60_receipt.json'
TOP5_REFERENCE = 'jiyujia/enveda-075-final-native-top5/1'
TOP5_SEAL = 'final_native_top5_receipt.json'
TOP5_SEAL_SHA = 'ca07e0cd897655f018a29716308653197d243ab1872ee91e143db81826dc4351'
TOP5_SHA = '9c3003a666e8f3d8cb6301bd8b8a96d691329e33b41b44fa99a616335ba49d2e'
FORMULA_RULE_SHA = 'f09d094a4df1041ebe1f6b746c693938f581019d4c9d6d962c240e59d5c90ca1'
BASE = 'jiyujia/enveda-075-np-recall-reseal/2'
SNAP = 'jiyujia/enveda-075-c39-snapshot-t4/1'


def run(snapshot, baseline_zip, top5_seal, out):
    from frozen_top5 import (BASE_SHA, LIBRARY_THRESHOLD, QUERY_SHA, dump, formula, hash_value,
                             native_evidence, original_inputs, rdBase, verify)

    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('Expected the contest RDKit version')
    previous = json.loads(verify(top5_seal.read_bytes(), TOP5_SEAL_SHA))
    original, ledgers, request, raw_scores, budget, hashes = original_inputs(snapshot, baseline_zip)
    membership = hash_value({m: sorted(v) for m, v in original.items()})
    if (previous['input_file_sha256'] != hashes or previous['files']['open_final.json'] != TOP5_SHA
            or previous['historical_final_membership_sha256'] != membership
            or Counter(map(len, original.values())) != {25: 59, 1: 1}):
        raise ValueError('The published NP60 top-five seal used different native scores or candidates')
    sealed = json.loads(verify(top5_seal.with_name('open_final.json').read_bytes(), TOP5_SHA))
    if sealed.keys() != original.keys():
        raise ValueError('Original and same-formula archives contain different queries')
    rankings = dict(top5=sealed, dual_single={}, dual_replicated={})
    cases = {}
    for mid, before in original.items():
        baseline = sealed[mid]
        formulas = {s: formula(s) for s in before[:PREFIX]}
        if (Counter(baseline[:5]) != Counter(before[:5]) or baseline[5:] != before[5:]
                or [formulas[s] for s in baseline[:5]] != [formulas[s] for s in before[:5]]):
            raise ValueError(f'Prior same-formula evidence changed its candidate or formula membership: {mid}')
        strength = ledgers[mid]['library_max']
        if not isfinite(strength):
            raise ValueError(f'Nonfinite library score: {mid}')
        evidence, paired = native_evidence(mid, before[:PREFIX], request, raw_scores,
                                          budget['cases'][mid]['predictor_budget'])
        ice, glacier = (evidence[model]['values'] for model in ('ice', 'gl'))
        scores = {s: (ice[s], glacier[s]) for s in ice.keys() & glacier.keys()}
        eligible = paired and strength < LIBRARY_THRESHOLD
        choices, details = swap_choices(before, baseline, formulas, scores if eligible else {})
        for arm, action in choices.items():
            ranked = list(baseline)
            if action is not None:
                i = action['prior_top5_rank'] - 1
                if not 1 <= i < min(PREFIX, len(ranked)) or action['formula'] == details['incumbent_formula']:
                    raise ValueError(f'Invalid cross-formula action: {mid}/{arm}')
                ranked[0], ranked[i] = ranked[i], ranked[0]
            if Counter(ranked) != Counter(baseline) or ranked[PREFIX:] != baseline[PREFIX:]:
                raise ValueError(f'Fixed candidate inventory or tail changed: {mid}/{arm}')
            rankings[arm][mid] = ranked
        cases[mid] = dict(library_max=strength, same_nonempty_native_spectra=paired, eligible=eligible,
                          jointly_finite=len(scores), actions=choices, **details)
    out.mkdir(parents=True, exist_ok=True)
    files = {f'{arm}_final.json': dump(out / f'{arm}_final.json', rows) for arm, rows in rankings.items()}
    if files['top5_final.json'] != TOP5_SHA:
        raise ValueError('Top-five reference changed when written')
    case_name = 'final_native_formula_np60_cases.json'
    files[case_name] = dump(out / case_name, cases)
    fired = {arm: [mid for mid, case in cases.items() if case['actions'][arm] is not None] for arm in ARMS[1:]}
    result = dict(status='sealed', arm_names=list(ARMS), molecules=len(original), candidate_budget=25,
                  actual_slots=sum(map(len, original.values())), new_predictor_requests=0,
                  baseline=BASE, baseline_archive_sha256=BASE_SHA, queries_sha256=QUERY_SHA,
                  first_prefix=PREFIX, source_60_rule_sha256=RULE_SHA, source_library_threshold=LIBRARY_THRESHOLD,
                  source_e180_formula_rule_sha256=FORMULA_RULE_SHA, top5_reference=TOP5_REFERENCE,
                  top5_seal_sha256=TOP5_SEAL_SHA, snapshot=SNAP, input_file_sha256=hashes,
                  files=files, arms=fired, historical_final_membership_sha256=membership)
    dump(out / SEAL, result)
    print('Blind NP60 cross-formula action counts', {arm: len(mids) for arm, mids in fired.items()}, flush=True)


def build(out):
    rule = HERE / 'build_final_native_formula_e180.py'
    if sha256(rule.read_bytes()).hexdigest() != FORMULA_RULE_SHA:
        raise ValueError('The previously evaluated E180 formula swap rule changed')
    constants = ('SEAL', 'ARMS', 'PREFIX', 'TOP5_SHA', 'TOP5_REFERENCE', 'TOP5_SEAL_SHA',
                 'BASE', 'SNAP', 'RULE_SHA', 'FORMULA_RULE_SHA')
    source = '\n'.join(('from collections import Counter', 'from functools import lru_cache',
                        'import json', 'from math import isfinite',
                        *(f'{name} = {globals()[name]!r}' for name in constants))) + '\n'
    source += '\n'.join(map(getsource, (identity, swap_choices, run)))
    payload = {}
    for name, body in {'frozen_top5.py': frozen_module(), 'np60_formula_rerank.py': source}.items():
        compile(body, name, 'exec')
        payload[name] = COMMON['pack'](body.encode())
    code = f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

inputs = Path('/kaggle/input')
{COMMON['METRIC_BOOTSTRAP']}
def mounted(reference, name):
    slug = reference.split('/')[1]
    item, = (p for p in inputs.rglob(name) if slug in p.parts)
    return item

snapshot = mounted({SNAP!r}, 'c39_snapshot_receipt.json').parent
baseline = mounted({BASE!r}, 'baseline.zip')
top5 = mounted({TOP5_REFERENCE!r}, {TOP5_SEAL!r})
with tempfile.TemporaryDirectory(prefix='np60-final-native-formula-') as directory:
    scratch = Path(directory)
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Embedded source changed: {{name}}')
        (scratch / name).write_bytes(raw)
    sys.path.insert(0, str(scratch))
    from np60_formula_rerank import run
    run(snapshot, baseline, top5, Path('/kaggle/working'))
'''
    compile(code, SLUG, 'exec')
    meta = dict(id='jiyujia/' + SLUG, title=SLUG, code_file='run_formula.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, docker_image=COMMON['IMAGE'],
                kernel_sources=[BASE, SNAP, TOP5_REFERENCE], dataset_sources=[COMMON['WHEEL']], competition_sources=[])
    out.mkdir(parents=True, exist_ok=True)
    (out / meta['code_file']).write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(sha256(code.encode()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    build(**vars(parser.parse_args()))
