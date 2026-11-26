"""Transfer the frozen E180 cross-formula action to the MassBank 101 archive."""

import argparse
from collections import Counter
from hashlib import sha256
from inspect import getsource
import json
from math import isfinite
from pathlib import Path

from build_final_native_formula_e180 import ARMS, PREFIX, identity, swap_choices
from build_final_native_top5_e180 import COMMON, HERE, RULE_SHA, frozen_module
from build_final_native_top5_mb101 import ARCHIVE, replay


SLUG = 'enveda-075-final-native-formula-mb101'
SEAL = 'final_native_formula_mb101_receipt.json'
TOP5_REFERENCE = 'jiyujia/enveda-075-final-native-top5-mb101/1'
TOP5_SEAL = 'final_native_top5_mb101_receipt.json'
TOP5_SEAL_SHA = '6501697f5b828003e540cea6b8ba82ce9e97120892894b60d99d04ac55e3ea68'
TOP5_SHA = '193914c5d2aebd8ad3cd56494eebaff4d452d58f324b5bf092fdabef25cab1b4'
ADAPTER_SHA = '1a8546bbc7e8d4755fe48e2e4c473a52b1801eb821bd9124e05a8860a65186b5'
E180_SOURCE_SHA = 'f09d094a4df1041ebe1f6b746c693938f581019d4c9d6d962c240e59d5c90ca1'


def run(archive, top5_seal, out):
    from archive_loader import ARCHIVE, ARCHIVE_SHA, NATIVE, ORIGINAL, PINS, load_archive
    from frozen_top5 import LIBRARY_THRESHOLD, dump, formula, hash_value, native_evidence, verify

    previous = json.loads(verify(top5_seal.read_bytes(), TOP5_SEAL_SHA))
    if (previous['status'] != 'sealed' or previous['molecules'] != 101
            or previous['actual_slots'] != 2454 or previous['archived_dataset'] != ARCHIVE
            or previous['archive_sha256'] != ARCHIVE_SHA or previous['input_file_sha256'] != PINS
            or previous['files']['open_final.json'] != TOP5_SHA):
        raise ValueError('The starting MassBank top-five archive changed')
    original, ledgers, request, raw_scores, budget, model_meta = load_archive(archive)
    sealed = json.loads(verify(top5_seal.with_name('open_final.json').read_bytes(), TOP5_SHA))
    membership = hash_value({mid: sorted(values) for mid, values in original.items()})
    if sealed.keys() != original.keys() or membership != previous['historical_final_membership_sha256']:
        raise ValueError('The frozen original and top-five candidate populations differ')

    rankings = dict(top5=sealed, dual_single={}, dual_replicated={})
    cases = {}
    for mid, before in original.items():
        baseline = sealed[mid]
        formulas = {text: formula(text) for text in before[:PREFIX]}
        if (Counter(baseline[:5]) != Counter(before[:5]) or baseline[5:] != before[5:]
                or [formulas[text] for text in baseline[:5]] != [formulas[text] for text in before[:5]]):
            raise ValueError(f'Former output changed candidate or formula membership: {mid}')
        strength = ledgers[mid]['library_max']
        if not isfinite(strength):
            raise ValueError(f'Nonfinite library score: {mid}')
        evidence, paired = native_evidence(mid, before[:PREFIX], request, raw_scores,
                                          budget['cases'][mid]['predictor_budget'])
        ice, glacier = (evidence[model]['values'] for model in ('ice', 'gl'))
        scores = {text: (ice[text], glacier[text]) for text in ice.keys() & glacier.keys()}
        eligible = paired and strength < LIBRARY_THRESHOLD
        choices, details = swap_choices(before, baseline, formulas, scores if eligible else {})
        for arm, action in choices.items():
            ranked = list(baseline)
            if action is not None:
                position = action['prior_top5_rank'] - 1
                if (not 1 <= position < min(PREFIX, len(baseline)) or not eligible
                        or baseline[position] != action['representatives'][0]
                        or action['formula'] == details['incumbent_formula']):
                    raise ValueError(f'Invalid cross-formula action: {mid}/{arm}')
                ranked[0], ranked[position] = ranked[position], ranked[0]
            if Counter(ranked) != Counter(baseline) or ranked[PREFIX:] != baseline[PREFIX:]:
                raise ValueError(f'Fixed candidate inventory or tail changed: {mid}/{arm}')
            rankings[arm][mid] = ranked
        cases[mid] = dict(library_max=strength, same_nonempty_native_spectra=paired, eligible=eligible,
                          jointly_finite=len(scores), actions=choices, **details)

    out.mkdir(parents=True, exist_ok=True)
    files = {f'{arm}_final.json': dump(out / f'{arm}_final.json', rows) for arm, rows in rankings.items()}
    if files['top5_final.json'] != TOP5_SHA:
        raise ValueError('Top-five reference changed when written')
    case_file = 'final_native_formula_mb101_cases.json'
    files[case_file] = dump(out / case_file, cases)
    fired = {arm: [mid for mid, case in cases.items() if case['actions'][arm] is not None] for arm in ARMS[1:]}
    result = dict(status='sealed', arm_names=list(ARMS), molecules=len(original), candidate_budget=25,
                  actual_slots=sum(map(len, original.values())), new_predictor_requests=0,
                  baseline=ORIGINAL, native_source=NATIVE, archived_dataset=ARCHIVE, archive_sha256=ARCHIVE_SHA,
                  base_submission_sha256=PINS['submission.csv'], first_prefix=PREFIX,
                  source_60_rule_sha256=RULE_SHA, frozen_e180_source_sha256=E180_SOURCE_SHA,
                  source_library_threshold=LIBRARY_THRESHOLD, top5_reference=TOP5_REFERENCE,
                  top5_seal_sha256=TOP5_SEAL_SHA, input_file_sha256=PINS, files=files, arms=fired,
                  native_forward_archives=model_meta, historical_final_membership_sha256=membership)
    dump(out / SEAL, result)
    print('Blind MassBank cross-formula action counts', {arm: len(mids) for arm, mids in fired.items()}, flush=True)


def build(out):
    adapter = (HERE / 'build_final_native_top5_mb101.py').read_bytes()
    source = (HERE / 'build_final_native_formula_e180.py').read_bytes()
    if sha256(adapter).hexdigest() != ADAPTER_SHA or sha256(source).hexdigest() != E180_SOURCE_SHA:
        raise ValueError('The verified MassBank adapter or frozen E180 formula source changed')
    loader = adapter.decode().split('\ndef replay(', 1)[0]
    loader += getsource(replay).split('    outputs,cases = apply(ranks,ledger,request,scores,budget)\n', 1)[0]
    loader = loader.replace('def replay(payload, out):', 'def load_archive(payload):')
    loader += '    return ranks, ledger, request, scores, budget, meta\n'
    constants = ('ARMS', 'PREFIX', 'SEAL', 'TOP5_REFERENCE', 'TOP5_SHA', 'TOP5_SEAL_SHA', 'RULE_SHA', 'E180_SOURCE_SHA')
    runner = '\n'.join(('from collections import Counter', 'from functools import lru_cache', 'import json',
                         'from math import isfinite', *(f'{name} = {globals()[name]!r}' for name in constants))) + '\n'
    runner += '\n'.join(map(getsource, (identity, swap_choices, run)))
    payload = {}
    for name, body in {'frozen_top5.py': frozen_module(), 'archive_loader.py': loader, 'formula_rerank.py': runner}.items():
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
archive, = (p for p in inputs.rglob('final_ranking_payload.bin') if {ARCHIVE.split('/')[1]!r} in p.parts)
top5, = (p for p in inputs.rglob({TOP5_SEAL!r}) if {TOP5_REFERENCE.split('/')[1]!r} in p.parts)
with tempfile.TemporaryDirectory(prefix='mb101-final-native-formula-') as directory:
    scratch = Path(directory)
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Embedded formula source changed: {{name}}')
        (scratch / name).write_bytes(raw)
    sys.path.insert(0, str(scratch))
    from formula_rerank import run
    run(archive, top5, Path('/kaggle/working'))
'''
    compile(code, SLUG, 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='run_formula.py', language='python',
                kernel_type='script', is_private=True, enable_gpu=False, enable_internet=False,
                docker_image=COMMON['IMAGE'], kernel_sources=[TOP5_REFERENCE],
                dataset_sources=[ARCHIVE, COMMON['WHEEL']], competition_sources=[])
    out.mkdir(parents=True, exist_ok=True)
    (out / meta['code_file']).write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(sha256(code.encode()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    build(**vars(parser.parse_args()))
