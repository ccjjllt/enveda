"""Package a blind fixed-slate comparison of native swap depths on the archived 121 E180 cases."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

from build_final_native_top5_e180 import COMMON, HERE, REFERENCE, RULE_SHA, frozen_module


SLUG = 'enveda-075-final-native-depth-e180'
TOP5_REFERENCE = 'jiyujia/enveda-075-final-native-top5-e180/1'
TOP5_SEAL = 'final_native_top5_e180_receipt.json'
TOP5_SEAL_SHA = '4c8607ccb5f87ad77de7861f705ed3e10878b8fd0dd4927dedcc2cd6986cd2ef'
ADAPTER_SHA = '81fc31042f26b0c9afe89abae530640a736175c792e6702cfedd4374a00abd4a'
DEPTHS = (5, 10, 25)


def swap_at_depth(original, formulas, scores, depth):
    groups = defaultdict(list)
    for rank, name in enumerate(formulas[:depth]):
        if name:
            groups[name].append(rank)
    ranking = list(original)
    swaps = 0
    for positions in groups.values():
        for i in range(1, len(positions)):
            for j in range(i, 0, -1):
                left, right = positions[j - 1], positions[j]
                incumbent, challenger = ranking[left], ranking[right]
                if (incumbent not in scores or challenger not in scores or not all(
                        new > old for new, old in zip(scores[challenger], scores[incumbent], strict=True))):
                    break
                ranking[left], ranking[right] = challenger, incumbent
                swaps += 1
    return ranking, swaps


def run(archive, top5_seal, out):
    from archive_loader import QUERY_SHA, SUBMISSION_SHA, load_archive
    from frozen_top5 import LIBRARY_THRESHOLD, dump, formula, hash_value, native_evidence, verify

    previous = json.loads(verify(top5_seal.read_bytes(), TOP5_SEAL_SHA))
    native_hashes = {name: digest for name, digest in previous['input_file_sha256'].items()
                     if name.startswith(('ice_work/', 'gl_work/'))}
    if len(native_hashes) != 6:
        raise ValueError('The original blind seal must identify all native request, score and metadata files')
    original, ledgers, request, raw_scores, budget, model_meta, hashes = load_archive(archive, native_hashes)
    rankings = {'historical': original, **{f'top{depth}': {} for depth in DEPTHS}}
    cases = {}
    for mid, before in original.items():
        strength = ledgers[mid]['library_max']
        if not math.isfinite(strength):
            raise ValueError(f'Nonfinite E1 library strength for {mid}')
        evidence, paired = native_evidence(mid, before, request, raw_scores, budget['cases'][mid]['predictor_budget'])
        formulas = [formula(smiles) for smiles in before]
        scores = {}
        if paired and strength < LIBRARY_THRESHOLD:
            ice, glacier = (evidence[model]['values'] for model in ('ice', 'gl'))
            scores = {smiles: (ice[smiles], glacier[smiles]) for smiles in ice.keys() & glacier.keys()}
        swaps = {}
        for depth in DEPTHS:
            arm = f'top{depth}'
            after, swaps[arm] = swap_at_depth(before, formulas, scores, depth)
            if (Counter(before) != Counter(after) or before[depth:] != after[depth:]
                    or formulas != [formula(smiles) for smiles in after]):
                raise ValueError(f'Candidate membership, formula position or excluded tail changed for {mid}/{arm}')
            rankings[arm][mid] = after
        cases[mid] = dict(library_max=strength, same_nonempty_native_spectra=paired,
                          dual_finite_in_prefix={f'top{n}': sum(s in scores for s in before[:n]) for n in DEPTHS},
                          swaps=swaps)
    out.mkdir(parents=True, exist_ok=True)
    files = {f'{arm}_final.json': dump(out / f'{arm}_final.json', lists) for arm, lists in rankings.items()}
    if (files['historical_final.json'] != previous['files']['historical_final.json']
            or files['top5_final.json'] != previous['files']['open_final.json']):
        raise ValueError('The historical or the top-five arm differs from the prior independent blind seal')
    files['final_native_depth_e180_cases.json'] = dump(out / 'final_native_depth_e180_cases.json', cases)
    arms = {}
    for arm, lists in rankings.items():
        if arm == 'historical':
            continue
        arms[arm] = dict(changed=[m for m, v in lists.items() if v != original[m]],
                         first_changed=[m for m, v in lists.items() if v[0] != original[m][0]],
                         top5_members_changed=[m for m, v in lists.items() if Counter(v[:5]) != Counter(original[m][:5])],
                         total_swaps=sum(case['swaps'][arm] for case in cases.values()))
    result = dict(status='sealed', arm_names=list(rankings), depths=DEPTHS, molecules=len(original),
                  candidate_budget=25, actual_slots=sum(map(len, original.values())), new_predictor_requests=0,
                  baseline=REFERENCE, base_submission_sha256=SUBMISSION_SHA, queries_sha256=QUERY_SHA,
                  source_60_rule_sha256=RULE_SHA, source_library_threshold=LIBRARY_THRESHOLD,
                  top5_reference=TOP5_REFERENCE, top5_seal_sha256=TOP5_SEAL_SHA,
                  input_file_sha256=hashes, files=files, arms=arms, native_forward_archives=model_meta,
                  historical_final_membership_sha256=hash_value({m: sorted(v) for m, v in original.items()}))
    dump(out / 'final_native_depth_e180_receipt.json', result)
    print('Blind fixed-slate comparison', {arm: len(data['changed']) for arm, data in arms.items()}, flush=True)


def build(out):
    # Embed the unchanged archive verification and input setup from the published transfer.
    raw_adapter = (HERE / 'final_native_top5_e180.py').read_bytes()
    if sha256(raw_adapter).hexdigest() != ADAPTER_SHA:
        raise ValueError('The verified archive input code changed')
    loader = raw_adapter.decode().split('    outputs, cases = apply(ranks, ledgers, request, scores, budget)\n')[0]
    loader = loader.replace('def run(root, out, native_hashes):', 'def load_archive(root, native_hashes):')
    loader += '    return ranks, ledgers, request, scores, budget, model_meta, hashes\n'
    payload = {
        'frozen_top5.py': frozen_module(),
        'archive_loader.py': loader,
        'depth.py': Path(__file__).read_text().split('\ndef build(out):')[0].replace(
            'from build_final_native_top5_e180 import COMMON, HERE, REFERENCE, RULE_SHA, frozen_module',
            f'REFERENCE = {REFERENCE!r}\nRULE_SHA = {RULE_SHA!r}'),
    }
    packed = {}
    for name, body in payload.items():
        compile(body, name, 'exec')
        packed[name] = COMMON['pack'](body.encode())
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
archive, = (p.parent for p in inputs.rglob('e180_ranked_input.json') if {REFERENCE.split('/')[1]!r} in p.parts)
top5, = (p for p in inputs.rglob({TOP5_SEAL!r}) if {TOP5_REFERENCE.split('/')[1]!r} in p.parts)
with tempfile.TemporaryDirectory(prefix='e180-final-native-depth-') as directory:
    scratch = Path(directory)
    for name, (blob, expected) in {packed!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Embedded depth source changed: {{name}}')
        (scratch / name).write_bytes(raw)
    sys.path.insert(0, str(scratch))
    from depth import run
    run(archive, top5, Path('/kaggle/working'))
'''
    compile(code, SLUG, 'exec')
    meta = dict(id='jiyujia/' + SLUG, title=SLUG, code_file='run_depth.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, docker_image=COMMON['IMAGE'],
                kernel_sources=[REFERENCE, TOP5_REFERENCE], dataset_sources=[COMMON['WHEEL']], competition_sources=[])
    out.mkdir(parents=True, exist_ok=True)
    (out / meta['code_file']).write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(sha256(code.encode()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    build(**vars(parser.parse_args()))
