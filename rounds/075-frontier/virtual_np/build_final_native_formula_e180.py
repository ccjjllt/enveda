"""Package a blind cross-formula head swap with and without second-structure support."""

import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from inspect import getsource
import json
from math import isfinite
from pathlib import Path

from build_final_native_depth_e180 import ADAPTER_SHA, TOP5_REFERENCE, TOP5_SEAL, TOP5_SEAL_SHA
from build_final_native_top5_e180 import COMMON, HERE, REFERENCE, RULE_SHA, frozen_module


SLUG = 'enveda-075-final-native-formula-e180'
SEAL = 'final_native_formula_e180_receipt.json'
ARMS = ('top5', 'dual_single', 'dual_replicated')
PREFIX = 10
TOP5_SHA = '0d2b52c1f14e1ed060ad8eaea44a9e3cd7eef54c6cf2cf3389732e9b59e54339'


@lru_cache(maxsize=10000)
def identity(smiles):
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    heavy = tuple(sorted(Counter(a.GetSymbol() for a in molecule.GetAtoms()
                                 if a.GetAtomicNum() > 1).items()))
    Chem.RemoveStereochemistry(molecule)
    canonical = rdMolStandardize.TautomerEnumerator().Canonicalize(molecule)
    key = Chem.MolToInchiKey(canonical)[:14]
    return (key, heavy) if len(key) == 14 else None


def swap_choices(original, baseline, by_formula, scores):
    first = baseline[0]
    original_order = {smiles: rank for rank, smiles in enumerate(original[:PREFIX], 1)}
    first_formula = by_formula[first]
    pairs = {}
    used = set()
    if first_formula and first in scores and (first_identity := identity(first)) is not None:
        pairs[first_formula] = [first]
        used.add(first_identity)
        for smiles in original[:PREFIX]:
            formula = by_formula[smiles]
            if not formula or smiles not in scores:
                continue
            representatives = pairs.setdefault(formula, [])
            if len(representatives) == 2 or (key := identity(smiles)) is None or key in used:
                continue
            representatives.append(smiles)
            used.add(key)

    def margin(challenger, incumbent):
        return [new - old for new, old in zip(scores[challenger], scores[incumbent], strict=True)]

    evidence = []
    for formula, reps in pairs.items():
        if formula == first_formula or not reps:
            continue
        first_delta = margin(reps[0], first)
        first_wins = all(value > 0 for value in first_delta)
        both_have_backup = len(reps) == len(pairs[first_formula]) == 2
        backup_delta = margin(reps[1], pairs[first_formula][1]) if both_have_backup else None
        evidence.append(dict(formula=formula, original_rank=original_order[reps[0]],
                             prior_top5_rank=baseline.index(reps[0]) + 1,
                             representatives=reps, first_delta=first_delta, backup_delta=backup_delta,
                             dual_single=first_wins,
                             dual_replicated=first_wins and backup_delta is not None
                             and all(value > 0 for value in backup_delta)))
    evidence.sort(key=lambda row: row['original_rank'])
    choices = {arm: next((row for row in evidence if row[arm]), None) for arm in ARMS[1:]}
    return choices, dict(incumbent_formula=first_formula, incumbent_representatives=pairs.get(first_formula, []),
                         alternatives=evidence)


def run(archive, top5_seal, out):
    from archive_loader import QUERY_SHA, SUBMISSION_SHA, load_archive
    from frozen_top5 import LIBRARY_THRESHOLD, dump, formula, hash_value, native_evidence, verify

    previous = json.loads(verify(top5_seal.read_bytes(), TOP5_SEAL_SHA))
    native_hashes = {name: digest for name, digest in previous['input_file_sha256'].items()
                     if name.startswith(('ice_work/', 'gl_work/'))}
    if len(native_hashes) != 6 or previous['files']['open_final.json'] != TOP5_SHA:
        raise ValueError('Missing original model artifacts or changed top-five result')
    original, ledgers, request, raw_scores, budget, model_meta, hashes = load_archive(archive, native_hashes)
    sealed = json.loads(verify(top5_seal.with_name('open_final.json').read_bytes(), TOP5_SHA))
    if sealed.keys() != original.keys():
        raise ValueError('Original and top-five archives contain different queries')
    rankings = dict(top5=sealed, dual_single={}, dual_replicated={})
    cases = {}
    for mid, before in original.items():
        baseline = sealed[mid]
        formulas = {s: formula(s) for s in before[:PREFIX]}
        if (Counter(baseline[:5]) != Counter(before[:5]) or baseline[5:] != before[5:]
                or [formulas[s] for s in baseline[:5]] != [formulas[s] for s in before[:5]]):
            raise ValueError(f'Former output changed candidate or formula membership: {mid}')
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
                if not 1 <= i < PREFIX or action['formula'] == details['incumbent_formula']:
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
    case_file = 'final_native_formula_e180_cases.json'
    files[case_file] = dump(out / case_file, cases)
    fired = {arm: [mid for mid, case in cases.items() if case['actions'][arm] is not None] for arm in ARMS[1:]}
    result = dict(status='sealed', arm_names=list(ARMS), molecules=len(original), candidate_budget=25,
                  actual_slots=sum(map(len, original.values())), new_predictor_requests=0,
                  baseline=REFERENCE, base_submission_sha256=SUBMISSION_SHA, queries_sha256=QUERY_SHA,
                  first_prefix=PREFIX, source_60_rule_sha256=RULE_SHA, source_library_threshold=LIBRARY_THRESHOLD,
                  top5_reference=TOP5_REFERENCE, top5_seal_sha256=TOP5_SEAL_SHA, input_file_sha256=hashes,
                  files=files, arms=fired, native_forward_archives=model_meta,
                  historical_final_membership_sha256=hash_value({m: sorted(v) for m, v in original.items()}))
    dump(out / SEAL, result)
    print('Blind cross-formula action counts', {arm: len(mids) for arm, mids in fired.items()}, flush=True)


def build(out):
    raw_adapter = (HERE / 'final_native_top5_e180.py').read_bytes()
    if sha256(raw_adapter).hexdigest() != ADAPTER_SHA:
        raise ValueError('The verified archive input code changed')
    loader = raw_adapter.decode().split('    outputs, cases = apply(ranks, ledgers, request, scores, budget)\n')[0]
    loader = loader.replace('def run(root, out, native_hashes):', 'def load_archive(root, native_hashes):')
    loader += '    return ranks, ledgers, request, scores, budget, model_meta, hashes\n'
    constants = ('SLUG', 'SEAL', 'ARMS', 'PREFIX', 'TOP5_SHA', 'TOP5_REFERENCE', 'TOP5_SEAL_SHA', 'REFERENCE', 'RULE_SHA')
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
archive, = (p.parent for p in inputs.rglob('e180_ranked_input.json') if {REFERENCE.split('/')[1]!r} in p.parts)
top5, = (p for p in inputs.rglob({TOP5_SEAL!r}) if {TOP5_REFERENCE.split('/')[1]!r} in p.parts)
with tempfile.TemporaryDirectory(prefix='e180-final-native-formula-') as directory:
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
    meta = dict(id='jiyujia/' + SLUG, title=SLUG, code_file='run_formula.py', language='python', kernel_type='script',
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
