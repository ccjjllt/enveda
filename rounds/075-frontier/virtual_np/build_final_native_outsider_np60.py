"""Package a blind final-list replacement of a single original-source outsider."""

import argparse
from collections import Counter
from hashlib import sha256
from inspect import getsource
import json
from math import isfinite
from pathlib import Path
from zipfile import ZipFile

from allocate_c39_shadow import SOURCE_CHANNELS, SHORTLISTS, UNSEEN_RECEIPT, UNSEEN_SHORTLIST
from build_final_native_formula_np60 import BASE, SNAP, TOP5_REFERENCE, TOP5_SEAL, TOP5_SEAL_SHA, TOP5_SHA
from build_final_native_top5_e180 import COMMON, HERE, RULE_SHA, frozen_module


SLUG = 'enveda-075-final-native-outsider-np60'
SEAL = 'final_native_outsider_np60_receipt.json'
UNSEEN = SHORTLISTS['unseen'][0]
TARGETS = {'rank5': 5, 'rank10': 10}
ARMS = ('top5', *TARGETS)


def replace_at_rank(baseline, shortlist, scores, slot, formula):
    if len(baseline) < slot:
        return baseline, None, 'no_slot'
    incumbent = baseline[slot - 1]
    if incumbent not in scores or (incumbent_formula := formula(incumbent)) is None:
        return baseline, None, 'no_joint_incumbent_score'
    for candidate in shortlist:
        challenger = candidate.smiles
        if challenger not in scores or formula(challenger) != incumbent_formula:
            continue
        deltas = [a - b for a, b in zip(scores[challenger], scores[incumbent], strict=True)]
        if all(delta > 0 for delta in deltas):
            after = list(baseline)
            after[slot - 1] = challenger
            return after, dict(shortlist_rank=int(candidate.rank), slot=slot, challenger=challenger,
                               incumbent=incumbent, formula=incumbent_formula,
                               difference=dict(ice=deltas[0], gl=deltas[1])), 'replaced'
    return baseline, None, 'no_dual_win'


def run(snapshot, baseline_zip, top5_seal, shortlist_dir, out):
    from allocate_c39_shadow import read_shortlist
    from c39_request_cache import identity_function
    from frozen_top5 import (BASE_SHA, LIBRARY_THRESHOLD, QUERY_SHA, dump, formula, hash_value,
                             native_evidence, original_inputs, verify)

    previous = json.loads(verify(top5_seal.read_bytes(), TOP5_SEAL_SHA))
    original, ledgers, request, raw_scores, budget, hashes = original_inputs(snapshot, baseline_zip)
    membership = hash_value({mid: sorted(values) for mid, values in original.items()})
    if (previous['input_file_sha256'] != hashes or previous['files']['open_final.json'] != TOP5_SHA
            or previous['historical_final_membership_sha256'] != membership
            or Counter(map(len, original.values())) != {25: 59, 1: 1}):
        raise ValueError('Different native scores or candidates from the final top-five control')
    sealed = json.loads(verify(top5_seal.with_name('open_final.json').read_bytes(), TOP5_SHA))
    identities = identity_function()
    with ZipFile(baseline_zip) as bundle:
        manifest = json.loads(bundle.read('pilot_arm_receipt.json'))
        source_raw = verify(bundle.read('four_source_lists.json'), manifest['hashes']['four_source_lists.json'])
    original_sources = json.loads(source_raw)
    if not sealed.keys() == original.keys() == original_sources.keys():
        raise ValueError('Different queries in original and frozen rankings or sources')
    known = {mid: {identities(text) for text in set(original[mid]) |
                   {s for channel in SOURCE_CHANNELS for s in original_sources[mid][channel]}}
             for mid in original}
    unseen = read_shortlist(shortlist_dir, known, identities, 'unseen')
    rankings = {arm: {} for arm in ARMS}
    cases = {}
    for mid, before in original.items():
        baseline = sealed[mid]
        if (Counter(baseline[:5]) != Counter(before[:5]) or baseline[5:] != before[5:]
                or [formula(s) for s in baseline[:5]] != [formula(s) for s in before[:5]]):
            raise ValueError(f'Final top-five control differs from the original fixed slate: {mid}')
        strength = ledgers[mid]['library_max']
        if not isfinite(strength):
            raise ValueError(f'Nonfinite E1 library maximum: {mid}')
        candidates = unseen[mid]
        if any(identities(candidate.smiles) in known[mid] for candidate in candidates):
            raise ValueError(f'An outsider matches an original-source identity: {mid}')
        request_set = set(baseline) | {candidate.smiles for candidate in candidates}
        audit, paired = native_evidence(mid, request_set, request, raw_scores, budget['cases'][mid]['predictor_budget'])
        ice, gl = (audit[model]['values'] for model in ('ice', 'gl'))
        scores = {s: (ice[s], gl[s]) for s in ice.keys() & gl.keys()}
        eligible = paired and strength < LIBRARY_THRESHOLD
        rankings['top5'][mid] = baseline
        actions = {}
        reasons = {}
        for arm, slot in TARGETS.items():
            if eligible:
                after, action, reason = replace_at_rank(baseline, candidates, scores, slot, formula)
            else:
                after, action, reason = baseline, None, 'no_shared_native_group_or_strong_library'
            changed = [i for i, (a, b) in enumerate(zip(baseline, after, strict=True), 1) if a != b]
            if changed != ([slot] if action else []):
                raise ValueError(f'Final slot budget or unchanged positions differ: {mid}/{arm}')
            if action and identities(after[slot - 1]) in known[mid]:
                raise ValueError(f'Treated list includes a duplicate or original-source outsider: {mid}/{arm}')
            rankings[arm][mid] = after
            actions[arm] = action
            reasons[arm] = reason
        cases[mid] = dict(library_max=strength, same_nonempty_native_spectra=paired,
                          jointly_native_scored_outsiders=sum(row.smiles in scores for row in candidates),
                          actions=actions, reasons=reasons)

    out.mkdir(parents=True, exist_ok=True)
    files = {f'{arm}_final.json': dump(out / f'{arm}_final.json', rows) for arm, rows in rankings.items()}
    if files['top5_final.json'] != TOP5_SHA:
        raise ValueError('Frozen control order changed')
    case_name = 'final_native_outsider_np60_cases.json'
    files[case_name] = dump(out / case_name, cases)
    fired = {arm: [mid for mid, case in cases.items() if case['actions'][arm] is not None] for arm in TARGETS}
    tally = {arm: Counter(case['reasons'][arm] for case in cases.values()) for arm in TARGETS}
    result = dict(status='sealed', arm_names=list(ARMS), molecules=len(original), candidate_budget=25,
                  actual_slots=sum(map(len, original.values())), new_predictor_requests=0,
                  baseline=BASE, baseline_archive_sha256=BASE_SHA, queries_sha256=QUERY_SHA,
                  targets=TARGETS, source_60_rule_sha256=RULE_SHA, source_library_threshold=LIBRARY_THRESHOLD,
                  top5_reference=TOP5_REFERENCE, top5_seal_sha256=TOP5_SEAL_SHA,
                  snapshot=SNAP, input_file_sha256=hashes, shortlist_ref=UNSEEN,
                  shortlist_sha256=UNSEEN_SHORTLIST, shortlist_receipt_sha256=UNSEEN_RECEIPT,
                  historical_final_membership_sha256=membership, files=files, arms=fired, reasons=tally)
    dump(out / SEAL, result)
    print('Blind final outsider replacement counts', {arm: len(mids) for arm, mids in fired.items()}, flush=True)


def build(out):
    constants = ('SEAL', 'ARMS', 'TARGETS', 'TOP5_SHA', 'TOP5_REFERENCE', 'TOP5_SEAL_SHA', 'BASE', 'SNAP', 'RULE_SHA',
                 'SOURCE_CHANNELS', 'UNSEEN', 'UNSEEN_SHORTLIST', 'UNSEEN_RECEIPT')
    source = '\n'.join(('from collections import Counter', 'import json', 'from math import isfinite',
                        'from zipfile import ZipFile', *(f'{name} = {globals()[name]!r}' for name in constants))) + '\n'
    source += '\n'.join(map(getsource, (replace_at_rank, run)))
    modules = {'frozen_top5.py': frozen_module(), 'np60_outsider.py': source,
               **{name: (HERE / name).read_text() for name in ('allocate_c39_shadow.py', 'c39_request_cache.py')}}
    payload = {}
    for name, body in modules.items():
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

with tempfile.TemporaryDirectory(prefix='np60-final-native-outsider-') as directory:
    scratch = Path(directory)
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Embedded source changed: {{name}}')
        (scratch / name).write_bytes(raw)
    sys.path.insert(0, str(scratch))
    from np60_outsider import run
    run(mounted({SNAP!r}, 'c39_snapshot_receipt.json').parent, mounted({BASE!r}, 'baseline.zip'),
        mounted({TOP5_REFERENCE!r}, {TOP5_SEAL!r}), mounted({UNSEEN!r}, 'unseen_relist_receipt.json').parent,
        Path('/kaggle/working'))
'''
    compile(code, SLUG, 'exec')
    meta = dict(id='jiyujia/' + SLUG, title=SLUG, code_file='run_outsider.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, docker_image=COMMON['IMAGE'],
                kernel_sources=[BASE, SNAP, TOP5_REFERENCE, UNSEEN], dataset_sources=[COMMON['WHEEL']],
                competition_sources=[])
    out.mkdir(parents=True, exist_ok=True)
    (out / meta['code_file']).write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(sha256(code.encode()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    build(**vars(parser.parse_args()))
