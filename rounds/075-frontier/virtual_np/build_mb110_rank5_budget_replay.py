"""Package the label-free MassBank 110 rank-5 prediction-budget replay."""

import argparse
from ast import FunctionDef, parse
from collections import Counter
from hashlib import file_digest, sha256
from inspect import getsource
import json
from pathlib import Path

from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack
from build_final_native_outsider_mb110 import BASE, DEPTH
from build_final_native_outsider_mb110_depth50 import (
    SEAL, STEM, SUPPLEMENT, SUPPLEMENT_PINS, byte_match, request_delta, spectral_array_bytes, supplement_inputs,
)
from build_final_native_outsider_np60 import replace_at_rank


SLUG = 'enveda-075-mb110-rank5-budget-replay'
REFERENCE = 'jiyujia/enveda-075-final-native-outsider-mb110-depth50/1'
REFERENCE_SHA = '0691bc1d3471b9869ae6f80ef42b453a4946a365987208ed41d033a9c1d5dcc8'
SUMMARY = 'mb110_rank5_budget_replay_receipt.json'
PER_QUERY = 'mb110_rank5_budget_replay_per_query.json'
CAPS = {'all': None, 'cap2': 2, 'cap4': 4, 'cap8': 8}
MODELS = ('ice', 'gl')


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def load_reference(path):
    if digest(path) != REFERENCE_SHA:
        raise ValueError('Expected the official sealed depth-50 version 1 receipt')
    receipt = json.loads(path.read_bytes())
    if (receipt['status'] != 'sealed' or receipt['molecules'] != 110 or receipt['candidate_budget'] != 25
            or receipt['actual_slots'] != 2750 or receipt['targets'] != {'rank5': 5, 'rank10': 10}
            or receipt['exact_spectra_queries'] != 110 or receipt['scores_reference'] != SUPPLEMENT
            or receipt['tail_ranks'] != [26, 100] or receipt['new_predictor_requests'] != 0):
        raise ValueError('The published depth-50 final-list experiment does not match this replay')
    return receipt


def admission(pool, incumbent, incumbent_formula, strength, proxy, same_spectra, augmented, threshold, formula):
    """Apply request-side conditions only, before seeing any physical score."""
    if not same_spectra or proxy or strength >= threshold or incumbent_formula is None:
        return []
    requested_in_both = set(augmented['ice']['cands']) & set(augmented['gl']['cands'])
    if incumbent not in requested_in_both:
        return []
    return [candidate for candidate in pool if 26 <= candidate.rank <= 50
            and candidate.smiles in requested_in_both and formula(candidate.smiles) == incumbent_formula]


def budget(selected, incumbent, stock_request, expanded_request):
    if not selected:
        return dict(challenger_requests=0, new_challenger_requests=0, incumbent_requests_if_rescored=0,
                    new_incumbent_requests_if_missing=0, incremental_if_stock_reused=0,
                    incremental_with_incumbent_rescore=0, same_run_requests=0)
    scans = len(expanded_request['spectra'])
    stock_literals = set(stock_request['cands'])
    challengers = len(selected) * scans
    new_challengers = sum(candidate.smiles not in stock_literals for candidate in selected) * scans
    new_incumbent = scans if incumbent not in stock_literals else 0
    return dict(challenger_requests=challengers, new_challenger_requests=new_challengers,
                incumbent_requests_if_rescored=scans, new_incumbent_requests_if_missing=new_incumbent,
                incremental_if_stock_reused=new_challengers + new_incumbent,
                incremental_with_incumbent_rescore=new_challengers + scans, same_run_requests=challengers + scans)


def run(inputs, out, rule_sha):
    from rdkit import RDLogger
    from c39_request_cache import identity_function, write_json
    from depth50_archive import byte_match, request_delta, supplement_inputs
    import final_native_outsider_mb110 as stock
    from outsider_rule import formula, replace_at_rank

    RDLogger.DisableLog('rdApp.*')
    ref_slug = REFERENCE.split('/')[1]
    receipt, = (p for p in inputs.rglob(SEAL) if ref_slug in p.parts)
    sealed = load_reference(receipt)
    frozen = {}
    for name, expected in sealed['files'].items():
        path = receipt.with_name(name)
        if digest(path) != expected:
            raise ValueError(f'Changed sealed depth-50 artifact: {name}')
        frozen[name] = json.loads(path.read_bytes())
    paths, extra = stock.verified_inputs(inputs), supplement_inputs(inputs)
    identity = identity_function()
    original, known, ledger, source_counts, source_queries = stock.historical(paths, identity)
    stock_inputs, stock_scores, stock_meta = stock.model_archive(paths, original)
    augmented, scores, aug_meta = stock.model_archive(extra, original)
    exact = byte_match(paths, extra, stock_inputs, augmented)
    delta = request_delta(stock_inputs, augmented, exact)
    if (sealed['input_file_sha256'] != stock.PINS or sealed['augmented_file_sha256'] != SUPPLEMENT_PINS
            or original != frozen['historical_final.json'] or set(original) != set(frozen['rank5_final.json'])
            or sealed['historical_cost'] != delta or sealed['source_channels'] != dict(source_counts)
            or sealed['source_query_counts'] != source_queries or sealed['native_forward_archives'] != stock_meta
            or sealed['augmented_forward_archives'] != aug_meta or sealed['source_60_rule_sha256'] != rule_sha
            or sum(value[0] for value in exact.values()) != 110
            or any(delta[m]['historical_added_literal_candidate_spectra'] != 3863 for m in MODELS)):
        raise ValueError('Replay inputs, accepted spectra, or historical 3,863 additional requests changed')
    outsiders, _ = stock.ranked_outsiders(paths, known, identity)
    reference_ranks = frozen['rank5_final.json']
    reference_cases = frozen[STEM + '_cases.json']
    ranks = {name: {} for name in CAPS}
    costs = {name: {model: Counter() for model in MODELS} for name in CAPS}
    mismatches = {name: [] for name in CAPS}
    fired = {name: [] for name in CAPS}
    admitted_n, active_n = Counter(), Counter()
    rows = []
    for mid, before in original.items():
        strength, proxy = ledger[mid]
        same, spectral_digests = exact[mid]
        model_requests = {model: augmented[model][mid] for model in MODELS}
        incumbent = before[4]
        incumbent_formula = formula(incumbent)
        admitted = admission(outsiders[mid], incumbent, incumbent_formula, strength, proxy, same, model_requests,
                             stock.LIBRARY_THRESHOLD, formula)
        both = {text: (value, scores['gl'][mid][text]) for text, value in scores['ice'][mid].items()
                if value is not None and scores['gl'][mid].get(text) is not None}
        full, full_action, _ = replace_at_rank(before, outsiders[mid], both, 5, formula) if (
            same and not proxy and strength < stock.LIBRARY_THRESHOLD) else (before, None, None)
        if (full != reference_ranks[mid] or full_action != reference_cases[mid]['actions']['rank5']
                or reference_cases[mid]['accepted_spectral_bytes_sha256'] != spectral_digests):
            raise ValueError(f'The unrestricted replay did not reproduce the published result: {mid}')
        incumbent_drift = {model: incumbent in stock_scores[model][mid] and incumbent in scores[model][mid]
                           and stock_scores[model][mid][incumbent] != scores[model][mid][incumbent]
                           for model in MODELS}
        row = dict(molecule_id=mid, admitted_c34_ranks=[item.rank for item in admitted],
                   archived_rank5_winner_c34_rank=full_action['shortlist_rank'] if full_action else None,
                   incumbent_score_differs_between_archives=incumbent_drift, budgets={})
        for name, cap in CAPS.items():
            selected = admitted if cap is None else admitted[:cap]
            after, action, _ = replace_at_rank(before, selected, both, 5, formula)
            if len(after) != 25 or after[:4] != before[:4] or after[5:] != before[5:]:
                raise ValueError(f'Fixed final roster or slots changed: {mid}/{name}')
            ranks[name][mid] = after
            admitted_n[name] += len(selected)
            active_n[name] += bool(selected)
            if action:
                fired[name].append(mid)
            if after != full:
                mismatches[name].append(mid)
            model_counts = {model: budget(selected, incumbent, stock_inputs[model][mid], augmented[model][mid])
                            for model in MODELS}
            for model in MODELS:
                costs[name][model].update(model_counts[model])
            row['budgets'][name] = dict(requests=model_counts,
                                        winner_c34_rank=action['shortlist_rank'] if action else None,
                                        reproduces_published_rank5=after == full)
        rows.append(row)

    out.mkdir(parents=True, exist_ok=True)
    files = {f'{name}_final.json': write_json(out / f'{name}_final.json', arm)
             for name, arm in {'historical': original, **ranks}.items()}
    files[PER_QUERY] = write_json(out / PER_QUERY, rows)
    if (files['historical_final.json'] != sealed['files']['historical_final.json']
            or (files['all_final.json'] == sealed['files']['rank5_final.json']) != (not mismatches['all'])):
        raise ValueError('Final artifact checksums disagree with the verified roster comparison')
    arms = {}
    for name, cap in CAPS.items():
        arms[name] = dict(cap=cap, admitted_query_candidates=admitted_n[name], active_queries=active_n[name],
                          triggered_queries=fired[name], differs_from_published_queries=mismatches[name],
                          reproduces_published_rank5=not mismatches[name], requests_by_model={})
        for model, by_scenario in costs[name].items():
            arms[name]['requests_by_model'][model] = dict(
                by_scenario, archived_depth50_added_requests=3863,
                savings_if_stock_reused=3863 - by_scenario['incremental_if_stock_reused'],
                savings_with_incumbent_rescore=3863 - by_scenario['incremental_with_incumbent_rescore'])
    summary = dict(status='sealed', reference=REFERENCE, reference_receipt_sha256=REFERENCE_SHA,
                   baseline=stock.BASE, candidate_archive=stock.DEPTH, physical_archive=SUPPLEMENT,
                   measured_spectra_match='byte equality across all four accepted spectra arrays',
                   molecules=110, candidate_budget=25, actual_slots=2750, slot=5, candidate_ranks=[26, 50],
                   predictor_requests_run=0, historical_cost=delta,
                   pre_score_admission='C34 rank 26–50, outside all original sources, literal in both augmented requests, '
                   'same formula as rank-5 incumbent, weak E1 without proxy and byte-identical accepted spectra',
                   budget_semantics='Request arithmetic on frozen spectra. Incumbent rescoring once per active query '
                   'is reported separately; cached-score replay does not establish fresh-run numerical equivalence.',
                   incumbent_score_differs_between_archives_active_queries={model: sum(
                       row['incumbent_score_differs_between_archives'][model] and bool(row['admitted_c34_ranks'])
                       for row in rows) for model in MODELS}, arms=arms, files=files)
    write_json(out / SUMMARY, summary)
    print('MassBank 110 rank-5 request budget replay', json.dumps(arms), flush=True)


def build(seal, out):
    reference = load_reference(seal)
    original = (HERE / 'final_native_top5.py').read_text()
    node, = (item for item in parse(original).body if isinstance(item, FunctionDef) and item.name == 'formula')
    rule = ('from functools import lru_cache\nfrom rdkit import Chem\n'
            'from rdkit.Chem.rdMolDescriptors import CalcMolFormula\n\n'
            + '\n'.join(original.splitlines()[node.decorator_list[0].lineno - 1:node.end_lineno])
            + '\n\n' + getsource(replace_at_rank))
    rule_sha = sha256(rule.encode()).hexdigest()
    if reference['source_60_rule_sha256'] != rule_sha:
        raise ValueError('The unchanged official fixed-slot rule is required')
    archive = ('from collections import Counter\nfrom hashlib import file_digest, sha256\n'
               'import json\nfrom pathlib import PurePosixPath\n'
               f'SUPPLEMENT = {SUPPLEMENT!r}\nSUPPLEMENT_PINS = {SUPPLEMENT_PINS!r}\n\n'
               + '\n'.join(map(getsource, (supplement_inputs, spectral_array_bytes, byte_match, request_delta))))
    constants = ('REFERENCE', 'REFERENCE_SHA', 'SEAL', 'STEM', 'SUPPLEMENT', 'SUPPLEMENT_PINS',
                 'SUMMARY', 'PER_QUERY', 'CAPS', 'MODELS')
    replay = ('from collections import Counter\nfrom hashlib import file_digest\nimport json\n\n'
              + '\n'.join(f'{name} = {globals()[name]!r}' for name in constants) + '\n\n'
              + '\n'.join(map(getsource, (digest, load_reference, admission, budget, run))))
    modules = {'outsider_rule.py': rule, 'depth50_archive.py': archive, 'budget_replay.py': replay,
               **{name: (HERE / name).read_text() for name in ('c39_request_cache.py', 'final_native_outsider_mb110.py')}}
    payload = {}
    for name, source in modules.items():
        compile(source, name, 'exec')
        payload[name] = pack(source.encode())
    code = f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='massbank110-rank5-budget-') as directory:
    scratch = Path(directory)
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Embedded source changed: {{name}}')
        (scratch / name).write_bytes(raw)
    sys.path.insert(0, directory)
    from budget_replay import run
    run(inputs, Path('/kaggle/working'), {rule_sha!r})
'''
    compile(code, SLUG, 'exec')
    metadata = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='replay.py', language='python', kernel_type='script',
                    is_private=True, enable_gpu=False, enable_internet=False, dataset_sources=[WHEEL],
                    kernel_sources=[BASE, DEPTH, SUPPLEMENT, REFERENCE], competition_sources=[], docker_image=IMAGE)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'replay.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return sha256(code.encode()).hexdigest()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--seal', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    print(build(**vars(cli.parse_args())))
