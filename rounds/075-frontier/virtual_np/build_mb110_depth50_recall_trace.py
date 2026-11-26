"""Package a post-seal Kaggle CPU audit of MassBank depth-50 recall losses."""

import argparse
from ast import FunctionDef, parse
from collections import Counter
import csv
from hashlib import sha256
from inspect import getsource
import json
from pathlib import Path


SLUG = 'enveda-075-mb110-depth50-trace'
OUTPUT = 'mb110_depth50_recall_trace'


def slot_trace(truth_candidate, incumbent, same_spectra, proxy, weak_e1, joint, selected, identity, formula):
    truth_formula = formula(truth_candidate.smiles) if truth_candidate is not None else None
    same_formula = truth_formula is not None and truth_formula == formula(incumbent)
    incumbent_joint = incumbent in joint
    truth_joint = truth_candidate is not None and truth_candidate.smiles in joint
    gate = bool(truth_joint and same_spectra and not proxy and weak_e1)
    comparable = gate and same_formula and incumbent_joint
    dual_win = comparable and all(a > b for a, b in zip(joint[truth_candidate.smiles],
                                                        joint[incumbent], strict=True))
    truth_selected = (selected is not None and truth_candidate is not None
                      and identity(selected['challenger']) == identity(truth_candidate.smiles))
    blocked_by_earlier = bool(dual_win and not truth_selected)
    if truth_selected and not dual_win:
        raise ValueError('The archived truth action did not pass the frozen rule')
    if blocked_by_earlier and (selected is None or selected['shortlist_rank'] >= truth_candidate.rank):
        raise ValueError('A passing truth candidate was not blocked by an earlier challenger')
    stages = (
        (truth_candidate is None, 'absent_from_c34_tail'),
        (not truth_joint, 'missing_literal_joint_scores'),
        (not same_spectra, 'different_accepted_spectra'),
        (proxy, 'proxy_gl'),
        (not weak_e1, 'strong_e1'),
        (not same_formula, 'different_or_missing_formula'),
        (not incumbent_joint, 'missing_incumbent_joint_scores'),
        (not dual_win, 'no_strict_dual_win'),
        (blocked_by_earlier, 'earlier_wrong_challenger'),
    )
    outcome = next((name for failed, name in stages if failed), 'truth_selected')
    return dict(same_formula=same_formula, incumbent_joint=incumbent_joint,
                eligible_comparison=bool(comparable), strict_dual_win=bool(dual_win),
                earlier_wrong_challenger=blocked_by_earlier, truth_selected=truth_selected, outcome=outcome)


def verify_blind(seal, ranks, cases, original, ledger, outsiders, tally, exact, score_banks):
    import final_native_outsider_mb110 as stock
    from outsider_rule import formula, replace_at_rank

    if ranks['historical'] != original or any(set(arm) != set(original) for arm in ranks.values()):
        raise ValueError('Blind ranks do not match the historical 110-query baseline')
    if cases.keys() != original.keys():
        raise ValueError('Blind cases and historical queries differ')
    totals, comparable_totals = Counter(), Counter()
    selected, evidence = {arm: [] for arm in stock.TARGETS}, {}
    for mid, before in original.items():
        state, (library, proxy), (same, digests) = cases[mid], ledger[mid], exact[mid]
        ice, gl = (score_banks[kind][mid] for kind in ('ice', 'gl'))
        joint = {text: (value, gl[text]) for text, value in ice.items()
                 if value is not None and gl.get(text) is not None}
        pool = outsiders[mid]
        joint_pool = [item for item in pool if item.smiles in joint]
        coverage = dict(tally[mid], ice_requested=sum(item.smiles in ice for item in pool),
                        gl_requested=sum(item.smiles in gl for item in pool),
                        both_requested=sum(item.smiles in ice and item.smiles in gl for item in pool),
                        jointly_finite=len(joint_pool), jointly_finite_same_native_spectra=len(joint_pool) if same and not proxy else 0)
        if (state['coverage'] != coverage or state['library_max'] != library or state['proxy_gl'] is not proxy
                or state['same_nonempty_native_spectra'] != same or state['accepted_spectral_bytes_sha256'] != digests
                or state['joint_outsiders'] != [dict(rank=item.rank, smiles=item.smiles) for item in joint_pool]):
            raise ValueError(f'Blind candidate coverage failed to reproduce: {mid}')
        eligible = same and not proxy and library < stock.LIBRARY_THRESHOLD
        for arm, slot in stock.TARGETS.items():
            incumbent = before[slot - 1]
            incumbent_formula = formula(incumbent)
            comparisons = sum(formula(item.smiles) == incumbent_formula for item in joint_pool) if (
                eligible and incumbent in joint and incumbent_formula is not None) else 0
            if eligible:
                after, action, reason = replace_at_rank(before, pool, joint, slot, formula)
            else:
                after, action = before, None
                reason = 'different_accepted_spectral_bytes' if not same else 'proxy_or_strong_library'
            if (after != ranks[arm][mid] or action != state['actions'][arm] or reason != state['reasons'][arm]
                    or comparisons != state['comparable_same_formula_pairs'][arm]):
                raise ValueError(f'Blind slot action failed to reproduce: {mid}/{arm}')
            comparable_totals[arm] += comparisons
            if action:
                selected[arm].append(mid)
        totals.update(coverage)
        evidence[mid] = dict(joint=joint, same_spectra=same, proxy=proxy,
                             weak_e1=library < stock.LIBRARY_THRESHOLD, eligible=eligible,
                             any_joint_outsider=bool(joint_pool))
    if (dict(totals) != seal['coverage'] or dict(comparable_totals) != seal['comparable_pairs']
            or any(sorted(ids) != sorted(seal['arms'][arm]) for arm, ids in selected.items())
            or sum(row['any_joint_outsider'] for row in evidence.values()) != seal['queries_with_joint_outsiders']
            or sum(row['same_spectra'] for row in evidence.values()) != seal['exact_spectra_queries']):
        raise ValueError('Blind coverage totals or trigger sets failed to reproduce')
    return evidence


def trace_misses(original, known, outsiders, evidence, cases, labels, score_banks, identity, formula):
    from final_native_outsider_mb110 import TARGETS

    rows = []
    for mid, record in sorted(labels.items()):
        truth = identity(record['truth_smiles'])
        if truth is None or truth[0] != record['candidate_id']:
            raise ValueError(f'Historical label failed archived normalization: {mid}')
        observed = truth in known[mid]
        candidate = next((item for item in outsiders[mid] if identity(item.smiles) == truth), None)
        ice, gl = (score_banks[kind][mid] for kind in ('ice', 'gl'))
        bank = evidence[mid]
        literal = candidate.smiles if candidate is not None else None
        row = dict(molecule_id=mid, seen_in_original_union=observed,
                   truth_c34_tail_rank=candidate.rank if candidate is not None else None,
                   truth_literal_requested_ice=literal in ice,
                   truth_literal_requested_gl=literal in gl,
                   truth_jointly_finite=literal in bank['joint'],
                   truth_joint_same_spectra=literal in bank['joint'] and bank['same_spectra'],
                   truth_joint_native=literal in bank['joint'] and bank['same_spectra'] and not bank['proxy'],
                   truth_passed_query_gate=literal in bank['joint'] and bank['eligible'],
                   exact_shared_spectra=bank['same_spectra'], no_proxy_gl=not bank['proxy'],
                   weak_e1=bank['weak_e1'], query_eligible=bank['eligible'])
        for arm, slot in TARGETS.items():
            selected = cases[mid]['actions'][arm]
            trace = slot_trace(candidate, original[mid][slot - 1], bank['same_spectra'],
                               bank['proxy'], bank['weak_e1'], bank['joint'], selected, identity, formula)
            if observed:
                trace['outcome'] = 'in_original_union'
            row.update({f'{arm}_{key}': value for key, value in trace.items()})
            row[f'{arm}_selected_c34_rank'] = selected['shortlist_rank'] if selected else None
        rows.append(row)
    return rows


def summarize(rows, seal):
    from final_native_outsider_mb110 import TARGETS

    misses = [row for row in rows if not row['seen_in_original_union']]
    if len(misses) != 45:
        raise ValueError('The pinned MassBank source union should miss exactly 45 of 110 truths')
    phases = ('truth_c34_tail_rank', 'truth_literal_requested_ice', 'truth_literal_requested_gl',
              'truth_jointly_finite', 'truth_joint_same_spectra', 'truth_joint_native', 'truth_passed_query_gate')
    counts = {name: sum(bool(row[name]) for row in misses) for name in phases}
    per_arm = {}
    for arm in TARGETS:
        fields = ('same_formula', 'incumbent_joint', 'eligible_comparison', 'strict_dual_win',
                  'earlier_wrong_challenger', 'truth_selected')
        passes_gate = [row for row in misses if row['truth_passed_query_gate']]
        outcomes = Counter(row[f'{arm}_outcome'] for row in misses)
        per_arm[arm] = dict(
            counts_among_gate_passes={name: sum(bool(row[f'{arm}_{name}']) for row in passes_gate) for name in fields},
            cumulative_same_formula_after_gate=sum(row[f'{arm}_same_formula'] for row in passes_gate),
            cumulative_joint_incumbent_after_formula=sum(row[f'{arm}_same_formula'] and row[f'{arm}_incumbent_joint']
                                                         for row in passes_gate),
            first_failure_or_result=dict(outcomes),
            passing_truth_ids=[row['molecule_id'] for row in misses if row[f'{arm}_strict_dual_win']],
            displaced_truth_ids=[row['molecule_id'] for row in misses if row[f'{arm}_earlier_wrong_challenger']],
            selected_truth_ids=[row['molecule_id'] for row in misses if row[f'{arm}_truth_selected']],
        )
    return dict(
        molecules=len(rows), original_all_source_misses=len(misses), miss_coverage=counts, miss_arms=per_arm,
        cohort=dict(query_gate_eligible=sum(row['query_eligible'] for row in rows),
                    exact_spectra_queries=seal['exact_spectra_queries'],
                    queries_with_joint_outsiders=seal['queries_with_joint_outsiders'],
                    coverage=seal['coverage'], comparable_pairs=seal['comparable_pairs'],
                    triggers={arm: len(ids) for arm, ids in seal['arms'].items()}),
    )


def run(inputs, out, seal_sha):
    import pyarrow.parquet as pq
    from rdkit import RDLogger
    from c39_request_cache import identity_function
    import final_native_outsider_mb110 as stock
    from depth50_inputs import SUPPLEMENT, SUPPLEMENT_PINS, byte_match, request_delta, supplement_inputs
    from outsider_rule import formula
    from score_outsider import LABELS, LABELS_SHA, artifact, names, pinned, read_json, validate_seal

    RDLogger.DisableLog('rdApp.*')
    identity = identity_function()
    blind_slug, stem = names('depth50')
    blind_ref = f'jiyujia/{blind_slug}/1'
    receipt = pinned(artifact(inputs, blind_ref, stem + '_receipt.json'), seal_sha)
    seal = validate_seal(receipt, 'depth50')
    if seal['input_file_sha256'] != stock.PINS or seal['augmented_file_sha256'] != SUPPLEMENT_PINS:
        raise ValueError('The blind seal names different archive inputs')
    sealed_files = {name: pinned(receipt.parent / name, digest) for name, digest in seal['files'].items()}
    ranks = {arm: read_json(sealed_files[f'{arm}_final.json']) for arm in stock.ARMS}
    cases = read_json(sealed_files[stem + '_cases.json'])
    paths, expanded = stock.verified_inputs(inputs), supplement_inputs(inputs)
    original, known, ledger, counts, source_queries = stock.historical(paths, identity)
    native, _, native_meta = stock.model_archive(paths, original)
    augmented, score_banks, expanded_meta = stock.model_archive(expanded, original)
    exact = byte_match(paths, expanded, native, augmented)
    if (dict(counts) != seal['source_channels'] or source_queries != seal['source_query_counts']
            or native_meta != seal['native_forward_archives'] or expanded_meta != seal['augmented_forward_archives']
            or request_delta(native, augmented, exact) != seal['historical_cost']):
        raise ValueError('Source export or model archive counts differ from the blind seal')
    outsiders, tally = stock.ranked_outsiders(paths, known, identity)
    evidence = verify_blind(seal, ranks, cases, original, ledger, outsiders, tally, exact, score_banks)
    for mid, row in read_json(paths['pc_lists.json']).items():
        known[mid].update(map(identity, row['pc']))

    # Only the independent audit reads labels; all archive and action checks have already passed.
    label_path = pinned(artifact(inputs, LABELS, 'labels.parquet'), LABELS_SHA)
    label_rows = pq.read_table(label_path, columns=['molecule_id', 'candidate_id', 'truth_smiles']).to_pylist()
    labels = {row['molecule_id']: row for row in label_rows}
    if len(label_rows) != 110 or labels.keys() != original.keys():
        raise ValueError('Labels do not describe the 110 verified queries')
    rows = trace_misses(original, known, outsiders, evidence, cases, labels, score_banks, identity, formula)
    report = summarize(rows, seal)
    out.mkdir(parents=True, exist_ok=True)
    per_query = out / (OUTPUT + '_per_query.csv')
    with per_query.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report.update(blind_receipt_sha256=seal_sha, labels_sha256=LABELS_SHA, score_archive=SUPPLEMENT,
                  per_query_sha256=sha256(per_query.read_bytes()).hexdigest(), candidate_slots=2750,
                  new_predictor_requests=0, score_scope='A diagnostic of the frozen action, not a re-ranked arm')
    (out / (OUTPUT + '.json')).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False), flush=True)


def build(seal, out):
    from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack
    from build_final_native_outsider_mb110_depth50 import (
        SUPPLEMENT, SUPPLEMENT_PINS, byte_match, request_delta, spectral_array_bytes, supplement_inputs,
    )
    from build_final_native_outsider_mb110_score import BASE, DEPTH, LABELS, checksum, names, validate_seal
    from build_final_native_outsider_np60 import replace_at_rank

    validate_seal(seal, 'depth50')
    original = (HERE / 'final_native_top5.py').read_text()
    node, = (item for item in parse(original).body if isinstance(item, FunctionDef) and item.name == 'formula')
    formula_body = '\n'.join(original.splitlines()[node.decorator_list[0].lineno - 1:node.end_lineno])
    rule = ('from functools import lru_cache\nfrom rdkit import Chem\n'
            'from rdkit.Chem.rdMolDescriptors import CalcMolFormula\n\n'
            + formula_body + '\n\n' + getsource(replace_at_rank))
    runtime = ('from collections import Counter\nfrom hashlib import file_digest, sha256\n'
               'import json\nfrom pathlib import PurePosixPath\n'
               f'SUPPLEMENT = {SUPPLEMENT!r}\nSUPPLEMENT_PINS = {SUPPLEMENT_PINS!r}\n\n'
               + '\n'.join(getsource(fn) for fn in (supplement_inputs, spectral_array_bytes, byte_match, request_delta)))
    modules = dict(
        outsider_rule=rule.encode(), depth50_inputs=runtime.encode(), recall_trace=Path(__file__).read_bytes(),
        score_outsider=(HERE / 'build_final_native_outsider_mb110_score.py').read_bytes(),
        **{name: (HERE / (name + '.py')).read_bytes()
           for name in ('c39_request_cache', 'final_native_outsider_mb110')},
    )
    payload = {}
    for module, source in modules.items():
        name = module + '.py'
        compile(source, name, 'exec')
        payload[name] = pack(source)
    code = f'''
import base64, subprocess, sys, tempfile, zlib
from hashlib import sha256
from pathlib import Path
inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='massbank110-depth50-trace-') as tmp:
    root = Path(tmp)
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Changed embedded audit source: {{name}}')
        (root / name).write_bytes(raw)
    sys.path.insert(0, str(root))
    from recall_trace import run
    run(inputs, Path('/kaggle/working'), {checksum(seal)!r})
'''
    compile(code, SLUG, 'exec')
    blind_slug, _ = names('depth50')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='run_trace.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, dataset_sources=[WHEEL],
                kernel_sources=[f'jiyujia/{blind_slug}/1', BASE, DEPTH, SUPPLEMENT, LABELS],
                competition_sources=[], docker_image=IMAGE)
    out.mkdir(exist_ok=True, parents=True)
    (out / 'run_trace.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return sha256(code.encode()).hexdigest()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--seal', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    print(build(**vars(cli.parse_args())))
