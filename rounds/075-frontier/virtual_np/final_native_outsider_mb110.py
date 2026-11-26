"""Blind final-slot challenger from the independent MassBank C34 PubChem tail."""

from collections import Counter, defaultdict, namedtuple
import csv
from hashlib import file_digest
import json
from math import isclose, isfinite
from pathlib import PurePosixPath

import pyarrow.parquet as pq
from rdkit import RDLogger

from c39_request_cache import checked_scores, identity_function, population, write_json
from outsider_rule import formula, replace_at_rank


BASE = 'jiyujia/enveda-075b-massbank-pipeline/1'
DEPTH = 'jiyujia/enveda-075b-massbank-pubchem-depth/2'
SLUG = 'enveda-075-final-native-outsider-mb110'
STEM = 'final_native_outsider_mb110'
SEAL = STEM + '_receipt.json'
PINS = {
    'submission.csv': 'a2277d569ccf838c11cc2b8fd36cf6bcb1b6565ffbe3c4312017480eaac7c5a1',
    'candidates.parquet': 'a9a63eb9a5c8f55d8965f499b447842ee679ef4d71ef0a0c00677eb97d6c95d7',
    'source_candidates.parquet': '805a7072a85aa53a0901387bd6b4799b98dc919e83577032749f4db19dfc09ac',
    'execution_receipt.json': '6800b2a2f0227c3ace77c696649a6e6ac90ae2bddff863f4539c3b639d69e851',
    'frozen_candidate_evidence.parquet': '0dcc671e581b5632d971c4422d8a748533dc688eab2d53ef5a41b923589d29f8',
    'pc_lists.json': '58092dc1f9db483f2681a7f90efb81d8f18fb6254c7982d90ca042d80c6d8f6c',
    'ice_work/ice_in.json': '03d101f53cdd58b4aa8894409fad731c7cb5f78001a8486c6c92a6ad52dbbe4b',
    'ice_work/ice_out.json': '7dc003598eca6fac8c2bb222b81b0c37dded983ade02242bdbd19a22af53111e',
    'ice_work/ice_out.json.meta.json': 'bd6dc59511dbb81316c5f319a55d82aba3ea6d8deb7bdc7afc568819b994aaea',
    'gl_work/gl_in.json': '03d101f53cdd58b4aa8894409fad731c7cb5f78001a8486c6c92a6ad52dbbe4b',
    'gl_work/gl_out.json': '264b3250616e6cc221e115ac85c2cecb06a5952b1c895b368a678df84ef2b32b',
    'gl_work/gl_out.json.meta.json': '06f3bcd22a66289d5ee053d31bf06c109e6dcfe74545a17db282a7c8d6427ffb',
    'pc_depth.json': 'ca1042629166b8935e1f08f726586246c763ee31f64de02865c41a19d5363f24',
}
TARGETS = {'rank5': 5, 'rank10': 10}
ARMS = ('historical', *TARGETS)
LIBRARY_THRESHOLD = 0.90
SOURCE_REQUIRED = {'engine1', 'engine2', 'pubchem', 'prefusion'}
SOURCE_KINDS = SOURCE_REQUIRED | {'fusion_overflow', 'apex_external'}
Candidate = namedtuple('Candidate', ('rank', 'smiles'))


def verified_inputs(root):
    paths = {}
    for name, expected in PINS.items():
        ref = DEPTH if name == 'pc_depth.json' else BASE
        slug = ref.split('/')[1]
        suffix = PurePosixPath(name).parts
        path, = (p for p in root.rglob(suffix[-1]) if slug in p.parts and p.parts[-len(suffix):] == suffix)
        with path.open('rb') as f:
            if file_digest(f, 'sha256').hexdigest() != expected:
                raise ValueError(f'Changed original blind archive input: {ref}/{name}')
        paths[name] = path
    return paths


def read_json(path):
    return json.loads(path.read_bytes())


def historical(paths, identity):
    execution = read_json(paths['execution_receipt.json'])
    if (execution['is_full_run'] is not True or execution['processed_molecules'] != 110
            or execution['source_sha256'] != 'bfcaed3c16beab294215fac37f50754ce1187f4e64cec1da634bb3dbe9b75c0c'):
        raise ValueError('The archived original MassBank run was not complete')
    with paths['submission.csv'].open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['molecule_id', 'smiles']:
            raise ValueError('Unexpected archived submission fields')
        rows = list(reader)
    final = {row['molecule_id']: row['smiles'].split(';') if row['smiles'] else [] for row in rows}
    if len(rows) != 110 or len(final) != 110 or any(not 1 <= len(listing) <= 25 for listing in final.values()):
        raise ValueError('Unexpected population or slots in the original final submission')

    columns = ['molecule_id', 'candidate_id', 'smiles', 'rank']
    original_final = defaultdict(dict)
    for row in pq.read_table(paths['candidates.parquet'], columns=columns).to_pylist():
        listing = original_final[row['molecule_id']]
        if row['rank'] in listing:
            raise ValueError('Duplicate position in the archived final-candidate export')
        listing[row['rank']] = row['smiles']
    if (set(original_final) != set(final) or any(original_final[mid] != dict(enumerate(listing, 1))
                                                for mid, listing in final.items())):
        raise ValueError('Final export differs from the original submission string order')

    known = {mid: {identity(smiles) for smiles in listing} for mid, listing in final.items()}
    source_counts = Counter()
    source_queries = defaultdict(set)
    positions = set()
    for row in pq.read_table(paths['source_candidates.parquet'], columns=[*columns, 'source']).to_pylist():
        mid, source, rank = row['molecule_id'], row['source'], row['rank']
        key = (mid, source, rank)
        normalized = identity(row['smiles'])
        if (mid not in final or source not in SOURCE_KINDS or not isinstance(rank, int) or rank <= 0
                or key in positions or normalized is None or normalized[0] != row['candidate_id']):
            raise ValueError('Invalid or repeated original source entry')
        positions.add(key)
        source_counts[source] += 1
        source_queries[source].add(mid)
        known[mid].add(normalized)
    if (not SOURCE_REQUIRED <= source_counts.keys() or any(None in identities for identities in known.values())
            or source_queries['pubchem'] != set(final)):
        raise ValueError('The original source union is incomplete or failed exact normalization')

    ledger = {}
    for row in pq.read_table(paths['frozen_candidate_evidence.parquet'],
                             columns=['molecule_id', 'library_max', 'processed', 'proxy_gl']).to_pylist():
        mid = row['molecule_id']
        value = row['library_max'], row['proxy_gl']
        if (mid not in final or row['processed'] is not True or value[0] is None or not isfinite(value[0])
                or type(value[1]) is not bool or ledger.setdefault(mid, value) != value):
            raise ValueError('Ambiguous original library strength, proxy status or processed query')
    if ledger.keys() != final.keys():
        raise ValueError('Original native LEDGER coverage differs from final submission')
    return final, known, ledger, source_counts, {key: len(values) for key, values in source_queries.items()}


def model_archive(paths, final):
    inputs, scores, metadata = {}, {}, {}
    for model in ('ice', 'gl'):
        items = population(read_json(paths[f'{model}_work/{model}_in.json']))
        values = read_json(paths[f'{model}_work/{model}_out.json'])
        meta = read_json(paths[f'{model}_work/{model}_out.json.meta.json'])
        if (meta['status'] != 'ok' or meta['device'] != 'cuda'
                or items.keys() != final.keys() or values.keys() != items.keys()):
            raise ValueError(f'Unexpected stock {model} request or archive status')
        finite = 0
        for mid, item in items.items():
            if not item['spectra'] or set(values[mid]) != set(item['cands']):
                raise ValueError(f'{model}/{mid} did not score the exact archived request')
            finite += checked_scores(values[mid])
        if (finite != meta['n_scored'] or meta['n_mols'] != len(items)
                or meta['n_cands'] != sum(len(item['cands']) for item in items.values())):
            raise ValueError(f'{model} scoring count differs from the archived metadata')
        inputs[model], scores[model] = items, values
        metadata[model] = {key: meta[key] for key in ('status', 'device', 'n_mols', 'n_cands', 'n_scored')}
    return inputs, scores, metadata


def ranked_outsiders(paths, known, identity):
    head, deep = map(read_json, (paths['pc_lists.json'], paths['pc_depth.json']))
    if head.keys() != known.keys() or deep.keys() != known.keys():
        raise ValueError('Different MassBank 110 query IDs in the stock and deep PubChem lists')
    outsiders, tally = {}, {}
    for mid in known:
        original = head[mid]
        expanded = deep[mid]
        if (len(original['pc']) != 25 or len(expanded['pc']) != 100
                or not isfinite(original['target']) or not isfinite(expanded['target'])
                or not isclose(original['target'], expanded['target'], rel_tol=0, abs_tol=1e-8)):
            raise ValueError(f'Different fixed precursor mass or list width: {mid}')
        old_head = {identity(smiles) for smiles in original['pc']}
        new_head = {identity(smiles) for smiles in expanded['pc'][:25]}
        if None in old_head | new_head or old_head != new_head:
            raise ValueError(f'Deep PubChem head disagrees with the original 25 structure identities: {mid}')
        seen = known[mid] | old_head
        chosen = []
        counts = Counter(dict(raw_tail_entries=75, tail_in_original_union=0, tail_duplicate=0, invalid_tail_entries=0))
        for rank, smiles in enumerate(expanded['pc'][25:100], 26):
            normalized = identity(smiles)
            if normalized is None:
                counts['invalid_tail_entries'] += 1
            elif normalized in known[mid] or normalized in old_head:
                counts['tail_in_original_union'] += 1
            elif normalized in seen:
                counts['tail_duplicate'] += 1
            else:
                chosen.append(Candidate(rank, smiles))
                seen.add(normalized)
        counts['unique_outsiders'] = len(chosen)
        if sum(counts[k] for k in ('tail_in_original_union', 'tail_duplicate', 'invalid_tail_entries',
                                    'unique_outsiders')) != counts['raw_tail_entries']:
            raise ValueError(f'Incomplete blind tail reconciliation: {mid}')
        outsiders[mid], tally[mid] = chosen, dict(counts)
    return outsiders, tally


def run(inputs, out, rule_sha):
    RDLogger.DisableLog('rdApp.*')
    identity = identity_function()
    paths = verified_inputs(inputs)
    final, known, ledger, source_counts, source_queries = historical(paths, identity)
    native, scores, metadata = model_archive(paths, final)
    outsiders, tally = ranked_outsiders(paths, known, identity)
    ranks = {'historical': final, **{name: {} for name in TARGETS}}
    cases, reasons, fired = {}, defaultdict(Counter), {arm: [] for arm in TARGETS}
    total = Counter()
    for mid, before in final.items():
        strength, proxy = ledger[mid]
        exact_spectra = bool(native['ice'][mid]['spectra'] and
                             native['ice'][mid]['spectra'] == native['gl'][mid]['spectra'])
        both = {smi: (value, scores['gl'][mid][smi]) for smi, value in scores['ice'][mid].items()
                if value is not None and scores['gl'][mid].get(smi) is not None}
        pool = outsiders[mid]
        n_requested = {kind: sum(row.smiles in scores[kind][mid] for row in pool) for kind in ('ice', 'gl')}
        joint_requested = sum(row.smiles in scores['ice'][mid] and row.smiles in scores['gl'][mid] for row in pool)
        joint = [row for row in pool if row.smiles in both]
        counts = dict(tally[mid], ice_requested=n_requested['ice'], gl_requested=n_requested['gl'],
                      both_requested=joint_requested, jointly_finite=len(joint),
                      jointly_finite_same_native_spectra=len(joint) if exact_spectra and not proxy else 0)
        total.update(counts)
        actions, why, comparable = {}, {}, {}
        for arm, slot in TARGETS.items():
            incumbent = before[slot - 1] if len(before) >= slot else None
            incumbent_formula = formula(incumbent) if incumbent is not None else None
            eligible = exact_spectra and not proxy and strength < LIBRARY_THRESHOLD
            comparable[arm] = sum(formula(c.smiles) == incumbent_formula for c in joint) if (
                eligible and incumbent in both and incumbent_formula is not None) else 0
            if eligible:
                after, action, reason = replace_at_rank(before, pool, both, slot, formula)
            else:
                after, action, reason = before, None, 'no_shared_native_group_or_strong_library'
            changed = [pos for pos, (a, b) in enumerate(zip(before, after, strict=True), 1) if a != b]
            if changed != ([slot] if action else []) or (action and identity(after[slot - 1]) in known[mid]):
                raise ValueError(f'Invalid fixed historical slot or known outsider: {mid}/{arm}')
            if action:
                fired[arm].append(mid)
            ranks[arm][mid] = after
            actions[arm], why[arm] = action, reason
            reasons[arm][reason] += 1
        cases[mid] = dict(library_max=strength, proxy_gl=proxy, same_nonempty_native_spectra=exact_spectra,
                          coverage=counts, comparable_same_formula_pairs=comparable, actions=actions, reasons=why,
                          joint_outsiders=[dict(rank=row.rank, smiles=row.smiles) for row in joint])
    out.mkdir(parents=True, exist_ok=True)
    files = {name + '_final.json': write_json(out / (name + '_final.json'), lists) for name, lists in ranks.items()}
    files[STEM + '_cases.json'] = write_json(out / (STEM + '_cases.json'), cases)
    info = dict(status='sealed', arm_names=list(ARMS), molecules=len(final), candidate_budget=25,
                actual_slots=sum(map(len, final.values())), new_predictor_requests=0, baseline=BASE,
                baseline_submission_sha256=PINS['submission.csv'], depth=DEPTH, tail_ranks=[26, 100],
                candidate_source='Fused-score C34 PubChem tail in frozen ascending rank; all original exported sources excluded',
                input_file_sha256=PINS, source_channels=dict(source_counts), source_query_counts=source_queries,
                original_lengths=dict(Counter(map(len, final.values()))), files=files, targets=TARGETS,
                source_library_threshold=LIBRARY_THRESHOLD, source_60_rule_sha256=rule_sha,
                native_forward_archives=metadata, arms=fired, reasons={arm: dict(reasons[arm]) for arm in TARGETS},
                coverage=dict(total), queries_with_joint_outsiders=sum(bool(row['joint_outsiders']) for row in cases.values()),
                exact_spectra_queries=sum(row['same_nonempty_native_spectra'] for row in cases.values()),
                comparable_pairs={arm: sum(row['comparable_same_formula_pairs'][arm] for row in cases.values())
                                  for arm in TARGETS})
    write_json(out / SEAL, info)
    print('MassBank 110 blind outsider eligibility', dict(total), {arm: len(mids) for arm, mids in fired.items()},
          flush=True)
