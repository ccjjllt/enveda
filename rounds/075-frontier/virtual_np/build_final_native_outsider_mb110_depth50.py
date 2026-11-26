"""Package an independent form of the MassBank fixed-slot test using an earlier depth-50 cache."""

import argparse
from ast import FunctionDef, parse
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
from inspect import getsource
import json
from pathlib import Path, PurePosixPath

from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack
from build_final_native_outsider_mb110 import BASE, DEPTH
from build_final_native_outsider_np60 import replace_at_rank


SLUG = 'enveda-075-final-native-outsider-mb110-depth50'
STEM = 'final_native_outsider_mb110_depth50'
SEAL = STEM + '_receipt.json'
SUPPLEMENT = 'jiyujia/enveda-075b-massbank-pubchem50/1'
SUPPLEMENT_PINS = {
    'execution_receipt.json': 'd002400433779081465d61f0703d0a1f69f11ea78c7a93202a641847c0a18590',
    'ice_work/ice_in.json': '0e901d2d761a849071e794bbae26e0f96459e4f67952607b8ee9e60698d986e6',
    'ice_work/ice_out.json': 'c8a423ee9dc99056fde3ce5c932e3cf84bb7abce937bce871b4f5661c68e75dd',
    'ice_work/ice_out.json.meta.json': '67eeed26db7d953b49d02f2e43a86cd4256d3f4f61e1def2bbb356128ec6bf12',
    'gl_work/gl_in.json': '0e901d2d761a849071e794bbae26e0f96459e4f67952607b8ee9e60698d986e6',
    'gl_work/gl_out.json': 'd73bb7946eae2c03fdfe2f0582a34b54d05d276e6b9ea2588eb7665e1148a6a6',
    'gl_work/gl_out.json.meta.json': 'bd4c2b3f71ab807b8dc73831607018ae0a04684128eeaba4b635c3c71372b7f0',
}


def supplement_inputs(inputs):
    slug = SUPPLEMENT.split('/')[1]
    paths = {}
    for name, expected in SUPPLEMENT_PINS.items():
        suffix = PurePosixPath(name).parts
        path, = (p for p in inputs.rglob(suffix[-1]) if slug in p.parts and p.parts[-len(suffix):] == suffix)
        with path.open('rb') as stream:
            if file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'Changed archived depth-50 input: {name}')
        paths[name] = path
    receipt = json.loads(paths['execution_receipt.json'].read_bytes())
    if (receipt['is_full_run'] is not True or receipt['processed_molecules'] != 110
            or receipt['source_sha256'] != '55ae490b9d43121b4dd0019affc06205dd00bb3189c2675765c75212e8c264ca'):
        raise ValueError('Expected the completed historical depth-50 request run')
    return paths


def spectral_array_bytes(path):
    """Return the original JSON byte span of each top-level request's spectra value."""
    text = path.read_bytes().decode('utf-8')
    decoder = json.JSONDecoder()

    def skip(at):
        while at < len(text) and text[at] in ' \r\n\t':
            at += 1
        return at

    def take(at, token):
        at = skip(at)
        if not text.startswith(token, at):
            raise ValueError(f'Missing JSON token {token!r} in {path.name}')
        return skip(at + len(token))

    position = take(0, '[')
    arrays = {}
    while not text.startswith(']', position):
        position = take(position, '{')
        row = {}
        raw = None
        while not text.startswith('}', position):
            name, end = decoder.raw_decode(text, position)
            start = take(end, ':')
            value, end = decoder.raw_decode(text, start)
            if not isinstance(name, str) or name in row:
                raise ValueError(f'Invalid duplicate JSON field in {path.name}')
            row[name] = value
            if name == 'spectra':
                raw = text[start:end].encode('utf-8')
            position = skip(end)
            if text.startswith('}', position):
                break
            position = take(position, ',')
        position = take(position, '}')
        mid = row['mid']
        if mid in arrays or not row['spectra'] or not isinstance(row['spectra'], list) or raw is None:
            raise ValueError(f'Missing, empty or duplicate original spectra in {path.name}')
        arrays[mid] = raw
        if text.startswith(']', position):
            break
        position = take(position, ',')
    if take(position, ']') != len(text):
        raise ValueError(f'Unexpected trailing archival input in {path.name}')
    return arrays


def byte_match(stock_paths, depth_paths, native, augmented):
    raw = {(run, kind): spectral_array_bytes(paths[f'{kind}_work/{kind}_in.json'])
           for run, paths in (('stock', stock_paths), ('depth50', depth_paths)) for kind in ('ice', 'gl')}
    expected = set(native['ice'])
    if any(set(arrays) != expected for arrays in raw.values()):
        raise ValueError('The historical accepted spectral byte ledgers disagree on query IDs')
    exact = {}
    for mid in expected:
        entries = {f'{run}_{kind}': arrays[mid] for (run, kind), arrays in raw.items()}
        same = len(set(entries.values())) == 1
        if same and any(augmented[kind][mid]['spectra'] != native[kind][mid]['spectra'] for kind in ('ice', 'gl')):
            raise ValueError(f'Accepted spectral bytes and parsed arrays disagree: {mid}')
        exact[mid] = same, {name: sha256(value).hexdigest() for name, value in entries.items()}
    return exact


def request_delta(native, augmented, exact):
    report = {}
    for model in ('ice', 'gl'):
        original_count, expanded_count, added, dropped = 0, 0, 0, 0
        report[model] = Counter()
        for mid, row in native[model].items():
            old, new = set(row['cands']), set(augmented[model][mid]['cands'])
            before, after = len(row['spectra']), len(augmented[model][mid]['spectra'])
            original_count += len(old) * before
            expanded_count += len(new) * after
            added += len(new - old) * after
            dropped += len(old - new) * before
            if exact[mid][0]:
                report[model].update(added_with_unchanged_accepted_spectra=len(new - old) * after,
                                     removed_with_unchanged_accepted_spectra=len(old - new) * before)
        report[model].update(historical_stock_candidate_spectra=original_count,
                             historical_depth50_candidate_spectra=expanded_count,
                             historical_added_literal_candidate_spectra=added,
                             historical_removed_literal_candidate_spectra=dropped)
    return {kind: dict(values) for kind, values in report.items()}


def run(inputs, out, rule_sha):
    from rdkit import RDLogger
    from c39_request_cache import identity_function, write_json
    import final_native_outsider_mb110 as stock
    from outsider_rule import formula, replace_at_rank

    RDLogger.DisableLog('rdApp.*')
    identity = identity_function()
    paths = stock.verified_inputs(inputs)
    depth_paths = supplement_inputs(inputs)
    final, known, ledger, source_counts, source_queries = stock.historical(paths, identity)
    native, _, stock_metadata = stock.model_archive(paths, final)
    augmented, scores, expanded_metadata = stock.model_archive(depth_paths, final)
    exact = byte_match(paths, depth_paths, native, augmented)
    costs = request_delta(native, augmented, exact)
    outsiders, tally = stock.ranked_outsiders(paths, known, identity)
    ranks = {'historical': final, **{name: {} for name in stock.TARGETS}}
    cases, reasons, fired = {}, defaultdict(Counter), {arm: [] for arm in stock.TARGETS}
    total = Counter()
    for mid, before in final.items():
        strength, proxy = ledger[mid]
        same, spectral_digests = exact[mid]
        both = {smi: (value, scores['gl'][mid][smi]) for smi, value in scores['ice'][mid].items()
                if value is not None and scores['gl'][mid].get(smi) is not None}
        pool = outsiders[mid]
        n_requested = {kind: sum(row.smiles in scores[kind][mid] for row in pool) for kind in ('ice', 'gl')}
        jointly_requested = sum(row.smiles in scores['ice'][mid] and row.smiles in scores['gl'][mid] for row in pool)
        joint = [row for row in pool if row.smiles in both]
        counts = dict(tally[mid], ice_requested=n_requested['ice'], gl_requested=n_requested['gl'],
                      both_requested=jointly_requested, jointly_finite=len(joint),
                      jointly_finite_same_native_spectra=len(joint) if same and not proxy else 0)
        total.update(counts)
        actions, why, comparable = {}, {}, {}
        eligible = same and not proxy and strength < stock.LIBRARY_THRESHOLD
        for arm, slot in stock.TARGETS.items():
            incumbent = before[slot - 1] if len(before) >= slot else None
            incumbent_formula = formula(incumbent) if incumbent is not None else None
            comparable[arm] = sum(formula(c.smiles) == incumbent_formula for c in joint) if (
                eligible and incumbent in both and incumbent_formula is not None) else 0
            if eligible:
                after, action, reason = replace_at_rank(before, pool, both, slot, formula)
            else:
                reason = 'different_accepted_spectral_bytes' if not same else 'proxy_or_strong_library'
                after, action = before, None
            changed = [pos for pos, (a, b) in enumerate(zip(before, after, strict=True), 1) if a != b]
            if changed != ([slot] if action else []) or (action and identity(after[slot - 1]) in known[mid]):
                raise ValueError(f'Invalid fixed slot or archived original identity: {mid}/{arm}')
            if action:
                fired[arm].append(mid)
            ranks[arm][mid] = after
            actions[arm], why[arm] = action, reason
            reasons[arm][reason] += 1
        cases[mid] = dict(library_max=strength, proxy_gl=proxy, same_nonempty_native_spectra=same,
                          accepted_spectral_bytes_sha256=spectral_digests, coverage=counts,
                          comparable_same_formula_pairs=comparable, actions=actions, reasons=why,
                          joint_outsiders=[dict(rank=row.rank, smiles=row.smiles) for row in joint])
    out.mkdir(parents=True, exist_ok=True)
    files = {name + '_final.json': write_json(out / (name + '_final.json'), lists) for name, lists in ranks.items()}
    files[STEM + '_cases.json'] = write_json(out / (STEM + '_cases.json'), cases)
    info = dict(status='sealed', arm_names=list(stock.ARMS), molecules=len(final), candidate_budget=25,
                actual_slots=sum(map(len, final.values())), new_predictor_requests=0, baseline=stock.BASE,
                baseline_submission_sha256=stock.PINS['submission.csv'], depth=stock.DEPTH, tail_ranks=[26, 100],
                scores_reference=SUPPLEMENT, scores_policy='Both incumbent and challenger use only the prior depth-50 run',
                equal_accepted_spectra='Raw JSON spectra array byte spans of stock ICE, stock GL, depth50 ICE and depth50 GL',
                historical_cost=costs, input_file_sha256=stock.PINS, augmented_file_sha256=SUPPLEMENT_PINS,
                source_channels=dict(source_counts), source_query_counts=source_queries,
                original_lengths=dict(Counter(map(len, final.values()))), files=files, targets=stock.TARGETS,
                source_library_threshold=stock.LIBRARY_THRESHOLD, source_60_rule_sha256=rule_sha,
                native_forward_archives=stock_metadata, augmented_forward_archives=expanded_metadata,
                arms=fired, reasons={arm: dict(reasons[arm]) for arm in stock.TARGETS}, coverage=dict(total),
                queries_with_joint_outsiders=sum(bool(row['joint_outsiders']) for row in cases.values()),
                exact_spectra_queries=sum(row['same_nonempty_native_spectra'] for row in cases.values()),
                comparable_pairs={arm: sum(row['comparable_same_formula_pairs'][arm] for row in cases.values())
                                  for arm in stock.TARGETS})
    write_json(out / SEAL, info)
    print('MassBank 110 prior depth-50 outsider eligibility', dict(total),
          {arm: len(mids) for arm, mids in fired.items()}, flush=True)


def build(out):
    original = (HERE / 'final_native_top5.py').read_text()
    node, = (item for item in parse(original).body if isinstance(item, FunctionDef) and item.name == 'formula')
    formula_source = '\n'.join(original.splitlines()[node.decorator_list[0].lineno - 1:node.end_lineno])
    rule = ('from functools import lru_cache\nfrom rdkit import Chem\n'
            'from rdkit.Chem.rdMolDescriptors import CalcMolFormula\n\n'
            + formula_source + '\n\n' + getsource(replace_at_rank))
    rule_sha = sha256(rule.encode()).hexdigest()
    constants = ('STEM', 'SEAL', 'SUPPLEMENT', 'SUPPLEMENT_PINS')
    runtime = ('from collections import Counter, defaultdict\nfrom hashlib import file_digest, sha256\n'
               'import json\nfrom pathlib import PurePosixPath\n'
               + '\n'.join(f'{name} = {globals()[name]!r}' for name in constants) + '\n\n'
               + '\n'.join(getsource(fn) for fn in (supplement_inputs, spectral_array_bytes, byte_match, request_delta, run)))
    modules = {'outsider_rule.py': rule, 'depth50.py': runtime,
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
with tempfile.TemporaryDirectory(prefix='massbank110-final-outsider-depth50-') as directory:
    scratch = Path(directory)
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Embedded source changed: {{name}}')
        (scratch / name).write_bytes(raw)
    sys.path.insert(0, directory)
    from depth50 import run
    run(inputs, Path('/kaggle/working'), {rule_sha!r})
'''
    compile(code, SLUG, 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='run_outsider.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, docker_image=IMAGE, dataset_sources=[WHEEL],
                kernel_sources=[BASE, DEPTH, SUPPLEMENT], competition_sources=[])
    out.mkdir(parents=True, exist_ok=True)
    (out / meta['code_file']).write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return sha256(code.encode()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    print(build(**vars(parser.parse_args())))
