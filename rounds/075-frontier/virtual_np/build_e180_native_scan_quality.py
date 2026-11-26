"""Package query-only scan weighting on the fixed native C41 PubChem archive."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from inspect import getsource
import json
from pathlib import Path

from build import ASSETS, IMAGE


SLUG = 'enveda-075b-e180-native-scan-quality'
FRAGMENTS = 'enveda-075b-e180-native-fragment-contrast'
REFERENCE = 'enveda-075b-e180-native-physical-ce-replay'
ARCHIVE = 'e180_native_ce_physical_locked.jsonl'
ARCHIVE_ARM = 'multi_fp_physics'
ORDERS = 'e180_native_scan_quality_orders.jsonl'
RECEIPT = 'e180_native_scan_quality_receipt.json'
ARMS = ('stock', 'effective_count', 'median')
PINS = {
    (REFERENCE, ARCHIVE): '7a197c0b33de616a60b7347457cd157b140f80670eb692d73fa653c235367958',
    (FRAGMENTS, 'native_frozen_in.json'): '5e14a44f743189dd1e86225f51d5eb7490c63e2b482062d1e176503ed65d7c05',
    (FRAGMENTS, 'e180_native_fragment_observations.parquet'): 'cc201273f4e36a9bf4faacc3423f14d4d4baee4dec96f25d07aa11f4095fe8c0',
    (FRAGMENTS, 'ice_per_scan.jsonl'): '681e1d9dc1e00b2ee88a242c9a5ec86cbfbb9b9aceaed74e2663c541774a7c8c',
    (FRAGMENTS, 'gl_per_scan.jsonl'): 'e95394158c46c824bb66fb453aa9d8769bc0f29511ddbeae25c6b8e6164878e3',
    (FRAGMENTS, 'ice_out.json'): '7a5a5f4ee3ab46081576c32e084610db46073889aff7671c4d58c2969dfcbff1',
    (FRAGMENTS, 'gl_out.json'): 'c619b14ad839d6f474bab1af1c82910e440f41425661830765c02c71ca78f945',
}


def effective_peaks(mz, intensity, precursor):
    import numpy as np

    mass, abundance = map(lambda x: np.asarray(x, dtype=float), (mz, intensity))
    if mass.ndim != 1 or abundance.shape != mass.shape or not np.isfinite(precursor) or precursor <= 35.5:
        raise ValueError('Invalid archived spectrum')
    valid = np.isfinite(mass) & np.isfinite(abundance) & (mass >= 35) & (mass < precursor - 0.5) & (abundance > 0)
    mass, abundance = mass[valid], abundance[valid]
    if not len(mass):
        return 1.0
    valid = abundance >= abundance.max() * 0.01
    mass, abundance = mass[valid], abundance[valid]
    ordered = np.argsort(mass, kind='stable')
    joined = []
    anchor = None
    for value, strength in zip(mass[ordered], abundance[ordered], strict=True):
        if anchor is None or value - anchor > max(0.01, value * 20e-6):
            joined.append(strength)
            anchor = value
        else:
            joined[-1] += strength
    probabilities = np.asarray(joined) / sum(joined)
    return float(1 / np.sum(probabilities ** 2))


def weights_from_counts(counts):
    import numpy as np

    values = np.asarray(counts, dtype=float)
    return np.clip(values / np.median(values), 0.5, 2.0) if len(values) else values


def read(inputs, origin, name):
    path, = (path for path in inputs.rglob(name) if origin in path.parts)
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != PINS[origin, name]:
        raise ValueError(f'Archived input changed: {origin}/{name}')
    return raw


def observations(inputs):
    from io import BytesIO
    import numpy as np
    import pyarrow.parquet as pq

    cols = ('query', 'scan_ordinal', 'spectrum_id', 'adduct', 'instrument', 'ce_bucket', 'ce_ev',
            'precursor_mz', 'mz', 'relative_intensity')
    table = pq.read_table(BytesIO(read(inputs, FRAGMENTS, 'e180_native_fragment_observations.parquet')), columns=list(cols))
    items = json.loads(read(inputs, FRAGMENTS, 'native_frozen_in.json'))
    native = {row['mid']: row for row in items}
    scans = defaultdict(list)
    seen = set()
    for row in table.to_pylist():
        if row['spectrum_id'] in seen:
            raise ValueError('Duplicate archived observation')
        seen.add(row['spectrum_id'])
        scans[row['query']].append(row)
    if len(items) != len(native) or len(native) != 58 or table.num_rows != 197 or scans.keys() != native.keys():
        raise ValueError('Unexpected measured scan population')
    for mid, rows in scans.items():
        rows.sort(key=lambda r: r['scan_ordinal'])
        for i, (row, spec) in enumerate(zip(rows, native[mid]['spectra'], strict=True)):
            if (row['scan_ordinal'] != i or row['precursor_mz'] != spec['prec'] or row['adduct'] != spec['adduct']
                    or row['ce_ev'] != spec['ce_ev'] or spec['ce_units'] != 'eV'
                    or not np.array_equal(row['mz'], spec['mz']) or not np.array_equal(row['relative_intensity'], spec['it'])):
                raise ValueError(f'Observed spectrum differs from the native model input: {mid}/{i}')
            row['effective_count'] = effective_peaks(row['mz'], row['relative_intensity'], row['precursor_mz'])
    return native, scans


def trace_scores(inputs, model, native, scans):
    import numpy as np

    aggregate = json.loads(read(inputs, FRAGMENTS, model + '_out.json'))
    trace = {}
    for line in read(inputs, FRAGMENTS, model + '_per_scan.jsonl').splitlines():
        row = json.loads(line)
        pair = row['query'], row['scan_ordinal']
        if pair in trace:
            raise ValueError(f'{model} duplicated scan {pair}')
        trace[pair] = row
    expected = {(mid, i) for mid, rows in scans.items() for i in range(len(rows))}
    if trace.keys() != expected or aggregate.keys() != native.keys():
        raise ValueError(f'{model} differs from the frozen native query/scan population')
    for mid, rows in scans.items():
        expected_candidates = set(native[mid]['cands'])
        if len(expected_candidates) != len(native[mid]['cands']) or set(aggregate[mid]) != expected_candidates:
            raise ValueError(f'{model}/{mid} differs from the native candidate request')
        for i, obs in enumerate(rows):
            scan = trace[mid, i]
            if (any(scan[k] != obs[k] for k in ('ce_bucket', 'instrument', 'adduct'))
                    or scan['scores'].keys() - expected_candidates):
                raise ValueError(f'{model}/{mid}/{i} changed acquisition metadata or candidate inventory')
        for candidate in expected_candidates:
            values = [trace[mid, i]['scores'].get(candidate) for i in range(len(rows))]
            scored = [v for v in values if v is not None]
            if any(not np.isfinite(v) for v in scored):
                raise ValueError(f'{model}/{mid} has a nonfinite native score')
            measured = float(np.mean(scored)) if scored else None
            if measured != aggregate[mid][candidate]:
                raise ValueError(f'{model}/{mid} native mean did not replay')
    return aggregate, trace


def adjust_scores(row, scans, native, banks):
    import numpy as np

    mid = row['query']
    head = row['candidates']
    scores = {arm: [dict(row[model]) for model in ('ice', 'gl')] for arm in ARMS}
    weights = weights_from_counts([scan['effective_count'] for scan in scans])
    groups = defaultdict(list)
    for candidate in head:
        groups[candidate['formula']].append(candidate['smiles'])
    eligible_groups = 0
    if native is not None:
        if any(set(row[model]) != set(native['cands']) for model in ('ice', 'gl')):
            raise ValueError(f'{mid} native scoring request differs from the C41 archive')
        for names in groups.values():
            if len(names) < 2 or not len(scans):
                continue
            matrices = [np.asarray([[trace[mid, i]['scores'].get(smi) for smi in names]
                                    for i in range(len(scans))], dtype=float) for _, trace in banks]
            if not all(np.isfinite(matrix).all() for matrix in matrices):
                continue
            eligible_groups += int(len(scans) >= 2)
            for j, model in enumerate(('ice', 'gl')):
                mean = matrices[j].mean(axis=0)
                old = [row[model][name] for name in names]
                if not np.array_equal(mean, old):
                    raise ValueError(f'{mid}/{model} complete-group arithmetic mean differs from archive')
                scores['stock'][j].update(zip(names, mean.tolist(), strict=True))
                if len(scans) >= 2:
                    weighted = np.average(matrices[j], axis=0, weights=weights).tolist()
                    median = np.median(matrices[j], axis=0).tolist()
                    scores['effective_count'][j].update(zip(names, weighted, strict=True))
                    scores['median'][j].update(zip(names, median, strict=True))
    elif row['ice'] or row['gl']:
        raise ValueError('Archived score missing its native input')
    return scores, weights.tolist(), eligible_groups


def run(inputs, out):
    fuser_path, = (p for p in inputs.rglob('gl_fuse.py')
                   if 'casmi26-glacier' in p.parts and ASSETS.split('/')[1] in p.parts)
    spec = spec_from_file_location('gl_fuse', fuser_path)
    fuser = module_from_spec(spec)
    spec.loader.exec_module(fuser)
    native, scans = observations(inputs)
    banks = [trace_scores(inputs, model, native, scans) for model in ('ice', 'gl')]
    raw_rows = read(inputs, REFERENCE, ARCHIVE).splitlines()
    rows = []
    for line in raw_rows:
        raw = json.loads(line)
        rows.append({k: raw[k] for k in ('query', 'candidates', 'orders', 'ice', 'gl')})
    mids = {row['query'] for row in rows}
    if len(rows) != len(mids) or len(mids) != 60 or not native.keys() <= mids:
        raise ValueError('The fixed C41 candidate population changed')
    changed, first_changed = Counter(), Counter()
    slots = 0
    eligible = {}
    out.mkdir(parents=True, exist_ok=True)
    rank_file = out / ORDERS
    with rank_file.open('w') as file:
        for row in rows:
            mid, head = row['query'], row['candidates']
            fields = [[item[k] for item in head] for k in ('smiles', 'key', 'score', 'formula')]
            slots += len(head)
            scored, weights, groups = adjust_scores(row, scans.get(mid, []), native.get(mid), banks)
            eligible[mid] = groups
            orders = {arm: [int(i) for i in fuser.rerank_multi(*fields, scored[arm], [1.0, 1.0], top_n=25)]
                      for arm in ARMS}
            baseline = row['orders'][ARCHIVE_ARM]
            if orders['stock'] != baseline:
                raise ValueError(f'Stock C41 ordering differs from the sealed archive: {mid}')
            for arm, order in orders.items():
                if (sorted(order) != list(range(len(head))) or [head[i]['formula'] for i in order] != fields[-1]
                        or (not groups and order != baseline)):
                    raise ValueError(f'{arm}/{mid} changes the fixed roster, formula slots or unsupported group')
                changed[arm] += int(order != baseline)
                first_changed[arm] += int(bool(order) and order[0] != baseline[0])
            roster = [{k: candidate[k] for k in ('smiles', 'key', 'formula')} for candidate in head]
            file.write(json.dumps(dict(query=mid, roster=roster, orders=orders, eligible_groups=groups,
                                       native_scans=len(weights), quality_weights=weights), allow_nan=False) + '\n')
    report = dict(status='sealed', arms=list(ARMS), molecules=len(mids), actual_slots=slots, observed_queries=len(scans),
                  observed_scans=sum(map(len, scans.values())), candidate_budget=25, new_predictor_requests=0,
                  scored_stage='Original PubChem C41 fixed candidates; no outer fusion',
                  archived_kernel=f'jiyujia/{REFERENCE}/1', archived_order=ARCHIVE_ARM,
                  archived_file=ARCHIVE, archived_sha256=PINS[REFERENCE, ARCHIVE],
                  observation_kernel=f'jiyujia/{FRAGMENTS}/1',
                  fuser_sha256=sha256(fuser_path.read_bytes()).hexdigest(),
                  orders_sha256=sha256(rank_file.read_bytes()).hexdigest(), fixed_formula_slots=True,
                  eligible_queries=sum(v > 0 for v in eligible.values()), eligible_formula_groups=sum(eligible.values()),
                  changed=changed, first_changed=first_changed,
                  support='All accepted scans finite for every candidate and both models per formula group; otherwise stock',
                  weighting='Effective count (inverse Simpson) on joined, normalized fragment intensities; ratio to query median clipped [0.5, 2]')
    (out / RECEIPT).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)


def build(out):
    names = ('SLUG', 'FRAGMENTS', 'REFERENCE', 'ARCHIVE', 'ARCHIVE_ARM', 'ORDERS', 'RECEIPT', 'ARMS', 'PINS', 'ASSETS')
    code = '\n'.join(('from collections import Counter, defaultdict', 'from hashlib import sha256',
                      'from importlib.util import module_from_spec, spec_from_file_location',
                      'import json', 'from pathlib import Path', *(f'{name} = {globals()[name]!r}' for name in names)))
    functions = (effective_peaks, weights_from_counts, read, observations, trace_scores, adjust_scores, run)
    code += '\n\n' + '\n\n'.join(map(getsource, functions)) + '\nrun(Path("/kaggle/input"), Path("/kaggle/working"))\n'
    compile(code, SLUG, 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='scan_quality.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, docker_image=IMAGE,
                dataset_sources=[ASSETS], kernel_sources=[f'jiyujia/{REF}/1' for REF in (REFERENCE, FRAGMENTS)],
                competition_sources=[])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'scan_quality.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(sha256(code.encode()).hexdigest(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    build(**vars(parser.parse_args()))
