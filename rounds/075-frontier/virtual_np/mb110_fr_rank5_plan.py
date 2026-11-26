"""Freeze matched C34 and FR challenger requests before scoring or answer access."""

from collections import Counter
from hashlib import file_digest, sha256
import json
from math import isfinite
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from rdkit import RDLogger

from c39_request_cache import identity_function, population, write_json
from depth50_archive import spectral_array_bytes
import final_native_outsider_mb110 as stock
from outsider_rule import formula


PANEL = 'jiyujia/enveda-075b-external-massbank-202611/1'
PANEL_SHA = '74929665fcceed3eb50e38f1bba500f7d208e21f7874c397b94df9071b6e75d5'
ORIGINAL_FINAL_SHA = '8568ed5abecb88ef43243b122fa726959c85893ff7ae32db22f07d72de0fa44d'
SUMMARY = 'mb110_fr_rank5_plan.json'
PER_QUERY = 'mb110_fr_rank5_plan_per_query.json'
MODELS = ('ice', 'gl')
ARMS = ('c34', 'fr')
CAP = 4
SLOT = 5
SOURCE_PINS = {name: expected for name, expected in stock.PINS.items() if '_out.json' not in name}
SPECTRAL_COLUMNS = ('molecule_id', 'precursor_mz', 'adduct', 'ionization_mode', 'ms2_mzs',
                    'ms2_normalized_intensities', 'collision_energy_ev')


def pinned_path(inputs, ref, name, expected):
    slug = ref.split('/')[1]
    suffix = PurePosixPath(name).parts
    path, = (path for path in inputs.rglob(suffix[-1]) if slug in path.parts and path.parts[-len(suffix):] == suffix)
    with path.open('rb') as stream:
        if file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError(f'Changed blind input: {ref}/{name}')
    return path


def accepted(paths, original):
    contexts = {model: population(stock.read_json(paths[f'{model}_work/{model}_in.json'])) for model in MODELS}
    raw = {model: spectral_array_bytes(paths[f'{model}_work/{model}_in.json']) for model in MODELS}
    if any(collection.keys() != original.keys() for collection in (*contexts.values(), *raw.values())):
        raise ValueError('Stock measured-spectrum and final roster query IDs disagree')
    same = {mid: raw['ice'][mid] == raw['gl'][mid] and
            contexts['ice'][mid]['spectra'] == contexts['gl'][mid]['spectra'] for mid in original}
    return contexts['ice'], raw['ice'], same


def coarse_pool(runtime, queries, plan):
    mass_arrays = [np.load(Path(runtime['PC_DIR']) / 'pc_mass.npy', mmap_mode='r')]
    mass_arrays += [np.load(Path(runtime['POP']) / f'{stem}_mass.npy', mmap_mode='r')
                    for stem in ('delta_pc', 'growth_pc')]
    config = {}
    for mid, item in plan.items():
        target = item['target']
        tol = target * 1e-5
        window = sum(int(masses.searchsorted(target + tol, 'right') - masses.searchsorted(target - tol, 'left'))
                     for masses in mass_arrays)
        config[mid] = dict(target=target, n_window=window, formula=item['formula'])
    with TemporaryDirectory(prefix='mb110-fr-rank5-c34-') as directory:
        scratch = Path(directory)
        config_path, panel_path, result_path = (scratch / name for name in ('scope.json', 'measured.parquet', 'pool.json'))
        config_path.write_text(json.dumps(config, allow_nan=False))
        queries.to_parquet(panel_path, index=False)
        (scratch / 'probe_core2.py').write_text(runtime['COARSE_WORKER'])
        runner = scratch / 'pc_runner.py'
        runner.write_text(runtime['COARSE_RUNNER'])
        env = {**os.environ, 'CASMI_FR_COARSE_COHORT': str(config_path), 'PYTHONHASHSEED': '0'}
        subprocess.run([sys.executable, '-u', str(runner), runtime['V3CODE'], runtime['V3MOD'],
                        runtime['POOL_DIR'], runtime['PC_DIR'], str(panel_path), str(result_path), '4'],
                       env=env, check=True)
        pool = json.loads(result_path.read_bytes())
    if pool.keys() != plan.keys():
        raise ValueError('C34 coarse lists do not cover the full frozen panel')
    return pool


def fr_spectra(spectra):
    adapted = []
    for scan in spectra:
        try:
            mode = {'positive': 1, 'negative': -1}[scan['mode']]
        except KeyError as error:
            raise ValueError('Unexpected native ionization mode in the frozen requests') from error
        adapted.append(dict(mz=np.asarray(scan['mz'], np.float64), it=np.asarray(scan['it'], np.float64),
                            adduct=scan['adduct'], mode=mode, prec=float(scan['prec'])))
    return adapted


def fr_order(rows, scans, featurizer, nets, score_list, excluded, pool_keys, native_key, identity, target_formula):
    candidates = {}
    for coarse_row, smiles in rows:
        key = identity(smiles)
        if key is None or key in excluded or key[0] in pool_keys:
            continue
        archive_key = native_key(smiles)
        if not archive_key or archive_key in pool_keys:
            continue
        if formula(smiles) != target_formula:
            raise ValueError('C34 export did not keep the rank-5 formula')
        candidates.setdefault(smiles, (coarse_row, key))
    literals = list(candidates)
    logits = []
    peaks = errors = 0
    for start in range(0, len(literals), 64):
        chunk = literals[start:start + 64]
        (scores,), peaks, failures = score_list(featurizer, nets, scans, chunk, device='cpu')
        errors += failures
        logits.extend((smiles, value) for smiles, value in zip(chunk, scores, strict=True) if isfinite(value))
    # Among exact ties only, Python preserves the independently frozen C34 coarse order.
    logits.sort(key=lambda pair: -pair[1])
    seen, order = set(), []
    for rank, (smiles, value) in enumerate(logits, 1):
        row, key = candidates[smiles]
        if key not in seen:
            seen.add(key)
            order.append(dict(smiles=smiles, rank=rank, c34_coarse_row=row, fr_logit=value))
    return order, dict(c34_coarse_same_formula=len(rows), fr_attempted_literals=len(literals),
                       fr_finite_literals=len(logits), fr_distinct_outsiders=len(order),
                       fr_merged_peaks=peaks, fr_feature_errors=errors)


def sealed_requests(path, rows, spectra_raw):
    def encode(value):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()

    data = b'[' + b','.join(b'{"mid":' + encode(row['mid']) + b',"spectra":' + spectra_raw[row['mid']]
                          + b',"cands":' + encode(row['cands']) + b'}' for row in rows) + b']\n'
    path.write_bytes(data)
    expected = {row['mid']: spectra_raw[row['mid']] for row in rows}
    if list(population(json.loads(data)).values()) != rows or spectral_array_bytes(path) != expected:
        raise ValueError(f'Stock native request serialization changed: {path.name}')
    return sha256(data).hexdigest()


def run(inputs, out, runtime, source_hashes):
    import torch

    torch.set_num_threads(2)
    RDLogger.DisableLog('rdApp.*')
    identity = identity_function()
    paths = {name: pinned_path(inputs, stock.DEPTH if name == 'pc_depth.json' else stock.BASE, name, digest)
             for name, digest in SOURCE_PINS.items()}
    original, known, ledger, source_counts, source_queries = stock.historical(paths, identity)
    if sum(map(len, original.values())) != 2750:
        raise ValueError('The original 110 final rosters no longer have exactly 25 slots')
    native, spectra_raw, same = accepted(paths, original)
    outsiders, _ = stock.ranked_outsiders(paths, known, identity)
    c34 = stock.read_json(paths['pc_depth.json'])
    queries = pd.read_parquet(pinned_path(inputs, PANEL, 'pipeline_queries.parquet', PANEL_SHA),
                              columns=list(SPECTRAL_COLUMNS))
    if set(queries['molecule_id']) != original.keys():
        raise ValueError('Pinned panel and stock query IDs disagree')
    plans = {}
    for mid, listing in original.items():
        incumbent = listing[SLOT - 1]
        key = formula(incumbent)
        if key is None:
            raise ValueError(f'Invalid archived rank-5 incumbent: {mid}')
        plans[mid] = dict(target=c34[mid]['target'], incumbent=incumbent, formula=key)
    wide = coarse_pool(runtime, queries, plans)

    sys.path.insert(0, runtime['V3CODE'])
    from casmi.chem import score_key

    pool_keys = set(pd.read_parquet(Path(runtime['POOL_DIR']) / 'pool_meta.parquet', columns=['key'])['key'])
    featurizer, nets, model = runtime['_f3_load'](runtime['_f3_find_assets'](inputs), 'cpu')
    masks = Counter()
    requests = {arm: [] for arm in ARMS}
    counts = {arm: Counter(dict(active_queries=0, challenger_query_candidates=0, incumbent_query_candidates=0,
                               challenger_candidate_spectra=0, incumbent_candidate_spectra=0,
                               total_candidate_spectra=0)) for arm in ARMS}
    per_query = {}
    for index, (mid, item) in enumerate(plans.items(), 1):
        if set(wide[mid]['formulas']) != {item['formula']}:
            raise ValueError(f'Unexpected FR coarse formula scope: {mid}')
        library, proxy = ledger[mid]
        scans = native[mid]['spectra']
        eligible = same[mid] and not proxy and library < stock.LIBRARY_THRESHOLD
        control = [{'smiles': candidate.smiles, 'rank': candidate.rank} for candidate in outsiders[mid]
                   if candidate.rank <= 50 and formula(candidate.smiles) == item['formula']]
        probe, fr_stats = [], {}
        if eligible:
            c34_keys = {identity(smiles) for smiles in c34[mid]['pc']}
            excluded = known[mid] | c34_keys
            probe, fr_stats = fr_order(wide[mid]['formulas'][item['formula']], fr_spectra(scans), featurizer, nets,
                                       runtime['_f3_score_list'], excluded, pool_keys, score_key, identity, item['formula'])
        k = min(CAP, len(control), len(probe)) if eligible else 0
        reason = ('missing_exact_spectra' if not same[mid] else 'proxy_gl' if proxy else 'strong_e1' if not eligible
                  else 'no_c34_outsider' if not control else 'no_fr_outsider' if not probe else 'paired')
        masks[reason] += 1
        arm_candidates = {'c34': control[:k], 'fr': probe[:k]}
        if ({identity(row['smiles']) for row in arm_candidates['c34']}
                & {identity(row['smiles']) for row in arm_candidates['fr']}):
            raise ValueError(f'FR and C34 nominees are not identity-disjoint: {mid}')
        row = dict(**item, library_max=library, proxy_gl=proxy, equal_accepted_spectra=same[mid], reason=reason,
                   archived_accepted_spectra_sha256=sha256(spectra_raw[mid]).hexdigest(), scans=len(scans),
                   coarse_rows=wide[mid]['coarse_union_rows'], window_rows=wide[mid]['window_rows'],
                   c34_available=len(control), fr_available=len(probe), budget=k, budgets={}, selected=arm_candidates,
                   **fr_stats)
        for arm, selected in arm_candidates.items():
            costs = dict(active_queries=int(k > 0), challenger_query_candidates=k,
                         incumbent_query_candidates=int(k > 0), challenger_candidate_spectra=k * len(scans),
                         incumbent_candidate_spectra=int(k > 0) * len(scans),
                         total_candidate_spectra=(k + int(k > 0)) * len(scans))
            counts[arm].update(costs)
            row['budgets'][arm] = costs
            if k:
                cands = [item['incumbent'], *(candidate['smiles'] for candidate in selected)]
                requests[arm].append(dict(mid=mid, spectra=scans, cands=cands))
        per_query[mid] = row
        print(f'FR rank-5 {index}/{len(plans)} {mid}: {reason}, paired={k}, FR available={len(probe)}', flush=True)
    if not counts['c34']['active_queries'] or counts['c34'] != counts['fr']:
        raise ValueError('Expected a nonempty paired selection with equal query and spectrum budgets')

    out.mkdir(parents=True, exist_ok=True)
    files = {f'{arm}_requests.json': sealed_requests(out / f'{arm}_requests.json', rows, spectra_raw)
             for arm, rows in requests.items()}
    files['historical_final.json'] = write_json(out / 'historical_final.json', original)
    if files['historical_final.json'] != ORIGINAL_FINAL_SHA:
        raise ValueError('Historical final roster changed on serialization')
    files[PER_QUERY] = write_json(out / PER_QUERY, per_query)
    summary = dict(status='sealed', molecules=len(original), candidate_budget=25, actual_slots=2750, slot=SLOT,
                   max_per_query_challengers=CAP, stock=stock.BASE, depth=stock.DEPTH, panel=PANEL,
                   entrant_formula='Archived final rank-5 incumbent',
                   c34_order='Original archived PubChem rank 26–50, same formula and outside original source union',
                   fr_order='Fresh FRNet descending logit on same-formula outsiders from the fixed PubChem coarse '
                   'window, excluding normalized C34 top 100 and the original source union',
                   admission='Exact same accepted ICE and GL spectrum bytes, weak E1, no GL proxy',
                   spectra='Original accepted model arrays embedded byte for byte in both request arms',
                   request_budget='min(4, eligible C34 candidates, eligible FR candidates) plus one incumbent per arm',
                   arms={arm: dict(totals=dict(totals), per_query={mid: row['budgets'][arm]
                                                                  for mid, row in per_query.items()})
                         for arm, totals in counts.items()},
                   routes=dict(masks), coarse_queries=len(wide), model=model['models'],
                   model_sha256=dict(runtime['_F3_MODELS']), model_code_sha256=runtime['_F3_PIN'],
                   source_pin_hashes=source_hashes, input_hashes={'stock': SOURCE_PINS, 'panel': PANEL_SHA},
                   source_counts=dict(source_counts), source_query_counts=source_queries, files=files)
    write_json(out / SUMMARY, summary)
    print('MassBank FR/C34 rank-5 sealed paired totals', json.dumps(counts), flush=True)
