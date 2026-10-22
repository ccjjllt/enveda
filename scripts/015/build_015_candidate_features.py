#!/usr/bin/env python3
"""015 有界候选 analog 特征。手动 CPU 运行，无网络/训练/新盲集。

输出逐 query metadata、逐 candidate 特征、参考身份。标签只用于预先冻结
全局 no-truth-reference 干预和后置评价字段，不进入候选/参考评分逻辑。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import signal
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from analog_evidence import (canonical_morgan, clean_peaks, exact_peak_hash,
                            neutral_mass, spectral_evidence, tanimoto_to_refs)

INPUTS = {'NP': 'artifacts/legacy/f1-bundle/inputs.json', 'G1': 'artifacts/legacy/g1-validation/inputs.json'}
ICE = {'NP': 'artifacts/legacy/f1-scores/scores/f1e-4_ice_top100_split.json',
       'G1': 'artifacts/legacy/g1-scores/scores/g1e-4_ice_top100_split.json'}
SCENARIOS = ('B_seen', 'A_absent')
BASELINE_FEATURES = ['fp_z', 'ice_z', 'ice_present', 'v9_reciprocal_rank']
ANALOG_FEATURES = ['direct_quality', 'analog_max_tanimoto', 'analog_weighted_tanimoto',
                   'analog_max_product', 'analog_top5mean_product', 'analog_best_spec_quality']
FEATURES = BASELINE_FEATURES + ANALOG_FEATURES
CONFIG = {
    'seed': 20261017, 'shortlist': 50, 'reference_target': 50000,
    'reference_stride_rule': 'max(1, floor(train_rows/50000)), row0 onward',
    'mass_window_da': 200.0, 'max_references_per_query': 2000,
    'top_reference_structures': 80, 'batch_rows': 8192, 'tolerance_da': 0.01,
    'intensity_floor': .002, 'peak_topk': 256, 'intensity_power': .5, 'quality_cosine_power': 3.2,
    'peak_removal': 'precursor +/-1.5Da; oligomer additionally known intact monomer +/-1.5Da; frozen experimental hypothesis',
    'query_spectrum': 'maximum original peak count, first on tie, inherited v9 representative',
    'candidate_shortlist': 'first 50 of reconstructed v9 order; keep full v9 tail unchanged',
    'reference_identity': '005 RDKit canonical_group_key; representative output_smiles lexical first per canonical',
    'hash_exclusion': 'all spectra of all 500 existing NP/G1 queries; exact raw float64 array SHA256',
    'no_truth_reference': 'remove all 500 canonical labels globally BEFORE query-specific reference retrieval',
    'reference_selection': 'same polarity; neutral mass +/-200; nearest2000 then top80 unique canonical by spectral best',
    'spectral_channels': 'union of direct and precursor-shift matching edges, product-descending greedy one-to-one; shift only both monomer singly charged and same polarity',
    'reference_quality': 'cosine**3.2 * min(matched_count/3,1) * (1-largest_pair_fraction)',
    'analog_aggregation': 'top80 positive-quality unique canonical; exclude candidate self, maxT,sum(qT)/sumq,max(qT),mean(top5qT),q_at_max(qT)',
    'shuffle_control': 'query/scenario reference qualities permuted with fixed seed20261017; same identities and Tanimoto, direct unchanged',
    'fingerprint': 'RDKit remove stereo + canonical tautomer + Morgan radius2, 2048 bits',
    'scope': 'repeatedly used development data; reference removal does not undo pretrained encoder exposure',
    'seconds_limit': 900,
}


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def file_info(path):
    path = ROOT / path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': digest}


def zscore(values):
    a = np.asarray(values, dtype=float)
    finite = np.isfinite(a)
    z = np.zeros(len(a))
    if finite.sum() > 1:
        sd = a[finite].std(ddof=1)
        if sd > 1e-12:
            z[finite] = (a[finite] - a[finite].mean()) / sd
    return z


def frozen_v9(molecule, raw):
    keys = molecule['order_top100']
    fp = np.asarray([molecule['base_scores_full_window'][k] for k in keys])
    cache = {r['canon']: r.get('fused_merge') for r in raw.values()}
    ice = np.asarray([cache.get(k) if cache.get(k) is not None else np.nan for k in keys], dtype=float)
    fp_z, ice_z = zscore(fp), zscore(ice)
    fused = fp_z + ice_z
    head = sorted(np.flatnonzero(np.isfinite(ice)), key=lambda i: (-fused[i], i))
    tail = sorted(np.flatnonzero(~np.isfinite(ice)), key=lambda i: (-fp[i], i))
    ix = head + tail
    features = {k: {'fp_score': float(fp[i]), 'fp_z': float(fp_z[i]),
                    'fp_rank': i + 1, 'ice_score': float(ice[i]) if np.isfinite(ice[i]) else None,
                    'ice_z': float(ice_z[i]), 'ice_missing': int(not np.isfinite(ice[i])),
                    'baseline_score': float(fused[i])} for i, k in enumerate(keys)}
    return [keys[i] for i in ix], features


def freeze_queries(out, limit):
    datasets = {ds: json.loads((ROOT / p).read_text())['molecules'] for ds, p in INPUTS.items()}
    ice = {ds: json.loads((ROOT / p).read_text()) for ds, p in ICE.items()}
    hashes, truth_set, labels, queries = set(), set(), {}, []
    # This is the only early use of truth: a global evaluation intervention,
    # fixed for all 500 queries even when --max-queries is used for a manual probe.
    for ds, molecules in datasets.items():
        for m in molecules:
            truth_set.add(m['truth_canon'])
            labels[f'{ds}:{m["key"]}'] = m['truth_canon']
            hashes.update(exact_peak_hash(s['mz'], s['it']) for s in m['spectra'])
    save(out / 'frozen_reference_exclusions.json', {'all_query_peak_hashes': sorted(hashes),
                                                  'all_query_canonical_groups': sorted(truth_set),
                                                  'scope_queries': len(labels)})
    for ds, molecules in datasets.items():
        for m in molecules:
            order, baseline = frozen_v9(m, ice[ds][m['key']])
            selected = max(range(len(m['spectra'])), key=lambda i: len(m['spectra'][i]['mz']))
            s = m['spectra'][selected]
            mz, intensity, clean_audit = clean_peaks(s['mz'], s['it'], s['prec'], s['adduct'])
            query = {'case': f'{ds}:{m["key"]}', 'dataset': ds,
                     'key': m['key'], 'selected_spectrum': selected,
                     'query_hash': exact_peak_hash(s['mz'], s['it']),
                     'adduct': s['adduct'], 'mode': s['mode'], 'instrument': s['instrument'],
                     'precursor_mz': float(s['prec']), 'neutral_mass': neutral_mass(s['prec'], s['adduct']),
                     'raw_peaks': len(s['mz']), 'clean_peaks': len(mz), 'mz': mz, 'intensity': intensity,
                     'clean_audit': clean_audit, 'n_spectra': len(m['spectra']), 'baseline_order': order,
                     'candidates': order[:CONFIG['shortlist']], 'baseline_features': baseline,
                     'rep_smiles': m['rep_smiles'], 'formulas': m['formulas'],
                     'ice_supported': any(v['ice_missing'] == 0 for v in baseline.values())}
            if not math.isfinite(query['neutral_mass']):
                raise ValueError(('Unknown query mass', query['case'], query['adduct']))
            queries.append(query)
    if limit:
        queries = queries[:limit]
    manifest = [{k: q[k] for k in ('case', 'dataset', 'selected_spectrum', 'query_hash', 'adduct',
                                  'mode', 'instrument', 'precursor_mz', 'neutral_mass', 'raw_peaks',
                                  'clean_peaks', 'n_spectra', 'baseline_order', 'candidates', 'ice_supported', 'clean_audit')}
                for q in queries]
    save(out / 'frozen_queries.json', manifest)
    return queries, hashes, truth_set, labels


def load_refs(query_hashes, excluded_groups):
    identity_path = ROOT / 'artifacts/runs/005/structure_identity_train.parquet'
    identity = pq.read_table(identity_path, columns=['raw_key', 'canonical_group_key', 'output_smiles']).to_pylist()
    raw_to_canonical = {r['raw_key']: r['canonical_group_key'] for r in identity}
    canonical_smiles = {}
    for r in identity:
        key, smiles = r['canonical_group_key'], r['output_smiles']
        if key and smiles and (key not in canonical_smiles or smiles < canonical_smiles[key]):
            canonical_smiles[key] = smiles
    del identity
    file = pq.ParquetFile(ROOT / 'data/raw/train.parquet')
    stride = max(1, file.metadata.num_rows // CONFIG['reference_target'])
    columns = ['inchikey14', 'adduct', 'precursor_mz', 'ingest_lib', 'ionization_mode',
               'instrument_type', 'ms2_mzs', 'ms2_normalized_intensities']
    refs, offset, counts = [], 0, Counter()
    for batch in file.iter_batches(columns=columns, batch_size=CONFIG['batch_rows']):
        local = np.arange((-offset) % stride, batch.num_rows, stride, dtype=np.int64)
        for idx, r in zip(local, batch.take(local).to_pylist()):
            counts['sampled_rows'] += 1
            canonical = raw_to_canonical.get(r['inchikey14'])
            nm = neutral_mass(r['precursor_mz'], r['adduct'])
            if canonical is None or canonical not in canonical_smiles:
                counts['unmapped'] += 1
                continue
            if not math.isfinite(nm):
                counts['unknown_mass'] += 1
                continue
            phash = exact_peak_hash(r['ms2_mzs'], r['ms2_normalized_intensities'])
            if phash in query_hashes:
                counts['exact_query_hash_removed'] += 1
                continue
            mz, intensity, clean_audit = clean_peaks(r['ms2_mzs'], r['ms2_normalized_intensities'], r['precursor_mz'], r['adduct'])
            if not len(mz):
                counts['empty_cleaned'] += 1
                continue
            refs.append({'row': offset + int(idx), 'raw_key': r['inchikey14'], 'canonical': canonical,
                         'smiles': canonical_smiles[canonical], 'adduct': r['adduct'],
                         'precursor_mz': float(r['precursor_mz']), 'neutral_mass': nm,
                         'source': r['ingest_lib'], 'mode': r['ionization_mode'],
                         'instrument': r['instrument_type'], 'hash': phash,
                         'mz': mz, 'intensity': intensity, 'clean_audit': clean_audit, 'no_truth_allowed': canonical not in excluded_groups})
        offset += batch.num_rows
    refs.sort(key=lambda r: (r['neutral_mass'], r['row']))
    counts['seen_independent_refs'] = len(refs)
    counts['no_truth_removed'] = sum(not r['no_truth_allowed'] for r in refs)
    counts['no_truth_refs'] = sum(r['no_truth_allowed'] for r in refs)
    counts['unique_canonical_refs'] = len({r['canonical'] for r in refs})
    return refs, {'train_rows': offset, 'stride': stride, **counts}


def reference_record(ref, score):
    return {**{k: ref[k] for k in ('row', 'raw_key', 'canonical', 'adduct', 'precursor_mz',
                                  'neutral_mass', 'source', 'mode', 'instrument', 'hash', 'clean_audit')}, **score}


def select_references(query, refs, mass, scenario):
    lo, hi = np.searchsorted(mass, [query['neutral_mass'] - CONFIG['mass_window_da'],
                                  query['neutral_mass'] + CONFIG['mass_window_da']])
    eligible = [i for i in range(lo, hi)
                if refs[i]['adduct'][-1:] == query['adduct'][-1:]
                and (scenario == 'B_seen' or refs[i]['no_truth_allowed'])]
    eligible.sort(key=lambda i: (abs(mass[i] - query['neutral_mass']), refs[i]['row']))
    return eligible[:CONFIG['max_references_per_query']]


def aggregate_analog(similarities, qualities):
    if not len(similarities) or not np.any(qualities > 0):
        return [0.0] * 5
    products = similarities * qualities
    best = int(np.argmax(products))
    return [float(similarities.max()), float(np.dot(similarities, qualities) / qualities.sum()),
            float(products[best]), float(np.sort(products)[-5:].mean()), float(qualities[best])]


def candidate_features(query, candidates, selected, refs, scores, rng):
    ordered = sorted(selected, key=lambda i: (-scores[i]['quality'], -scores[i]['cosine'], refs[i]['row']))
    unique, seen = [], set()
    for i in ordered:
        if scores[i]['quality'] <= 0:
            continue
        canonical = refs[i]['canonical']
        if canonical not in seen:
            unique.append(i)
            seen.add(canonical)
            if len(unique) == CONFIG['top_reference_structures']:
                break
    qualities = np.asarray([scores[i]['quality'] for i in unique])
    shuffled = rng.permutation(qualities)
    ref_smiles = [refs[i]['smiles'] for i in unique]
    by_canonical = {}
    for i in selected:
        by_canonical.setdefault(refs[i]['canonical'], []).append(i)
    X, XS, best_refs = [], [], []
    for baseline_rank, candidate in enumerate(candidates, 1):
        similarities = tanimoto_to_refs(query['rep_smiles'][candidate], ref_smiles) if unique else np.array([])
        keep = np.asarray([refs[i]['canonical'] != candidate for i in unique], dtype=bool)
        refs_same = by_canonical.get(candidate, [])
        direct_quality = max((scores[i]['quality'] for i in refs_same), default=0.0)
        base = query['baseline_features'][candidate]
        row = [base['fp_z'], base['ice_z'], 1 - base['ice_missing'], 1.0 / baseline_rank, direct_quality]
        analog = aggregate_analog(similarities[keep], qualities[keep])
        analog_shuffled = aggregate_analog(similarities[keep], shuffled[keep])
        X.append(row + analog)
        XS.append(row + analog_shuffled)
        remaining = np.flatnonzero(keep)
        best_j = int(remaining[np.argmax(similarities[keep] * qualities[keep])]) if len(remaining) else None
        best_refs.append({'candidate': candidate,
                          'reference_row': refs[unique[best_j]]['row'] if best_j is not None else None,
                          'reference_canonical': refs[unique[best_j]]['canonical'] if best_j is not None else None,
                          'tanimoto': float(similarities[best_j]) if best_j is not None else 0.0,
                          'product': analog[2]})
    evidence = [reference_record(refs[i], scores[i]) for i in unique]
    for record, shuffled_quality in zip(evidence, shuffled):
        record['shuffled_quality'] = float(shuffled_quality)
    return X, XS, evidence, best_refs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/runs/015-analog/features')
    parser.add_argument('--max-queries', type=int)
    args = parser.parse_args()
    start = time.perf_counter()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('015 fixed 900s CPU experiment budget')))
    signal.alarm(CONFIG['seconds_limit'])
    RDLogger.DisableLog('rdApp.warning')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'config.json', CONFIG)
    queries, hashes, excluded, labels = freeze_queries(out, args.max_queries)
    print(f'frozen {len(queries)} queries, {len(hashes)} unique hashes, {len(excluded)} canonical groups', flush=True)
    refs, ref_meta = load_refs(hashes, excluded)
    mass = np.asarray([r['neutral_mass'] for r in refs])
    print(f'loaded references {ref_meta}, wall={time.perf_counter()-start:.1f}s', flush=True)
    row_counts, query_stats = Counter(), []
    all_X, all_XS, qindex, sindex, positions = [], [], [], [], []
    rng = np.random.default_rng(CONFIG['seed'])
    reference_file = (out / 'reference_evidence.jsonl').open('w')
    try:
        for qi, query in enumerate(queries):
            pools = {scenario: select_references(query, refs, mass, scenario) for scenario in SCENARIOS}
            union = sorted(set(pools[SCENARIOS[0]]) | set(pools[SCENARIOS[1]]))
            scores = {i: spectral_evidence(query, refs[i]) for i in union}
            for si, (scenario, pool) in enumerate(pools.items()):
                X, XS, evidence, best_refs = candidate_features(query, query['candidates'], pool, refs, scores, rng)
                all_X.extend(X)
                all_XS.extend(XS)
                qindex.extend([qi] * len(X))
                sindex.extend([si] * len(X))
                positions.extend(range(len(X)))
                reference_file.write(json.dumps({'case': query['case'], 'scenario': scenario,
                                                 'references': evidence, 'candidate_best_refs': best_refs},
                                                 ensure_ascii=False, allow_nan=False) + '\n')
                row_counts[scenario] += len(X)
                query_stats.append({'case': query['case'], 'scenario': scenario, 'references_scored': len(pool),
                                    'reference_structures': len(evidence), 'candidate_rows': len(X),
                                    'truth_reference_count_for_audit_only': sum(refs[i]['canonical'] == labels[query['case']] for i in pool)})
            if (qi + 1) % 25 == 0 or qi == len(queries) - 1:
                print(f'features {qi+1}/{len(queries)}, wall={time.perf_counter()-start:.1f}s', flush=True)
    finally:
        reference_file.close()
    # All scoring is frozen above. Only now attach labels and evaluation metadata.
    X, XS = np.asarray(all_X, dtype=np.float32), np.asarray(all_XS, dtype=np.float32)
    y = np.asarray([queries[qi]['candidates'][position] == labels[queries[qi]['case']]
                    for qi, position in zip(qindex, positions)], dtype=np.uint8)
    np.savez_compressed(out / 'features.npz', X=X, X_shuffled=XS, y=y,
                        query_index=np.asarray(qindex, dtype=np.int32),
                        scenario_index=np.asarray(sindex, dtype=np.uint8),
                        candidate_position=np.asarray(positions, dtype=np.int16))
    manifest = {'feature_names': FEATURES, 'baseline_feature_names': BASELINE_FEATURES,
                'analog_feature_names': ANALOG_FEATURES, 'scenarios': list(SCENARIOS), 'queries': [
                    {'dataset': q['dataset'], 'key': q['key'], 'truth': labels[q['case']],
                     'baseline_order': q['baseline_order'], 'candidate_keys': q['candidates'],
                     'candidate_formulas': [q['formulas'].get(k) for k in q['candidates']],
                     'adduct': q['adduct'], 'ice_supported': q['ice_supported'], 'clean_audit': q['clean_audit']}
                    for q in queries]}
    save(out / 'manifest.json', manifest)
    sources = {name: file_info(path) for name, path in {**{f'inputs_{k}': v for k, v in INPUTS.items()},
                                                       **{f'ice_{k}': v for k, v in ICE.items()},
                                                       'identity': 'artifacts/runs/005/structure_identity_train.parquet'}.items()}
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        peak_rss *= 1024
    summary = {'complete': True, 'queries': len(queries), 'candidate_rows': dict(row_counts),
               'reference_meta': ref_meta, 'query_stats': query_stats, 'sources': sources,
               'config': CONFIG, 'environment': {'python': platform.python_version(), 'numpy': np.__version__,
               'pyarrow': pyarrow.__version__, 'rdkit': rdkit.__version__, 'device': 'local CPU',
               'wall_seconds': time.perf_counter() - start, 'peak_rss_bytes': peak_rss,
               'finished_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()},
               'fingerprint_cache': dict(zip(canonical_morgan.cache_info()._fields, canonical_morgan.cache_info())),
               'feature_shape': list(X.shape), 'features_finite': bool(np.isfinite(X).all() and np.isfinite(XS).all()),
               'query_empty_after_removal': sum(q['clean_peaks'] == 0 for q in queries),
               'output_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'features.npz',out/'manifest.json']}}
    save(out / 'summary.json', summary)
    signal.alarm(0)
    print(json.dumps({k: summary[k] for k in ('complete', 'queries', 'candidate_rows', 'environment')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
