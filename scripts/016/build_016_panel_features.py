"""016/017 panel features, corrected FP+ICE baseline (manual CPU research).

Reuse the frozen 015 v9 parser and ranking. 017 defaults preserve historical
016 outputs. Reference definitions, matching and trained model inputs remain
unchanged: B excludes the new panel representative hashes; A also excludes
all panel canonical structures. --cross-external-only isolates external
references for cross_source queries. The four-query probe independently
compares the accelerated reference selection to the original Python loop.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from analog_evidence import clean_peaks, neutral_mass, spectral_evidence, tanimoto_to_refs
sys.path[:0] = [str(ROOT / "scripts/015"), str(ROOT / "scripts/016")]
from build_015_candidate_features import frozen_v9
from build_016_reference_index import load_index, read_peaks

OUT = ROOT / "artifacts/runs/017-corrected-features"
ENVIDA_LIBS = {"enveda-180", "enveda-np-examples"}
FEATURES = ['fp_z', 'ice_z', 'ice_present', 'v9_reciprocal_rank', 'direct_quality',
            'analog_max_tanimoto', 'analog_weighted_tanimoto', 'analog_max_product',
            'analog_top5mean_product', 'analog_best_spec_quality']
SEED = 20261017
WIN_DA = 200.0
MAXREF = 2000
TOPREF = 80


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def aggregate_analog(similarities, qualities):
    if not len(similarities) or not np.any(qualities > 0):
        return [0.0] * 5
    products = similarities * qualities
    best = int(np.argmax(products))
    return [float(similarities.max()), float(np.dot(similarities, qualities) / qualities.sum()),
            float(products[best]), float(np.sort(products)[-5:].mean()), float(qualities[best])]


def smiles_map():
    import pyarrow.parquet as pq
    m = {}
    for path in ["artifacts/runs/005/structure_identity_train.parquet",
                 "artifacts/runs/005/structure_identity_coco.parquet"]:
        for r in pq.read_table(ROOT / path, columns=["canonical_group_key", "output_smiles"]).to_pylist():
            k, s = r["canonical_group_key"], r["output_smiles"]
            if k and s and (k not in m or s < m[k]):
                m[k] = s
    return m


def prepare_eligible(idx, qhashes, targets):
    """Precompute original filters; all arrays retain global mass order."""
    valid = np.fromiter((str(h) not in qhashes for h in idx['phash']), bool, len(idx['phash']))
    absent = np.fromiter((str(c) not in targets for c in idx['canon']), bool, len(idx['canon']))
    external = np.fromiter((str(x) not in ENVIDA_LIBS for x in idx['lib']), bool, len(idx['lib']))
    polarity = np.fromiter((str(x)[-1:] == '+' for x in idx['adduct']), bool, len(idx['adduct']))
    by_condition = {}
    for sign in ('+', '-'):
        # The original final-character equality is retained, not inferred mode.
        mode = polarity if sign == '+' else np.fromiter(
            (str(x)[-1:] == '-' for x in idx['adduct']), bool, len(idx['adduct']))
        for scenario in ('B_seen', 'A_absent'):
            mask = valid & mode
            if scenario == 'A_absent':
                mask &= absent
            for ext in (False, True):
                indices = np.flatnonzero(mask & external if ext else mask)
                by_condition[(sign, scenario, ext)] = (indices, idx['nm'][indices])
    return by_condition


def select_pool(q, scenario, idx, prepared, external_only):
    """Nearest 2,000, expanding boundary ties before row-order tie breaking."""
    ext = external_only and q['group'] == 'cross_source'
    indices, masses = prepared[(q['adduct'][-1:], scenario, ext)]
    mass = q['neutral_mass']
    lo = int(np.searchsorted(masses, mass - WIN_DA, 'left'))
    hi = int(np.searchsorted(masses, mass + WIN_DA, 'right'))
    center = int(np.searchsorted(masses, mass, 'left'))
    left, right = max(lo, center - MAXREF), min(hi, center + MAXREF)
    if left < right:
        left = max(lo, int(np.searchsorted(masses, masses[left], 'left')))
        right = min(hi, int(np.searchsorted(masses, masses[right - 1], 'right')))
    possible = indices[left:right]
    order = np.lexsort((idx['row'][possible], np.abs(idx['nm'][possible] - mass)))
    return possible[order[:MAXREF]].tolist(), hi - lo


def select_pool_original(q, scenario, idx, qhashes, targets, external_only):
    mass = idx['nm']
    lo = int(np.searchsorted(mass, q['neutral_mass'] - WIN_DA, 'left'))
    hi = int(np.searchsorted(mass, q['neutral_mass'] + WIN_DA, 'right'))
    eli = [i for i in range(lo, hi)
           if str(idx['adduct'][i])[-1:] == q['adduct'][-1:]
           and str(idx['phash'][i]) not in qhashes
           and (scenario == 'B_seen' or str(idx['canon'][i]) not in targets)
           and not (external_only and q['group'] == 'cross_source'
                    and str(idx['lib'][i]) in ENVIDA_LIBS)]
    eli.sort(key=lambda i: (abs(float(mass[i]) - q['neutral_mass']), int(idx['row'][i])))
    return eli[:MAXREF], len(eli)


def ref_record(i, idx, scores, retained):
    return {'train_row': int(idx['row'][i]), 'canonical': str(idx['canon'][i]),
            'source': str(idx['lib'][i]), 'instrument': str(idx['instrument'][i]),
            'mode': str(idx['mode'][i]), 'adduct': str(idx['adduct'][i]),
            'precursor_mz': float(idx['prec'][i]), 'neutral_mass': float(idx['nm'][i]),
            'peak_hash': str(idx['phash'][i]), 'clean_peaks': int(idx['n_clean'][i]),
            'signal_retained_raw': float(retained[int(idx['row'][i])]), **scores[i]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--output-dir', type=str, default=None)
    ap.add_argument('--inputs-path', type=str, default=str(ROOT / 'artifacts/runs/017-panel-inputs/inputs.json'))
    ap.add_argument('--ice-path', type=str, default=str(ROOT / 'artifacts/runs/017-panel-inputs/ice_scores.json'))
    ap.add_argument('--cross-external-only', action='store_true')
    args = ap.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else OUT
    t0 = time.time()
    panel_path = ROOT / 'artifacts/runs/016-panel/panel_manifest.json'
    inputs_path, ice_path = Path(args.inputs_path), Path(args.ice_path)
    panel = json.load(open(panel_path))['panel']
    inputs = {m['key']: m for m in json.load(open(inputs_path))['molecules']}
    ice = json.load(open(ice_path))
    smap = smiles_map()
    idx, n = load_index()
    # Retention is diagnostic only; keep indexed by original train row.
    import pyarrow.parquet as pq
    audit_t = pq.read_table(ROOT / 'artifacts/runs/016-index/index.parquet', columns=['row', 'sig_ret'])
    retained = np.full(int(np.max(idx['row'])) + 1, np.nan)
    retained[np.asarray(audit_t.column('row'))] = np.asarray(audit_t.column('sig_ret'))
    del audit_t
    qhashes = {p['query_peak_hash'] for p in panel if p.get('query_peak_hash')}
    targets = {p['canonical'] for p in panel}
    prepared = prepare_eligible(idx, qhashes, targets)
    shard_cache = {}
    queries = []
    for p in panel:
        m = inputs[p['canonical']]
        fp_order = m['order_top100']
        if len(set(fp_order)) != len(fp_order):
            raise ValueError(('Duplicate canonical candidates', p['canonical']))
        frozen_input = {**m, 'base_scores_full_window': dict(zip(fp_order, m['scores_top100']))}
        order100, base = frozen_v9(frozen_input, ice.get(p['canonical'], {}))
        # Confirm canonical-based cache lookup really recovered the neural signal.
        cache_canons = {r['canon'] for r in ice.get(p['canonical'], {}).values()
                        if r.get('fused_merge') is not None and np.isfinite(r['fused_merge'])}
        recovered = {c for c, v in base.items() if v['ice_missing'] == 0}
        if recovered != cache_canons.intersection(fp_order):
            raise ValueError(('ICE alignment', p['canonical']))
        selected = max(range(len(m['spectra'])), key=lambda i: len(m['spectra'][i]['mz']))
        s = m['spectra'][selected]
        cmz, cit, audit = clean_peaks(np.asarray(s['mz'], float), np.asarray(s['it'], float),
                                     float(s['prec']), s['adduct'])
        queries.append({'key': p['canonical'], 'group': p['group'], 'truth': p['canonical'],
                        'adduct': s['adduct'], 'mode': s['mode'], 'precursor_mz': float(s['prec']),
                        'neutral_mass': neutral_mass(s['prec'], s['adduct']),
                        'cmz': cmz, 'cit': cit, 'baseline_order': order100,
                        'candidates': order100[:50], 'base': base,
                        'rep_smiles': m['rep_smiles'], 'formulas': m['formulas'],
                        'ice_supported': bool(recovered), 'finite_ice_count': len(recovered),
                        'fp_order': fp_order, 'clean_audit': audit})
    if args.probe:
        queries = [q for q in queries if q['group'] == 'cross_source'][:2] + \
                  [q for q in queries if q['group'] == 'natural_missing'][:2]
    all_X, qindex, sindex, positions, manifest_q, evidence = [], [], [], [], [], []
    selection_checks = []
    for qi, q in enumerate(queries):
        pools, eligible_counts = {}, {}
        for scenario in ('B_seen', 'A_absent'):
            pools[scenario], eligible_counts[scenario] = select_pool(q, scenario, idx, prepared, args.cross_external_only)
            if args.probe:
                old, old_n = select_pool_original(q, scenario, idx, qhashes, targets, args.cross_external_only)
                match = old == pools[scenario] and old_n == eligible_counts[scenario]
                selection_checks.append({'case': q['key'], 'scenario': scenario,
                                         'match': match, 'eligible': old_n, 'selected': len(old)})
                if not match:
                    raise ValueError(('Accelerated selection differs', q['key'], scenario))
        union = sorted(set(pools['B_seen']) | set(pools['A_absent']))
        refs = {}
        for i in union:
            rmz, rit = read_peaks(shard_cache, int(idx['shard'][i]), int(idx['off'][i]), int(idx['n_clean'][i]))
            refs[i] = {'mz': rmz, 'intensity': rit, 'adduct': str(idx['adduct'][i]),
                       'precursor_mz': float(idx['prec'][i]), 'canonical': str(idx['canon'][i])}
        scores = {i: spectral_evidence({'mz': q['cmz'], 'intensity': q['cit'],
                                       'adduct': q['adduct'], 'precursor_mz': q['precursor_mz']}, refs[i]) for i in union}
        for si, scenario in enumerate(('B_seen', 'A_absent')):
            pool = pools[scenario]
            ordered = sorted(pool, key=lambda i: (-scores[i]['quality'], -scores[i]['cosine'], int(idx['row'][i])))
            unique, seen = [], set()
            for i in ordered:
                if scores[i]['quality'] <= 0:
                    continue
                c = refs[i]['canonical']
                if c not in seen:
                    unique.append(i)
                    seen.add(c)
                    if len(unique) == TOPREF:
                        break
            qualities = np.asarray([scores[i]['quality'] for i in unique])
            ref_smiles = [smap.get(refs[i]['canonical']) for i in unique]
            if any(s is None for s in ref_smiles):
                raise ValueError(('Reference without smiles', q['key']))
            by_canon = {}
            for i in pool:
                by_canon.setdefault(refs[i]['canonical'], []).append(i)
            direct_records = []
            for pos, cand in enumerate(q['candidates']):
                sims = tanimoto_to_refs(q['rep_smiles'][cand], ref_smiles) if unique else np.array([])
                keep = np.asarray([refs[i]['canonical'] != cand for i in unique], dtype=bool)
                direct_rows = by_canon.get(cand, [])
                best = min(direct_rows, key=lambda i: (-scores[i]['quality'], -scores[i]['cosine'], int(idx['row'][i]))) if direct_rows else None
                dq = scores[best]['quality'] if best is not None else 0.0
                b = q['base'][cand]
                row = [b['fp_z'], b['ice_z'], 1 - b['ice_missing'], 1.0 / (pos + 1), dq]
                row += aggregate_analog(sims[keep], qualities[keep])
                all_X.append(row)
                qindex.append(qi)
                sindex.append(si)
                positions.append(pos)
                direct_records.append({'candidate': cand, 'candidate_position': pos,
                                       'reference_count': len(direct_rows),
                                       'best_reference': ref_record(best, idx, scores, retained) if best is not None else None})
            lib_counts = {}
            for i in pool:
                lib = str(idx['lib'][i]); lib_counts[lib] = lib_counts.get(lib, 0) + 1
            evidence.append({'case': q['key'], 'group': q['group'], 'scenario': scenario,
                             'query_clean_audit': q['clean_audit'], 'eligible': eligible_counts[scenario],
                             'selected': len(pool), 'truncated': max(0, eligible_counts[scenario] - len(pool)),
                             'selected_unique_structures': len(by_canon),
                             'quality_positive_spectra': sum(scores[i]['quality'] > 0 for i in pool),
                             'positive_analog_structures': len(unique), 'sources': lib_counts,
                             'candidate_direct': direct_records,
                             'analog_references': [ref_record(i, idx, scores, retained) for i in unique]})
        manifest_q.append({'key': q['key'], 'group': q['group'], 'truth': q['truth'],
                           'baseline_order': q['baseline_order'], 'candidate_keys': q['candidates'],
                           'candidate_formulas': [q['formulas'].get(k) for k in q['candidates']],
                           'adduct': q['adduct'], 'ice_supported': q['ice_supported'],
                           'finite_ice_count': q['finite_ice_count'],
                           'fp_order_changed': q['fp_order'] != q['baseline_order'],
                           'fp_top50_set_changed': set(q['fp_order'][:50]) != set(q['candidates'])})
        if (qi + 1) % 50 == 0 or qi == len(queries) - 1:
            print(f"features {qi+1}/{len(queries)} wall={time.time()-t0:.0f}s", flush=True)
    X = np.asarray(all_X, dtype=np.float32)
    y = np.asarray([manifest_q[qidx]['candidate_keys'][pos] == manifest_q[qidx]['truth']
                    for qidx, pos in zip(qindex, positions)], dtype=np.uint8)
    hash_paths = {'inputs': inputs_path, 'ice': ice_path, 'panel': panel_path,
                  'reference_index': ROOT / 'artifacts/runs/016-index/index.parquet',
                  'builder': Path(__file__), 'frozen_v9_source': ROOT / 'scripts/015/build_015_candidate_features.py',
                  'reference_loader': ROOT / 'scripts/016/build_016_reference_index.py',
                  'spectral_evidence': ROOT / 'src/analog_evidence.py'}
    hashes = {name: {'path': str(path), 'sha256': sha_file(path)} for name, path in hash_paths.items()}
    summary = {'queries': len(queries), 'rows': len(all_X), 'finite': bool(np.isfinite(X).all()),
               'cross_external_only': args.cross_external_only,
               'ice_supported_queries': sum(q['ice_supported'] for q in queries),
               'finite_ice_candidates_full100': sum(q['finite_ice_count'] for q in queries),
               'finite_ice_feature_rows': int((X[:, 2] > 0).sum()),
               'fp_order_changed_queries': sum(q['fp_order_changed'] for q in manifest_q),
               'fp_top50_set_changed_queries': sum(q['fp_top50_set_changed'] for q in manifest_q),
               'reference_count': n, 'seconds': round(time.time() - t0, 1),
               'selection_checks': selection_checks, 'hashes': hashes,
               'reference_contract': 'Keep 016: exclude current400 representative exact hashes; A additionally removes all400 canonical. Old-panel hashes are not extra reference exclusions.',
               'numeric_contract': '016 float32 stored reference peaks converted to float64; query cleaned at float64.'}
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.probe:
        np.savez_compressed(out_dir / 'features.npz', X=X, y=y,
                            query_index=np.asarray(qindex, dtype=np.int32),
                            scenario_index=np.asarray(sindex, dtype=np.uint8),
                            candidate_position=np.asarray(positions, dtype=np.int16))
        json.dump({'feature_names': FEATURES, 'scenarios': ['B_seen', 'A_absent'],
                   'queries': manifest_q, 'hashes': hashes}, open(out_dir / 'manifest.json', 'w'))
        with open(out_dir / 'reference_evidence.jsonl', 'w') as f:
            for row in evidence:
                f.write(json.dumps(row) + '\n')
    json.dump(summary, open(out_dir / ('probe_summary.json' if args.probe else 'summary.json'), 'w'), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('hashes', 'selection_checks')}, indent=2))
    if args.probe:
        print('selection_checks', json.dumps(selection_checks), flush=True)


if __name__ == '__main__':
    main()
