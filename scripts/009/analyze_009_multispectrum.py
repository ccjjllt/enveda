"""手动固定多谱证据实验：只读研究集缓存，无训练/远端/保留集。

同一质量窗、代表SMILES与原FP top100上比较代表谱、logits均值、adduct平衡均值。
新增证据只改变FP评分；ICE缓存按原代表SMILES取值，不为缺失缓存造0分。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/007'))
from analyze_007_g1_evidence import compare, zscore


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rr(order, truth):
    return 1.0 / (order.index(truth) + 1) if truth in order[:25] else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'artifacts/runs/009-multispectrum/results.json')
    ap.add_argument('--g1-windows', type=Path, default=ROOT / 'artifacts/runs/009-multispectrum/g1_windows.npz')
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    tk = np.load(ROOT / 'artifacts/shared/pool-train/keys.npy', allow_pickle=True).astype(str)
    cm = pickle.loads((ROOT / 'artifacts/runs/005/coco/coco_meta.pkl').read_bytes())
    ck = np.asarray(cm['keys']).astype(str)
    assert not (set(tk) & set(ck)), 'Union index requires no shipped-key overlap.'
    keys = np.concatenate([tk, ck])
    smi = (ROOT / 'artifacts/shared/pool-train/smiles.txt').read_text().splitlines() + list(cm['smiles'])
    tfp = np.load(ROOT / 'artifacts/shared/pool-train/fp.npy', mmap_mode='r')
    cfp = np.load(ROOT / 'artifacts/runs/005/coco/coco_fp.npy', mmap_mode='r')
    ntr = len(tk)
    maps = []
    for name in ['train', 'coco']:
        tab = pq.read_table(ROOT / f'artifacts/runs/005/structure_identity_{name}.parquet',
                            columns=['raw_key', 'canonical_group_key']).to_pydict()
        maps.append(dict(zip(tab['raw_key'], tab['canonical_group_key'])))
    canons = np.asarray([maps[0][k] for k in tk] + [maps[1][k] for k in ck])
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': 20261012,
              'device': 'local CPU, cached FPNet logits; no model training or forward',
              'protocol': 'pre-fixed representative/all_mean/adduct_balanced; fixed original representatives and top100; full-window FP diagnostic also reported',
              'limitations': ['development sets reused; no heldout or leaderboard evaluation',
                              'full new shortlist and ICE for newly entering candidates NOT evaluated',
                              'public FP checkpoint training exposure unknown'], 'domains': {}}
    for domain, bundle, score, windows in [
            ('np', 'f1-bundle', 'f1-scores/scores/f1e-4_ice_top100_split.json', ROOT / 'artifacts/runs/005/e0_cq_windows.npz'),
            ('g1', 'g1-validation', 'g1-scores/scores/g1e-4_ice_top100_split.json', args.g1_windows)]:
        inp = ROOT / f'artifacts/{bundle}/inputs.json'
        lp = ROOT / f'artifacts/{bundle}/inputs_e0_logits.npy'
        sp = ROOT / f'artifacts/{score}'
        molecules = json.loads(inp.read_text())['molecules']
        logits = np.load(lp, mmap_mode='r')
        cache = json.loads(sp.read_text())
        ws = np.load(windows, allow_pickle=True)
        rows, scores_out, offset = [], {}, 0
        max_baseline_error = 0.
        for m in molecules:
            key = m['key']
            specs = m['spectra']
            nq = len(specs)
            select_local = max(range(nq), key=lambda i: len(specs[i]['mz']))
            assert m['qidx'] == offset + select_local, (domain, key, 'row alignment')
            qs = np.asarray(logits[offset:offset + nq], dtype=np.float64)
            groups = defaultdict(list)
            for i, spec in enumerate(specs):
                groups[(spec['mode'], spec['adduct'])].append(i)
            qmethods = {'representative': qs[select_local], 'all_mean': qs.mean(axis=0),
                        'adduct_balanced': np.mean([qs[ix].mean(axis=0) for ix in groups.values()], axis=0)}
            offset += nq
            w = np.asarray(ws[f'union|single|10.0|{key}'], dtype=np.int64)
            packs = np.stack([tfp[j] if j < ntr else cfp[j - ntr] for j in w])
            bits = np.unpackbits(packs, axis=1, bitorder='big')[:, :6930].astype(np.float64)
            single = bits @ qmethods['representative']
            # Reconstruct original max-over-shipped representative without labels.
            best = {}
            for pos, j in enumerate(w):
                canon = str(canons[j])
                if canon not in best or single[pos] > single[best[canon]]:
                    best[canon] = pos
            assert set(best) == set(m['base_scores_full_window'])
            for canon, pos in best.items():
                err = abs(float(single[pos]) - m['base_scores_full_window'][canon])
                max_baseline_error = max(max_baseline_error, err)
                assert err < 1e-6, (domain, key, canon, err)
                assert smi[int(w[pos])] == m['rep_smiles'][canon], (domain, key, 'representative')
            full_keys = list(best)
            rep_bits = bits[[best[k] for k in full_keys]]
            scoremaps = {name: dict(zip(full_keys, (rep_bits @ q).tolist())) for name, q in qmethods.items()}
            short = m['order_top100']
            sims = [cache[key].get(m['rep_smiles'][k], {}).get('fused_merge') for k in short]
            finite = np.asarray([v is not None and np.isfinite(v) for v in sims])
            row = {'key': key, 'n_spectra': nq, 'n_adduct_groups': len(groups),
                   'modes': sorted({s['mode'] for s in specs}), 'has_ice': bool(finite.any()),
                   'n_window': len(full_keys), 'orders_top25': {}}
            scores_out[key] = {'full_keys': full_keys, 'fp': scoremaps, 'frozen_top100': short,
                               'ice_sim': sims, 'representative_pool_indices': {k: int(w[v]) for k, v in best.items()}}
            for name, scores in scoremaps.items():
                whole = sorted(full_keys, key=lambda k: (-scores[k], k))
                row[name + '_fp_full'] = rr(whole, m['truth_canon'])
                row[name + '_shortlist_recall100'] = m['truth_canon'] in whole[:100]
                base = np.asarray([scores[k] for k in short])
                f = zscore(base) + zscore(sims)
                ix = np.arange(len(short))
                fused = sorted(ix[finite], key=lambda i: (-f[i], i))
                fused += sorted(ix[~finite], key=lambda i: (-base[i], i))
                order = [short[i] for i in fused]
                row[name + '_ice_fixed100'] = rr(order, m['truth_canon'])
                row['orders_top25'][name] = order[:25]
            rows.append(row)
        assert offset == logits.shape[0]
        expected = .6689965804009282 if domain == 'np' else .9070219389114126
        observed = np.mean([r['representative_ice_fixed100'] for r in rows])
        # Published figures rounded: preserve exact computed result in output.
        assert abs(observed - expected) < 1e-6, (domain, observed)
        strata = {'all': rows, 'multi': [r for r in rows if r['n_spectra'] > 1],
                  'dual_polarity': [r for r in rows if len(r['modes']) > 1],
                  'ice_uncovered': [r for r in rows if not r['has_ice']]}
        comparisons = {}
        for sname, subset in strata.items():
            if subset:
                comparisons[sname] = {name + '_' + scope: compare(subset, 'representative_' + scope, name + '_' + scope, 20261012, 20000)
                                     for name in ['all_mean', 'adduct_balanced']
                                     for scope in ['fp_full', 'ice_fixed100']}
        result['domains'][domain] = {'n_molecules': len(rows), 'n_spectra': offset,
            'baseline_max_abs_error': max_baseline_error,
            'sources_sha256': {str(p): digest(p) for p in [inp, lp, sp, windows]},
            'comparisons': comparisons, 'rows': rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / f'{domain}_candidate_scores.json').write_text(json.dumps(scores_out))
        print(domain, json.dumps(comparisons['all']), flush=True)
    result['wall_seconds'] = time.monotonic() - started
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('saved', args.output, 'seconds', result['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
