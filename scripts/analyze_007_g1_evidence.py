"""006 G1/F1 手动独立复算；只读研究250及NP250，不读heldout或运行模型。

Run manually: python scripts/analyze_007_g1_evidence.py --output artifacts/007/g1-evidence.json
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def zscore(values):
    values = np.array([np.nan if x is None else float(x) for x in values])
    ok = np.isfinite(values)
    result = np.zeros(values.shape)
    if ok.sum() >= 2:
        std = values[ok].std(ddof=1)
        if std > 0:
            result[ok] = (values[ok] - values[ok].mean()) / std
    return result


def order(mol, scores=None, field='fused'):
    keys = mol['order_top100']
    if scores is None:
        return keys, False, None
    base = np.array([mol['base_scores_full_window'][k] for k in keys])
    sims = [scores.get(mol['rep_smiles'][k], {}).get(field) for k in keys]
    finite = np.array([s is not None and np.isfinite(s) for s in sims])
    combined = zscore(base) + zscore(sims)
    ix = np.arange(len(keys))
    if finite.any():
        ranked = sorted(ix[finite], key=lambda i: (-combined[i], i))
        ranked += sorted(ix[~finite], key=lambda i: (-base[i], i))
    else:
        ranked = sorted(ix, key=lambda i: (-combined[i], i))
    return [keys[i] for i in ranked], bool(finite.any()), sims


def compare(rows, a, b, seed, draws):
    x, y = (np.array([r[k] for r in rows]) for k in (a, b))
    delta = y - x
    bootstrap = np.random.default_rng(seed).integers(0, len(rows), size=(draws, len(rows)))
    return {
        'n': len(rows), 'mrr_a': float(x.mean()), 'mrr_b': float(y.mean()),
        'delta_mrr': float(delta.mean()),
        'ci95': np.quantile(delta[bootstrap].mean(axis=1), [.025, .975]).tolist(),
        'improved': int((delta > 1e-15).sum()), 'worsened': int((delta < -1e-15).sum()),
        'top1_a': int((x == 1).sum()), 'top1_b': int((y == 1).sum()),
        'top1_wrong_to_right': int(((x < 1) & (y == 1)).sum()),
        'top1_right_to_wrong': int(((x == 1) & (y < 1)).sum()),
        'top25_a': int((x > 0).sum()), 'top25_b': int((y > 0).sum()),
    }


def load_sources(repo, prefix):
    if prefix == 'g1':
        paths = ['artifacts/g1-validation/inputs.json',
                 'artifacts/g1-scores/scores/g1e-2_ice_top100_mean.json',
                 'artifacts/g1-scores/scores/g1e-4_ice_top100_split.json']
    else:
        paths = ['artifacts/f1-bundle/inputs.json',
                 'artifacts/f1-scores/scores/f1e-2_ice_top100_mean.json',
                 'artifacts/f1-scores/scores/f1e-4_ice_top100_split.json']
    objs, source_meta = [], {}
    for path in paths:
        raw = (repo / path).read_bytes()
        objs.append(json.loads(raw))
        source_meta[path] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return objs, source_meta


def calculate(repo, prefix, seed, draws):
    (inputs, mean, split), source_meta = load_sources(repo, prefix)
    molecules = inputs['molecules']
    rows = []
    spectra_composition = Counter()
    supported = {'[M+H]+', '[M+Na]+'}
    ce = Counter()
    score_check = {}
    for name, data in [('mean', mean), ('split', split)]:
        counts = Counter()
        maximum_error = 0.
        for candidates in data.values():
            for result in candidates.values():
                counts['candidate_records'] += 1
                for field, detail in [('fused', 'sims'), ('fused_merge', 'sims_merge')]:
                    if field not in result:
                        continue
                    score = result[field]
                    sims = [x['sim'] for x in result[detail]]
                    if score is not None:
                        counts[field + '_finite'] += int(np.isfinite(score))
                        maximum_error = max(maximum_error, abs(score - float(np.mean(sims))))
                    else:
                        counts[field + '_missing'] += 1
                    if (score is None) != (not sims):
                        counts[field + '_inconsistent_missing'] += 1
        score_check[name] = {**dict(counts), 'maximum_fused_vs_detail_mean_error': maximum_error}
    for mol in molecules:
        truth = mol['truth_canon']
        row = {'key': mol['key'], 'n_spectra': len(mol['spectra']),
               'instruments': sorted({s['instrument'] for s in mol['spectra']}),
               'adduct_counts': dict(Counter(s['adduct'] for s in mol['spectra'])),
               'n_window': mol['n_win_canon'], 'base_rank_full': mol['base_rank_full'],
               'truth_in_window': truth in mol['base_scores_full_window'],
               'truth_in_top100': truth in mol['order_top100']}
        for method, data, field in [('base', None, 'fused'), ('mean', mean, 'fused'),
                                    ('merge', split, 'fused_merge'), ('simsmean', split, 'fused')]:
            ranked, covered, sim = order(mol, data[mol['key']] if data is not None else None, field)
            rank = ranked.index(truth) + 1 if truth in ranked else None
            row[method + '_rank'] = rank
            row[method] = 1. / rank if rank is not None and rank <= 25 else 0.
            if method == 'merge':
                row['covered'] = covered
                row['truth_forward_score'] = sim[mol['order_top100'].index(truth)] if truth in mol['order_top100'] else None
                row['top1'] = ranked[0]
                row['top1_same_formula'] = mol['formulas'][ranked[0]] == mol['formulas'][truth]
        if row['merge'] < 1:
            if not row['truth_in_window']:
                row['error_category'] = 'truth_outside_window'
            elif not row['truth_in_top100']:
                row['error_category'] = 'truth_outside_shortlist'
            elif not row['covered']:
                row['error_category'] = 'no_forward_coverage'
            else:
                row['error_category'] = 'forward_covered_ranking'
        rows.append(row)
        spectra_composition.update(s['instrument'] for s in mol['spectra'])
        selected = [s for s in mol['spectra'] if s['adduct'] in supported]
        multi = [s for s in selected if len(s['ce_ev']) > 1]
        single = {(s['adduct'], s['instrument'], float(s['ce_ev'][0])) for s in selected if len(s['ce_ev']) == 1}
        repeated = [s for s in multi if all((s['adduct'], s['instrument'], float(v)) in single for v in s['ce_ev'])]
        ce['supported_spectra'] += len(selected)
        ce['multiCE_spectra'] += len(multi)
        ce['molecules_with_multiCE'] += bool(multi)
        ce['multiCE_with_all_corresponding_singleCE_spectra'] += len(repeated)
        ce['molecules_all_multiCE_have_corresponding_singles'] += bool(multi) and len(multi) == len(repeated)
    comparisons = {b + '-' + a: compare(rows, a, b, seed, draws) for a, b in
                   [('base', 'mean'), ('base', 'merge'), ('mean', 'merge'), ('simsmean', 'merge')]}
    wrong = [r for r in rows if r['merge'] < 1]
    uncovered = [r for r in rows if not r['covered']]
    strata = {}
    for label, selected in [('covered', [r for r in rows if r['covered']]), ('uncovered', uncovered),
                            ('timsTOF_only', [r for r in rows if r['instruments'] == ['timsTOF']]),
                            ('includes_other_instrument', [r for r in rows if r['instruments'] != ['timsTOF']])]:
        if selected:
            strata[label] = {b + '-' + a: compare(selected, a, b, seed, draws) for a, b in
                             [('base', 'merge'), ('mean', 'merge')]}
    groups = {}
    for row in uncovered:
        key = ' / '.join(sorted(row['adduct_counts']))
        if key not in groups:
            groups[key] = {'n': 0, 'top1_errors': 0, 'same_formula_top1_errors': 0, 'keys': []}
        groups[key]['n'] += 1
        groups[key]['top1_errors'] += int(row['merge'] < 1)
        groups[key]['same_formula_top1_errors'] += int(row['merge'] < 1 and row['top1_same_formula'])
        groups[key]['keys'].append(row['key'])
    result = {'n': len(rows), 'sources': source_meta, 'comparisons': comparisons,
              'strata': strata, 'spectra_instruments': dict(spectra_composition),
              'multiCE_composition': dict(ce), 'score_integrity': score_check,
              'coverage': {'truth_in_window': sum(r['truth_in_window'] for r in rows),
                           'truth_in_top100': sum(r['truth_in_top100'] for r in rows),
                           'wrong_top1': len(wrong),
                           'wrong_top1_truth_in_top100': sum(r['truth_in_top100'] for r in wrong),
                           'wrong_top1_same_formula': sum(r['top1_same_formula'] for r in wrong),
                           'wrong_top1_mutually_exclusive_categories': {k: sum(r['error_category'] == k for r in wrong) for k in
                               ['truth_outside_window', 'truth_outside_shortlist', 'no_forward_coverage', 'forward_covered_ranking']},
                           'same_formula_errors_by_category': {k: sum(r['error_category'] == k and r['top1_same_formula'] for r in wrong) for k in
                               ['truth_outside_window', 'truth_outside_shortlist', 'no_forward_coverage', 'forward_covered_ranking']}},
              'uncovered_adduct_groups': groups, 'top25_misses': [r for r in rows if r['merge'] == 0]}
    if prefix == 'g1':
        result['top1_errors'] = wrong
        result['uncovered_molecules'] = uncovered
        result['CE_changed_RR'] = [r for r in rows if r['merge'] != r['mean']]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('artifacts/007/g1-evidence.json'))
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--bootstrap', type=int, default=20000)
    args = parser.parse_args()
    report = {'audit': '007 G1/F1 independent manual metric recalculation',
              'seed': args.seed, 'bootstrap_draws': args.bootstrap,
              'caliber': 'canonical truth; MRR@25; top100; z(base)+z(sim), ddof=1; finite scored candidates first, unscored base tail; ties by shortlist index',
              'limits': ['Only G1 research250 and NP development250 read; heldout not accessed.',
                         'No model execution; CIs do not correct for previous tuning on NP250.',
                         'Top1 correctness strata are descriptive, not an inference-time routing feature.'],
              'g1': calculate(args.repo, 'g1', args.seed, args.bootstrap),
              'np_f1': calculate(args.repo, 'f1', args.seed, args.bootstrap)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'g1_coverage': report['g1']['coverage'],
                      'uncovered_adduct_groups': report['g1']['uncovered_adduct_groups']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
