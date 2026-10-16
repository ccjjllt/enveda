"""手动独立复算009 LMSD的逐例指标；不运行模型、不新增评估标签。"""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    start = time.perf_counter()
    p = ROOT / 'artifacts/009-source/recall_eval_50.json'
    data = json.loads(p.read_text())
    rows = data['per_molecule']
    def rr(rank):
        return 1. / rank if rank is not None and 1 <= rank <= 25 else 0.
    failures, strata = [], {}
    for row in rows:
        for side in ['base', 'expanded']:
            observed = rr(row[f'final_top25_{side}_rank_oracle'])
            if abs(observed - row[f'final_top25_{side}_RR']) > 1e-12:
                failures.append([row['canonical_key'], side])
    for name, subset in [('all', rows), ('A', [r for r in rows if r['class_AB'] == 'A']),
                         ('B_known', [r for r in rows if r['class_AB'] == 'B' and not r['mass_uncertain']])]:
        before = [rr(r['final_top25_base_rank_oracle']) for r in subset]
        after = [rr(r['final_top25_expanded_rank_oracle']) for r in subset]
        strata[name] = {'n': len(subset), 'mrr_base': sum(before) / len(subset),
            'mrr_expanded': sum(after) / len(subset),
            'new_true_structures_in_full_index': sum(r['full_expanded_covered_oracle'] and not r['full_base_covered_oracle'] for r in subset),
            'top25_base': sum(v > 0 for v in before), 'top25_expanded': sum(v > 0 for v in after),
            'added_window_pairs': sum(r['window_expanded_shipped'] - r['window_base_shipped'] for r in subset),
            'new_candidate_top25_pairs': sum(r['n_new_in_top25'] for r in subset),
            'displaced_old_top25_pairs': sum(r['n_displaced_top25'] for r in subset),
            'rank_changed': sum(r['final_top25_base_rank_oracle'] != r['final_top25_expanded_rank_oracle'] for r in subset)}
    result = {'source': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
        'rows': len(rows), 'uncertain_mass': sum(r['mass_uncertain'] for r in rows),
        'rr_recomputation_mismatches': failures, 'strata': strata,
        'conclusion': 'Zero new truths and zero RR change on this selected non-lipid R3 sample. This does not establish the value of LMSD on lipids or justify stopping all other structure sources.',
        'scope': 'Stored ranks independently converted to reciprocal-rank metrics; no re-scan of 50k SDF records and no model execution.',
        'seconds': time.perf_counter() - start}
    out = ROOT / 'artifacts/010-source-audit/summary.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
