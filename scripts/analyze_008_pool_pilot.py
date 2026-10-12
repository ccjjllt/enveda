"""Manual JSON-only audit of R3 oracle pool pilot and observation coverage.
No model execution, downloads, training, spectra reads or repository changes.
"""
import argparse
import collections
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def summarize_observations(doc):
    result = {'n_molecules': len(doc['molecules']),
              'n_spectra': sum(len(m['spectra']) for m in doc['molecules']),
              'by_class': {}}
    for cls in ('A', 'B'):
        mols = [m for m in doc['molecules'] if m['class_AB'] == cls]
        specs = [s for m in mols for s in m['spectra']]
        result['by_class'][cls] = {
            'n_molecules': len(mols), 'n_spectra': len(specs),
            'mass_median_min': min(m['mass_median'] for m in mols),
            'mass_median_max': max(m['mass_median'] for m in mols),
            'instrument_families': dict(collections.Counter(s['instrument_family'] for s in specs)),
            'ionization_modes': dict(collections.Counter(s['ionization_mode'] for s in specs)),
            'source_accession_prefixes': dict(collections.Counter(m['external_structure_ids'][0].rsplit('-', 1)[0] for m in mols)),
            'license_strings_as_recorded': dict(collections.Counter(m['license'] for m in mols)),
            'exposure_as_recorded': dict(collections.Counter(m['exposure'] for m in mols)),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--output', default='artifacts/008/pool-pilot-audit.json')
    args = ap.parse_args()
    base = Path(args.repo) / 'artifacts/007-r3'
    pilot = read_json(base / 'pool_pilot.json')
    obs10 = read_json(base / 'observations_10.json')
    obs50 = read_json(base / 'observations_50.json')
    coverage = read_json(base / 'coverage_50.json')
    b_rows = [r for r in pilot['per_molecule'] if r['class_AB'] == 'B']
    ranks = [r['base_rank_pilot'] for r in b_rows]
    b_truths = {r['canonical_key'] for r in b_rows}
    inserted = {r['canonical_key'] for r in pilot['pilot_pool']}
    match_report = []
    for query in pilot['per_molecule']:
        qm = query['QM_med']
        matches = [r['canonical_key'] for r in pilot['pilot_pool']
                   if abs(r['mass_theo'] - qm) <= abs(qm) * 1e-5]
        match_report.append({'query_key': query['canonical_key'], 'mol_temp': query['mol_temp'],
                             'class_AB': query['class_AB'], 'inserted_candidates_in_window': matches,
                             'only_own_truth_or_none': matches == ([query['canonical_key']] if query['class_AB'] == 'B' else [])})
    unknown = [{'query_key': m['canonical_key'], 'spectrum_id': s['spectrum_id'],
                'adduct': s['adduct'], 'precursor_mz': s['precursor_mz'],
                'query_neutral_mass': s['query_neutral_mass'], 'qc_pass': s['qc_pass'],
                'qc_notes': s['qc_notes']}
               for m in obs50['molecules'] for s in m['spectra'] if not s['known_adduct']]
    selected = {s['spectrum_id'] for m in obs50['molecules'] for s in m['spectra']}
    report = {
        'scope': 'JSON-only manual verification. Pilot inserts known truth structures; no unlabeled candidate discovery is evaluated. License values below are source strings, not legal conclusions.',
        'input_files': [str(base / n) for n in ('pool_pilot.json', 'observations_10.json', 'observations_50.json', 'coverage_50.json')],
        'pilot': {
            'n_inserted': len(pilot['pilot_pool']), 'inserted_identities_equal_five_B_truths': inserted == b_truths,
            'B_ranks_after': ranks,
            'B_mrr_at_25_recomputed': sum(1/r for r in ranks if r is not None and r <= 25) / len(ranks),
            'B_top1_n': sum(r == 1 for r in ranks),
            'B_top25_n': sum(r is not None and r <= 25 for r in ranks),
            'B_canonical_window_sizes_after': [r['n_win_canon_pilot'] for r in b_rows],
            'guaranteed_B_top25_from_window_size_alone': sum(r['n_win_canon_pilot'] <= 25 for r in b_rows),
            'stop_threshold_top25_le': pilot['stop_decision']['threshold_top25_le1_stop'],
            'candidate_window_matches': match_report,
            'all_insertions_only_enter_own_query_window': all(r['only_own_truth_or_none'] for r in match_report),
            'source_code_evidence': {'oracle_smiles_read': 'r3_pool_pilot.py:143-169', 'only_five_added': 'r3_pool_pilot.py:170-184', 'stop_decision': 'r3_pool_pilot.py:466-474'},
        },
        'observations_10': summarize_observations(obs10),
        'observations_50': summarize_observations(obs50),
        'coverage_50': {
            'ranking_method_as_recorded': coverage['meta']['inputs']['top100'],
            'maximum_shipped_window_size': max(r['window_size_shipped_10ppm'] for r in coverage['per_molecule']),
            'n_windows_above100': sum(r['window_size_shipped_10ppm'] > 100 for r in coverage['per_molecule']),
            'summary_as_recorded': coverage['summary'],
        },
        'unknown_adduct_spectra': unknown,
        'unknown_adduct_spectrum_count': len(unknown),
        'selected_accessions_also_in_historical_discards': len({r['accession'] for r in obs50['discards'] if r['accession'] in selected}),
        'interpretation': 'Conditional ranking is observed after oracle insertion; four Top25 hits and the CONTINUE criterion are already guaranteed by small windows. A stability and no added decoys do not measure realistic expansion interference, since inserted structures enter only their own B query windows. Coverage50 uses mass ranking, not FP or full v9 ranking.',
    }
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'B_ranks': ranks, 'B_MRR': report['pilot']['B_mrr_at_25_recomputed'],
                      'Top25_guaranteed_by_window': report['pilot']['guaranteed_B_top25_from_window_size_alone'],
                      'only_own_windows': report['pilot']['all_insertions_only_enter_own_query_window'],
                      'max_window50': report['coverage_50']['maximum_shipped_window_size'],
                      'spectra50': report['observations_50']['n_spectra'], 'unknown_adduct': len(unknown)}, ensure_ascii=False))
    print('wrote', args.output)


if __name__ == '__main__':
    main()
