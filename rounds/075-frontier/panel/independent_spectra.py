"""Measure FLARE rank changes after cross-source experimental spectrum retrieval."""

from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')
PANEL = INPUT / 'notebooks/jiyujia/enveda-075b-mass-queries'
FLARE = INPUT / 'notebooks/jiyujia/enveda-075b-flare-mist/scores.parquet'
ADDUCT = '[M+H]+'
PEAK_COLUMNS = ['ms2_mzs', 'ms2_normalized_intensities']
REFERENCE_COLUMNS = ['inchikey14', 'adduct', 'ingest_lib', 'precursor_mz']


def candidate_raw_keys(candidates):
    try:
        installed = version('rdkit')
    except PackageNotFoundError:
        installed = None
    if installed not in ('2026.3.3', '2026.03.3'):
        abi = f'cp{sys.version_info.major}{sys.version_info.minor}'
        wheel = next(path for path in INPUT.rglob('rdkit-*.whl') if f'-{abi}-' in path.name)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps',
                        '--no-index', str(wheel)], check=True)
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog('rdApp.*')
    aliases = defaultdict(set)
    for candidate in candidates.itertuples(index=False):
        molecule = Chem.MolFromSmiles(candidate.smiles)
        aliases[candidate.candidate_id].add(candidate.candidate_id)
        if molecule is not None:
            raw = Chem.MolToInchiKey(molecule)[:14]
            if raw:
                aliases[raw].add(candidate.candidate_id)
    return {raw: next(iter(canonical)) for raw, canonical in aliases.items() if len(canonical) == 1}


def preprocess(mz, intensities, precursor):
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensities, dtype=np.float64)
    if mz.ndim != 1 or mz.shape != intensity.shape or not np.isfinite(precursor):
        return np.empty((0, 2), dtype=np.float64)
    mask = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    mask &= (mz <= precursor + 1) & (np.abs(mz - precursor) > 0.05)
    mz, intensity = mz[mask], intensity[mask]
    if not len(mz):
        return np.empty((0, 2), dtype=np.float64)
    by_intensity = np.argsort(-intensity, kind='stable')
    mz, intensity = mz[by_intensity], intensity[by_intensity]
    _, first = np.unique(np.round(mz, 4), return_index=True)
    selected = np.sort(first)[:60]
    weights = np.sqrt(intensity[selected] / intensity[0])
    selected, weights = selected[weights > 0.001], weights[weights > 0.001]
    return np.column_stack([mz[selected], weights / np.linalg.norm(weights)])


def similarity(query, reference):
    if not len(query) or not len(reference):
        return 0.0, False
    tolerance = np.maximum(0.01, query[:, 0] * 20e-6)
    left, right = np.nonzero(np.abs(query[:, 0, None] - reference[:, 0]) <= tolerance[:, None])
    if not len(left):
        return 0.0, False
    weights = query[left, 1] * reference[right, 1]
    used_left, used_right = set(), set()
    score = 0.0
    for index in np.argsort(-weights, kind='stable'):
        if left[index] not in used_left and right[index] not in used_right:
            used_left.add(left[index])
            used_right.add(right[index])
            score += float(weights[index])
    near_duplicate = score >= 0.995 and len(used_left) >= 0.95 * max(len(query), len(reference))
    return min(score, 1.0), near_duplicate


def query_inputs(train, queries):
    locations = defaultdict(dict)
    for query in queries.itertuples(index=False):
        group, index = map(int, query.query_id.split('-'))
        locations[group][index] = query
    molecules = defaultdict(list)
    excluded_rows = set()
    for group, items in locations.items():
        indices = sorted(items)
        sources = train.read_row_group(group, columns=['ingest_lib']).take(indices).column(0).to_pylist()
        for index, source in zip(indices, sources, strict=True):
            query = items[index]
            excluded_rows.add((group, index))
            if source and query.adduct == ADDUCT:
                spectrum = preprocess(query.mz, query.intensity, float(query.precursor_mz))
                if len(spectrum) >= 3:
                    molecules[query.molecule_id].append((str(source).strip().casefold(), spectrum))
    return molecules, excluded_rows


def search(train, candidates, queries):
    raw_keys = candidate_raw_keys(candidates)
    known_raw = pa.array(sorted(raw_keys), type=pa.string())
    molecules, excluded_rows = query_inputs(train, queries)
    sources = {mid: {source for source, _ in spectra} for mid, spectra in molecules.items()}
    owners = candidates.groupby('candidate_id')['molecule_id'].agg(lambda mids: set(mids)).to_dict()
    best = {}
    selected_references, duplicates = 0, 0

    for group in range(train.num_row_groups):
        meta = train.read_row_group(group, columns=REFERENCE_COLUMNS)
        selected = pc.and_(pc.is_in(pc.cast(meta['inchikey14'], pa.string()), value_set=known_raw),
                           pc.equal(meta['adduct'], ADDUCT))
        indices = pc.indices_nonzero(selected).to_pylist()
        if not indices:
            continue
        wanted = []
        for index, row in zip(indices, meta.take(indices).to_pylist(), strict=True):
            ref_lib = str(row['ingest_lib']).strip().casefold() if row['ingest_lib'] else ''
            if (group, index) in excluded_rows or not ref_lib or row['precursor_mz'] is None:
                continue
            cid = raw_keys[row['inchikey14']]
            mids = [mid for mid in owners[cid] if mid in sources and ref_lib not in sources[mid]]
            if mids:
                wanted.append((index, cid, mids, float(row['precursor_mz'])))
        if not wanted:
            continue
        spectra = train.read_row_group(group, columns=PEAK_COLUMNS).take([row[0] for row in wanted])
        for (_, cid, mids, precursor), row in zip(wanted, spectra.to_pylist(), strict=True):
            reference = preprocess(row[PEAK_COLUMNS[0]], row[PEAK_COLUMNS[1]], precursor)
            if len(reference) < 3:
                continue
            selected_references += 1
            for mid in mids:
                matched = [similarity(query, reference) for _, query in molecules[mid]]
                if any(near_duplicate for _, near_duplicate in matched):
                    duplicates += 1
                    continue
                values = np.array([score for score, _ in matched])
                pair = mid, cid
                best[pair] = np.maximum(best[pair], values) if pair in best else values
    return {pair: float(values.mean()) for pair, values in best.items()}, selected_references, duplicates, molecules


def reorder_with_flare(library, flare):
    reranked = library.copy()
    for _, candidates in library.groupby('formula', sort=False):
        ids = candidates['candidate_id'].tolist()
        if len(ids) < 2:
            continue
        completed = flare.reindex(columns=ids).dropna()
        if completed.empty:
            continue
        ordered = candidates.assign(score=candidates['candidate_id'].map(completed.mean()))
        order = ordered.sort_values(['score', 'library_rank'], ascending=[False, True]).index
        reranked.loc[order, 'flare_rank'] = sorted(candidates['library_rank'])
    return reranked


def require_ce_agreement(ranking, flare, queries):
    """Keep the library leader unless every scored spectrum supports its challenger."""
    revised = ranking.copy()
    original = revised.loc[revised['library_rank'].eq(1)].iloc[0]
    challenger = revised.loc[revised['flare_rank'].eq(1)].iloc[0]
    if original['candidate_id'] == challenger['candidate_id']:
        return revised
    old_id, new_id = original['candidate_id'], challenger['candidate_id']
    paired = flare.reindex(columns=[old_id, new_id]).dropna().join(queries[['adduct', 'collision_energy']])
    measured = paired.dropna(subset=['adduct', 'collision_energy'])
    ce_count = measured.groupby('adduct')['collision_energy'].nunique()
    if ce_count.ge(2).any() and measured[new_id].gt(measured[old_id]).all():
        return revised
    group = revised[revised['formula'].eq(original['formula'])].sort_values('flare_rank')
    order = [original.name, *group.index[group.index != original.name]]
    revised.loc[order, 'ce_agreement_rank'] = sorted(group['flare_rank'])
    return revised


def rank_comparison(results):
    comparison = {}
    for strategy, rank in (('spectral_library', 'library_rank'), ('flare', 'flare_rank'),
                           ('flare_ce_agreement', 'ce_agreement_rank')):
        rr = (1.0 / results[rank]).fillna(0.0)
        delta = rr - results['rr_before']
        old_first, new_first = results['library_rank'].eq(1), results[rank].eq(1)
        comparison[strategy] = {
            'molecules': len(results), 'mrr': float(rr.mean()) if len(results) else None,
            'delta_mrr': float(delta.mean()) if len(results) else None,
            'total_delta_rr': float(delta.sum()), 'top1': int(new_first.sum()),
            'promoted_to_first': int((new_first & ~old_first).sum()),
            'lost_first': int((old_first & ~new_first).sum()),
        }
    return comparison


def main(panel=PANEL, flare_path=FLARE, ce_agreement=False):
    started = time.monotonic()
    train = pq.ParquetFile(next(path for path in INPUT.rglob('train.parquet')
                                if path.parent.name.lower() == 'enveda-casmi26-molecule-id-mass-spectra'))
    queries = pd.read_parquet(panel / 'queries.parquet')
    candidates = pd.read_parquet(panel / 'candidates.parquet').sort_values('rank')
    candidates = candidates.drop_duplicates(['molecule_id', 'candidate_id'])
    cosine, reference_count, near_duplicates, usable_query = search(train, candidates, queries)
    print(f'{reference_count} cross-source reference spectra considered, {near_duplicates} near-duplicate reference–molecule pairs removed', flush=True)
    flare = pd.read_parquet(flare_path)
    flare = flare[flare['model'].eq('flare')]
    flare_by_molecule = {mid: group.pivot_table(index='query_id', columns='candidate_id', values='score')
                         for mid, group in flare.groupby('molecule_id', sort=False)}
    query_metadata = queries.set_index('query_id')
    reranked = []
    for mid, group in candidates.groupby('molecule_id', sort=False):
        library = group.assign(library_cosine=[cosine.get((mid, cid), np.nan) for cid in group['candidate_id']])
        library = library.assign(sort_score=library['library_cosine'].fillna(0.0))
        library = library.sort_values(['sort_score', 'rank'], ascending=[False, True]).drop(columns='sort_score')
        library['library_rank'] = library['flare_rank'] = np.sort(group['rank'].to_numpy())
        if mid in flare_by_molecule and library['library_cosine'].gt(0).any():
            library = reorder_with_flare(library, flare_by_molecule[mid])
        if ce_agreement:
            library['ce_agreement_rank'] = library['flare_rank']
            if mid in flare_by_molecule and library['library_cosine'].gt(0).any():
                library = require_ce_agreement(library, flare_by_molecule[mid], query_metadata)
        reranked.append(library)
    ranking = pd.concat(reranked)

    # Labels are read only after the retrieval and both rankings are complete.
    labels = pd.read_parquet(panel / 'labels.parquet')[['molecule_id', 'candidate_id']]
    results = labels.merge(ranking, on=['molecule_id', 'candidate_id'], how='left')
    leaders = ranking.loc[ranking['library_rank'].eq(1)].set_index('molecule_id')['library_cosine']
    results['leader_cosine'] = results['molecule_id'].map(leaders)
    flare_leaders = ranking.loc[ranking['flare_rank'].eq(1)].set_index('molecule_id')['library_cosine']
    results['flare_leader_library_cosine'] = results['molecule_id'].map(flare_leaders)
    results['cross_source_candidate_match'] = results['leader_cosine'].gt(0)
    results['true_has_independent_reference'] = results['library_cosine'].notna()
    results['rr_before'] = (1.0 / results['library_rank']).fillna(0)
    results['rr_after'] = (1.0 / results['flare_rank']).fillna(0)
    results['delta_rr'] = results['rr_after'] - results['rr_before']
    covered = results[results['cross_source_candidate_match']]
    true_covered = results[results['true_has_independent_reference']]
    anchored = true_covered[true_covered['library_rank'].eq(1) & true_covered['library_cosine'].gt(0)]
    lost = anchored[anchored['flare_rank'].gt(1)]
    summary = {
        'molecules': len(results), 'queries_used': sum(map(len, usable_query.values())),
        'reference_spectra': reference_count, 'excluded_near_duplicate_pairs': near_duplicates,
        'covered_any_candidate': len(covered), 'covered_true_candidate': len(true_covered),
        'all_mrr_before': float(results['rr_before'].mean()), 'all_mrr_after': float(results['rr_after'].mean()),
        'covered_mrr_before': float(covered['rr_before'].mean()) if len(covered) else None,
        'covered_mrr_after': float(covered['rr_after'].mean()) if len(covered) else None,
        'true_covered_mrr_before': float(true_covered['rr_before'].mean()) if len(true_covered) else None,
        'true_covered_mrr_after': float(true_covered['rr_after'].mean()) if len(true_covered) else None,
        'correct_library_leaders': len(anchored), 'correct_library_leaders_displaced': len(lost),
        'displaced_leader_cosines': lost['library_cosine'].tolist(),
        'seconds': time.monotonic() - started,
    }
    if ce_agreement:
        results['rr_ce_agreement'] = (1.0 / results['ce_agreement_rank']).fillna(0)
        results['delta_rr_ce_agreement'] = results['rr_ce_agreement'] - results['rr_before']
        summary['comparison'] = {
            'all': rank_comparison(results),
            'any_candidate_cross_source_match': rank_comparison(covered),
            'truth_has_independent_reference': rank_comparison(true_covered),
        }
    results.drop(columns=['smiles', 'formula', 'mass_error_ppm'], errors='ignore').to_parquet(
        OUTPUT / 'independent_spectrum_ranks.parquet', index=False)
    (OUTPUT / 'independent_spectrum_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
