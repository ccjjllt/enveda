"""同式实测谱跨能量配对。运行 python isomer_energy_pairs.py data/raw/train.parquet。"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


LIMIT = 16384
PAIR_LIMIT = 48
ENERGIES = (20, 40, 60)
TOLERANCE = 0.01


def stable_key(value):
    return sha256(('isomer-energy-v1/' + value).encode()).digest()


def proposed_pairs(parquet):
    cols = ['molecular_formula', 'inchikey14', 'normalized_smiles', 'ingest_lib', 'instrument_type',
            'adduct', 'collision_energy_ev', 'collision_energy_orig', 'collision_energy_orig_units', 'num_peaks', 'precursor_mz']
    meta = next(parquet.iter_batches(batch_size=LIMIT, row_groups=[0], columns=cols)).to_pylist()
    scans = defaultdict(lambda: defaultdict(dict))
    smiles = {}
    for index, r in enumerate(meta):
        if (r['ingest_lib'], r['instrument_type'], r['adduct']) != ('enveda-180', 'timsTOF', '[M+H]+'):
            continue
        ce = r['collision_energy_orig']
        if ce not in ('20', '40', '60') or r['collision_energy_orig_units'] != 'eV' or r['collision_energy_ev'] != [float(ce)]:
            continue
        key, formula = r['inchikey14'], r['molecular_formula']
        if not (key and formula and r['normalized_smiles'] and r['num_peaks'] >= 5):
            continue
        scans[formula][key].setdefault(int(ce), index)
        smiles[key] = r['normalized_smiles']
    eligible = {formula: sorted((key for key, rows in group.items() if len(rows) == 3), key=stable_key)
                for formula, group in scans.items()}
    eligible = {formula: keys for formula, keys in eligible.items() if len(keys) >= 2}
    tautomer = rdMolStandardize.TautomerEnumerator()
    chosen, difficult = [], []
    fingerprinter = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for formula in sorted(eligible, key=stable_key):
        ids, rows, fingerprints, masses = [], [], [], []
        for key in eligible[formula]:
            mol = Chem.MolFromSmiles(smiles[key])
            if mol is None:
                continue
            Chem.RemoveStereochemistry(mol)
            canonical = tautomer.Canonicalize(mol)
            identity = Chem.MolToInchiKey(canonical).split('-')[0]
            if identity and identity not in ids:
                ids.append(identity)
                fingerprints.append(fingerprinter.GetFingerprint(canonical))
                rows.append(scans[formula][key])
                masses.append(rdMolDescriptors.CalcExactMolWt(canonical))
        if len(ids) >= 2 and len(chosen) < PAIR_LIMIT:
            tanimoto = DataStructs.TanimotoSimilarity(*fingerprints[:2])
            chosen.append((formula, rows[:2], tanimoto))
        best = None
        for i, j in combinations(range(len(ids)), 2):
            if abs(masses[i] - masses[j]) > 1e-6:
                continue
            pair = [rows[i], rows[j]]
            precursors = [meta[r]['precursor_mz'] for group in pair for r in group.values()]
            if not all(m is not None and np.isfinite(m) and m > 0 for m in precursors):
                continue
            if np.ptp(precursors) / np.mean(precursors) > 10e-6:
                continue
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            if similarity >= 0.6 and (best is None or similarity > best[2]):
                best = (formula, pair, similarity)
        if best is not None:
            difficult.append(best)
    return chosen, difficult, len(eligible)


def cleaned_peaks(row):
    mass = np.asarray(row['ms2_mzs'], dtype=float)
    intensity = np.asarray(row['ms2_normalized_intensities'], dtype=float)
    keep = np.isfinite(mass) & np.isfinite(intensity) & (intensity > 0) & (mass >= 15.) & (mass < row['precursor_mz'] - 1.5)
    mass, intensity = mass[keep], intensity[keep]
    order = np.argsort(mass, kind='stable')
    intensity = np.sqrt(intensity[order])
    norm = np.linalg.norm(intensity)
    return mass[order], intensity / norm if norm else intensity


def cosine(left, right):
    lm, li = left
    rm, ri = right
    low, high = np.searchsorted(rm, lm - TOLERANCE), np.searchsorted(rm, lm + TOLERANCE, side='right')
    candidates = [(li[i] * ri[j], i, j) for i in range(len(lm)) for j in range(low[i], high[i])]
    used_left, used_right = set(), set()
    total = 0.
    for product, i, j in sorted(candidates, reverse=True):
        if i not in used_left and j not in used_right:
            used_left.add(i)
            used_right.add(j)
            total += product
    return total


def summarize(scores):
    # 每式固定20 eV身份，分别独立置换40和60 eV的两个标签。
    if not scores:
        return {'n_formula_pairs': 0}
    margins = np.asarray(scores)
    avg = margins.mean(axis=1)
    rng = np.random.default_rng(7522)
    sign40, sign60 = rng.choice([-1, 1], size=(2, 10000, len(avg)))
    permuted_totals = sign40 * margins[:, 0] + sign60 * margins[:, 1] + sign40 * sign60 * margins[:, 2]
    null = permuted_totals.mean(axis=1) / 3.
    exceedances = int((null >= avg.mean() - 1e-12).sum())
    return {'n_formula_pairs': len(avg), 'energy_pairs_eV': [[20, 40], [20, 60], [40, 60]],
            'energy_mean_diagonal_minus_swap': margins.mean(axis=0).tolist(),
            'mean_diagonal_minus_swap': float(avg.mean()), 'median_diagonal_minus_swap': float(np.median(avg)),
            'formula_wins_ties_losses': [int((avg > 1e-12).sum()), int((np.abs(avg) <= 1e-12).sum()), int((avg < -1e-12).sum())],
            'null_mean': float(null.mean()), 'permutations': len(null), 'exceedances': exceedances,
            'one_sided_permutation_p': (exceedances + 1) / (len(null) + 1)}


def read_peaks(parquet, wanted):
    peaks = {}
    start = 0
    # Parquet 可按批解码，Arrow take 后才将所选谱转为 Python 对象。
    for batch in parquet.iter_batches(batch_size=512, row_groups=[0], columns=['precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']):
        end = start + len(batch)
        rows = [r for r in wanted if start <= r < end]
        if rows:
            chosen = batch.take([r - start for r in rows]).to_pylist()
            peaks.update((r, cleaned_peaks(s)) for r, s in zip(rows, chosen))
        if end >= LIMIT:
            break
        start = end
    return peaks


def evaluate(selected, peaks):
    discarded = Counter()
    valid, filtered = [], []
    tanimotos = []
    for _, pair, tanimoto in selected:
        ids = [rows[e] for rows in pair for e in ENERGIES]
        if any(len(peaks[r][0]) < 4 for r in ids):
            discarded['under_four_fragment_peaks'] += 1
            continue
        similarities = {tuple(sorted((i, j))): cosine(peaks[i], peaks[j]) for i, j in combinations(ids, 2)}
        def score(i, j):
            return similarities[tuple(sorted((i, j)))]
        a, b = pair
        deltas = [(score(a[i], a[j]) + score(b[i], b[j]) - score(a[i], b[j]) - score(b[i], a[j])) / 2
                  for i, j in combinations(ENERGIES, 2)]
        valid.append(deltas)
        if max(similarities.values()) >= 0.995:
            discarded['any_of_six_near_duplicate'] += 1
            continue
        filtered.append(deltas)
        tanimotos.append(tanimoto)
    wanted = sorted({r for _, pair, _ in selected for rows in pair for r in rows.values()})
    return {'selected_formula_pairs': len(selected), 'selected_spectra': len(wanted),
            'indices_sha256': sha256(','.join(map(str, wanted)).encode()).hexdigest(),
            'filtered_morgan_tanimoto_quartiles': np.quantile(tanimotos, [0.25, 0.5, 0.75]).tolist() if tanimotos else [],
            'filtered_morgan_tanimoto_ge_0_6': sum(value >= 0.6 for value in tanimotos),
            'excluded': dict(discarded), 'all_with_four_fragments': summarize(valid),
            'without_near_duplicates': summarize(filtered)}


def main(path):
    parquet = pq.ParquetFile(path)
    random_pairs, hard_pairs, eligible = proposed_pairs(parquet)
    wanted = sorted({r for _, pair, _ in random_pairs + hard_pairs for rows in pair for r in rows.values()})
    peaks = read_peaks(parquet, wanted)
    return {'source': 'train.parquet', 'row_group': 0, 'metadata_rows': LIMIT,
            'eligible_formula_groups_stored_keys': eligible, 'max_hash_pairs': PAIR_LIMIT,
            'total_distinct_spectra_read': len(peaks), 'rdkit': rdBase.rdkitVersion,
            'null_design': '每式固定 20 eV 的结构标签，独立交换 40 与 60 eV 的标签。三项配对符号依次为 s40、s60、s40*s60。',
            'hash_selected': evaluate(random_pairs, peaks), 'hard_selected': evaluate(hard_pairs, peaks),
            'hard_selection': '在同一固定片段内，仅按结构选择每式 Morgan 相似度最大的结构对，最低 0.60。两结构精确质量相差最多 1e-6 Da，六条实测前体的极差不超过均值的 10 ppm。',
            'scope': '比赛训练首个行组前16384行。真实参考谱两选一，只能衡量同仪器已见结构的跨能量鉴别。不同扫描行，源文件未提供独立采集批次号。'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parquet')
    args = parser.parse_args()
    print(json.dumps(main(args.parquet), indent=2, ensure_ascii=False))
