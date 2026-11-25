"""Load sealed training-only dual-polarity pairs and their original spectra."""

from bisect import bisect_right
from collections import defaultdict
from hashlib import sha256
from itertools import accumulate
import json
import math
from pathlib import Path

import pyarrow.parquet as pq


_ADDUCTS = ('[M+H]+', '[M-H]-')
_ENERGIES = (20, 40, 60)
_FOLDS = 5
_CONDITIONS = {f'{adduct}/{energy}eV': (adduct, energy) for adduct in _ADDUCTS for energy in _ENERGIES}
_COLUMNS = (
    'ingest_lib', 'instrument_type', 'adduct', 'collision_energy_orig', 'collision_energy_orig_units', 'collision_energy_ev',
    'precursor_mz', 'num_peaks', 'ms2_mzs', 'ms2_normalized_intensities',
)
_OPTIONAL_COLUMNS = ('spectrum_id', 'ionization_mode')


def _selection_key(pair):
    left, right = pair['left'], pair['right']
    structures = '/'.join(sorted((left['key'], right['key'])))
    value = f"dual-training-fold-selection-v1/{pair['fold']}/{left['formula']}/{structures}/{pair['match']}"
    return len(pair['matched_dual_energies']) < 2, sha256(value.encode()).digest()


def _shared_energies(left, right):
    return [energy for energy in _ENERGIES
            if all(f'{adduct}/{energy}eV' in left and f'{adduct}/{energy}eV' in right
                   for adduct in _ADDUCTS)]


def _select_pairs(raw, receipt, per_fold, minimum_similarity):
    if sha256(raw).hexdigest() != receipt['pair_export_sha256']:
        raise ValueError('The dual-energy pair export differs from its receipt')
    pairs = [json.loads(line) for line in raw.splitlines()]
    if len(pairs) != receipt['pair_export_count']:
        raise ValueError('The number of exported pairs differs from its receipt')
    folds = defaultdict(list)
    for pair in pairs:
        fold = pair['fold']
        left, right = pair['left'], pair['right']
        left_rows, right_rows = left['train_row_0based'], right['train_row_0based']
        if (type(fold) is not int or not 0 <= fold < _FOLDS
                or left['fold'] != fold or right['fold'] != fold
                or left['formula'] != right['formula'] or left['key'] == right['key']
                or set(left_rows) - _CONDITIONS.keys() or set(right_rows) - _CONDITIONS.keys()):
            raise ValueError('Invalid same-formula, within-fold pair metadata')
        shared = _shared_energies(left_rows, right_rows)
        if 40 not in shared or pair['matched_dual_energies'] != shared or pair['match'] != f'dual_{len(shared)}_energy':
            raise ValueError('The exported paired energy conditions disagree with their scan references')
        if pair['morgan_tanimoto'] >= minimum_similarity:
            folds[fold].append(pair)
    return [pair for fold in sorted(folds) for pair in sorted(folds[fold], key=_selection_key)[:per_fold]]


def _expected_rows(pairs, row_count):
    expected = {}
    for pair in pairs:
        for side in ('left', 'right'):
            for condition, row in pair[side]['train_row_0based'].items():
                if type(row) is not int or not 0 <= row < row_count:
                    raise ValueError(f'Invalid global training row for {side} {condition}')
                requirement = _CONDITIONS[condition]
                if expected.setdefault(row, requirement) != requirement:
                    raise ValueError(f'Training row {row} was assigned two different scan conditions')
    return expected


def _check_scan(position, scan, expected):
    adduct, energy = expected
    try:
        measured_ev = abs(float(scan['collision_energy_orig']))
    except (TypeError, ValueError):
        measured_ev = None
    if (scan['adduct'] != adduct
            or scan['collision_energy_orig_units'] != 'eV'
            or measured_ev != energy or scan['collision_energy_ev'] != [float(energy)]
            or (scan['ingest_lib'], scan['instrument_type']) != ('enveda-180', 'timsTOF')):
        raise ValueError(f'Train row {position} does not contain the specified {adduct}/{energy}eV measured scan')
    mz, intensity = scan['ms2_mzs'], scan['ms2_normalized_intensities']
    precursor = scan['precursor_mz']
    if not mz or not intensity or len(mz) != len(intensity) or precursor is None or not math.isfinite(precursor) or precursor <= 0:
        raise ValueError(f'Train row {position} has malformed m/z or intensity data')


def load_pairs(pair_path, receipt_path, train_path, per_fold=400, *, minimum_similarity=0.0):
    """Return selected JSONL pair dictionaries and raw Parquet scan dictionaries by global row.

    Each selected pair keeps its original scaffold hashes and both condition-to-row
    mappings. The returned scan table includes all references in those mappings.
    """
    if type(per_fold) is not int or per_fold < 1:
        raise ValueError('per_fold must be a positive integer')
    pair_path, receipt_path, train_path = map(Path, (pair_path, receipt_path, train_path))
    receipt = json.loads(receipt_path.read_bytes())
    if train_path.stat().st_size != receipt['training_parquet_bytes']:
        raise ValueError('The official training file size differs from the sealed pair input')
    selected = _select_pairs(pair_path.read_bytes(), receipt, per_fold, minimum_similarity)
    with pq.ParquetFile(train_path) as train:
        row_count = train.metadata.num_rows
        if row_count != receipt['metadata_counts']['rows_read']:
            raise ValueError('The official training row count differs from the sealed pair input')
        names = set(train.schema_arrow.names)
        if missing := set(_COLUMNS) - names:
            raise ValueError(f'The training Parquet is missing required columns: {sorted(missing)}')
        columns = [*_COLUMNS, *(name for name in _OPTIONAL_COLUMNS if name in names)]
        expected = _expected_rows(selected, row_count)
        offsets = [0, *accumulate(train.metadata.row_group(i).num_rows for i in range(train.num_row_groups))]
        groups = defaultdict(list)
        for row in sorted(expected):
            groups[bisect_right(offsets, row) - 1].append(row)
        scans = {}
        for group, positions in groups.items():
            subset = train.read_row_group(group, columns=columns).take(
                [row - offsets[group] for row in positions])
            for position, scan in zip(positions, subset.to_pylist(), strict=True):
                _check_scan(position, scan, expected[position])
                scans[position] = scan
    return selected, scans
