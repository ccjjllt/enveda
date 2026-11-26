"""Scan the staged November 2026 COCONUT release against sealed C34 mass windows."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import csv
from io import TextIOWrapper
from itertools import islice
import json
from math import isfinite
from multiprocessing import get_context
from pathlib import Path
import re
import time
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from generate import AUTHOR_PPM, C34_PPM, RDKit_VERSION, file_hash, heavy_atoms
from scan_rebuilt_parent_window import ARCHIVED_C34, FROZEN_TARGETS_SHA, read_legacy


ARCHIVE = 'coconut_csv_lite-11-2026.zip'
STAGE_RECEIPT = 'coconut_2026_source_receipt.json'
WINDOW = 'coconut_2026_window.parquet'
RECORDS = 'coconut_2026_window_sources.parquet'
REPORT = 'coconut_2026_window_receipt.json'
BATCH_SIZE = 1024
WINDOW_SCHEMA = pa.schema([
    ('molecule_id', pa.string()), ('smiles', pa.string()), ('score_key', pa.string()), ('heavy_atoms', pa.string()),
    ('formula', pa.string()), ('mass', pa.float64()), ('mass_error_ppm', pa.float64()), ('formal_charge', pa.int16()),
    ('in_author_window', pa.bool_()), ('legacy_same_window', pa.bool_()), ('legacy_same_author_window', pa.bool_()),
    ('source_row', pa.int64()), ('source_records_in_window', pa.int32()),
])
RECORD_SCHEMA = pa.schema([
    ('source_row', pa.int64()), ('source_id', pa.string()), ('source_identifier', pa.string()),
    ('original_smiles', pa.string()), ('score_key', pa.string()), ('heavy_atoms', pa.string()),
    ('mass', pa.float64()), ('query_ids', pa.list_(pa.string())), ('author_window_query_ids', pa.list_(pa.string())),
    ('original_fields_json', pa.string()),
])


def validate_stage_receipt(receipt):
    columns = receipt['columns']
    provenance = receipt['provenance_columns']
    if (receipt['version'] != '2026-11' or receipt['filename'] != ARCHIVE or receipt['license'] != 'CC0-1.0'
            or receipt['zip_crc_checked'] is not True or receipt['bytes'] != 199_108_039
            or receipt['expected_object_bytes'] != receipt['bytes']
            or receipt['expected_multipart_etag'] != 'ff30cff5276dfe8880fb3c8ce539b7ef-24'
            or receipt['csv_member'] != 'coconut_csv_lite-11-2026.csv'
            or not re.fullmatch(r'[0-9a-f]{64}', receipt['sha256'])
            or not re.fullmatch(r'[0-9a-f]{8}', receipt['csv_crc32'])
            or not 0 < receipt['csv_bytes'] <= 4 * 1024 ** 3
            or not isinstance(columns, list) or len(set(columns)) != len(columns)
            or not all(isinstance(field, str) and field.strip() for field in columns)
            or not {'id', 'identifier', 'canonical_smiles', 'standard_inchi_key'} <= set(columns)
            or 'canonical_smiles' not in receipt['structure_columns']
            or not set(provenance) <= set(columns)):
        raise ValueError('Invalid official COCONUT 2026 stage receipt')


def worker_init(targets):
    global TAUTOMER, FRAGMENTS, TARGETS, LOW, HIGH
    RDLogger.DisableLog('rdApp.*')
    TAUTOMER = rdMolStandardize.TautomerEnumerator()
    FRAGMENTS = rdMolStandardize.LargestFragmentChooser()
    TARGETS = sorted(targets.items(), key=lambda item: (item[1], item[0]))
    LOW = [mass - mass * C34_PPM * 1e-6 for _, mass in TARGETS]
    HIGH = [mass + mass * C34_PPM * 1e-6 for _, mass in TARGETS]


def scan_batch(texts):
    counts = Counter()
    matched = []
    for index, text in enumerate(texts):
        try:
            molecule = Chem.MolFromSmiles(text)
        except (RuntimeError, ValueError):
            molecule = None
        if molecule is None:
            counts['rdkit_parse_failures'] += 1
            continue
        counts['parsed_structures'] += 1
        try:
            if len(Chem.GetMolFrags(molecule)) > 1:
                molecule = FRAGMENTS.choose(molecule)
                counts['largest_fragment_reduced'] += 1
        except (RuntimeError, ValueError):
            counts['fragment_selection_failures'] += 1
            continue
        if not molecule.GetNumHeavyAtoms():
            counts['no_heavy_atoms'] += 1
            continue
        try:
            Chem.RemoveStereochemistry(molecule)
            mass = float(rdMolDescriptors.CalcExactMolWt(molecule))
        except (RuntimeError, ValueError):
            mass = float('nan')
        if not isfinite(mass) or mass <= 0:
            counts['mass_failures'] += 1
            continue
        counts['exact_mass_scanned'] += 1
        first, stop = bisect_left(HIGH, mass), bisect_right(LOW, mass)
        if first == stop:
            counts['outside_broad_window_rows'] += 1
            continue
        counts['inside_broad_window_rows'] += 1
        try:
            key = Chem.MolToInchiKey(TAUTOMER.Canonicalize(molecule))[:14]
            heavy = heavy_atoms(molecule)
            smiles = Chem.MolToSmiles(molecule)
            formula = rdMolDescriptors.CalcMolFormula(molecule)
            charge = Chem.GetFormalCharge(molecule)
        except (RuntimeError, ValueError):
            key = ''
        if len(key) != 14 or not heavy or not smiles or not formula:
            counts['window_normalization_failures'] += 1
            continue
        windows = []
        for mid, target in TARGETS[first:stop]:
            error = (target - mass) / mass * 1e6
            windows.append((mid, error, AUTHOR_PPM[0] <= error <= AUTHOR_PPM[1]))
        counts['valid_window_source_rows'] += 1
        counts['window_source_query_occurrences'] += len(windows)
        matched.append((index, key, heavy, smiles, formula, charge, mass, windows))
    return counts, matched


def csv_rows(archive, receipt, counts, present):
    with ZipFile(archive) as source:
        info = source.getinfo(receipt['csv_member'])
        if info.file_size != receipt['csv_bytes'] or f'{info.CRC:08x}' != receipt['csv_crc32']:
            raise ValueError('Mounted COCONUT CSV metadata differs from the sealed release')
        with source.open(info) as entry, TextIOWrapper(entry, encoding='utf-8-sig', newline='') as stream:
            rows = csv.reader(stream, strict=True)
            columns = next(rows)
            if columns != receipt['columns']:
                raise ValueError('Mounted COCONUT CSV header differs from the sealed release')
            smiles_column = columns.index('canonical_smiles')
            for ordinal, fields in enumerate(rows, 1):
                counts['source_rows'] += 1
                if len(fields) != len(columns):
                    counts['malformed_width_rows'] += 1
                    continue
                present.update(name for name, value in zip(columns, fields, strict=True) if value.strip())
                text = fields[smiles_column].strip()
                if not text:
                    counts['missing_canonical_smiles'] += 1
                    continue
                counts['submitted_smiles_rows'] += 1
                yield ordinal, fields, text


def aggregate(picks, source_row, description):
    _, key, heavy, smiles, formula, charge, mass, windows = description
    for mid, error, narrow in windows:
        identity = key, heavy
        rank = (not narrow, smiles, mass, source_row)
        previous = picks[mid].get(identity)
        row = dict(molecule_id=mid, smiles=smiles, score_key=key, heavy_atoms=heavy, formula=formula,
                   mass=mass, mass_error_ppm=error, formal_charge=charge, in_author_window=narrow,
                   source_row=source_row)
        if previous is None:
            picks[mid][identity] = rank, row, 1
        elif rank < previous[0]:
            picks[mid][identity] = rank, row, previous[2] + 1
        else:
            picks[mid][identity] = previous[0], previous[1], previous[2] + 1


def check_counts(counts, selected, sources):
    partitions = {
        'source_rows': ('malformed_width_rows', 'missing_canonical_smiles', 'submitted_smiles_rows'),
        'submitted_smiles_rows': ('rdkit_parse_failures', 'parsed_structures'),
        'parsed_structures': ('fragment_selection_failures', 'no_heavy_atoms', 'mass_failures', 'exact_mass_scanned'),
        'exact_mass_scanned': ('inside_broad_window_rows', 'outside_broad_window_rows'),
        'inside_broad_window_rows': ('window_normalization_failures', 'valid_window_source_rows'),
    }
    if any(counts[total] != sum(counts[part] for part in parts) for total, parts in partitions.items()):
        raise ValueError('COCONUT scan row totals failed to reconcile')
    if (counts['valid_window_source_rows'] != sources
            or counts['window_source_query_occurrences'] != sum(item[2] for group in selected.values() for item in group.values())):
        raise ValueError('Matched source or deduplication counts failed to reconcile')
    for name in {part for parts in partitions.values() for part in parts} | set(partitions) | {
            'largest_fragment_reduced', 'window_source_query_occurrences'}:
        counts.setdefault(name, 0)


def run(stage_receipt, stage_receipt_sha, legacy, out, archived_pc_sha, archived_targets_sha, workers):
    start = time.monotonic()
    if rdBase.rdkitVersion != RDKit_VERSION or workers < 1:
        raise ValueError(f'Expected contest RDKit {RDKit_VERSION} and at least one worker')
    if file_hash(stage_receipt) != stage_receipt_sha:
        raise ValueError('Mounted COCONUT source receipt differs from the packaged receipt')
    stage = json.loads(stage_receipt.read_bytes())
    validate_stage_receipt(stage)
    archive = stage_receipt.with_name(ARCHIVE)
    if archive.stat().st_size != stage['bytes'] or file_hash(archive) != stage['sha256']:
        raise ValueError('Mounted COCONUT ZIP differs from the staged release')
    targets, old_broad, old_narrow = read_legacy(legacy, archived_pc_sha, archived_targets_sha)
    out.mkdir(exist_ok=True, parents=True)
    if any((out / name).exists() for name in (WINDOW, RECORDS, REPORT)):
        raise FileExistsError('Existing COCONUT 2026 window output')
    window_tmp, records_tmp = (out / (name + '.part') for name in (WINDOW, RECORDS))
    csv.field_size_limit(16 * 1024 * 1024)
    counts, present = Counter(), Counter()
    picks = {mid: {} for mid in targets}
    records, blocks = 0, 0
    cols = stage['columns']
    try:
        with (closing(csv_rows(archive, stage, counts, present)) as rows,
              ProcessPoolExecutor(max_workers=workers, initializer=worker_init, initargs=(targets,),
                                  mp_context=get_context('spawn')) as executor,
              pq.ParquetWriter(records_tmp, RECORD_SCHEMA, compression='zstd') as writer):
            while batches := [batch for _ in range(2 * workers) if (batch := list(islice(rows, BATCH_SIZE)))]:
                work = [[text for _, _, text in batch] for batch in batches]
                for batch, (batch_counts, matched) in zip(batches, executor.map(scan_batch, work), strict=True):
                    counts.update(batch_counts)
                    output = []
                    for result in matched:
                        index, key, heavy, _, _, _, mass, windows = result
                        ordinal, fields, _ = batch[index]
                        aggregate(picks, ordinal, result)
                        metadata = dict(zip(cols, fields, strict=True))
                        output.append(dict(source_row=ordinal, source_id=metadata['id'],
                                           source_identifier=metadata['identifier'],
                                           original_smiles=metadata['canonical_smiles'], score_key=key, heavy_atoms=heavy,
                                           mass=mass, query_ids=[item[0] for item in windows],
                                           author_window_query_ids=[mid for mid, _, narrower in windows if narrower],
                                           original_fields_json=json.dumps(metadata, ensure_ascii=False, separators=(',', ':'))))
                    if output:
                        writer.write_table(pa.Table.from_pylist(output, schema=RECORD_SCHEMA))
                    records += len(output)
                    blocks += 1
                    if blocks % 32 == 0:
                        print('COCONUT_2026_WINDOW_PROGRESS', counts['exact_mass_scanned'],
                              counts['valid_window_source_rows'], flush=True)
        check_counts(counts, picks, records)
        cases, totals = {}, Counter()
        with pq.ParquetWriter(window_tmp, WINDOW_SCHEMA, compression='zstd') as writer:
            for mid, target in sorted(targets.items()):
                broad = set(picks[mid])
                narrow = {key for key, (_, row, _) in picks[mid].items() if row['in_author_window']}
                output = [dict(row, legacy_same_window=key in old_broad[mid],
                               legacy_same_author_window=key in old_narrow[mid],
                               source_records_in_window=occurrences)
                          for key, (_, row, occurrences) in sorted(picks[mid].items())]
                writer.write_table(pa.Table.from_pylist(output, schema=WINDOW_SCHEMA))
                stats = dict(coconut_broad=len(broad), legacy_broad=len(old_broad[mid]),
                             shared_broad=len(broad & old_broad[mid]), added_to_legacy_broad=len(broad - old_broad[mid]),
                             coconut_author=len(narrow), legacy_author=len(old_narrow[mid]),
                             shared_author=len(narrow & old_narrow[mid]), added_to_legacy_author=len(narrow - old_narrow[mid]))
                totals.update(stats)
                cases[mid] = dict(target_mass=target, **stats)
        checksums = {WINDOW: file_hash(window_tmp), RECORDS: file_hash(records_tmp)}
        window_tmp.replace(out / WINDOW)
        records_tmp.replace(out / RECORDS)
    finally:
        window_tmp.unlink(missing_ok=True)
        records_tmp.unlink(missing_ok=True)
    report = dict(status='complete', scope='coconut-2026-exact-direct-structure-window', query_count=len(cases),
                  rdkit=RDKit_VERSION, archived_c34=ARCHIVED_C34, archived_pc_lists_sha256=archived_pc_sha,
                  exact_targets_sha256=FROZEN_TARGETS_SHA, broad_window_ppm=C34_PPM, author_window_ppm=AUTHOR_PPM,
                  stage_receipt_sha256=stage_receipt_sha, archive_sha256=stage['sha256'],
                  legacy_window_receipt_sha256=file_hash(legacy / 'blind_virtual_np_receipt.json'),
                  legacy_window_sha256=file_hash(legacy / 'known_main.parquet'),
                  csv_member=stage['csv_member'], columns=cols, provenance_columns=stage['provenance_columns'],
                  field_nonempty_counts={name: present[name] for name in cols}, source_counts=dict(sorted(counts.items())),
                  source_query_duplicates=counts['window_source_query_occurrences'] - totals['coconut_broad'],
                  totals=dict(totals), cases=cases, outputs=checksums, workers=workers,
                  mass_method='CalcExactMolWt on all parsable largest fragments, preserving formal charge; no archive-mass prefilter',
                  identity='Canonical tautomer InChIKey14 with heavy-atom composition; enumerate only broad-window structures',
                  dedup='Choose an author-window representative if any, then canonical SMILES, exact mass and CSV row',
                  source_scope='Release records with structures only; overlap is evaluated within the frozen mass windows',
                  seconds=time.monotonic() - start)
    (out / REPORT).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('COCONUT_2026_WINDOW', len(cases), dict(totals), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('stage-receipt', 'legacy', 'out'):
        cli.add_argument('--' + name, type=Path, required=True)
    for name in ('stage-receipt-sha', 'archived-pc-sha', 'archived-targets-sha'):
        cli.add_argument('--' + name, required=True)
    cli.add_argument('--workers', type=int, default=4)
    run(**vars(cli.parse_args()))
