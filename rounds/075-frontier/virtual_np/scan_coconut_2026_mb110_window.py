"""Blind exact COCONUT 2026-lite inventory transfer to the 110 MassBank windows."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import csv
from hashlib import sha256
from itertools import islice
import json
from math import isfinite
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import rdBase

from generate import AUTHOR_PPM, C34_PPM, RDKit_VERSION, file_hash
from rebuild_parents import COCONUT
from scan_coconut_2026_window import (
    ARCHIVE, BATCH_SIZE, RECORD_SCHEMA, WINDOW_SCHEMA, aggregate, check_counts, csv_rows,
    scan_batch, validate_stage_receipt, worker_init,
)
from scan_rebuilt_parent_window import read_rebuilt


BASE = 'jiyujia/enveda-075b-massbank-pipeline/1'
REBUILT = 'jiyujia/enveda-075-parent-v2-merge/1'
STAGE = 'jiyujia/enveda-075-coconut-2026-source/1'
PC_SHA = '58092dc1f9db483f2681a7f90efb81d8f18fb6254c7982d90ca042d80c6d8f6c'
WINDOW = 'coconut_2026_mb110_window.parquet'
OLD_WINDOW = 'coconut_rebuilt_mb110_window.parquet'
RECORDS = 'coconut_2026_mb110_window_sources.parquet'
REPORT = 'coconut_2026_mb110_window_receipt.json'
NEW_SCHEMA = pa.schema([
    field for field in WINDOW_SCHEMA if field.name not in {'legacy_same_window', 'legacy_same_author_window'}
] + [
    ('rebuilt_source_mask_in_broad', pa.int8()), ('rebuilt_source_mask_in_author', pa.int8()),
])
OLD_SCHEMA = pa.schema([
    ('molecule_id', pa.string()), ('smiles', pa.string()), ('score_key', pa.string()), ('heavy_atoms', pa.string()),
    ('formula', pa.string()), ('mass', pa.float64()), ('mass_error_ppm', pa.float64()), ('formal_charge', pa.int16()),
    ('in_author_window', pa.bool_()), ('source_mask', pa.int8()),
])


def frozen_targets(pc_lists):
    if file_hash(pc_lists) != PC_SHA:
        raise ValueError('Use the exact historical MassBank C34 output digest')
    records = json.loads(pc_lists.read_bytes())
    if len(records) != 110 or any(
        not isinstance(row, dict) or type(row.get('target')) not in (int, float)
        or not isfinite(row['target']) or row['target'] <= 0
        or not isinstance(row.get('pc'), list) or len(row['pc']) != 25
        or any(not isinstance(text, str) or not text for text in row['pc'])
        for row in records.values()
    ):
        raise ValueError('Expected the 110 sealed C34 targets and historical candidate widths')
    targets = {mid: row['target'] for mid, row in records.items()}
    encoded = json.dumps(targets, sort_keys=True, separators=(',', ':')).encode()
    return targets, sha256(encoded).hexdigest()


def scan_archive(archive, stage, targets, workers, output_path):
    """Reuse the 60-query chemistry, CSV accounting and representative policy."""
    csv.field_size_limit(16 * 1024 * 1024)
    counts, nonempty = Counter(), Counter()
    picks = {mid: {} for mid in targets}
    records, blocks = 0, 0
    cols = stage['columns']
    with (closing(csv_rows(archive, stage, counts, nonempty)) as rows,
          ProcessPoolExecutor(max_workers=workers, initializer=worker_init, initargs=(targets,),
                              mp_context=get_context('spawn')) as executor,
          pq.ParquetWriter(output_path, RECORD_SCHEMA, compression='zstd') as writer):
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
                                       mass=mass, query_ids=[mid for mid, _, _ in windows],
                                       author_window_query_ids=[mid for mid, _, narrow in windows if narrow],
                                       original_fields_json=json.dumps(metadata, ensure_ascii=False, separators=(',', ':'))))
                if output:
                    writer.write_table(pa.Table.from_pylist(output, schema=RECORD_SCHEMA))
                records += len(output)
                blocks += 1
                if blocks % 32 == 0:
                    print('COCONUT_2026_MB110_PROGRESS', counts['exact_mass_scanned'],
                          counts['valid_window_source_rows'], flush=True)
    check_counts(counts, picks, records)
    return picks, counts, dict(nonempty)


def rebuilt_window(mid, target, table, masses):
    tolerance = target * C34_PPM * 1e-6
    start = int(np.searchsorted(masses, target - tolerance, side='left'))
    end = int(np.searchsorted(masses, target + tolerance, side='right'))
    rows = {}
    for item in table.slice(start, end - start).to_pylist():
        key = item['key'], item['heavy_atoms']
        if key in rows:
            raise ValueError(f'Duplicate independently normalized 2024 parent in {mid}')
        mass = item['mass']
        error = (target - mass) / mass * 1e6
        rows[key] = dict(molecule_id=mid, smiles=item['smiles'], score_key=key[0], heavy_atoms=key[1],
                         formula=item['formula'], mass=mass, mass_error_ppm=error,
                         formal_charge=item['formal_charge'], in_author_window=AUTHOR_PPM[0] <= error <= AUTHOR_PPM[1],
                         source_mask=item['src'])
    return rows


def compare_window(picks, old_rows):
    new_broad = set(picks)
    new_author = {key for key, (_, row, _) in picks.items() if row['in_author_window']}
    all_old = set(old_rows)
    old_author = {key for key, row in old_rows.items() if row['in_author_window']}
    coconut_old = {key for key, row in old_rows.items() if row['source_mask'] & COCONUT}
    counts = {}
    for suffix, new, rebuilt, coconut in (
        ('broad', new_broad, all_old, coconut_old),
        ('author', new_author, old_author, coconut_old & old_author),
    ):
        for name, old in [('rebuilt', rebuilt), ('coconut_2024', coconut)]:
            counts[f'{name}_{suffix}'] = len(old)
            counts[f'shared_with_{name}_{suffix}'] = len(new & old)
            counts[f'added_to_{name}_{suffix}'] = len(new - old)
            counts[f'union_with_{name}_{suffix}'] = len(new | old)
        counts[f'coconut_2026_{suffix}'] = len(new)
    rows = []
    for key, (_, item, occurrences) in sorted(picks.items()):
        old = old_rows.get(key)
        source = old['source_mask'] if old is not None else 0
        rows.append(dict(item, source_records_in_window=occurrences, rebuilt_source_mask_in_broad=source,
                         rebuilt_source_mask_in_author=source if key in old_author else 0))
    return rows, counts


def run(stage_receipt, stage_sha, rebuilt_receipt, rebuilt_sha, pc_lists, out, workers):
    start = time.monotonic()
    if rdBase.rdkitVersion != RDKit_VERSION or not 1 <= workers <= 8:
        raise ValueError(f'Expected official RDKit {RDKit_VERSION} and one to eight workers')
    if file_hash(stage_receipt) != stage_sha or file_hash(rebuilt_receipt) != rebuilt_sha:
        raise ValueError('Mounted stage or merge receipt changed after packaging')
    stage = json.loads(stage_receipt.read_bytes())
    validate_stage_receipt(stage)
    archive = stage_receipt.with_name(ARCHIVE)
    if archive.stat().st_size != stage['bytes'] or file_hash(archive) != stage['sha256']:
        raise ValueError('Mounted COCONUT archive differs from stage')
    targets, targets_sha = frozen_targets(pc_lists)
    old_table, old_masses, merge = read_rebuilt(rebuilt_receipt)
    out.mkdir(exist_ok=True, parents=True)
    outputs = (WINDOW, OLD_WINDOW, RECORDS)
    if any((out / name).exists() for name in (*outputs, REPORT)):
        raise FileExistsError('Existing MB110 COCONUT window output')
    temporary = {name: out / (name + '.part') for name in (*outputs, REPORT)}
    try:
        picks, counts, present = scan_archive(archive, stage, targets, workers, temporary[RECORDS])
        cases, totals = {}, Counter()
        with (pq.ParquetWriter(temporary[WINDOW], NEW_SCHEMA, compression='zstd') as new_writer,
              pq.ParquetWriter(temporary[OLD_WINDOW], OLD_SCHEMA, compression='zstd') as old_writer):
            for mid, target in sorted(targets.items()):
                old = rebuilt_window(mid, target, old_table, old_masses)
                new, comparison = compare_window(picks[mid], old)
                new_writer.write_table(pa.Table.from_pylist(new, schema=NEW_SCHEMA))
                old_writer.write_table(pa.Table.from_pylist(list(old.values()), schema=OLD_SCHEMA))
                cases[mid] = dict(target_mass=target, **comparison)
                totals.update(comparison)
        report = dict(status='complete', scope='massbank110-coconut-2026-exact-window-inventory', query_count=len(cases),
                      archived_c34=BASE, archived_pc_lists_sha256=PC_SHA, exact_targets_sha256=targets_sha,
                      stage=STAGE, stage_receipt_sha256=stage_sha, archive_sha256=stage['sha256'],
                      old_inventory=REBUILT, rebuilt_receipt_sha256=rebuilt_sha,
                      rebuilt_parent_sha256=merge['output_sha256'], rebuilt_parent_identities=merge['identities'],
                      rebuilt_source_bits=merge['source_bits'],
                      rdkit=RDKit_VERSION, broad_window_ppm=C34_PPM, author_window_ppm=AUTHOR_PPM,
                      mass_method='Calculate exact mass of every parsed largest fragment before evaluating windows',
                      identity='Canonical tautomer InChIKey14 with heavy-atom composition',
                      scope_note='Per-query window additions. No evaluation truth, global inventory claim or physical scores.',
                      columns=stage['columns'], provenance_columns=stage['provenance_columns'],
                      field_nonempty_counts={name: present.get(name, 0) for name in stage['columns']},
                      source_counts=dict(sorted(counts.items())),
                      source_query_duplicates=counts['window_source_query_occurrences'] - totals['coconut_2026_broad'],
                      cases=cases, totals=dict(totals), outputs={name: file_hash(temporary[name]) for name in outputs},
                      workers=workers, seconds=time.monotonic() - start)
        temporary[REPORT].write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        for name in (*outputs, REPORT):
            temporary[name].replace(out / name)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    print('COCONUT_2026_MB110_WINDOW', len(cases), dict(totals), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('stage-receipt', 'rebuilt-receipt', 'pc-lists', 'out'):
        cli.add_argument('--' + name, type=Path, required=True)
    for name in ('stage-sha', 'rebuilt-sha'):
        cli.add_argument('--' + name, required=True)
    cli.add_argument('--workers', type=int, default=4)
    run(**vars(cli.parse_args()))
