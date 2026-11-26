"""Scan staged NP-MRD structures in the sealed MassBank 110 windows."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import csv
import gzip
from itertools import islice
import json
from math import isfinite
from multiprocessing import get_context
from pathlib import Path
import re
import time

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import rdBase

from generate import AUTHOR_PPM, C34_PPM, RDKit_VERSION, file_hash
from scan_coconut_2026_mb110_window import (
    BASE, NEW_SCHEMA, OLD_SCHEMA, OLD_WINDOW, PC_SHA, RECORDS as COCONUT_RECORDS,
    WINDOW as COCONUT_WINDOW, frozen_targets,
)
from scan_coconut_2026_window import BATCH_SIZE, aggregate, check_counts, scan_batch, worker_init
from stage_npmrd_2025 import ARCHIVES, COLUMNS, FINGERPRINT_BYTES, LAST_MODIFIED, LICENSE_URL, RELEASE_DATE, RELEASE_URL, UPSTREAM


STAGE = 'jiyujia/enveda-075-npmrd-2025-source/1'
COCONUT_SCAN = 'jiyujia/enveda-075-coconut-2026-mb110-window/1'
SCOPE = 'massbank110-npmrd-2025-exact-window-inventory'
WINDOW = 'npmrd_mb110_window.parquet'
REPORT = 'npmrd_mb110_window_receipt.json'
SCHEMA = pa.schema(list(NEW_SCHEMA) + [
    ('npmrd_id', pa.string()), ('coconut_2026_in_broad', pa.bool_()), ('coconut_2026_in_author', pa.bool_()),
    ('residual_in_broad', pa.bool_()), ('residual_in_author', pa.bool_()),
])


def read_stage(receipt, expected_sha):
    if file_hash(receipt) != expected_sha:
        raise ValueError('Mounted NP-MRD source receipt differs from the packaged receipt')
    report = json.loads(receipt.read_bytes())
    expected = dict(release_date=RELEASE_DATE, release_url=RELEASE_URL, license='CC-BY-NC-4.0',
                    license_url=LICENSE_URL, expected_last_modified=LAST_MODIFIED,
                    fingerprint_bytes=FINGERPRINT_BYTES, columns=list(COLUMNS), gzip_integrity_checked=True,
                    total_bytes=sum(spec.size for spec in ARCHIVES))
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError('Unexpected NP-MRD source receipt')
    paths = []
    for spec, item in zip(ARCHIVES, report['artifacts'], strict=True):
        expected = dict(filename=spec.filename, upstream=f'{UPSTREAM}/{spec.filename}', bytes=spec.size,
                        expected_etag=spec.etag, head_sha256=spec.head_sha256, tail_sha256=spec.tail_sha256)
        if (any(item.get(key) != value for key, value in expected.items())
                or not re.fullmatch(r'[0-9a-f]{64}', item.get('sha256', ''))):
            raise ValueError(f'Unexpected staged metadata for {spec.filename}')
        path = receipt.with_name(spec.filename)
        if path.stat().st_size != spec.size or file_hash(path) != item['sha256']:
            raise ValueError(f'Mounted bytes differ from the staged NP-MRD archive: {spec.filename}')
        paths.append(path)
    return paths, report


def structure_rows(paths, counts):
    for path in paths:
        with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as stream:
            rows = csv.reader(stream, strict=True)
            if tuple(next(rows)) != COLUMNS:
                raise ValueError(f'Unexpected NP-MRD CSV header in {path.name}')
            for fields in rows:
                counts['source_rows'] += 1
                if len(fields) != len(COLUMNS):
                    counts['malformed_width_rows'] += 1
                    continue
                _, accession, text = (field.strip() for field in fields)
                if not text:
                    counts['missing_canonical_smiles'] += 1
                    continue
                if re.fullmatch(r'NP[0-9]{7}', accession) is None:
                    raise ValueError(f'Invalid NP-MRD accession in {path.name}')
                counts['submitted_smiles_rows'] += 1
                yield counts['source_rows'], accession, text


def scan_archives(paths, targets, workers):
    csv.field_size_limit(16 * 1024 * 1024)
    counts = Counter()
    picks = {mid: {} for mid in targets}
    accessions = {}
    blocks = 0
    with (closing(structure_rows(paths, counts)) as rows,
          ProcessPoolExecutor(max_workers=workers, initializer=worker_init, initargs=(targets,),
                              mp_context=get_context('spawn')) as executor):
        while batches := [batch for _ in range(2 * workers) if (batch := list(islice(rows, BATCH_SIZE)))]:
            work = [[text for _, _, text in batch] for batch in batches]
            for batch, (batch_counts, matched) in zip(batches, executor.map(scan_batch, work), strict=True):
                counts.update(batch_counts)
                for result in matched:
                    ordinal, accession, _ = batch[result[0]]
                    accessions[ordinal] = accession
                    aggregate(picks, ordinal, result)
                blocks += 1
                if blocks % 32 == 0:
                    print('NP_MRD_MB110_PROGRESS', counts['exact_mass_scanned'], len(accessions), flush=True)
    check_counts(counts, picks, len(accessions))
    return picks, accessions, counts


def load_comparator(path, expected_sha, schema, targets):
    if file_hash(path) != expected_sha:
        raise ValueError(f'Archived comparator checksum mismatch for {path.name}')
    handle = pq.ParquetFile(path)
    if not handle.schema_arrow.equals(schema, check_metadata=False):
        raise ValueError(f'Archived comparator schema mismatch for {path.name}')
    cases = {mid: {} for mid in targets}
    columns = ['molecule_id', 'score_key', 'heavy_atoms', 'mass', 'mass_error_ppm', 'in_author_window']
    if 'source_mask' in schema.names:
        columns.append('source_mask')
    for batch in handle.iter_batches(columns=columns):
        if any(column.null_count for column in batch.columns):
            raise ValueError(f'Unexpected missing archived comparator field in {path.name}')
        for row in batch.to_pylist():
            mid, mass = row['molecule_id'], row['mass']
            key = row['score_key'], row['heavy_atoms']
            if mid not in targets or key in cases[mid] or not isfinite(mass) or mass <= 0:
                raise ValueError(f'Invalid archived comparator identity or mass in {path.name}')
            target = targets[mid]
            tolerance = target * C34_PPM * 1e-6
            error = (target - mass) / mass * 1e6
            if (not target - tolerance <= mass <= target + tolerance or row['mass_error_ppm'] != error
                    or row['in_author_window'] != (AUTHOR_PPM[0] <= error <= AUTHOR_PPM[1])
                    or ('source_mask' in row and row['source_mask'] not in (1, 2, 3))):
                raise ValueError(f'Invalid archived comparator window or source in {path.name} for {mid}')
            cases[mid][key] = row
    return cases


def read_comparator(receipt, expected_sha, targets, targets_sha):
    if file_hash(receipt) != expected_sha:
        raise ValueError('Mounted COCONUT MB110 receipt differs from the packaged receipt')
    report = json.loads(receipt.read_bytes())
    expected = dict(status='complete', scope='massbank110-coconut-2026-exact-window-inventory', query_count=len(targets),
                    archived_c34=BASE, archived_pc_lists_sha256=PC_SHA, exact_targets_sha256=targets_sha,
                    rdkit=RDKit_VERSION, broad_window_ppm=C34_PPM, author_window_ppm=list(AUTHOR_PPM))
    if (any(report.get(key) != value for key, value in expected.items())
            or set(report['cases']) != set(targets) or set(report['outputs']) != {COCONUT_WINDOW, OLD_WINDOW, COCONUT_RECORDS}):
        raise ValueError('Unexpected sealed COCONUT MB110 inventory or target roster')
    for mid, target in targets.items():
        if report['cases'][mid]['target_mass'] != target:
            raise ValueError(f'Historical target mass mismatch for {mid}')
    newer = load_comparator(receipt.with_name(COCONUT_WINDOW), report['outputs'][COCONUT_WINDOW], NEW_SCHEMA, targets)
    rebuilt = load_comparator(receipt.with_name(OLD_WINDOW), report['outputs'][OLD_WINDOW], OLD_SCHEMA, targets)
    return newer, rebuilt, report


def compare_window(picks, accessions, newer, rebuilt):
    rows = []
    for key, (_, row, occurrences) in sorted(picks.items()):
        old_mask = rebuilt[key]['source_mask'] if key in rebuilt else 0
        old_narrow = bool(old_mask and rebuilt[key]['in_author_window'])
        new_narrow = key in newer and newer[key]['in_author_window']
        rows.append(dict(row, source_records_in_window=occurrences,
                         rebuilt_source_mask_in_broad=old_mask,
                         rebuilt_source_mask_in_author=old_mask if old_narrow else 0,
                         npmrd_id=accessions[row['source_row']],
                         coconut_2026_in_broad=key in newer, coconut_2026_in_author=new_narrow,
                         residual_in_broad=not old_mask and key not in newer,
                         residual_in_author=row['in_author_window'] and not old_narrow and not new_narrow))

    def narrower(group):
        return {key for key, row in group.items() if row['in_author_window']}

    counts = {}
    for suffix, np_keys, old_keys, new_keys in (
        ('broad', set(picks), set(rebuilt), set(newer)),
        ('author', {key for key, (_, row, _) in picks.items() if row['in_author_window']},
         narrower(rebuilt), narrower(newer)),
    ):
        union = old_keys | new_keys
        counts.update({f'npmrd_{suffix}': len(np_keys), f'rebuilt_{suffix}': len(old_keys),
                       f'coconut_2026_{suffix}': len(new_keys), f'reference_union_{suffix}': len(union),
                       f'shared_with_reference_{suffix}': len(np_keys & union),
                       f'residual_{suffix}': len(np_keys - union)})
    return rows, counts


def run(stage_receipt, stage_sha, coconut_receipt, coconut_sha, pc_lists, out, workers, stage_version):
    started = time.monotonic()
    if rdBase.rdkitVersion != RDKit_VERSION or not 1 <= workers <= 8 or stage_version < 1:
        raise ValueError(f'Expected metric RDKit {RDKit_VERSION}, one to eight workers and a pinned source version')
    paths, stage = read_stage(stage_receipt, stage_sha)
    targets, targets_sha = frozen_targets(pc_lists)
    newer, rebuilt, comparator = read_comparator(coconut_receipt, coconut_sha, targets, targets_sha)
    out.mkdir(exist_ok=True, parents=True)
    if any((out / name).exists() for name in (WINDOW, REPORT)):
        raise FileExistsError('Existing NP-MRD MB110 scan output')
    window_tmp, report_tmp = (out / (name + '.part') for name in (WINDOW, REPORT))
    try:
        picks, accessions, source_counts = scan_archives(paths, targets, workers)
        cases, totals = {}, Counter()
        with pq.ParquetWriter(window_tmp, SCHEMA, compression='zstd') as writer:
            for mid, target in sorted(targets.items()):
                rows, counts = compare_window(picks[mid], accessions, newer[mid], rebuilt[mid])
                for name in ('rebuilt', 'coconut_2026'):
                    for suffix in ('broad', 'author'):
                        field = f'{name}_{suffix}'
                        if counts[field] != comparator['cases'][mid][field]:
                            raise ValueError(f'Comparator count changed for {mid}, {field}')
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                cases[mid] = dict(target_mass=target, **counts)
                totals.update(counts)

        report = dict(status='complete', scope=SCOPE, query_count=len(cases), archived_c34=BASE,
                      archived_pc_lists_sha256=PC_SHA, exact_targets_sha256=targets_sha,
                      stage=f'{STAGE.rsplit("/", 1)[0]}/{stage_version}', stage_receipt_sha256=stage_sha,
                      source_release_date=stage['release_date'], source_license=stage['license'],
                      source_artifacts_sha256={item['filename']: item['sha256'] for item in stage['artifacts']},
                      coconut_window=COCONUT_SCAN, coconut_window_receipt_sha256=coconut_sha,
                      coconut_window_outputs={name: comparator['outputs'][name] for name in (OLD_WINDOW, COCONUT_WINDOW)},
                      rdkit=RDKit_VERSION, broad_window_ppm=C34_PPM, author_window_ppm=AUTHOR_PPM,
                      identity='Canonical tautomer InChIKey14 and heavy-atom composition',
                      broad_mass_method='Exact largest-fragment mass within the target-centred 10 ppm window',
                      residual='Per-query NP-MRD window membership minus the union of rebuilt parent and COCONUT 2026',
                      source_counts=dict(sorted(source_counts.items())),
                      source_query_duplicates=source_counts['window_source_query_occurrences'] - totals['npmrd_broad'],
                      cases=cases, totals=dict(totals), outputs={WINDOW: file_hash(window_tmp)},
                      workers=workers, seconds=time.monotonic() - started)
        report_tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        window_tmp.replace(out / WINDOW)
        report_tmp.replace(out / REPORT)
    finally:
        window_tmp.unlink(missing_ok=True)
        report_tmp.unlink(missing_ok=True)
    print('NP_MRD_MB110_WINDOW', len(cases), dict(totals), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('stage-receipt', 'coconut-receipt', 'pc-lists', 'out'):
        cli.add_argument('--' + name, type=Path, required=True)
    for name in ('stage-sha', 'coconut-sha'):
        cli.add_argument('--' + name, required=True)
    cli.add_argument('--workers', type=int, default=4)
    cli.add_argument('--stage-version', type=int, default=1)
    run(**vars(cli.parse_args()))
