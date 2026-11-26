"""Scan the independent parent inventory against the sealed 60 legacy C34 windows."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import rdBase

from generate import AUTHOR_PPM, C34_PPM, POOL_ROWS, POOL_SHA256, RDKit_VERSION, file_hash
from rebuild_parents import PARENT_SCHEMA
import rebuild_parents


FROZEN_TARGETS_SHA = 'a0e3c181dbf80030605b258a60cda4489abe317eaf8c7ba3e666a98dbf8b4edb'
ARCHIVED_C34 = 'jiyujia/enveda-075-np-recall-base/2'
WINDOW = 'rebuilt_parent_window.parquet'
SCHEMA = pa.schema([
    ('molecule_id', pa.string()), ('smiles', pa.string()), ('score_key', pa.string()),
    ('heavy_atoms', pa.string()), ('formula', pa.string()), ('mass', pa.float64()),
    ('mass_error_ppm', pa.float64()), ('formal_charge', pa.int16()), ('src', pa.int8()),
    ('in_author_window', pa.bool_()), ('legacy_stored_key_global', pa.bool_()),
    ('legacy_same_window', pa.bool_()), ('legacy_same_author_window', pa.bool_()),
])


def read_legacy(directory, archived_pc_sha, archived_targets_sha):
    blind_path = directory / 'blind_virtual_np_receipt.json'
    detail_path = directory / 'receipt.json'
    blind, detail = (json.loads(path.read_bytes()) for path in (blind_path, detail_path))
    if (blind['generation_scope'] != 'full-main-window' or blind['query_count'] != 60
            or blind['archived_c34'] != ARCHIVED_C34
            or blind['archived_pc_lists_sha256'] != archived_pc_sha
            or blind['target_receipt_sha256'] != archived_targets_sha
            or blind['detailed_receipt_sha256'] != file_hash(detail_path)
            or detail['generation_scope'] != 'full-main-window' or detail['queries'] != 60
            or detail['rdkit'] != RDKit_VERSION or detail['pool_sha256'] != POOL_SHA256
            or detail['pool_rows'] != POOL_ROWS or detail['window_ppm'] != C34_PPM
            or detail['author_window_ppm'] != list(AUTHOR_PPM)
            or detail['target_masses_sha256'] != FROZEN_TARGETS_SHA
            or blind['output_sha256'] != detail['outputs']):
        raise ValueError('Expected the original sealed legacy C34 window scan')
    for name in ('known_main.parquet', 'stats.parquet'):
        if file_hash(directory / name) != detail['outputs'][name]:
            raise ValueError(f'Legacy scan changed: {name}')
    stats = pq.read_table(directory / 'stats.parquet', columns=[
        'molecule_id', 'target_mass', 'known_main', 'author_window_known_main']).to_pylist()
    targets = {row['molecule_id']: row['target_mass'] for row in stats}
    serialized = json.dumps(targets, sort_keys=True, separators=(',', ':')).encode()
    if len(stats) != len(targets) or len(targets) != 60 or sha256(serialized).hexdigest() != FROZEN_TARGETS_SHA:
        raise ValueError('Legacy statistics do not contain the exact 60 historical mass centres')
    broad = {mid: set() for mid in targets}
    narrow = {mid: set() for mid in targets}
    table = pq.read_table(directory / 'known_main.parquet', columns=[
        'molecule_id', 'score_key', 'heavy_atoms', 'mass', 'in_author_window'])
    for row in table.to_pylist():
        mid, identity = row['molecule_id'], (row['score_key'], row['heavy_atoms'])
        mass, target = row['mass'], targets[mid]
        tolerance = target * C34_PPM * 1e-6
        error = (target - mass) / mass * 1e6
        if (identity in broad[mid] or not target - tolerance <= mass <= target + tolerance
                or row['in_author_window'] != (AUTHOR_PPM[0] <= error <= AUTHOR_PPM[1])):
            raise ValueError(f'Invalid legacy window candidate: {mid}, {identity}')
        broad[mid].add(identity)
        if row['in_author_window']:
            narrow[mid].add(identity)
    if any(len(broad[row['molecule_id']]) != row['known_main'] or
           len(narrow[row['molecule_id']]) != row['author_window_known_main'] for row in stats):
        raise ValueError('Legacy per-query mass-window counts changed')
    return targets, broad, narrow


def read_rebuilt(receipt_path):
    source = receipt_path.with_name('rebuilt_parent_meta.parquet')
    receipt = json.loads(receipt_path.read_bytes())
    if (receipt['version'] != 2 or receipt['status'] != 'complete' or receipt['rdkit'] != RDKit_VERSION
            or receipt['legacy_pool_sha256'] != POOL_SHA256 or receipt['output_sha256'] != file_hash(source)
            or receipt['chem_source_sha256'] != file_hash(Path(rebuild_parents.__file__))):
        raise ValueError('Expected the completed, checksum-verified v2 parent inventory')
    table = pq.read_table(source)
    if (not table.schema.equals(PARENT_SCHEMA, check_metadata=False) or table.num_rows != receipt['identities']
            or any(column.null_count for column in table.columns)):
        raise ValueError('Rebuilt inventory schema or count differs from the merge receipt')
    actual_sources = {str(row['values']): row['counts'] for row in table['src'].value_counts().to_pylist()}
    if actual_sources != receipt['structure_counts'] or actual_sources.keys() - {'1', '2', '3'}:
        raise ValueError('Rebuilt source inventory counts differ from the merge receipt')
    masses = table['mass'].to_numpy()
    if not np.isfinite(masses).all() or (masses <= 0).any() or (np.diff(masses) < 0).any():
        raise ValueError('Rebuilt inventory is not sorted by positive finite mass')
    return table, masses, receipt


def scan_query(mid, target, table, masses, keys, old_broad, old_narrow):
    tolerance = target * C34_PPM * 1e-6
    first = int(np.searchsorted(masses, target - tolerance, side='left'))
    stop = int(np.searchsorted(masses, target + tolerance, side='right'))
    broad, narrow, rows = set(), set(), []
    sources = Counter()
    for candidate in table.slice(first, stop - first).to_pylist():
        key = candidate['key']
        identity = (key, candidate['heavy_atoms'])
        mass = candidate['mass']
        error = (target - mass) / mass * 1e6
        in_author_window = AUTHOR_PPM[0] <= error <= AUTHOR_PPM[1]
        if identity in broad:
            raise ValueError(f'Duplicate normalized rebuilt parent in {mid}: {identity}')
        broad.add(identity)
        if in_author_window:
            narrow.add(identity)
        known_key, in_legacy = key in keys, identity in old_broad
        sources[str(candidate['src'])] += 1
        rows.append(dict(molecule_id=mid, score_key=key, smiles=candidate['smiles'],
                         heavy_atoms=candidate['heavy_atoms'], formula=candidate['formula'], mass=mass,
                         mass_error_ppm=error, formal_charge=candidate['formal_charge'], src=candidate['src'],
                         in_author_window=in_author_window, legacy_stored_key_global=known_key,
                         legacy_same_window=in_legacy, legacy_same_author_window=identity in old_narrow))

    stats = dict(rebuilt_window=len(broad), legacy_window=len(old_broad),
                 shared_window=len(broad & old_broad), rebuilt_only_window=len(broad - old_broad),
                 legacy_only_window=len(old_broad - broad), rebuilt_author_window=len(narrow),
                 legacy_author_window=len(old_narrow), shared_author_window=len(narrow & old_narrow),
                 rebuilt_only_author_window=len(narrow - old_narrow), legacy_only_author_window=len(old_narrow - narrow),
                 rebuilt_no_legacy_stored_key=sum(not row['legacy_stored_key_global'] for row in rows),
                 rebuilt_extra_with_legacy_stored_key=sum(row['legacy_stored_key_global'] and
                                                         not row['legacy_same_window'] for row in rows))
    return rows, stats, sources


def run(rebuilt_receipt, legacy_window, legacy_pool, out, archived_pc_sha, archived_targets_sha):
    if rdBase.rdkitVersion != RDKit_VERSION:
        raise ValueError(f'Expected contest RDKit {RDKit_VERSION}')
    targets, old_broad, old_narrow = read_legacy(legacy_window, archived_pc_sha, archived_targets_sha)
    if file_hash(legacy_pool) != POOL_SHA256:
        raise ValueError('Legacy global key lookup must use the published 710,701-row pool')
    keys = set(pq.read_table(legacy_pool, columns=['key'])['key'].to_pylist())
    if len(keys) != POOL_ROWS or None in keys:
        raise ValueError('Unexpected legacy stored keys')
    table, masses, merge = read_rebuilt(rebuilt_receipt)
    out.mkdir(exist_ok=True, parents=True)
    window, report = out / WINDOW, out / 'rebuilt_parent_window_receipt.json'
    if window.exists() or report.exists():
        raise FileExistsError('Existing rebuilt parent scan in output directory')
    total, sources, cases = Counter(), Counter(), {}
    with pq.ParquetWriter(window, SCHEMA, compression='zstd') as writer:
        for mid, target in sorted(targets.items()):
            rows, stats, provenance = scan_query(mid, target, table, masses, keys, old_broad[mid], old_narrow[mid])
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
            total.update(stats)
            sources.update(provenance)
            cases[mid] = dict(stats, target_mass=target, rebuilt_window_by_src=dict(provenance))
    receipt = dict(status='complete', scope='rebuilt-parent-direct-window', query_count=len(cases),
                   rdkit=RDKit_VERSION, archived_c34=ARCHIVED_C34, archived_pc_lists_sha256=archived_pc_sha,
                   exact_targets_sha256=FROZEN_TARGETS_SHA, broad_window_ppm=C34_PPM, author_window_ppm=AUTHOR_PPM,
                   rebuilt_receipt_sha256=file_hash(rebuilt_receipt), rebuilt_inventory_sha256=merge['output_sha256'],
                   rebuilt_inventory_rows=merge['identities'], legacy_pool_sha256=POOL_SHA256,
                   legacy_window_receipt_sha256=file_hash(legacy_window / 'blind_virtual_np_receipt.json'),
                   legacy_window_sha256=file_hash(legacy_window / 'known_main.parquet'),
                   identity='First 14 characters of canonical tautomer InChIKey and heavy-atom composition',
                   legacy_global_identity='Exact stored-key membership, independent of window; no alias inference',
                   source_bits=merge['source_bits'], totals=dict(total), rebuilt_window_by_src=dict(sources),
                   cases=cases, outputs={WINDOW: file_hash(window)})
    report.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print('REBUILT_PARENT_WINDOW', len(cases), dict(total), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rebuilt-receipt', 'legacy-window', 'legacy-pool', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--archived-pc-sha', required=True)
    parser.add_argument('--archived-targets-sha', required=True)
    run(**vars(parser.parse_args()))
