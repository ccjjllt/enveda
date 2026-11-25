"""Enumerate the pinned main pool within all 60 frozen native C34 mass windows."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

import generate


COUNTS = ('window_rows', 'unusable_identity_rows', 'identity_alias_rows',
          'known_main', 'author_window_known_main')
STATS = pa.schema([('molecule_id', pa.string()), ('target_mass', pa.float64()),
                   *((name, pa.int32()) for name in COUNTS), ('seconds', pa.float64())])


def scan(item):
    mid, target = item
    start = time.monotonic()
    mass, texts = generate.MASSES, generate.SMILES
    tolerance = target * generate.C34_PPM * 1e-6
    first = int(np.searchsorted(mass, target - tolerance, side='left'))
    stop = int(np.searchsorted(mass, target + tolerance, side='right'))
    selected = {}
    rejected = 0
    for index in range(first, stop):
        text, stored_mass = texts[index], float(mass[index])
        identity = generate.normalize(text)
        if identity is None:
            rejected += 1
            continue
        error = (target - stored_mass) / stored_mass * 1e6
        in_author = generate.AUTHOR_PPM[0] <= error <= generate.AUTHOR_PPM[1]
        rank = (not in_author, abs(error), text, index)
        old = selected.get(identity)
        if old is None or rank < old[0]:
            selected[identity] = (rank, stored_mass, error)

    rows = []
    for (key, heavy), ((outside_author, _, text, _), stored_mass, error) in sorted(selected.items()):
        formula = CalcMolFormula(Chem.MolFromSmiles(text))
        rows.append(dict(molecule_id=mid, smiles=text, score_key=key, heavy_atoms=heavy, formula=formula,
                         mass=stored_mass, mass_error_ppm=error, in_author_window=not outside_author,
                         in_main_pool=True, n_parents=0, n_routes=0, transforms=[]))
    stats = dict(molecule_id=mid, target_mass=target, window_rows=stop - first,
                 unusable_identity_rows=rejected, identity_alias_rows=(stop - first) - rejected - len(rows),
                 known_main=len(rows), author_window_known_main=sum(row['in_author_window'] for row in rows),
                 seconds=round(time.monotonic() - start, 3))
    return rows, stats


def run(targets, pool, out, workers):
    if rdBase.rdkitVersion != generate.RDKit_VERSION or workers < 1:
        raise ValueError('Expected official RDKit and at least one worker')
    queries = json.loads(targets.read_text())
    if len(queries) != 60 or any(not isinstance(mass, (int, float)) or not 0 < mass < 10_000
                                 for mass in queries.values()):
        raise ValueError('Expected 60 finite positive native C34 mass centers')
    pool_rows = generate.load_pool(pool)
    generate.initialize(pool_rows, None, (), {})
    out.mkdir(parents=True, exist_ok=True)
    names = ('known_main.parquet', 'proposals.parquet', 'stats.parquet', 'receipt.json')
    if any((out / name).exists() for name in names):
        raise FileExistsError(f'An existing candidate artifact occupies {out}')
    totals = Counter()
    started = time.monotonic()
    with (ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as executor,
          pq.ParquetWriter(out / 'known_main.parquet', generate.PROPOSALS, compression='zstd') as proposals,
          pq.ParquetWriter(out / 'stats.parquet', STATS, compression='zstd') as scope):
        for rows, stats in executor.map(scan, sorted(queries.items())):
            proposals.write_table(pa.Table.from_pylist(rows, schema=generate.PROPOSALS))
            scope.write_table(pa.Table.from_pylist([stats], schema=STATS))
            totals.update({field: stats[field] for field in COUNTS})
    pq.write_table(pa.Table.from_pylist([], schema=generate.PROPOSALS), out / 'proposals.parquet', compression='zstd')
    if pq.ParquetFile(out / 'stats.parquet').metadata.num_rows != len(queries):
        raise ValueError('Full-main scan did not account for each query')
    receipt = dict(generation_scope='full-main-window', queries=len(queries),
                   rdkit=rdBase.rdkitVersion, target_masses_sha256=generate.file_hash(targets),
                   pool_sha256=generate.POOL_SHA256, pool_rows=generate.POOL_ROWS,
                   window_ppm=generate.C34_PPM, author_window_ppm=generate.AUTHOR_PPM,
                   selection='All stored pool masses within target ±10 ppm, inclusive',
                   identity='Stereo-free canonical tautomer InChIKey14 and heavy-element composition',
                   representative='Author window first, absolute ppm error, stored SMILES, pool row ID',
                   counts=dict(totals), seconds=round(time.monotonic() - started, 3),
                   outputs={name: generate.file_hash(out / name) for name in names[:-1]})
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('FULL_MAIN_WINDOW', dict(totals), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('targets', 'pool', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    run(**vars(parser.parse_args()))
