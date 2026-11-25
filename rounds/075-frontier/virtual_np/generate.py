"""Enumerate one-edit products for blinded precursor masses using the public CASMI pool."""

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from itertools import islice
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


RDKit_VERSION = '2026.03.3'
SOURCE_HASHES = {
    'code/casmi/derive.py': '7c1e8c3afb4c17a3ae58fbc6c36fd4cfc552da97b9129c9b46cca3cdc6ee4892',
    'code/casmi/chem.py': '99474f23d8a67040b0c0f8c70e73592a6f7f656f50103e350ca473451e2169f1',
    'MANIFEST.json': 'e64b960037769971e146a35932e29ed3f042de1716e4dbe281ebd84a7cfc3b6a',
    'LICENSE-NOTICE.txt': 'b9c26d77338c03cb4387ae61bf4f6524eb77ee59c8530bc9f8d33a742a913a3b',
}
POOL_SHA256 = '82b8c40315611289b9a4004b3af18d6d3402b3cd5baa4ee5723da07b7e2f6bb0'
POOL_ROWS = 710_701
C34_PPM = 10
AUTHOR_PPM = (-3.0, 4.6)
DELTA_CORRECTIONS = {'carboxyl_CH3': {'O': 2, 'H': -2}}
# Guard against the last decimal places of the pool mass and atomic-mass constants.
# The measured C34 window is applied to the product itself with no extra tolerance.
INDEX_SLACK_DA = 0.000_01

PROPOSALS = pa.schema([
    ('molecule_id', pa.string()), ('smiles', pa.string()),
    ('score_key', pa.string()), ('heavy_atoms', pa.string()), ('formula', pa.string()),
    ('mass', pa.float64()), ('mass_error_ppm', pa.float64()),
    ('in_author_window', pa.bool_()), ('in_main_pool', pa.bool_()),
    ('n_parents', pa.int32()), ('n_routes', pa.int32()), ('transforms', pa.list_(pa.string())),
])
STATS = pa.schema([
    ('molecule_id', pa.string()), ('target_mass', pa.float64()), ('status', pa.string()),
    ('parent_rows', pa.int32()), ('candidate_parent_transform_pairs', pa.int32()),
    ('scanned_parent_transform_pairs', pa.int32()), ('invalid_parent_rows', pa.int32()),
    ('raw_products', pa.int32()), ('invalid_products', pa.int32()),
    ('out_of_window_products', pa.int32()), ('valid_product_occurrences', pa.int32()),
    ('identities', pa.int32()), ('novel', pa.int32()), ('known_main', pa.int32()),
    ('author_window_identities', pa.int32()), ('author_window_novel', pa.int32()),
    ('author_window_known_main', pa.int32()), ('seconds', pa.float64()),
])


def file_hash(path):
    h = sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def public_source(derive_path):
    root = derive_path.resolve().parents[2]
    if derive_path.resolve() != root / 'code/casmi/derive.py':
        raise ValueError('Use the mounted casmi26-v4b-models/code/casmi/derive.py.')
    for name, expected in SOURCE_HASHES.items():
        if file_hash(root / name) != expected:
            raise ValueError(f'Unexpected source asset: {name}')
    hashes = json.loads((root / 'MANIFEST.json').read_text())['files']
    for name in ('code/casmi/derive.py', 'code/casmi/chem.py'):
        if hashes.get(name) != SOURCE_HASHES[name]:
            raise ValueError(f'Published manifest disagrees with the pinned file: {name}')
    sys.path.insert(0, str(root / 'code'))
    from casmi import chem, derive

    if (Path(derive.__file__).resolve() != derive_path.resolve()
            or Path(chem.__file__).resolve() != root / 'code/casmi/chem.py'):
        raise ValueError('Another casmi package was imported before the pinned asset.')
    return chem, derive


def load_pool(path):
    if file_hash(path) != POOL_SHA256:
        raise ValueError('Use the pinned v2 pool_meta.parquet.')
    table = pq.read_table(path, columns=['key', 'smiles', 'mass'])
    masses = np.asarray(table['mass'], dtype=np.float64)
    if (len(masses) != POOL_ROWS or not np.isfinite(masses).all()
            or not (masses > 0).all() or np.any(np.diff(masses) < 0)
            or table['smiles'].null_count or table['key'].null_count):
        raise ValueError('Expected the complete mass-sorted 710,701-row public pool.')
    keys = table['key'].to_pylist()
    by_key = defaultdict(list)
    for row, key in enumerate(keys):
        by_key[key].append(row)
    return masses, table['smiles'].to_pylist(), by_key


def query_masses(path, chem):
    fields = set(pq.read_schema(path).names)
    if {'molecule_id', 'target_mass'} <= fields:
        rows = pq.read_table(path, columns=['molecule_id', 'target_mass']).to_pylist()
        result = [(str(r['molecule_id']), r['target_mass']) for r in rows]
        if (any(r['molecule_id'] is None for r in rows) or len(dict(result)) != len(result)
                or any(m is None or not np.isfinite(m) or m <= 0 for _, m in result)):
            raise ValueError('Expected one positive finite native C34 target per molecule.')
        return result

    rows = pq.read_table(path, columns=['molecule_id', 'precursor_mz', 'adduct'])
    masses = chem.neutral_mass(rows['precursor_mz'].to_numpy(), rows['adduct'].to_pylist())
    groups = defaultdict(list)
    for mid, mass in zip(rows['molecule_id'].to_pylist(), masses):
        if mid is None:
            raise ValueError('A spectrum lacks a molecule_id.')
        values = groups[str(mid)]
        if np.isfinite(mass) and mass > 0:
            values.append(mass)
    return [(mid, float(np.median(values)) if values else None)
            for mid, values in groups.items()]


def transforms(derive):
    records = []
    corrections = {}
    known = set()
    for name, (smarts, elements) in derive._TRANSFORMS.items():
        rule = (smarts, tuple(sorted(elements.items())))
        if rule in known:
            continue
        known.add(rule)
        delta = derive._delta_mass(elements)
        if name in DELTA_CORRECTIONS:
            # The published reaction converts CH3 to COOH and loses two H.
            corrected = derive._delta_mass(DELTA_CORRECTIONS[name])
            corrections[name] = {'declared_delta_da': delta, 'used_delta_da': corrected}
            delta = corrected
        records.append((name, delta))
    if len(records) != 45:
        raise ValueError('Expected 45 different one-step reactions in the source table.')
    return records, corrections


def initialize(pool, derive, records, loss_rules):
    global MASSES, SMILES, MAIN_BY_KEY, INDEX, REACTIONS, TAUTOMER, LOSS_RULES, STATS_SCHEMA
    RDLogger.DisableLog('rdApp.*')
    MASSES, SMILES, MAIN_BY_KEY = pool
    LOSS_RULES = loss_rules
    STATS_SCHEMA = STATS.append(pa.field('rejected_exact_loss_products', pa.int32())) if loss_rules else STATS
    INDEX = defaultdict(list)
    REACTIONS = {}
    for name, delta in records:
        INDEX[delta].append(name)
        REACTIONS[name] = loss_rules[name]['run'] if loss_rules else derive._rxn(name).RunReactants
    TAUTOMER = rdMolStandardize.TautomerEnumerator()


def heavy_atoms(molecule):
    counts = Counter(a.GetSymbol() for a in molecule.GetAtoms() if a.GetAtomicNum() > 1)
    return ','.join(f'{element}:{n}' for element, n in sorted(counts.items()))


def composition(molecule):
    atoms = molecule.GetAtoms()
    counts = Counter(atom.GetSymbol() for atom in atoms)
    hydrogens = sum(atom.GetTotalNumHs() for atom in atoms)
    if hydrogens:
        counts['H'] += hydrogens
    return counts


@lru_cache(maxsize=32_768)
def normalize(text):
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        normalized = TAUTOMER.Canonicalize(mol)
        key = Chem.MolToInchiKey(normalized)
        if key:
            return key[:14], heavy_atoms(normalized)
    except (RuntimeError, ValueError):
        return None
    return None


def propose(molecule, target):
    start = time.monotonic()
    counts = Counter()
    stats = {field.name: 0 for field in STATS_SCHEMA if pa.types.is_integer(field.type)}
    stats.update(molecule_id=molecule, target_mass=target, status='no_target_mass', seconds=0.0)
    if target is None:
        return [], [], stats

    tolerance = target * C34_PPM * 1e-6
    lo, hi = target - tolerance, target + tolerance
    parent_rules = defaultdict(list)
    for delta, names in INDEX.items():
        first = np.searchsorted(MASSES, lo - delta - INDEX_SLACK_DA, side='left')
        stop = np.searchsorted(MASSES, hi - delta + INDEX_SLACK_DA, side='right')
        for row in range(first, stop):
            parent_rules[row].extend(names)
    counts['parent_rows'] = len(parent_rules)
    counts['candidate_parent_transform_pairs'] = sum(map(len, parent_rules.values()))
    rows = {}
    for parent_row, names in parent_rules.items():
        parent = Chem.MolFromSmiles(SMILES[parent_row])
        if parent is None:
            counts['invalid_parent_rows'] += 1
            continue
        if LOSS_RULES:
            if len(Chem.GetMolFrags(parent)) != 1:
                counts['invalid_parent_rows'] += 1
                continue
            parent_mass = rdMolDescriptors.CalcExactMolWt(parent)
            parent_formula = composition(parent)
            parent_charge = Chem.GetFormalCharge(parent)
        for name in names:
            counts['scanned_parent_transform_pairs'] += 1
            for products in REACTIONS[name]((parent,), 0):
                counts['raw_products'] += 1
                product = products[0]
                try:
                    Chem.SanitizeMol(product)
                    Chem.RemoveStereochemistry(product)
                    mass = rdMolDescriptors.CalcExactMolWt(product)
                except (RuntimeError, ValueError):
                    counts['invalid_products'] += 1
                    continue
                if not lo <= mass <= hi:
                    counts['out_of_window_products'] += 1
                    continue
                if LOSS_RULES:
                    rule = LOSS_RULES[name]
                    if (abs(mass - parent_mass - rule['delta']) > INDEX_SLACK_DA
                            or len(Chem.GetMolFrags(product)) != 1
                            or Chem.GetFormalCharge(product) != parent_charge
                            or composition(product) + rule['loss'] != parent_formula):
                        counts['rejected_exact_loss_products'] += 1
                        continue
                text = Chem.MolToSmiles(product)
                identity = normalize(text)
                if identity is None:
                    counts['invalid_products'] += 1
                    continue
                counts['valid_product_occurrences'] += 1
                error = (target - mass) / mass * 1e6
                is_narrow = AUTHOR_PPM[0] <= error <= AUTHOR_PPM[1]
                representative = (not is_narrow, abs(error), text)
                if identity not in rows:
                    rows[identity] = (representative, mass, error, rdMolDescriptors.CalcMolFormula(product), set())
                elif representative < rows[identity][0]:
                    _, _, _, _, routes = rows[identity]
                    rows[identity] = (representative, mass, error, rdMolDescriptors.CalcMolFormula(product), routes)
                rows[identity][4].add((parent_row, name))

    novel, known = [], []
    for (key, heavy), (representative, mass, error, formula, routes) in sorted(rows.items()):
        outside_author, _, text = representative
        in_main = any(normalize(SMILES[i]) == (key, heavy) for i in MAIN_BY_KEY.get(key, ()))
        row = dict(molecule_id=molecule, smiles=text, score_key=key, heavy_atoms=heavy,
                   formula=formula, mass=mass, mass_error_ppm=error, in_main_pool=in_main,
                   in_author_window=not outside_author, n_parents=len({i for i, _ in routes}),
                   n_routes=len(routes), transforms=sorted({name for _, name in routes}))
        (known if in_main else novel).append(row)
    stats.update(counts)
    narrow_novel = sum(r['in_author_window'] for r in novel)
    narrow_known = sum(r['in_author_window'] for r in known)
    stats.update(status='ok', identities=len(rows), novel=len(novel), known_main=len(known),
                 author_window_identities=narrow_novel + narrow_known,
                 author_window_novel=narrow_novel, author_window_known_main=narrow_known,
                 seconds=time.monotonic() - start)
    return novel, known, stats


def query_results(executor, queries, workers):
    remaining = iter(queries)
    pending = deque(executor.submit(propose, *query) for query in islice(remaining, 2 * workers))
    while pending:
        result = pending.popleft().result()
        query = next(remaining, None)
        if query is not None:
            pending.append(executor.submit(propose, *query))
        yield result


def generate(args, *, build_loss_rules=None):
    if rdBase.rdkitVersion != RDKit_VERSION:
        raise ValueError(f'Use official RDKit {RDKit_VERSION}, found {rdBase.rdkitVersion}.')
    if args.workers < 1:
        raise ValueError('Choose at least one worker.')
    chem, derive = public_source(args.derive_source)
    pool = load_pool(args.pool_meta)
    queries = query_masses(args.queries, chem)
    if not queries:
        raise ValueError('No query molecules.')
    loss_rules = {}
    if build_loss_rules is None:
        records, corrections = transforms(derive)
        if args.only_rule:
            records = [record for record in records if record[0] == args.only_rule]
            if len(records) != 1:
                raise ValueError(f'Expected one published edit: {args.only_rule}')
            corrections = {name: correction for name, correction in corrections.items() if name == args.only_rule}
    else:
        loss_rules = build_loss_rules(derive)
        if not loss_rules:
            raise ValueError('The supplemental loss rule set is empty.')
        records = [(name, rule['delta']) for name, rule in loss_rules.items()]
        corrections = {}
    initialize(pool, derive, records, loss_rules)

    args.outdir.mkdir(parents=True, exist_ok=True)
    files = [args.outdir / name for name in ('proposals.parquet', 'known_main.parquet', 'stats.parquet')]
    receipt = args.outdir / 'receipt.json'
    if receipt.exists() or any(path.exists() for path in files):
        raise FileExistsError(f'An existing output occupies {args.outdir}.')
    total_stats = Counter()
    start = time.monotonic()
    with (
        ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as executor,
        pq.ParquetWriter(files[0], PROPOSALS, compression='zstd') as novel_writer,
        pq.ParquetWriter(files[1], PROPOSALS, compression='zstd') as known_writer,
        pq.ParquetWriter(files[2], STATS_SCHEMA, compression='zstd') as stats_writer,
    ):
        for i, (novel, known, stats) in enumerate(query_results(executor, queries, args.workers), 1):
            novel_writer.write_table(pa.Table.from_pylist(novel, schema=PROPOSALS))
            known_writer.write_table(pa.Table.from_pylist(known, schema=PROPOSALS))
            stats_writer.write_table(pa.Table.from_pylist([stats], schema=STATS_SCHEMA))
            total_stats.update({field.name: stats[field.name] for field in STATS_SCHEMA
                                if pa.types.is_integer(field.type)})
            print(f'{i}/{len(queries)} {stats["molecule_id"]}: {len(novel)} novel, '
                  f'{len(known)} known, {stats["scanned_parent_transform_pairs"]} pairs, '
                  f'{stats["seconds"]:.1f}s', flush=True)
    if pq.ParquetFile(files[2]).metadata.num_rows != len(queries):
        raise ValueError('Generation statistics are incomplete.')
    details = {
        'queries': len(queries), 'query_sha256': file_hash(args.queries),
        'pool_rows': POOL_ROWS, 'pool_sha256': POOL_SHA256, 'source_files': SOURCE_HASHES,
        'rdkit': rdBase.rdkitVersion, 'c34_ppm': C34_PPM, 'author_ppm': AUTHOR_PPM,
        'mass_error_ppm': '(target_mass - candidate_mass) / candidate_mass * 1e6',
        'parent_index_slack_da': INDEX_SLACK_DA, 'transforms': dict(records),
        'transform_delta_corrections': corrections,
        'counts': dict(total_stats), 'seconds': time.monotonic() - start,
        'outputs': {path.name: file_hash(path) for path in files},
    }
    if loss_rules:
        details['exact_loss_rules'] = {
            name: {'formula': dict(rule['loss']), 'smarts': rule['smarts']}
            for name, rule in loss_rules.items()
        }
    receipt.write_text(json.dumps(details, indent=2) + '\n')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--pool-meta', type=Path, required=True)
    parser.add_argument('--derive-source', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--only-rule')
    return parser.parse_args()


if __name__ == '__main__':
    generate(arguments())
