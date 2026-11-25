"""Build a structure-only parent inventory from competition train and the COCONUT 2.0 archive."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from hashlib import file_digest, md5, sha256
from io import TextIOWrapper
from itertools import islice
import json
from pathlib import Path
import time
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


RDKit_VERSION = '2026.03.3'
TRAIN, COCONUT = 1, 2
COCONUT_RECORD = 'https://zenodo.org/records/13692394'
COCONUT_DOI = '10.5281/zenodo.13692394'
COCONUT_MD5 = '439fca469248b2eb5194ffb75ddf62ed'
LEGACY_SHA256 = '82b8c40315611289b9a4004b3af18d6d3402b3cd5baa4ee5723da07b7e2f6bb0'
LEGACY_ROWS = 710_701
BATCH_SIZE = 512
PARENT_SCHEMA = pa.schema([
    ('key', pa.string()), ('smiles', pa.string()), ('mass', pa.float64()),
    ('formula', pa.string()), ('n_heavy', pa.int16()), ('heavy_atoms', pa.string()),
    ('formal_charge', pa.int16()), ('src', pa.int8()),
])


def checksum(path):
    with path.open('rb') as file:
        return file_digest(file, 'sha256').hexdigest()


def verified_release(path, receipt_path):
    receipt = json.loads(receipt_path.read_text())
    sums = (md5(usedforsecurity=False), sha256())
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            for digest in sums:
                digest.update(block)
    if (receipt['doi'] != COCONUT_DOI or receipt['release'] != COCONUT_RECORD
            or receipt['author_md5'] != COCONUT_MD5 or sums[0].hexdigest() != COCONUT_MD5
            or sums[1].hexdigest() != receipt['sha256'] or path.stat().st_size != receipt['bytes']):
        raise ValueError('Expected the original, checksum-verified September 2024 COCONUT archive.')
    return receipt


def training_smiles(path, counts):
    seen = set()
    for batch in pq.ParquetFile(path).iter_batches(columns=['normalized_smiles'], batch_size=65536):
        for text in batch.column(0).to_pylist():
            counts['source_rows'] += 1
            if text is None or not text.strip():
                counts['missing_smiles'] += 1
                continue
            if text in seen:
                counts['duplicate_raw_smiles'] += 1
                continue
            seen.add(text)
            yield text


def coconut_smiles(path, receipt, counts):
    with ZipFile(path) as zip_file, zip_file.open(receipt['csv_member']) as member:
        rows = csv.DictReader(TextIOWrapper(member, encoding='utf-8-sig', newline=''))
        if (rows.fieldnames != receipt['columns']
                or not {'canonical_smiles', 'identifier'} <= set(rows.fieldnames)):
            raise ValueError('Unexpected header in the pinned COCONUT CSV.')
        for row in rows:
            counts['source_rows'] += 1
            text = row['canonical_smiles']
            if text is None or not text.strip():
                counts['missing_smiles'] += 1
                continue
            yield text


def initialize():
    global TAUTOMERS, FRAGMENTS
    RDLogger.DisableLog('rdApp.*')
    TAUTOMERS = rdMolStandardize.TautomerEnumerator()
    FRAGMENTS = rdMolStandardize.LargestFragmentChooser()


def describe(text):
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None, False
    salt = len(Chem.GetMolFrags(mol)) > 1
    if salt:
        mol = FRAGMENTS.choose(mol)
    if not mol.GetNumHeavyAtoms():
        return None, salt
    Chem.RemoveStereochemistry(mol)
    heavy_counts = Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    heavy = ','.join(f'{element}:{count}' for element, count in sorted(heavy_counts.items()))
    key = Chem.MolToInchiKey(TAUTOMERS.Canonicalize(mol))[:14]
    if len(key) != 14:
        return None, salt
    return (key, Chem.MolToSmiles(mol), float(rdMolDescriptors.CalcExactMolWt(mol)),
            rdMolDescriptors.CalcMolFormula(mol), mol.GetNumHeavyAtoms(), heavy,
            Chem.GetFormalCharge(mol)), salt


def describe_batch(batch):
    result = []
    for text in batch:
        try:
            result.append(describe(text))
        except (RuntimeError, ValueError):
            result.append((None, False))
    return result


def add_source(identities, iterator, source, counts, workers, executor):
    while True:
        batches = [tuple(islice(iterator, BATCH_SIZE)) for _ in range(2 * workers)]
        batches = [batch for batch in batches if batch]
        if not batches:
            break
        for data in executor.map(describe_batch, batches):
            for row, salt in data:
                counts['multicomponent_reduced'] += salt
                if row is None:
                    counts['invalid_structures'] += 1
                    continue
                counts['valid_structures'] += 1
                counts['valid_net_charged'] += row[-1] != 0
                identity = (row[0], row[5])
                previous = identities.get(identity)
                if previous is None:
                    identities[identity] = (source, row)
                    continue
                bits, representative = previous
                if bits & source:
                    counts['duplicate_normalized_identity'] += 1
                # The training representation wins across sources. Within a source use sorted SMILES.
                if (source == TRAIN or not bits & TRAIN) and row[1] < representative[1]:
                    representative = row
                identities[identity] = (bits | source, representative)
    print(f'source {source}: {counts["source_rows"]:,} records, {len(identities):,} pooled identities',
          flush=True)


def compare(identities, legacy_path):
    if checksum(legacy_path) != LEGACY_SHA256:
        raise ValueError('Expected the historical public train/COCONUT pool metadata.')
    table = pq.read_table(legacy_path, columns=['key', 'smiles', 'mass', 'src'])
    legacy = {r['key']: r for r in table.to_pylist()}
    if len(legacy) != LEGACY_ROWS or len(legacy) != len(table):
        raise ValueError('Different historical parent inventory.')
    counts, overlap_by_source = Counter(), Counter()
    new_bits = {}
    for (key, _), (bits, row) in identities.items():
        new_bits[key] = new_bits.get(key, 0) | bits
        old = legacy.get(key)
        if old is not None:
            counts['shared_rebuilt_identities'] += 1
            counts['same_smiles'] += row[1] == old['smiles']
            counts['same_mass_within_1e-6_da'] += abs(row[2] - old['mass']) <= 1e-6
            counts['same_mass_within_10_ppm'] += abs(row[2] - old['mass']) <= old['mass'] * 10e-6
    shared = new_bits.keys() & legacy.keys()
    for key in shared:
        overlap_by_source[f'{legacy[key]["src"]}->{new_bits[key]}'] += 1
    return dict(
        legacy_keys=len(legacy), rebuilt_keys=len(new_bits), shared_keys=len(shared),
        legacy_only_keys=len(legacy) - len(shared), rebuilt_only_keys=len(new_bits) - len(shared),
        rebuilt_keys_with_distinct_heavy_compositions=len(identities) - len(new_bits),
        legacy_source_bits=dict(Counter(row['src'] for row in legacy.values())),
        shared_by_legacy_to_rebuilt_source_bits=dict(overlap_by_source), **counts,
    )


def build(args):
    if rdBase.rdkitVersion != RDKit_VERSION:
        raise ValueError(f'Use official RDKit {RDKit_VERSION}, found {rdBase.rdkitVersion}.')
    if args.workers < 1:
        raise ValueError('Choose at least one worker.')
    start = time.monotonic()
    receipt = verified_release(args.coconut_archive, args.coconut_receipt)
    sources = {TRAIN: Counter(), COCONUT: Counter()}
    identities = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as executor:
        add_source(identities, training_smiles(args.train, sources[TRAIN]),
                   TRAIN, sources[TRAIN], args.workers, executor)
        add_source(identities, coconut_smiles(args.coconut_archive, receipt, sources[COCONUT]),
                   COCONUT, sources[COCONUT], args.workers, executor)
    overlap = compare(identities, args.legacy_pool)
    ordered = sorted(identities.values(), key=lambda entry: (entry[1][2], entry[1][0], entry[1][5]))
    source_counts = Counter(bits for bits, _ in ordered)
    args.outdir.mkdir(parents=True, exist_ok=True)
    output = args.outdir / 'rebuilt_parent_meta.parquet'
    report = args.outdir / 'rebuilt_parent_receipt.json'
    if output.exists() or report.exists():
        raise FileExistsError(f'Parent inventory already exists in {args.outdir}.')
    names = PARENT_SCHEMA.names
    with pq.ParquetWriter(output, PARENT_SCHEMA, compression='zstd') as writer:
        for offset in range(0, len(ordered), 32768):
            package = [dict(zip(names, row + (bits,))) for bits, row in ordered[offset:offset + 32768]]
            writer.write_table(pa.Table.from_pylist(package, schema=PARENT_SCHEMA))
    report.write_text(json.dumps(dict(
        rdkit=rdBase.rdkitVersion, train_sha256=checksum(args.train),
        coconut_release=receipt, legacy_pool_sha256=LEGACY_SHA256,
        inventory='Independent structure-only inventory',
        source_bits={'1': 'competition train', '2': 'COCONUT', '3': 'both'},
        normalization=[
            'Read normalized_smiles from train and canonical_smiles from the pinned COCONUT CSV.',
            'For a multicomponent record, use the RDKit largest fragment. Preserve its net formal charge.',
            'Remove stereochemistry. Use canonical tautomer InChIKey14 and heavy-element counts for identity.',
            'Store the de-stereochemized parent as the chemistry input. Training wins across sources.',
            'Choose lexicographically smallest canonical SMILES within a source. Apply no numeric cutoffs.',
        ],
        source_input_counts={'train': dict(sources[TRAIN]), 'coconut': dict(sources[COCONUT])},
        structure_counts=dict(source_counts), identities=len(ordered),
        legacy_overlap=overlap, output_sha256=checksum(output), seconds=time.monotonic() - start,
    ), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('train', 'coconut-archive', 'coconut-receipt', 'legacy-pool', 'outdir'):
        parser.add_argument(f'--{arg}', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    build(parser.parse_args())
