"""Generate a public structure-only, endpoint-resolved training supply for a transform prior."""

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from itertools import islice
import json
import math
from multiprocessing import get_context
from pathlib import Path
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

import generate


BANDS = 12
SAMPLE_SIZE = 8_192
SEED = 20_261_125
PRODUCT_CAP = 8
ENDPOINTS = 'transform_prior_endpoints.parquet'
EDGES = 'transform_prior_edges.parquet'
RECEIPT = 'transform_prior_supply_receipt.json'
ENDPOINT_SCHEMA = pa.schema([
    ('endpoint_id', pa.string()), ('score_key', pa.string()), ('heavy_atoms', pa.string()),
    ('smiles', pa.string()), ('family', pa.string()), ('formula', pa.string()),
    ('mass_da', pa.float64()), ('charge', pa.int8()), ('pool_key_present', pa.bool_()),
    ('pool_match', pa.bool_()), ('pool_src', pa.int16()), ('sampled_parent', pa.bool_()),
])
EDGE_SCHEMA = pa.schema([
    ('parent_id', pa.string()), ('product_id', pa.string()), ('rule', pa.string()),
    ('multiplicity', pa.int32()), ('parent_band', pa.int8()), ('parent_pool_row', pa.int32()),
    ('at_product_cap', pa.bool_()),
])


def read_pool(path):
    if generate.file_hash(path) != generate.POOL_SHA256:
        raise ValueError('Expected the pinned public 710,701-row parent pool')
    table = pq.read_table(path, columns=['key', 'smiles', 'mass', 'src'])
    masses = table['mass'].to_numpy()
    if (table.num_rows != generate.POOL_ROWS or any(column.null_count for column in table.columns)
            or not np.isfinite(masses).all() or (masses <= 0).any() or (np.diff(masses) < 0).any()):
        raise ValueError('Expected the complete sorted mass index and source codes')
    keys = table['key'].to_pylist()
    key_rows = defaultdict(list)
    for row, key in enumerate(keys):
        key_rows[key].append(row)
    return keys, table['smiles'].to_pylist(), masses, table['src'].to_pylist(), dict(key_rows)


def sample_rows(keys, masses, size, bands, seed):
    if not 1 <= bands <= size <= len(keys) or len(keys) != len(masses):
        raise ValueError('Invalid mass-stratified sample dimensions')
    rows, strata = [], []
    for band, indices in enumerate(np.array_split(np.arange(len(keys)), bands)):
        budget = size // bands + int(band < size % bands)
        if budget > len(indices):
            raise ValueError('Mass stratum does not have its fixed parent budget')
        ranked = sorted((int(row) for row in indices), key=lambda row: (
            sha256(f'{seed}|{band}|{row}|{keys[row]}'.encode()).digest(), row))
        selected = sorted(ranked[:budget])
        rows.extend((band, row) for row in selected)
        strata.append(dict(band=band, start=int(indices[0]), stop=int(indices[-1]) + 1,
                           min_mass_da=float(masses[indices[0]]), max_mass_da=float(masses[indices[-1]]),
                           sample_count=budget))
    return rows, strata


def chemistry_filter(molecule):
    if molecule is None:
        return 'unparseable'
    if not any(atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()):
        return 'no_heavy_atoms'
    if len(Chem.GetMolFrags(molecule)) != 1:
        return 'disconnected'
    if Chem.GetFormalCharge(molecule) != 0:
        return 'nonneutral'
    if any(atom.GetIsotope() for atom in molecule.GetAtoms()):
        return 'explicit_isotope'
    return None


@lru_cache(maxsize=8_192)
def endpoint(smiles):
    normalized = generate.normalize(smiles)
    if normalized is None or len(normalized[0]) != 14:
        return None
    key, heavy = normalized
    molecule = Chem.MolFromSmiles(smiles)
    try:
        core = MurckoScaffold.GetScaffoldForMol(molecule)
        family = (Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core), isomericSmiles=False)
                  if core.GetNumAtoms() else '@acyclic')
    except (RuntimeError, ValueError):
        family = '@unresolved'
    return dict(endpoint_id=f'{key}|{heavy}', score_key=key, heavy_atoms=heavy, smiles=smiles, family=family,
                formula=rdMolDescriptors.CalcMolFormula(molecule), mass_da=float(rdMolDescriptors.CalcExactMolWt(molecule)),
                charge=Chem.GetFormalCharge(molecule))


def keep_endpoint(mapping, item):
    key = item['endpoint_id']
    previous = mapping.get(key)
    if previous is not None:
        if (previous['formula'] != item['formula'] or previous['charge'] != item['charge']
                or abs(previous['mass_da'] - item['mass_da']) > 1e-6):
            raise ValueError(f'One normalized neutral identity has conflicting composition: {key}')
        priority = (not item.get('sampled_parent', False), item['smiles'])
        existing = (not previous.get('sampled_parent', False), previous['smiles'])
        if priority >= existing:
            return
    mapping[key] = item


def worker_init():
    RDLogger.DisableLog('rdApp.*')
    generate.TAUTOMER = rdMolStandardize.TautomerEnumerator()


def generate_parent(band, row, parent_text, reactions, cap):
    parent = Chem.MolFromSmiles(parent_text)
    counts = Counter()
    rejection = chemistry_filter(parent)
    if rejection:
        counts[f'parent_{rejection}'] += 1
        return None, {}, [], counts, Counter()
    parent_descriptor = endpoint(Chem.MolToSmiles(parent, isomericSmiles=False))
    if parent_descriptor is None:
        counts['parent_identity_failed'] += 1
        return None, {}, [], counts, Counter()
    counts['eligible_parent'] += 1
    parent_id = parent_descriptor['endpoint_id']
    products_by_id, edges, rule_counts = {}, [], Counter()
    for name, reaction, template in reactions:
        if not parent.HasSubstructMatch(template):
            continue
        raw = reaction.RunReactants((parent,), cap)
        counts['matched_parent_rules'] += 1
        counts['raw_products'] += len(raw)
        capped = len(raw) == cap
        counts['parent_rules_at_cap'] += capped
        smiles_counts = Counter()
        for components in raw:
            if len(components) != 1:
                counts['product_disconnected'] += 1
                continue
            product = components[0]
            try:
                Chem.SanitizeMol(product)
            except (RuntimeError, ValueError):
                counts['product_sanitize_failed'] += 1
                continue
            rejection = chemistry_filter(product)
            if rejection:
                counts[f'product_{rejection}'] += 1
                continue
            smiles_counts[Chem.MolToSmiles(product, isomericSmiles=False)] += 1

        route_counts = Counter()
        for smiles, multiplicity in smiles_counts.items():
            description = endpoint(smiles)
            if description is None:
                counts['product_identity_failed'] += multiplicity
                continue
            product_id = description['endpoint_id']
            if parent_id == product_id:
                counts['product_self_outcomes'] += multiplicity
                continue
            route_counts[product_id] += multiplicity
            keep_endpoint(products_by_id, description)
        for product_id, multiplicity in sorted(route_counts.items()):
            edges.append(dict(parent_id=parent_id, product_id=product_id, rule=name, multiplicity=multiplicity,
                              parent_band=band, parent_pool_row=row, at_product_cap=capped))
        rule_counts[name] += len(route_counts)
    return parent_descriptor, products_by_id, edges, counts, rule_counts


def run_batch(tasks):
    return [generate_parent(band, row, POOL_SMILES[row], REACTIONS, CURRENT_CAP) for band, row in tasks]


def batches(executor, tasks, workers):
    remaining = iter(tasks)
    pending = deque()
    for _ in range(2 * workers):
        part = list(islice(remaining, 16))
        if not part:
            break
        pending.append((part, executor.submit(run_batch, part)))
    while pending:
        part, future = pending.popleft()
        yield from zip(part, future.result(), strict=True)
        next_part = list(islice(remaining, 16))
        if next_part:
            pending.append((next_part, executor.submit(run_batch, next_part)))


def attach_membership(item, parent_rows, key_rows, texts, sources):
    identity = (item['score_key'], item['heavy_atoms'])
    exact = set(parent_rows)
    same_keys = key_rows.get(item['score_key'], ())
    for row in same_keys:
        if row not in exact and generate.normalize(texts[row]) == identity:
            exact.add(row)
    return dict(item, pool_key_present=bool(same_keys), pool_match=bool(exact),
                pool_src=int(sources[min(exact)]) if exact else None, sampled_parent=bool(parent_rows))


def run(pool, derive_source, out, workers, sample_size, bands, seed, cap):
    global POOL_SMILES, REACTIONS, CURRENT_CAP
    if rdBase.rdkitVersion != generate.RDKit_VERSION or not 1 <= workers <= 6 or cap < 1:
        raise ValueError('Use the metric RDKit version and a positive worker and outcome cap')
    keys, POOL_SMILES, masses, sources, key_rows = read_pool(pool)
    _, derive = generate.public_source(derive_source)
    rules, _ = generate.transforms(derive)
    REACTIONS = []
    for name, _ in rules:
        reaction = derive._rxn(name)
        if reaction.GetNumReactantTemplates() != 1:
            raise ValueError(f'Expected one parent in route {name}')
        REACTIONS.append((name, reaction, reaction.GetReactantTemplate(0)))
    CURRENT_CAP = cap
    sample, strata = sample_rows(keys, masses, sample_size, bands, seed)
    sample_hash = sha256(json.dumps(sample, separators=(',', ':')).encode()).hexdigest()
    out.mkdir(exist_ok=True, parents=True)
    if any((out / name).exists() for name in (EDGES, ENDPOINTS, RECEIPT)):
        raise FileExistsError('Output directory already has transform prior artifacts')

    counts, rule_counts, unique, records = Counter(), Counter(), {}, {}
    role_flags = set()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='transform-prior-output-', dir=out) as path:
        scratch = Path(path)
        with (ProcessPoolExecutor(max_workers=workers, mp_context=get_context('fork'), initializer=worker_init) as executor,
              pq.ParquetWriter(scratch / EDGES, EDGE_SCHEMA, compression='zstd') as writer):
            edges_pending = []
            for (band, row), (parent, products, edges, tally, per_rule) in batches(executor, sample, workers):
                counts.update(tally)
                counts['sampled_rows_processed'] += 1
                if parent is not None:
                    parent_id = parent['endpoint_id']
                    if parent_id in unique:
                        unique[parent_id].append(row)
                        counts['duplicate_sampled_parent_identity'] += 1
                    else:
                        unique[parent_id] = [row]
                        keep_endpoint(records, dict(parent, sampled_parent=True))
                        for product in products.values():
                            keep_endpoint(records, product)
                        role_flags.update(products)
                        rule_counts.update(per_rule)
                        counts['edges'] += len(edges)
                        counts['edge_multiplicity'] += sum(edge['multiplicity'] for edge in edges)
                        edges_pending.extend(edges)
                        if len(edges_pending) >= 8_192:
                            writer.write_table(pa.Table.from_pylist(edges_pending, schema=EDGE_SCHEMA))
                            edges_pending.clear()
                if counts['sampled_rows_processed'] % 512 == 0:
                    print('sampled parents', counts['sampled_rows_processed'], '/', sample_size,
                          'edges', counts['edges'], 'elapsed', round(time.monotonic() - started), flush=True)
            if edges_pending:
                writer.write_table(pa.Table.from_pylist(edges_pending, schema=EDGE_SCHEMA))

        if not counts['edges'] or counts['sampled_rows_processed'] != sample_size:
            raise ValueError('Missing normalized interaction graph or incomplete parent sample')
        worker_init()
        counts.update(endpoints=len(records), parents=len(unique), products=len(role_flags))
        ordered = sorted(records)
        with pq.ParquetWriter(scratch / ENDPOINTS, ENDPOINT_SCHEMA, compression='zstd') as writer:
            for offset in range(0, len(ordered), 8_192):
                items = []
                for key in ordered[offset:offset + 8_192]:
                    item = attach_membership(records[key], unique.get(key, ()), key_rows, POOL_SMILES, sources)
                    if not math.isfinite(item['mass_da']) or item['mass_da'] <= 0 or item['charge'] != 0:
                        raise ValueError('Invalid formula-matched endpoint')
                    items.append(item)
                    counts['endpoints_pool_match'] += item['pool_match']
                    counts['ambiguous_pool_key'] += item['pool_key_present'] and not item['pool_match']
                    counts['products_pool_match'] += key in role_flags and item['pool_match']
                    counts['products_pool_absent_key'] += key in role_flags and not item['pool_key_present'] and not item['pool_match']
                    counts['endpoints_acyclic'] += item['family'] == '@acyclic'
                    counts['endpoints_unresolved_family'] += item['family'] == '@unresolved'
                writer.write_table(pa.Table.from_pylist(items, schema=ENDPOINT_SCHEMA))

        outputs = (ENDPOINTS, EDGES)
        if any(pq.ParquetFile(scratch / file).metadata.num_rows != counts[field]
               for file, field in zip(outputs, ('endpoints', 'edges'), strict=True)):
            raise ValueError('Output accounting differs from the completed graph')
        report = dict(status='complete', rdkit=rdBase.rdkitVersion,
                      pool_sha256=generate.POOL_SHA256, source_pool_rows=generate.POOL_ROWS,
                      chem_source_sha256=generate.SOURCE_HASHES['code/casmi/derive.py'],
                      module_sha256=generate.file_hash(Path(__file__)),
                      sample_size=sample_size, mass_bands=strata, sample_sha256=sample_hash, seed=seed,
                      sampling='Equal-size mass strata in the pinned pool; deterministic SHA-256 row selection',
                      sampled_parent_row_indices=[dict(band=band, parent_pool_row=row) for band, row in sample],
                      molecule_filter='Only neutral connected molecules without explicit isotope labels and with heavy atoms',
                      identity='Stereo-free tautomer InChIKey14 and exact heavy-atom composition',
                      family='Generic RDKit Murcko framework; empty scaffold assigned to @acyclic; failures grouped at @unresolved',
                      membership='Exact normalized hit in a published row with the same stored key or a verified sampled parent; all other hits from the index are ambiguous',
                      negative_status='Missing stored key means unobserved in this index, not proof of nonexistence',
                      outcome_cap_per_parent_rule=cap, count_at_cap_note='At-cap multiplicity is a lower bound',
                      rules=[name for name, _ in rules], distinct_edges_by_rule={name: rule_counts[name] for name, _ in rules},
                      counts=dict(counts), output_sha256={name: generate.file_hash(scratch / name) for name in outputs},
                      seconds=round(time.monotonic() - started, 3))
        for name in outputs:
            (scratch / name).replace(out / name)
        receipt = scratch / RECEIPT
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        receipt.replace(out / RECEIPT)
    print('TRANSFORM_PRIOR_SUPPLY', json.dumps(dict(counts), sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pool', 'derive-source', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--sample-size', type=int, default=SAMPLE_SIZE)
    parser.add_argument('--bands', type=int, default=BANDS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--cap', type=int, default=PRODUCT_CAP)
    run(**vars(parser.parse_args()))
