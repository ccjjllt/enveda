"""Measure actual one-edit mass deltas on a deterministic sample of the public parent pool."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolDescriptors

from generate import INDEX_SLACK_DA, POOL_ROWS, POOL_SHA256, RDKit_VERSION, file_hash, public_source, transforms


BANDS = 12
MASS_BIN_DA = 0.000001


def audit(pool, source, output, seed, sample_per_band, max_parents_per_band):
    RDLogger.DisableLog('rdApp.*')
    if rdBase.rdkitVersion != RDKit_VERSION or file_hash(pool) != POOL_SHA256:
        raise ValueError('Expected the official RDKit and pinned public parent pool')
    _, derive = public_source(source)
    rules, corrections = transforms(derive)
    prepared = [(name, expected, derive._rxn(name)) for name, expected in rules]
    if any(rxn.GetNumReactantTemplates() != 1 for _, _, rxn in prepared):
        raise ValueError('Reverse-mass generation requires one parent molecule')

    table = pq.read_table(pool, columns=['smiles', 'mass'])
    masses = table['mass'].to_numpy()
    if (len(masses) != POOL_ROWS or not np.isfinite(masses).all()
            or np.any(np.diff(masses) < 0)):
        raise ValueError('The parent pool must be mass-sorted')
    templates = {name: rxn.GetReactantTemplate(0) for name, _, rxn in prepared}
    declared = {name: derive._delta_mass(derive._TRANSFORMS[name][1]) for name, _ in rules}
    rng = np.random.default_rng(seed)
    totals = {name: Counter() for name, _ in rules}
    histograms = {name: Counter() for name, _ in rules}
    query_bands = []
    present = Counter()
    start = time.monotonic()

    for band in np.array_split(np.arange(len(masses)), BANDS):
        rows = rng.choice(band, size=sample_per_band, replace=False)
        coverage = Counter()
        for row in rows:
            parent = Chem.MolFromSmiles(table['smiles'][int(row)].as_py())
            if parent is None:
                present['invalid_parents'] += 1
                continue
            present['valid_parents'] += 1
            parent_mass = rdMolDescriptors.CalcExactMolWt(parent)
            stored_mass = float(masses[row])
            if abs(parent_mass - stored_mass) > INDEX_SLACK_DA:
                present['stored_mass_beyond_index_slack'] += 1
            for name, expected, rxn in prepared:
                if coverage[name] >= max_parents_per_band or not parent.HasSubstructMatch(templates[name]):
                    continue
                coverage[name] += 1
                counts = totals[name]
                parent_products = set()
                products = rxn.RunReactants((parent,), 8)
                counts['reactant_matches'] += 1
                counts['reached_product_cap'] += len(products) == 8
                for components in products:
                    product = components[0]
                    counts['raw_product_rows'] += 1
                    try:
                        Chem.SanitizeMol(product)
                        key = Chem.MolToSmiles(product, isomericSmiles=False)
                        product_mass = rdMolDescriptors.CalcExactMolWt(product)
                    except (RuntimeError, ValueError):
                        counts['invalid_product_rows'] += 1
                        continue
                    if key in parent_products:
                        continue
                    parent_products.add(key)
                    delta = product_mass - parent_mass
                    histograms[name][round(delta / MASS_BIN_DA)] += 1
                    counts['unique_parent_product_pairs'] += 1
                    counts['declared_delta_mismatch'] += abs(delta - declared[name]) > INDEX_SLACK_DA
                    counts['used_delta_mismatch'] += abs(delta - expected) > INDEX_SLACK_DA
                    counts['reverse_index_residual'] += abs(product_mass - stored_mass - expected) > INDEX_SLACK_DA
                    counts['multifragment_products'] += len(Chem.GetMolFrags(product)) > 1
                counts['parents_with_valid_products'] += bool(parent_products)
        query_bands.append(dict(parent_mass_min=float(masses[band[0]]),
                                parent_mass_max=float(masses[band[-1]]), matches=dict(coverage)))
        print('parent mass stratum', len(query_bands), '/', BANDS,
              'parents', present['valid_parents'], 'seconds', round(time.monotonic() - start, 1), flush=True)

    evidence = {name: dict(declared_da=float(declared[name]), used_da=float(expected),
                           counts={field: int(value) for field, value in totals[name].items()},
                           observed_mass_deltas=[{'da': round(k * MASS_BIN_DA, 6), 'pairs': int(n)}
                                                 for k, n in histograms[name].most_common()])
                for name, expected in rules}
    print('mass delta evidence', report_summary(totals, histograms), flush=True)
    report = dict(rdkit=rdBase.rdkitVersion, pool_sha256=POOL_SHA256, seed=seed,
                  sample_per_band=sample_per_band, max_parents_per_rule_per_band=max_parents_per_band,
                  reactant_matches_product_cap=8, index_slack_da=INDEX_SLACK_DA,
                  parent_totals=dict(present), mass_bands=query_bands, corrections=corrections,
                  transform_count=len(rules), observed_transform_count=sum(bool(h) for h in histograms.values()),
                  rules=evidence)
    output.write_text(json.dumps(report, indent=2) + '\n')


def report_summary(counts, histograms):
    return dict(observed_rules=sum(bool(h) for h in histograms.values()),
                unobserved_rules=[name for name, values in histograms.items() if not values],
                mismatches={name: int(values['used_delta_mismatch']) for name, values in counts.items()
                            if values['used_delta_mismatch']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261125)
    parser.add_argument('--sample-per-band', type=int, default=2000)
    parser.add_argument('--max-parents-per-band', type=int, default=64)
    audit(**vars(parser.parse_args()))
