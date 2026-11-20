"""Prepare fixed [M+H]+ queries and mass candidates on Kaggle."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")
COMPETITION = "enveda-casmi26-molecule-id-mass-spectra"
PANEL_SIZE = 80
MASS_PPM = 15
PROTON_MASS = 1.007276466621


def setup_rdkit():
    try:
        from importlib.metadata import version
        if version("rdkit") in ("2026.3.3", "2026.03.3"):
            return
    except ImportError:
        pass
    wheels = sorted(INPUT.rglob("rdkit-2026.3.3-*.whl")) + sorted(
        INPUT.rglob("rdkit-2026.03.3-*.whl")
    )
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheel = next((path for path in wheels if f"-{abi}-" in path.name), None)
    if wheel is None:
        raise RuntimeError(f"No RDKit 2026.03.3 wheel for {abi} in {wheels}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", str(wheel)],
        check=True,
    )


def init_chemistry():
    global Chem, tautomer, exact_mass, formula, scaffold_smiles
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    RDLogger.DisableLog("rdApp.*")
    tautomer = rdMolStandardize.TautomerEnumerator()
    exact_mass = rdMolDescriptors.CalcExactMolWt
    formula = rdMolDescriptors.CalcMolFormula
    scaffold_smiles = MurckoScaffoldSmiles


def neutral_molecule(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or Chem.GetFormalCharge(molecule) or "." in smiles:
        return None
    return molecule


def measure_mass(item):
    raw_key, smiles = item
    molecule = neutral_molecule(smiles)
    if molecule is not None:
        return raw_key, smiles, exact_mass(molecule)


def describe_structure(item):
    raw_key, smiles = item
    molecule = neutral_molecule(smiles)
    if molecule is None:
        return None
    Chem.RemoveStereochemistry(molecule)
    molecule = tautomer.Canonicalize(molecule)
    key = Chem.MolToInchiKey(molecule)[:14]
    if not key:
        return None
    return (
        raw_key, key, smiles, scaffold_smiles(mol=molecule),
        exact_mass(molecule), formula(molecule),
    )


def index_train(train, cross_source=False):
    structures = {}
    observations = defaultdict(lambda: defaultdict(dict))
    sources = defaultdict(set)
    spectra_columns = [
        "inchikey14", "adduct", "precursor_mz", "collision_energy_ev",
        "instrument_type", "ingest_lib",
    ]
    for group in range(train.num_row_groups):
        identities = train.read_row_group(group, columns=["inchikey14", "normalized_smiles"])
        for raw_key, smiles in zip(
            identities["inchikey14"].to_pylist(), identities["normalized_smiles"].to_pylist()
        ):
            if raw_key and smiles:
                structures.setdefault(raw_key, smiles)

        table = train.read_row_group(group, columns=spectra_columns)
        protonated = pc.equal(table["adduct"], "[M+H]+")
        if cross_source:
            source_table = table.select(["inchikey14", "ingest_lib"]).filter(protonated)
            for key, source in zip(source_table[0].to_pylist(), source_table[1].to_pylist()):
                if key and source:
                    sources[key].add(str(source).strip().casefold())
        eligible = pc.and_(protonated, pc.equal(pc.list_value_length(table["collision_energy_ev"]), 1))
        indices = pc.indices_nonzero(eligible)
        for row_index, row in zip(indices.to_pylist(), table.take(indices).to_pylist()):
            precursor = row["precursor_mz"]
            energy = row["collision_energy_ev"][0]
            raw_key = row["inchikey14"]
            if raw_key and precursor and energy is not None and math.isfinite(energy):
                method = (row["ingest_lib"], row["instrument_type"])
                observations[raw_key][method].setdefault(
                    round(energy, 3), (group, row_index, precursor)
                )
    if cross_source:
        observations = {
            key: {method: spectra for method, spectra in methods.items() if method[0]}
            for key, methods in observations.items() if len(sources.get(key, ())) >= 2
        }
        observations = {key: methods for key, methods in observations.items() if methods}
    return structures, observations


def exclude_queries(train, path):
    positions = defaultdict(list)
    for query_id in pd.read_parquet(path, columns=["query_id"])["query_id"]:
        group, index = map(int, query_id.split("-"))
        positions[group].append(index)
    return {key for group, indices in positions.items()
            for key in train.read_row_group(group, columns=["inchikey14"]).take(indices).column(0).to_pylist()}


def choose_queries(observations, structure_keys, lookup_label, masses, candidates_in_window, panel_size, excluded_identities):
    rng = np.random.default_rng(7502)
    raw_keys = sorted(observations.keys() & structure_keys)
    selected = {}
    scaffolds = set()
    for raw_key in rng.permutation(raw_keys):
        groups = sorted(observations[raw_key].values(), key=len, reverse=True)
        if len(groups[0]) < 2:
            continue
        energies = sorted(groups[0])
        if len(energies) > 4:
            energies = [energies[position] for position in np.linspace(0, len(energies) - 1, 4).round().astype(int)]
        spectra = [groups[0][energy] for energy in energies]
        mass = float(np.median([row[2] for row in spectra])) - PROTON_MASS
        tolerance = mass * MASS_PPM / 1_000_000
        lo, hi = masses.searchsorted([mass - tolerance, mass + tolerance])
        if hi - lo < 2:
            continue
        info = lookup_label(raw_key)
        if (info is None or info["candidate_id"] in excluded_identities
                or not info["scaffold"] or info["scaffold"] in scaffolds):
            continue
        nearby = candidates_in_window(lo, hi)
        if len(nearby) < 2:
            continue
        molecule_id = f"U{len(selected) + 1:03}"
        selected[molecule_id] = (info, mass, nearby, spectra)
        scaffolds.add(info["scaffold"])
        if len(selected) == panel_size:
            break
    return selected, scaffolds


def load_queries(train, selected, test_columns):
    desired_by_group = defaultdict(dict)
    for molecule_id, (_, _, _, spectra) in selected.items():
        for group, row_index, _ in spectra:
            desired_by_group[group][row_index] = molecule_id
    queries, pipeline_queries = [], []
    source_columns = [name for name in test_columns if name not in ("molecule_id", "spectrum_id")]
    for group, desired in sorted(desired_by_group.items()):
        indices = sorted(desired)
        rows = train.read_row_group(group, columns=source_columns).take(indices).to_pylist()
        for index, row in zip(indices, rows):
            molecule_id = desired[index]
            query_id = f"{group}-{index}"
            queries.append(dict(
                query_id=query_id, molecule_id=molecule_id,
                mz=row["ms2_mzs"], intensity=row["ms2_normalized_intensities"],
                precursor_mz=row["precursor_mz"], adduct=row["adduct"],
                collision_energy=row["collision_energy_ev"][0],
                collision_energy_orig=row["collision_energy_orig"],
                collision_energy_orig_units=row["collision_energy_orig_units"],
                instrument_type=row["instrument_type"], polarity=row["ionization_mode"],
            ))
            pipeline_queries.append(dict(molecule_id=molecule_id, spectrum_id=query_id, **row))
    return pd.DataFrame(queries), pd.DataFrame(pipeline_queries, columns=test_columns)


def main(mass_only=False, cross_source=False, exclude=None):
    setup_rdkit()
    source = next(
        path for path in INPUT.rglob("train.parquet") if path.parent.name.lower() == COMPETITION
    ).parent
    train = pq.ParquetFile(source / "train.parquet")
    test_columns = pq.ParquetFile(source / "test.parquet").schema_arrow.names
    structures, observations = index_train(train, cross_source=cross_source)
    excluded_identities = set()
    if exclude is not None:
        for key in exclude_queries(train, exclude):
            observations.pop(key, None)
        excluded_identities.update(pd.read_parquet(exclude.parent / "labels.parquet", columns=["candidate_id"])["candidate_id"])
    print(f"{len(structures)} raw structure identities, {len(observations)} with eligible spectra", flush=True)
    structure_columns = ["raw_key", "candidate_id", "smiles", "scaffold", "mass", "formula"]
    columns = ["raw_key", "smiles", "mass"] if mass_only else structure_columns
    describe = measure_mass if mass_only else describe_structure
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1), initializer=init_chemistry) as workers:
        results = workers.map(describe, structures.items(), chunksize=512 if mass_only else 128)
        table = pd.DataFrame.from_records((row for row in results if row is not None), columns=columns)

    if mass_only:
        init_chemistry()
        pool = table.sort_values("mass", kind="stable").reset_index(drop=True)
        normalize = lru_cache(maxsize=8192)(describe_structure)

        def lookup_label(key):
            result = normalize((key, structures[key]))
            return dict(zip(structure_columns, result)) if result else None

        def candidates_in_window(lo, hi):
            rows = pool.iloc[lo:hi][["raw_key", "smiles"]].itertuples(index=False, name=None)
            result = (normalize(row) for row in rows)
            canonical = pd.DataFrame.from_records((row for row in result if row), columns=structure_columns)
            return canonical.drop_duplicates("candidate_id").sort_values(["mass", "candidate_id"])
    else:
        pool = table.drop_duplicates("candidate_id").sort_values(["mass", "candidate_id"]).reset_index(drop=True)
        labels_by_key = table.set_index("raw_key")[["candidate_id", "scaffold"]].to_dict("index")
        lookup_label = labels_by_key.get

        def candidates_in_window(lo, hi):
            return pool.iloc[lo:hi]

    masses = pool["mass"].to_numpy()
    print(f"{len(pool)} eligible structure records in the mass index", flush=True)
    selected, scaffolds = choose_queries(observations, set(table["raw_key"]), lookup_label, masses,
                                        candidates_in_window, 120 if cross_source else PANEL_SIZE, excluded_identities)
    if not selected:
        raise RuntimeError("No multi-energy [M+H]+ structures with competing candidates")
    queries, pipeline_queries = load_queries(train, selected, test_columns)
    panel_name = "S_cross_source" if cross_source else "mass_only" if mass_only else "U"
    labels, candidates = [], []
    for molecule_id, (info, mass, nearby, _) in selected.items():
        labels.append(dict(
            molecule_id=molecule_id, candidate_id=info["candidate_id"],
            scaffold=info["scaffold"], panel=panel_name,
        ))
        nearby = nearby[["candidate_id", "smiles", "formula", "mass"]].copy()
        nearby["mass_error_ppm"] = 1e6 * (nearby["mass"] - mass) / mass
        nearby = nearby.iloc[np.argsort(np.abs(nearby["mass_error_ppm"].to_numpy()), kind="stable")[:25]]
        nearby["molecule_id"] = molecule_id
        nearby["rank"] = np.arange(1, len(nearby) + 1)
        candidates.append(nearby.drop(columns="mass"))
    labels = pd.DataFrame(labels)
    candidates = pd.concat(candidates, ignore_index=True)
    queries.to_parquet(OUTPUT / "queries.parquet", index=False)
    candidates.to_parquet(OUTPUT / "candidates.parquet", index=False)
    labels.to_parquet(OUTPUT / "labels.parquet", index=False)
    excluded_references = None
    if not mass_only:
        held_out = table[table["scaffold"].isin(scaffolds)]
        held_out = table[table["candidate_id"].isin(held_out["candidate_id"])]
        held_out[["raw_key", "candidate_id"]].to_parquet(OUTPUT / "heldout_structures.parquet", index=False)
        pipeline_queries.to_parquet(OUTPUT / "pipeline_queries.parquet", index=False)
        excluded_references = len(held_out)
    hits = labels.merge(candidates, on=["molecule_id", "candidate_id"], how="left")
    report = {
        "molecules": len(labels), "spectra": len(queries),
        "panel": panel_name,
        "excluded_reference_identities": excluded_references,
        "median_candidates": float(candidates.groupby("molecule_id").size().median()),
        "recall_at_25": float(hits["rank"].notna().mean()),
        "mass_order_mrr_at_25": float((1 / hits["rank"]).fillna(0).mean()),
    }
    (OUTPUT / "panel.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass-only", action="store_true", help="Normalize only labels and mass-window candidates")
    parser.add_argument("--cross-source", action="store_true", help="Select structures present in multiple library sources")
    parser.add_argument("--exclude", type=Path, help="Exclude the raw identities from another queries.parquet file")
    args = parser.parse_args()
    main(mass_only=args.mass_only or args.cross_source, cross_source=args.cross_source, exclude=args.exclude)
