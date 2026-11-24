"""Audit published MassSpecGym train/val/test identities against the external panel."""
import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re

import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize


ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_counts(formula):
    count = Counter()
    for atom, number in ELEMENT.findall(formula):
        if atom != "H":
            count[atom] += int(number or 1)
    return tuple(sorted(count.items()))


def run(labels, tables):
    if rdBase.rdkitVersion != "2026.03.3":
        raise RuntimeError("Use the competition RDKit build")
    RDLogger.DisableLog("rdApp.*")
    tautomer = rdMolStandardize.TautomerEnumerator()

    @lru_cache(maxsize=None)
    def identity(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, ()
        counts = Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        Chem.RemoveStereochemistry(mol)
        key = Chem.MolToInchiKey(tautomer.Canonicalize(mol)).split("-")[0]
        return key, tuple(sorted(counts.items()))

    targets, raw, compositions = set(), set(), set()
    for row in pd.read_parquet(labels).itertuples(index=False):
        key, counts = identity(row.truth_smiles)
        if key != row.truth_key:
            raise ValueError("Panel identity disagrees with the competition build")
        targets.add(key)
        raw.add(Chem.MolToInchiKey(Chem.MolFromSmiles(row.truth_smiles)).split("-")[0])
        compositions.add(counts)
    result = {}
    for table in tables:
        matches = defaultdict(Counter)
        total = 0
        with table.open() as stream:
            for r in csv.DictReader(stream, delimiter="\t"):
                total += 1
                if r["inchikey"][:14] not in raw and formula_counts(r["formula"]) not in compositions:
                    continue
                key, counts = identity(r["smiles"])
                if key in targets and counts in compositions:
                    matches[key][r["fold"]] += 1
        with table.open("rb") as stream:
            digest = sha256()
            for block in iter(lambda: stream.read(4 << 20), b""):
                digest.update(block)
        result[table.name] = {"sha256": digest.hexdigest(), "spectra": total, "matching_folds": matches}
    return {"rdkit": rdBase.rdkitVersion, "panel_identities": len(targets),
            "dataset": "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/tree/018a954/data",
            "scope": "Published MassSpecGym folds, exact normalized identity. Exposure in other pretrained weights and shared scaffolds requires separate evidence.",
            "tables": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--tables", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.labels, args.tables), indent=2) + "\n")
