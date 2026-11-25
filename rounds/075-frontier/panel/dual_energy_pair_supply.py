"""Build training-only dual-polarity same-formula A/B pairs on Kaggle CPU."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


ENERGIES = (20, 40, 60)
ADDUCTS = ('[M+H]+', '[M-H]-')
PROTON = 1.007276466621
FOLDS = 5
PER_FORMULA_CAP = 32
COLUMNS = ('molecular_formula', 'inchikey14', 'normalized_smiles', 'ingest_lib',
           'instrument_type', 'adduct', 'collision_energy_ev', 'collision_energy_orig',
           'collision_energy_orig_units', 'num_peaks', 'precursor_mz')


def stable_hash(value):
    return sha256(f'dual-energy-pairs-v1/{value}'.encode()).digest()


def collision(row):
    if row['collision_energy_orig_units'] != 'eV':
        return None
    try:
        measured = abs(float(row['collision_energy_orig']))
    except (TypeError, ValueError):
        return None
    if measured not in ENERGIES or row['collision_energy_ev'] != [measured]:
        return None
    return int(measured)


def inventory(path):
    by_formula, counts = defaultdict(dict), Counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32768, columns=list(COLUMNS)):
        for row in batch.to_pylist():
            position = counts['rows_read']
            counts['rows_read'] += 1
            if (row['ingest_lib'], row['instrument_type']) != ('enveda-180', 'timsTOF') or row['adduct'] not in ADDUCTS:
                continue
            energy = collision(row)
            if energy is None:
                continue
            formula, key, smiles = (row[name] for name in ('molecular_formula', 'inchikey14', 'normalized_smiles'))
            precursor = row['precursor_mz']
            if not (formula and key and smiles and row['num_peaks'] and row['num_peaks'] >= 5
                    and precursor and math.isfinite(precursor) and precursor > PROTON):
                counts['invalid_input_metadata'] += 1
                continue
            adduct = row['adduct']
            neutral = precursor - PROTON if adduct == ADDUCTS[0] else precursor + PROTON
            record = by_formula[formula].setdefault((key, smiles), {})
            condition = (adduct, energy)
            record.setdefault(condition, (position, neutral))
            counts[f'{adduct}/{energy}eV'] += 1
    return by_formula, counts


def canonical_candidates(group, tautomer, fingerprints, counts):
    candidates = {}
    for (_, text), measured in group:
        if not {(adduct, 40) for adduct in ADDUCTS} <= measured.keys():
            continue
        molecule = Chem.MolFromSmiles(text)
        if molecule is None or Chem.GetFormalCharge(molecule) or len(Chem.GetMolFrags(molecule)) != 1:
            counts['invalid_or_nonneutral'] += 1
            continue
        Chem.RemoveStereochemistry(molecule)
        heavy = tuple(sorted(Counter(a.GetSymbol() for a in molecule.GetAtoms() if a.GetSymbol() != 'H').items()))
        try:
            molecule = tautomer.Canonicalize(molecule)
            native_key = Chem.MolToInchiKey(molecule)[:14]
            scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
            scaffold_key = (Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold))
                            if scaffold.GetNumAtoms() else 'acyclic')
        except (RuntimeError, ValueError):
            counts['normalization_failed'] += 1
            continue
        if len(native_key) != 14:
            counts['normalization_failed'] += 1
            continue
        formula = rdMolDescriptors.CalcMolFormula(molecule)
        mass = rdMolDescriptors.CalcExactMolWt(molecule)
        eligible = {cond: spectrum for cond, (spectrum, observed) in measured.items()
                    if abs(observed - mass) <= mass * 10e-6}
        if not {(adduct, 40) for adduct in ADDUCTS} <= eligible.keys():
            counts['precursors_outside_10ppm'] += 1
            continue
        key = (native_key, heavy)
        previous = candidates.get(key)
        if previous is not None:
            counts['normalization_aliases'] += 1
            for condition, row_index in eligible.items():
                previous['scans'][condition] = min(row_index, previous['scans'].get(condition, row_index))
            continue
        fold = int.from_bytes(stable_hash(scaffold_key)[:4], 'big') % FOLDS
        candidate = dict(smiles=Chem.MolToSmiles(molecule), key=native_key, formula=formula, mass=mass,
                         fold=fold, scaffold_sha256=sha256(scaffold_key.encode()).hexdigest(), scans=eligible,
                         fingerprint=fingerprints.GetFingerprint(molecule))
        candidates[key] = candidate
    return list(candidates.values())


def render_candidate(candidate):
    return {key: value for key, value in candidate.items() if key not in ('fingerprint', 'scans')} | {
        'train_row_0based': {f'{adduct}/{energy}eV': row for (adduct, energy), row in candidate['scans'].items()}}


def emit(groups):
    counts, formulas, chosen = Counter(), Counter(), []
    tautomer = rdMolStandardize.TautomerEnumerator()
    fingerprints = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    at_40 = {(adduct, 40) for adduct in ADDUCTS}
    for formula, raw in sorted(groups.items()):
        matched = [(key, scans) for key, scans in raw.items() if at_40 <= scans.keys()]
        if len({key for (key, _), _ in matched}) < 2:
            continue
        counts['metadata_formula_groups_with_two_raw_keys_dual_40'] += 1
        counts['formula_groups_sampled_to_cap'] += len(matched) > PER_FORMULA_CAP
        sampled = sorted(matched, key=lambda item: stable_hash(formula+'/'+item[0][0]+'/'+item[0][1]))[:PER_FORMULA_CAP]
        candidates = canonical_candidates(sampled, tautomer, fingerprints, counts)
        best = {}
        levels = set()
        for left, right in combinations(candidates, 2):
            if left['formula'] != right['formula'] or abs(left['mass'] - right['mass']) > 1e-6:
                continue
            common = left['scans'].keys() & right['scans'].keys()
            dual = [e for e in ENERGIES if {(adduct, e) for adduct in ADDUCTS} <= common]
            mode = f'dual_{len(dual)}_energy'
            levels.add(mode)
            counts[f'checked_{mode}_pairs'] += 1
            similarity = float(DataStructs.TanimotoSimilarity(left['fingerprint'], right['fingerprint']))
            counts[f'checked_{mode}_similarity_ge_0_6_pairs'] += similarity >= 0.6
            counts[f'checked_{mode}_similarity_ge_0_8_pairs'] += similarity >= 0.8
            if left['fold'] != right['fold']:
                counts['cross_fold_pairs_excluded_from_output'] += 1
                continue
            key = (left['fold'], mode)
            if key not in best or similarity > best[key][0]:
                best[key] = similarity, dual, left, right
        for level in levels:
            formulas[level] += 1
        for (fold, mode), (similarity, energies, left, right) in sorted(best.items()):
            chosen.append(dict(fold=fold, morgan_tanimoto=similarity, match=mode, matched_dual_energies=energies,
                               left=render_candidate(left), right=render_candidate(right)))
    return counts, formulas, chosen


def run(train, out):
    if rdBase.rdkitVersion != '2026.03.3':
        raise RuntimeError('The training identity must use the official contest RDKit')
    RDLogger.DisableLog('rdApp.*')
    groups, metadata = inventory(train)
    counts, formulas, rows = emit(groups)
    out.mkdir(parents=True, exist_ok=True)
    path = out/'dual_energy_training_pairs.jsonl'
    with path.open('w') as file:
        for row in rows:
            file.write(json.dumps(row, separators=(',', ':'))+'\n')
    summary = dict(scope='Competition training only. No held-out or answer input. No ranking score.',
                   metadata_counts=dict(metadata), sampled_pair_counts=dict(counts), sampled_formula_groups=dict(formulas),
                   pair_export_count=len(rows), pair_export_sha256=sha256(path.read_bytes()).hexdigest(),
                   export_by_fold_and_shared_energy={f'{f}/{m}':n for (f,m),n in Counter((r['fold'],r['match']) for r in rows).items()},
                   strict_both_polarities_2_or_3_energy_pairs=sum(len(r['matched_dual_energies'])>=2 for r in rows),
                   exported_hard_pairs_ge_0_6=sum(r['morgan_tanimoto']>=0.6 for r in rows),
                   per_formula_input_cap=PER_FORMULA_CAP, rdkit=rdBase.rdkitVersion,
                   sampling_note='One score-key-distinct pair per formula, scaffold fold and exact common dual-energy count.',
                   duplicate_scan_policy='Use the first parquet row per raw identity and condition, then apply the mass audit.',
                   training_parquet_bytes=train.stat().st_size,
                   allowed_cross_validation='Split by fold. Formulas may occur in multiple folds. Group by formula as well if formula transfer is the claim.')
    (out/'dual_energy_pair_supply.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(**vars(parser.parse_args()))
