"""Evaluate final rankings against the privately retained E180 normalized structure keys."""
import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPECTRAL_FIELDS = ('precursor_mz', 'base_peak_intensity', 'adduct', 'ionization_mode',
                   'instrument_type', 'collision_energy_orig', 'collision_energy_orig_units',
                   'collision_energy_ev', 'ms2_mzs', 'ms2_normalized_intensities')


def scan_digest(values):
    metadata, arrays = values[:7], values[7:]
    digest = sha256(json.dumps(list(metadata), separators=(',', ':')).encode())
    for array in arrays:
        data = np.asarray(array, dtype='<f8')
        digest.update(len(data).to_bytes(8, 'big'))
        digest.update(data.tobytes())
    return digest.hexdigest()


def blind_labels(blind_path, raw_path):
    def signatures(path):
        scans = pd.read_parquet(path, columns=['molecule_id', *SPECTRAL_FIELDS])
        groups = scans.groupby('molecule_id', sort=False)
        if len(scans) != 299 or len(groups) != 121:
            raise ValueError('Expected 299 spectra for 121 structures.')
        return {tuple(sorted(scan_digest(values) for values in
                             group[list(SPECTRAL_FIELDS)].itertuples(index=False, name=None))): molecule
                for molecule, group in groups}

    blind, raw = signatures(blind_path), signatures(raw_path)
    if len(blind) != 121 or blind.keys() != raw.keys():
        raise ValueError('Could not link the two complete sets of measured spectra uniquely.')
    return pd.Series({molecule: raw[signature] for signature, molecule in blind.items()},
                     name='truth_key')


def candidate_key_function():
    from rdkit import Chem, rdBase
    from rdkit.Chem.MolStandardize import rdMolStandardize

    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError(f'Use the evaluation RDKit version 2026.03.3, found {rdBase.rdkitVersion}.')
    tautomer = rdMolStandardize.TautomerEnumerator()

    @lru_cache(maxsize=10000)
    def key(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToInchiKey(tautomer.Canonicalize(mol))[:14]

    return key


def ranks(directory, receipt, labels, key):
    run = json.loads((directory / 'e180_ranked_input.json').read_text())
    if run['inference'] != receipt:
        raise ValueError(f'{directory} did not use this panel input.')
    final = directory / 'submission.csv'
    if sha256(final.read_bytes()).hexdigest() != run['submission_sha256']:
        raise ValueError(f'{directory} final rankings differ from the completed run.')
    frame = pd.read_csv(final, dtype=str, keep_default_na=False)
    if len(frame) != len(labels) or not frame.molecule_id.is_unique or set(frame.molecule_id) != set(labels.index):
        raise ValueError(f'{directory} final submission has different or duplicate identities.')
    result = {}
    for row in frame.itertuples(index=False):
        candidates = [s.strip() for s in row.smiles.split(';') if s.strip()]
        if len(candidates) > 25:
            raise ValueError(f'Invalid candidate list for {row.molecule_id}')
        result[row.molecule_id] = next((i for i, smiles in enumerate(candidates, 1)
                                        if key(smiles.strip()) == labels[row.molecule_id]), None)
    return pd.Series(result, dtype='Int64')


def compare(receipt, labels, control, expanded, output, rescued_keys=None):
    if len(labels) != receipt['molecules'] or not labels.index.is_unique or not labels.is_unique:
        raise ValueError('Invalid scoring identity map.')
    key = candidate_key_function()
    frame = pd.DataFrame({arm: ranks(path, receipt, labels, key)
                          for arm, path in [('control', control), ('expanded', expanded)]})
    frame.index.name = 'molecule_id'
    rr = frame.rdiv(1).fillna(0)
    delta = rr.expanded - rr.control
    frame['delta_rr'] = delta
    review = dict(molecules=len(frame), mrr25=rr.mean().to_dict(),
                  hits25=frame[['control', 'expanded']].count().to_dict(),
                  top5=frame[['control', 'expanded']].le(5).sum().to_dict(),
                  top1=frame[['control', 'expanded']].eq(1).sum().to_dict(),
                  delta_mrr25=float(delta.mean()), improved=int(delta.gt(0).sum()),
                  worsened=int(delta.lt(0).sum()),
                  top1_gained=int((frame.expanded.eq(1).fillna(False) &
                                   ~frame.control.eq(1).fillna(False)).sum()),
                  top1_lost=int((frame.control.eq(1).fillna(False) &
                                 ~frame.expanded.eq(1).fillna(False)).sum()))
    if rescued_keys is not None:
        if len(rescued_keys) != 15 or len(set(rescued_keys)) != 15 or not set(rescued_keys) <= set(labels):
            raise ValueError('The source experiment must supply all 15 unique recovered structures.')
        review['previous_15'] = []
        inverse = {key: molecule for molecule, key in labels.items()}
        audit_file = expanded / 'pc_first_formula_audit.json'
        applied = json.loads(audit_file.read_text())['applied_queries'] if audit_file.exists() else None
        review['expansion_applied_queries'] = len(applied) if applied is not None else None
        for i, identity in enumerate(rescued_keys, 1):
            molecule = inverse[identity]
            record = frame.loc[molecule]
            patch = (applied or {}).get(molecule)
            pubchem_rank = next((rank for rank, smiles in enumerate(
                patch.get('ranked_smiles', []), 1) if key(smiles) == identity), None) if patch else None
            review['previous_15'].append(dict(
                case=i, expansion_applied=None if applied is None else patch is not None,
                expanded_pubchem_rank=pubchem_rank,
                baseline_rank=None if pd.isna(record.control) else int(record.control),
                expanded_rank=None if pd.isna(record.expanded) else int(record.expanded)))
        review['previous_15_expanded_rank_counts'] = dict(Counter(
            str(record['expanded_rank'] or '>25') for record in review['previous_15']))
    if output:
        frame.to_csv(output)
    return review


def score(scoring, control, expanded, output):
    receipt = json.loads((scoring / 'e180_scoring_input.json').read_text())['inference']
    labels = pd.read_parquet(scoring / 'blind_labels.parquet').set_index('molecule_id').truth_key
    return compare(receipt, labels, control, expanded, output)


def score_kaggle():
    import importlib.metadata
    import subprocess
    import sys

    inputs, output = Path('/kaggle/input'), Path('/kaggle/working')
    try:
        version = importlib.metadata.version('rdkit')
    except importlib.metadata.PackageNotFoundError:
        version = ''
    if version.replace('.03.', '.3.') != '2026.3.3':
        tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
        wheel = next(inputs.rglob(f'rdkit-2026.3.3-{tag}-*.whl'))
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', str(wheel)], check=True)
    manifest, = inputs.rglob('e180_blind_input.json')
    raw, = inputs.rglob('e180_scan_pipeline_queries_ce.parquet')
    reserve, = inputs.rglob('e180_reserve_first_formula_c41_summary.json')
    blind = manifest.parent / 'test.parquet'
    receipt = json.loads(manifest.read_text())
    if receipt['sha256']['test.parquet'] != sha256(blind.read_bytes()).hexdigest():
        raise ValueError('The blind query file does not match the input receipt.')
    labels = blind_labels(blind, raw)
    ranks_by_arm = {}
    for run in inputs.rglob('e180_ranked_input.json'):
        arm = json.loads(run.read_text())['arm']
        if arm in ranks_by_arm:
            raise ValueError(f'Duplicate result for {arm}.')
        ranks_by_arm[arm] = run.parent
    report = compare(receipt, labels, ranks_by_arm['baseline'], ranks_by_arm['expanded'],
                     output / 'e180_pair_ranks.csv',
                     json.loads(reserve.read_text())['expanded_vs_control']['open']['top25_gains'])
    for arm, directory in ranks_by_arm.items():
        completed = json.loads((directory / 'e180_ranked_input.json').read_text())
        report.setdefault('runs', {})[arm] = {key: value for key, value in completed.items() if key != 'inference'}
    (output / 'e180_pair_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('scoring', 'control', 'expanded'):
        parser.add_argument('--'+arg, type=Path)
    parser.add_argument('--per-query', type=Path)
    args = parser.parse_args()
    paths = (args.scoring, args.control, args.expanded)
    if all(paths):
        report = score(*paths, args.per_query)
    elif any(paths) or args.per_query:
        parser.error('Supply scoring, control, and expanded for local execution.')
    else:
        report = score_kaggle()
    print(json.dumps(report, indent=2))
