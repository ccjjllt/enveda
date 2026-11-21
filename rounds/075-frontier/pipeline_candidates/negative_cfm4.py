"""Run stock CFM-ID 4 on the fixed negative-ion front-five formula groups."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from math import isclose
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd


INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')
PANEL = 'enveda-075b-external-massbank-adducts'
PIPELINE = 'enveda-075b-massbank-adducts-pipeline'
ASSETS = 'enveda-075b-cfm4-runtime'
RUNTIME = Path('/tmp/enveda_cfm4_runtime')
ENERGIES = ('energy0', 'energy1', 'energy2')
METHODS = ('original', 'native_cfm4')


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_runtime():
    receipt = json.loads(artifact(ASSETS, 'cfm4_runtime.json').read_text())
    archive = artifact(ASSETS, 'cfm4_runtime.tar.gz')
    if sha256(archive) != receipt['archive_sha256']:
        raise RuntimeError('Offline CFM runtime archive checksum mismatch')
    subprocess.run(['tar', '-xzf', str(archive), '-C', str(RUNTIME.parent)], check=True)
    files = [RUNTIME / receipt[key] for key in ('binary', 'params', 'config')]
    for key, path in zip(('binary', 'params', 'config'), files, strict=True):
        if sha256(path) != receipt[f'{key}_sha256']:
            raise RuntimeError(f'Packaged CFM {key} checksum mismatch')
    return [str(path) for path in files], receipt


def read_peaks(path):
    peaks = {}
    energy = None
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0].lower() in ENERGIES:
            energy = fields[0].lower()
            peaks[energy] = []
            continue
        if energy is not None and len(fields) >= 2:
            try:
                peaks[energy].append((float(fields[0]), float(fields[1])))
            except ValueError:
                continue
    if set(peaks) != set(ENERGIES):
        return None
    return {name: tuple(zip(*peaks[name])) if peaks[name] else ((), ()) for name in ENERGIES}


def predict(smiles, cli, params, config, env):
    with tempfile.TemporaryDirectory(prefix='cfm4_') as work:
        output = Path(work) / 'spectrum.log'
        try:
            result = subprocess.run([cli, smiles, '0.001', params, config, '0', str(output), '1', '1'],
                                    cwd=work, env=env, capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            return None, 'timeout'
        if result.returncode or not output.is_file():
            return None, 'prediction_failed'
        spectra = read_peaks(output)
        return spectra, 'ok' if spectra else 'missing_energy'


def read_panel():
    fields = ['molecule_id', 'adduct', 'ionization_mode', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']
    queries = pd.read_parquet(artifact(PANEL, 'pipeline_queries.parquet'), columns=fields)
    queries = queries.loc[queries['adduct'].eq('[M-H]-') & queries['ionization_mode'].eq('negative')]
    if len(queries) != 47 or not queries['molecule_id'].is_unique:
        raise ValueError('Expected the same 47 single-spectrum negative queries')
    fields = ['molecule_id', 'candidate_id', 'smiles', 'formula', 'rank']
    candidates = pd.read_parquet(artifact(PIPELINE, 'candidates.parquet'), columns=fields)
    candidates = candidates.loc[candidates['molecule_id'].isin(queries['molecule_id'])]
    if (set(candidates['molecule_id']) != set(queries['molecule_id'])
            or not candidates['rank'].between(1, 25).all()
            or candidates.duplicated(['molecule_id', 'rank']).any()):
        raise ValueError('The original final ranking does not align with the negative panel')
    rankings = {mid: list(frame.itertuples(index=False))
                for mid, frame in candidates.sort_values('rank', kind='stable').groupby('molecule_id')}
    return queries.set_index('molecule_id'), rankings


def eligible_slots(entries):
    formula = entries[0].formula
    if pd.isna(formula) or not formula:
        return []
    chosen = [i for i, item in enumerate(entries[:5]) if pd.notna(item.formula) and item.formula == formula]
    return chosen if len(chosen) >= 2 else []


def main():
    started = time.monotonic()
    queries, rankings = read_panel()
    slots = {mid: eligible_slots(entries) for mid, entries in rankings.items()}
    smi_list = sorted({rankings[mid][i].smiles for mid, positions in slots.items() for i in positions})
    print(f'Fixed {len(queries)} negative queries, {sum(bool(p) for p in slots.values())} eligible, '
          f'{sum(map(len, slots.values()))} candidate entries, {len(smi_list)} unique SMILES', flush=True)
    (cli, params, config), receipt = load_runtime()
    env = {**os.environ, 'LD_LIBRARY_PATH': str(RUNTIME / 'lib'), 'OMP_NUM_THREADS': '1'}
    prediction_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(4, len(os.sched_getaffinity(0)))) as pool:
        prediction = dict(zip(smi_list, pool.map(lambda smi: predict(smi, cli, params, config, env), smi_list),
                              strict=True))
    prediction_seconds = time.monotonic() - prediction_started
    print(f'CFM-ID unique structures {dict(Counter(status for _, status in prediction.values()))}, '
          f'{prediction_seconds:.1f}s', flush=True)

    report_rows, output, coverage = [], {}, Counter()
    for mid, entries in rankings.items():
        query = queries.loc[mid]
        observed = clean_peaks(query.ms2_mzs, query.ms2_normalized_intensities, query.precursor_mz)
        scores = {}
        for i in slots[mid]:
            item = entries[i]
            predicted, status = prediction[item.smiles]
            per_ce = {}
            if predicted is not None and len(observed[0]):
                cleaned = {name: clean_peaks(*predicted[name], query.precursor_mz) for name in ENERGIES}
                if any(len(masses) for masses, _ in cleaned.values()):
                    per_ce = {name: float(cosine(observed, cleaned[name])) for name in ENERGIES}
                    scores[i] = sum(per_ce.values()) / len(ENERGIES)
                else:
                    status = 'no_fragments_after_cleaning'
            elif not len(observed[0]):
                status = 'no_measured_fragments'
            coverage[status] += 1
            report_rows.append(dict(molecule_id=mid, candidate_id=item.candidate_id, original_rank=item.rank,
                                    formula=item.formula, status=status, mean_cosine=scores.get(i),
                                    **{f'cosine_{energy}ev': per_ce.get(name)
                                       for energy, name in zip((10, 20, 40), ENERGIES, strict=True)}))
        original = [item.candidate_id for item in entries]
        modified = original.copy()
        covered = 0 in scores and len(scores) > 1
        if covered:
            for position, candidate in zip(scores, sorted(scores, key=lambda i: -scores[i]), strict=True):
                modified[position] = original[candidate]
        output[mid] = dict(original=original, native_cfm4=modified, eligible=bool(slots[mid]), comparable=covered)

    labels = pd.read_parquet(artifact(PANEL, 'labels.parquet'), columns=['molecule_id', 'candidate_id'])
    labels = labels.loc[labels['molecule_id'].isin(output)].copy()
    if len(labels) != 47 or not labels['molecule_id'].is_unique:
        raise ValueError('Labels do not align with the fixed negative panel')
    for method in METHODS:
        labels[method] = [next((i for i, key in enumerate(output[mid][method], 1) if key == truth), 0)
                          for mid, truth in labels[['molecule_id', 'candidate_id']].itertuples(index=False, name=None)]
    labels['eligible'] = labels['molecule_id'].map({mid: row['eligible'] for mid, row in output.items()})
    labels['comparable'] = labels['molecule_id'].map({mid: row['comparable'] for mid, row in output.items()})
    metrics = summaries(labels, METHODS, 'original')
    original = metrics['original']
    if (original['rank1'], original['found_at_25']) != (17, 28) or not isclose(
        original['mrr_at_25'], 0.41061589120099756, abs_tol=1e-10
    ):
        raise ValueError(f'Original ranking does not reproduce the approved control: {original}')
    summary = dict(molecules=len(labels), eligible_molecules=int(labels['eligible'].sum()),
                   comparable_molecules=int(labels['comparable'].sum()), unique_smiles=len(smi_list),
                   candidate_coverage=dict(coverage), params_sha256=receipt['params_sha256'],
                   prediction_seconds=round(prediction_seconds, 2), seconds=round(time.monotonic() - started, 2),
                   methods=metrics, mrr_delta=metrics['native_cfm4']['mrr_at_25'] - original['mrr_at_25'])
    pd.DataFrame(report_rows).to_parquet(OUTPUT / 'negative_cfm4_scores.parquet', index=False)
    labels.to_csv(OUTPUT / 'negative_cfm4_ranks.csv', index=False)
    (OUTPUT / 'negative_cfm4_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
