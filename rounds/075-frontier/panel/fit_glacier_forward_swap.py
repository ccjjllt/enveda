"""Score the frozen ICEBERG hard pairs with GLACIER and retain every requested prediction."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import monotonic
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from fit_forward_swap import FEATURES, ION, score_pair, summarize
from forward_ce import clean_peaks
from np_glacier_reference import RUNNER_SHA, WEIGHTS_SHA, checked
from read_dual_training_scans import load_pairs


def load_glacier(inputs, device):
    runner, = inputs.rglob('gl_runner.py')
    ice_runner, = inputs.rglob('ice_runner.py')
    package = checked(runner, RUNNER_SHA).parent
    checked(package/'ckpt/glacier.pt', WEIGHTS_SHA)
    sys.path.insert(0, str(package))
    import gl_runner as gl

    log = gl.Log()
    rdkit = gl.setup_env(SimpleNamespace(pkg=str(package), site='/tmp/np_glacier_site',
                                         wheels=str(ice_runner.parent/'wheels'), any_rdkit=False), log)
    import torch

    if device == 'cuda:0' and not torch.cuda.is_available():
        raise RuntimeError('The GLACIER comparison requires the configured Kaggle T4')
    torch.set_num_threads(2)
    predictor = gl.Predictor(gl.load_model(str(package), device, log), device, 24, log)
    runtime = dict(device=device, rdkit=rdkit, torch=str(torch.__version__))
    if device == 'cuda:0':
        runtime['gpu'] = torch.cuda.get_device_name(0)
    return gl, predictor, runtime


def write_jsonl(path, records):
    with path.open('wb') as out:
        for row in records:
            out.write((json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n').encode())
    return sha256(path.read_bytes()).hexdigest()


def store_predictions(predictor, conditions, path):
    import torch

    ordered = sorted(conditions)
    rows, peaks = [], {}
    for start in range(0, len(ordered), 128):
        jobs = ordered[start:start + 128]
        with torch.inference_mode():
            if not predictor.predict(set(jobs), float('inf')):
                raise RuntimeError('The native prediction loop exited before processing this chunk')
        for job in jobs:
            smiles, adduct, instrument, energy = job
            value = predictor.cache[job]
            peaks[job] = value
            rows.append(dict(smiles=smiles, adduct=adduct, instrument=instrument, energy_ev=energy,
                             available=value is not None, mz=None if value is None else value[0],
                             intensity=None if value is None else value[1]))
        predictor.cache.clear()
        predictor.pcache.clear()
    schema = pa.schema([('smiles', pa.string()), ('adduct', pa.string()), ('instrument', pa.string()),
                        ('energy_ev', pa.float64()), ('available', pa.bool_()),
                        ('mz', pa.list_(pa.float64())), ('intensity', pa.list_(pa.float64()))])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return peaks


def run(train, pairs, receipt, out, per_fold, device):
    start = monotonic()
    chosen, scans = load_pairs(pairs, receipt, train, per_fold=per_fold, minimum_similarity=0.6)
    out.mkdir(parents=True, exist_ok=True)
    gl, predictor, runtime = load_glacier(Path('/kaggle/input'), device)
    instrument = gl.instr_token({'instrument': 'timsTOF'})
    if instrument != 'QTOF' or ION not in gl.COV:
        raise ValueError('The native model does not cover the frozen positive timsTOF conditions')
    supported = {name: gl.smiles_ok(name)[0] for name in
                 sorted({pair[side]['smiles'] for pair in chosen for side in ('left', 'right')})}
    selection, planned, jobs = [], [], set()
    for pair in chosen:
        sides = [pair[side] for side in ('left', 'right')]
        energies = pair['matched_dual_energies']
        row = dict(fold=pair['fold'], tanimoto=pair['morgan_tanimoto'], structures=[s['key'] for s in sides],
                   smiles=[s['smiles'] for s in sides], measured_energies_ev=energies,
                   positive_train_rows=[[s['train_row_0based'][f'{ION}/{en}eV'] for en in energies] for s in sides])
        selection.append(row)
        if len(energies) < 2:
            row['status'] = 'single_measured_energy'
        elif not all(supported[s['smiles']] for s in sides):
            row['status'] = 'unsupported_structure'
        else:
            planned.append((pair, row))
            jobs.update((s['smiles'], ION, instrument, float(en)) for s in sides for en in energies)
    print('GLACIER hard pairs', len(planned), 'unique model conditions', len(jobs), flush=True)
    peak_file = out/'gl_forward_swap_prediction_peaks.parquet'
    peaks = store_predictions(predictor, jobs, peak_file)
    samples, evidence = [], []
    for pair, row in planned:
        energies = row['measured_energies_ev']
        names = row['smiles']
        if any(peaks[(name, ION, instrument, float(en))] is None for name in names for en in energies):
            row['status'] = 'missing_model_prediction'
            continue
        precursor = min(scans[i]['precursor_mz'] for side_rows in row['positive_train_rows'] for i in side_rows)
        predicted = {name: {en: clean_peaks(*peaks[(name, ION, instrument, float(en))], precursor)
                            for en in energies} for name in names}
        values = score_pair(pair, scans, gl, predicted)
        if values is None:
            row['status'] = 'unusable_peak_series'
            continue
        row['status'] = 'scored'
        samples.append((pair, values))
        evidence.append({k: row[k] for k in ('fold', 'tanimoto', 'structures', 'measured_energies_ev')}
                        | {'oriented_scores': values.tolist()})
    feature_sha = write_jsonl(out/'gl_forward_swap_features.jsonl', evidence)
    selection_sha = write_jsonl(out/'gl_forward_swap_selection.jsonl', selection)
    result = dict(scope='Competition training, same-formula hard pairs, positive ion only',
                  pair_export_sha256=json.loads(receipt.read_bytes())['pair_export_sha256'],
                  selected_pairs=len(chosen), planned_pairs=len(planned), scored_pairs=len(samples),
                  pairs_per_fold=per_fold, selected_pair_status=dict(Counter(row['status'] for row in selection)),
                  selected_per_fold=dict(Counter(row['fold'] for row in selection)),
                  shared_instrument=instrument, adduct=ION, measured_energy_ev=[20, 40, 60],
                  model_conditions=len(jobs), missing_conditions=sum(v is None for v in peaks.values()),
                  model_predictions=predictor.n_pred, prediction_failures=predictor.n_fail,
                  featurization_failures=predictor.n_feat_fail, model_sha256=WEIGHTS_SHA, runner_sha256=RUNNER_SHA,
                  runtime=runtime, features=FEATURES, evidence_sha256=feature_sha, selection_sha256=selection_sha,
                  prediction_peaks_sha256=sha256(peak_file.read_bytes()).hexdigest(),
                  paired_evaluation=summarize(samples), elapsed_seconds=monotonic()-start)
    (out/'gl_forward_swap_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['paired_evaluation']['models']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'pairs', 'receipt', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--per-fold', type=int, default=48)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    run(**vars(parser.parse_args()))
