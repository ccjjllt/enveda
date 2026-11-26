"""Replay held-out fragment reconstruction on frozen GLACIER hard pairs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path
from time import monotonic

from build_fragment_reliability import COMPETITION, GL, HERE, PAIRS, READER_SHA


SLUG = 'enveda-075-fragment-reconstruction-gl'
SUMMARY = 'fragment_reconstruction_receipt.json'
PER_PAIR = 'fragment_reconstruction_per_pair.jsonl'
ARMS = ('cosine', 'heldout_template', 'heldout_with_common')
CONTROLS = ('measured', 'mismatch')


def comparison(observed, modeled, precursor, energies, expected):
    import numpy as np
    from fragment_reconstruction import ordered_scores

    values = {name: np.asarray(matrix, dtype=float)
              for name, matrix in ordered_scores(observed, modeled, precursor, energies).items()}
    swapped = {name: np.asarray(matrix, dtype=float)
               for name, matrix in ordered_scores(observed[::-1], modeled, precursor, energies).items()}
    if set(values) != set(ARMS) or values.keys() != swapped.keys():
        raise ValueError('Three predeclared reconstruction arms required')
    for arm, matrix in values.items():
        if matrix.shape != (2, 2) or not np.isfinite(matrix).all() or swapped[arm].shape != (2, 2):
            raise ValueError(f'Invalid paired scores for {arm}')
        if not np.allclose(swapped[arm], matrix[::-1], atol=1e-10, rtol=0):
            raise ValueError(f'Swapping the measured sides changed the {arm} score function')
    difference = float(np.max(np.abs(values['cosine'] - expected)))
    if difference > 1e-12:
        raise ValueError(f'Native GLACIER cosine failed the exact archived replay: {difference}')
    return {arm: matrix.tolist() for arm, matrix in values.items()}, difference


def control_spectra(case, donor):
    keys = '/'.join((*case['key'][1], *donor['key'][1]))
    ordered = donor['observed'][::-1] if sha256(keys.encode()).digest()[0] & 1 else donor['observed']
    indices = [donor['energies'].index(energy) for energy in case['energies']]
    return [[side[i] for i in indices] for side in ordered]


def summarize(rows):
    import numpy as np
    from fit_fragment_reliability import summarize as pair_metrics

    if not rows:
        return dict(pairs=0)
    metrics = {}
    for control in CONTROLS:
        metrics[control] = {}
        for arm in ARMS:
            tensor = np.asarray([row['scores'][control][arm] for row in rows])
            metrics[control][arm] = pair_metrics(tensor[:, :, 0] - tensor[:, :, 1])
    for control in CONTROLS:
        stock = metrics[control]['cosine']['mean_mrr_at_two']
        for result in metrics[control].values():
            result['delta_mrr2_vs_cosine'] = result['mean_mrr_at_two'] - stock
    return dict(pairs=len(rows), **metrics)


def run(train, pairs, receipt, gl, out):
    import numpy as np
    from fit_fragment_reliability import FILES, GL_SUMMARY_SHA, PAIR_EXPORT_SHA, load, mismatch

    started = monotonic()
    cases, archived_error = load(train, pairs, receipt, gl)
    controls = mismatch(cases)
    indexed = {case['key']: case for case in cases}
    if len(cases) != 239 or set(controls) != indexed.keys():
        raise ValueError('The archived donor mechanism did not cover all 239 hard pairs')
    rows, errors = [], dict.fromkeys(CONTROLS, 0.)
    for case in cases:
        negative, donor_key, delta = controls[case['key']]
        observe = {'measured': (case['observed'], case['data']),
                   'mismatch': (control_spectra(case, indexed[donor_key]), negative)}
        scores = {}
        for name, (spectra, reference) in observe.items():
            expected = np.array([[np.mean([value for value, _ in scan]) for scan in row] for row in reference])
            scores[name], error = comparison(spectra, case['modeled'], case['precursor'], case['energies'], expected)
            errors[name] = max(errors[name], error)
        rows.append(dict(fold=case['fold'], structures=case['key'][1], energy_ev=case['energies'],
                         tanimoto=case['similarity'], positive_train_rows=case['row_ids'],
                         donor_structures=donor_key[1], donor_precursor_delta_da=delta, scores=scores))
    out.mkdir(parents=True, exist_ok=True)
    path = out / PER_PAIR
    path.write_text(''.join(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n' for row in rows))
    deltas = [row['donor_precursor_delta_da'] for row in rows]
    result = dict(status='complete', scope='Frozen positive Enveda-180 same-formula hard pairs, MRR@2',
                  archive=GL, archive_receipt_sha256=GL_SUMMARY_SHA, input_sha256=FILES,
                  pair_export_sha256=PAIR_EXPORT_SHA, new_predictor_requests=0, reference_pairs=len(cases),
                  measured_and_mismatch_pairs=len(rows), archived_cosine_margin_max_error=archived_error,
                  exact_expected_cosine_max_error=errors, verified_measured_side_swaps=len(rows) * len(CONTROLS),
                  wrong_scan_control='Nearest precursor independent pair in the same fold and measured eV; matched instrument/adduct; no shared identity or scaffold',
                  donor_precursor_delta_da=dict(median=float(np.median(deltas)), maximum=max(deltas)),
                  outcomes=summarize(rows), by_fold={str(fold): summarize([r for r in rows if r['fold'] == fold])
                                                     for fold in range(5)},
                  similarity_at_least_080=summarize([r for r in rows if r['tanimoto'] >= 0.8]),
                  within_5da_control=summarize([r for r in rows if r['donor_precursor_delta_da'] <= 5.0]),
                  per_pair_sha256=sha256(path.read_bytes()).hexdigest(), elapsed_seconds=monotonic() - started)
    (out / SUMMARY).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['outcomes'], allow_nan=False), flush=True)


def build(out):
    common = run_path(str(HERE.parent / 'virtual_np' / 'build.py'))
    reader = (HERE / 'read_dual_training_scans.py').read_bytes()
    if sha256(reader).hexdigest() != READER_SHA:
        raise ValueError('Historical training scan reader changed')
    # Only the runtime half of this entry point is needed in Kaggle.
    source = Path(__file__).read_text()
    runtime = ('from hashlib import sha256\nimport json\nfrom time import monotonic\n'
               + f'GL = {GL!r}\n' + source[source.index('SLUG = '):source.index('\ndef build(')])
    modules = {'run_reconstruction.py': runtime.encode(), 'read_dual_training_scans.py': reader}
    for name in ('fit_fragment_reliability.py', 'fragment_reconstruction.py'):
        modules[name] = (HERE / name).read_bytes()
    for name, raw in modules.items():
        compile(raw, name, 'exec')
    payload = {name: common['pack'](raw) for name, raw in modules.items()}
    code = f'''
import base64, os, sys, tempfile, zlib
from hashlib import sha256
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
inputs = Path('/kaggle/input')
def mounted(name, slug):
    path, = (p for p in inputs.rglob(name) if slug.lower() in (part.lower() for part in p.parts))
    return path

train = mounted('train.parquet', {COMPETITION!r})
receipt = mounted('dual_energy_pair_supply.json', {PAIRS.split('/')[1]!r})
gl = mounted('gl_forward_swap_summary.json', {GL.split('/')[1]!r}).parent
with tempfile.TemporaryDirectory(prefix='gl-fragment-reconstruction-') as tmp:
    for name, (encoded, digest) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(encoded))
        if sha256(raw).hexdigest() != digest:
            raise ValueError(f'Embedded scoring source changed: {{name}}')
        Path(tmp, name).write_bytes(raw)
    sys.path.insert(0, tmp)
    from run_reconstruction import run
    run(train, receipt.with_name('dual_energy_training_pairs.jsonl'), receipt, gl, Path('/kaggle/working'))
'''
    compile(code, SLUG, 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='run.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, docker_image=common['IMAGE'],
                dataset_sources=[], kernel_sources=[PAIRS, GL], competition_sources=[COMPETITION])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'run.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return sha256(code.encode()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    print(build(parser.parse_args().out))
