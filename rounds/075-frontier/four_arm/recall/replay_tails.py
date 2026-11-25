"""Package a blind Kaggle CPU replay of NPX, E180X and sealing for the saved T4 v2 runs."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import zlib

from build_recall import FOUR_ARM, SOURCE_SHA, adapt_tails, assemble, masked


SLUG = 'enveda-075-np-recall-reseal'
RUNS = {'baseline': 'base', 'early_score': 'early'}
INPUTS = {
    'submission_pre_e180x.csv', 'four_source_lists.json', 'pilot_forward_budget.json',
    'execution_receipt.json', 'v43_execution_receipt.json', 'e004_verification.json',
    'four_reference_audit.json', 'four_inner_reference.json', 'spectral_parallax_report.json',
    'merge_report.json', 'v8_report.json', 'apex_extlib_audit.json', 'pilot_provenance_audit.json',
    'n1c_omni_audit.json', 'n1c_dl1_audit.json', 'dual_sync_audit.json', 'p5tail_audit.json',
    'fr3to25_audit.json', 'aprior_audit.json', 'uniq_audit.json', 'npx_audit.json',
    'submission_pre_n1c.csv', 'submission_pre_dualsync.csv', 'submission_pre_apex_extlib.csv',
}


def pack(contents):
    return base64.b64encode(zlib.compress(contents, 9)).decode(), sha256(contents).hexdigest()


PARALLEL_FEATURES = '''
import four_reference_mask as _four_mask


def _npx_record_feature(record, featurize):
    lab, smiles = record
    if lab == 1:
        try:
            if _four_mask.excluded_smiles(smiles):
                return lab, None
        except (ValueError, RuntimeError):
            return lab, None
    return lab, featurize(smiles)


def _npx_feature_worker_init():
    global _npx_worker_features
    _npx_worker_features = npx_featurizer()


def _npx_feature_worker(record):
    return _npx_record_feature(record, _npx_worker_features)
'''


def prepare_parallel_npx(source):
    begin = source.index('    X, y = [], []\n')
    end = source.index('    perm = rng.permutation(len(y))', begin)
    before = source[begin:end]
    if 'X, y = np.stack(X), np.asarray(y)' not in before or '_four_mask.excluded_smiles' not in before:
        raise ValueError('The pinned masked NPX feature loop changed')
    after = '''    from multiprocessing import get_context
    from os import sched_getaffinity
    records = [(1, s) for s in pos] + [(0, s) for s in neg]
    workers = min(4, len(sched_getaffinity(0)))
    X, y = [], []
    print('NPX featurisation', len(records), 'rows;', workers, 'workers', flush=True)
    with get_context('spawn').Pool(workers, initializer=_npx_feature_worker_init) as pool:
        sample = records[:12] + records[len(pos):len(pos)+12]
        for record, actual in zip(sample, pool.imap(_npx_feature_worker, sample), strict=True):
            serial = _npx_record_feature(record, feats)
            if serial[0] != actual[0] or (serial[1] is None) != (actual[1] is None):
                raise ValueError('Parallel NPX changed the sampled holdout decisions')
            if serial[1] is not None and serial[1].tobytes() != actual[1].tobytes():
                raise ValueError('Parallel NPX changed a sampled feature vector')
        for i, (lab, f) in enumerate(pool.imap(_npx_feature_worker, records, chunksize=48), 1):
            if f is not None:
                X.append(f)
                y.append(lab)
            if i % 4000 == 0:
                print('NPX features', i, '/', len(records), 'retained', len(X), flush=True)
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError('NPX featurisation over budget')
    X, y = np.stack(X), np.asarray(y)
    feature_receipt = dict(rows=int(len(X)), columns=int(X.shape[1]), workers=workers,
                           features_sha256=hashlib.sha256(memoryview(X).cast('B')).hexdigest(),
                           labels_sha256=hashlib.sha256(memoryview(y).cast('B')).hexdigest())
'''
    updated = masked.rewrite(source, before, after)
    updated = masked.rewrite(updated, "    info['table_keys'] = len(table)",
                             "    info['table_keys'] = len(table)\n    info['feature_receipt'] = feature_receipt")
    featurizer = source[source.index('def npx_featurizer():'):source.index('def npx_keyer():')]
    worker_source = PARALLEL_FEATURES + '\n' + featurizer
    compile(worker_source, 'npx_parallel_workers.py', 'exec')
    bootstrap_workers = f"""
worker_path = Path('/tmp/npx_parallel_workers.py')
worker_path.write_text({worker_source!r})
sys.path.insert(0, str(worker_path.parent))
from npx_parallel_workers import _npx_record_feature, _npx_feature_worker, _npx_feature_worker_init
"""
    return bootstrap_workers + '\n' + updated


def build(source, metadata, panel, artifacts, out):
    raw = source.read_bytes()
    if sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('Expected the unmodified public Pablo v29 source')
    book = json.loads(raw)
    bootstrap = masked.route_engine(book, bundle_index=10, driver_index=11, allow_empty_extension=True)
    adapt_tails(book)
    config = masked.rewrite(masked.cell(book, 1), 'NPX_BUDGET_S = 4200.0', 'NPX_BUDGET_S = 7200.0')
    parallel_npx = prepare_parallel_npx(masked.cell(book, 35))
    prepare = masked.prepare_source(panel / 'blind', panel / 'private/holdout.json', bootstrap)
    blobs = {}
    for arm, directory in RUNS.items():
        saved = artifacts / directory
        audit = json.loads((saved / 'npx_audit.json').read_text())
        if not (audit['status'] == 'failed closed: original bytes kept' and
                'AtomValenceException' in audit['exception']):
            raise ValueError(f'{arm}: the published pre-E180X list is not the NPX input')
        files = INPUTS | ({'submission_pre_n1c_dl1.csv'}
                          if json.loads((saved / 'n1c_dl1_audit.json').read_text())['changed_lists'] else set())
        blobs[arm] = {name: pack((saved / name).read_bytes()) for name in sorted(files)}

    replay = f'''
import shutil
import tempfile
import pickle
from zipfile import ZipFile, ZIP_DEFLATED

blobs = {blobs!r}
runtime = {assemble.pack(FOUR_ARM / 'retrieval_pilot_score.py')!r}

def restore(payload, path):
    blob, expected = payload
    data = zlib.decompress(base64.b64decode(blob))
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'Changed replay artifact: {{path}}')
    path.write_bytes(data)

program = Path('/tmp/pilot_fusion.py')
restore(runtime, program)
work = ROOT
real_npx_train = npx_train
npx_model = None

def npx_train(*args, **kwargs):
    global npx_model
    if npx_model is None:
        npx_model = real_npx_train(*args, **kwargs)
        model, _, table, fit = npx_model
        with (work / 'shared_npx_model.pickle').open('wb') as output:
            pickle.dump(dict(model=model, table=table, fit=fit, reference_counts=_four_train_stats), output, protocol=5)
        (work / 'shared_npx_fit.json').write_text(json.dumps(fit, indent=2))
    return npx_model

with tempfile.TemporaryDirectory() as scratch:
    for arm, files in blobs.items():
        target = Path(scratch) / arm
        target.mkdir()
        for name, payload in files.items():
            restore(payload, target / name)
        original_reference = json.loads((target / 'four_reference_audit.json').read_text())
        if original_reference != _four_train_stats:
            raise ValueError(f'{{arm}}: CPU masking differs from the original T4 run')
        shutil.copyfile(target / 'submission_pre_e180x.csv', target / 'submission.csv')
        npx = npx_run(str(target), str(INPUT), True, enabled=True, cap_s=NPX_BUDGET_S)
        print(arm, 'NPX', npx, flush=True)
        e180x = ex_run(str(target), str(INPUT), True, enabled=True, cap_s=E180X_BUDGET_S)
        print(arm, 'E180X', e180x, flush=True)
        subprocess.run([sys.executable, str(program), 'seal', '--arm', arm, '--work',
                        str(target), '--queries', str(COMP / 'test.parquet')], check=True)
        receipt_path = target / 'pilot_arm_receipt.json'
        receipt = json.loads(receipt_path.read_text())
        with ZipFile(work / f'{{arm}}.zip', 'w', compression=ZIP_DEFLATED) as output:
            for name in ('pilot_arm_receipt.json', *receipt['hashes']):
                output.write(target / name, name)
'''
    setup = """from pathlib import Path
ROOT = Path('/kaggle/working')
INPUT = Path('/kaggle/input')
"""
    tails = "_NPX_NOTEBOOK_RUN = _EX_NOTEBOOK_RUN = False\n"
    book['cells'] = [
        assemble.code_cell('configure', config),
        assemble.code_cell('prepare-masked-input', setup + prepare),
        assemble.code_cell('original-tail-functions', tails + parallel_npx + '\n' + masked.cell(book, 37)),
        assemble.code_cell('reseal-saved-runs', replay),
    ]
    for cell in book['cells']:
        compile(''.join(cell['source']), cell['id'], 'exec')
    options = json.loads(metadata.read_text())
    needed = ('casmi26-natural-product-panel-benchmark', 'casmi26-natural-product-knowledge-table',
              'casmi26-attributed-v4g-reproduction-assets', 'casmi26-pubchem-popularity-prior',
              'rdkit-2026-3-3-wheel')
    options.update(id=f'jiyujia/{SLUG}', title=SLUG, code_file='replay.ipynb',
                   is_private=True, enable_internet=False, enable_gpu=False, kernel_sources=[],
                   dataset_sources=[src for src in options['dataset_sources'] if src.split('/')[-1] in needed])
    options.pop('machine_shape', None)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'kernel-metadata.json').write_text(json.dumps(options, indent=2) + '\n')
    (out / options['code_file']).write_text(json.dumps(book, ensure_ascii=False, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('source', 'metadata', 'panel', 'artifacts', 'out'):
        parser.add_argument('--' + arg, required=True, type=Path)
    build(**vars(parser.parse_args()))
