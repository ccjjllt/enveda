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


def build(source, metadata, panel, artifacts, out):
    raw = source.read_bytes()
    if sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('Expected the unmodified public Pablo v29 source')
    book = json.loads(raw)
    bootstrap = masked.route_engine(book, bundle_index=10, driver_index=11, allow_empty_extension=True)
    adapt_tails(book)
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
        assemble.code_cell('configure', masked.cell(book, 1)),
        assemble.code_cell('prepare-masked-input', setup + prepare),
        assemble.code_cell('original-tail-functions', tails + masked.cell(book, 35) + '\n' + masked.cell(book, 37)),
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
