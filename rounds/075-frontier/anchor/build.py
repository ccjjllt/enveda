"""Package a private Kaggle CPU notebook for blind anchor reachability."""

import argparse
import base64
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
from runpy import run_path
import zlib


HERE = Path(__file__).resolve().parent
FOUR_ARM = HERE.parent / 'four_arm'
SLUG = 'enveda-075-np-anchor-preflight'
METRIC_BOOTSTRAP = run_path(str(HERE.parent / 'virtual_np/build.py'))['METRIC_BOOTSTRAP']


def build(args):
    spec = importlib.util.spec_from_file_location('anchor_notebook_cells', FOUR_ARM / 'assemble.py')
    cells = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cells)
    expected = json.loads((args.panel / 'panel_manifest.json').read_text())['output_sha256']
    blind = args.panel / 'blind/test.parquet'
    groups_path = args.panel / 'private/holdout.json'
    groups_raw = groups_path.read_bytes()
    groups = json.loads(groups_raw)
    if sha256(groups_raw).hexdigest() != expected['private/holdout.json']:
        raise ValueError('The panel holdout differs from the frozen manifest')
    hashed = lambda values: sorted({sha256(value.encode()).hexdigest() for value in values if value})
    sources = {re.sub(r'[^a-z0-9]+', '', source.lower()) for source in groups['blocked_sources']}
    mask = dict(keys=hashed(groups['blocked_keys']), scaffolds=hashed(groups['blocked_scaffolds']),
                sources=sorted(sources - {''}), block_acyclic=bool(groups['block_acyclic']))
    mask_raw = json.dumps(mask).encode()
    files = {
        'test.parquet': cells.pack(blind),
        'mask.json': (base64.b64encode(zlib.compress(mask_raw, 9)).decode(), sha256(mask_raw).hexdigest()),
        'four_reference_mask.py': cells.pack(FOUR_ARM / 'baseline/reference_mask.py'),
        'retrieval_pilot_score.py': cells.pack(FOUR_ARM / 'retrieval_pilot_score.py'),
        'probe.py': cells.pack(HERE / 'probe.py'),
    }
    if files['test.parquet'][1] != expected['blind/test.parquet']:
        raise ValueError('The blind input differs from the frozen manifest')

    run = f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import zlib

scratch = Path('/tmp/four_arm_comp')
scratch.mkdir(exist_ok=True)
inputs = Path('/kaggle/input')
for name, (blob, digest) in {files!r}.items():
    raw = zlib.decompress(base64.b64decode(blob))
    if sha256(raw).hexdigest() != digest:
        raise ValueError(f'Embedded file digest differs: {{name}}')
    (scratch / name).write_bytes(raw)

def only(filename, component):
    matches = {{path.resolve() for path in inputs.rglob(filename)
                if component.lower() in {{part.lower() for part in path.parts}}}}
    if len(matches) != 1:
        raise ValueError(f'Expected one {{filename}} from {{component}}; got {{matches}}')
    return next(iter(matches))

{METRIC_BOOTSTRAP}

baseline = only('baseline.zip', {args.baseline_kernel.split('/')[1]!r})
train = only('train.parquet', 'enveda-casmi26-molecule-id-mass-spectra')
structures = only('structures.parquet', 'casmi26-identity-disjoint-validation-folds')
subprocess.run([sys.executable, str(scratch / 'probe.py'), '--baseline', str(baseline),
                '--queries', str(scratch / 'test.parquet'), '--mask', str(scratch / 'mask.json'),
                '--structures', str(structures), '--train', str(train), '--out', '/kaggle/working'],
               check=True)
'''
    compile(run, SLUG, 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        cells=[cells.code_cell('blind-anchor-reachability', run)])
    metadata = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='anchor.ipynb',
                    language='python', kernel_type='notebook', is_private=True,
                    enable_gpu=False, enable_tpu=False, enable_internet=False,
                    kernel_sources=[args.baseline_kernel],
                    dataset_sources=['dmitriigluzdov/casmi26-natural-product-panel-benchmark',
                                     'metric/rdkit-2026-3-3-wheel'],
                    competition_sources=['enveda-CASMI26-molecule-id-mass-spectra'])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'anchor.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (args.out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Created private anchor preflight in {args.out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--baseline-kernel', default='jiyujia/enveda-075-np-recall-reseal/1')
    build(parser.parse_args())
