"""Package the private, offline Kaggle feasibility pilot with blind spectra only."""

import argparse
import ast
import base64
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path
import zlib


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075-np-relative-feasibility'
SOURCE_SHA = 'a183882abe0b131581109455e1b704ec6d93f27fb99e7b5c2f07d40063e9c5af'
BASELINE_KERNEL = 'jiyujia/enveda-075-np-recall-reseal/2'
KAGGLE_IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'
METRIC_BOOTSTRAP = run_path(str(HERE.parent / 'virtual_np/build.py'))['METRIC_BOOTSTRAP']


def native_fragment_source(notebook):
    raw = notebook.read_bytes()
    if sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('Expected the published, unchanged Pablo v29 notebook')
    cells = json.loads(raw)['cells']
    for cell in cells:
        source = ''.join(cell.get('source', []))
        if cell['cell_type'] != 'code' or '_physical =' not in source:
            continue
        for node in ast.parse(source).body:
            if (isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == '_physical' for target in node.targets)):
                result = ast.literal_eval(node.value)
                compile(result, 'frag_rescore.py', 'exec')
                return result.encode()
    raise ValueError('Native Parallax fragment proposal implementation missing')


def pack(raw):
    return base64.b64encode(zlib.compress(raw, 9)).decode(), sha256(raw).hexdigest()


def build(args):
    manifest = json.loads((args.panel / 'panel_manifest.json').read_text())
    queries = (args.panel / 'blind/test.parquet').read_bytes()
    if sha256(queries).hexdigest() != manifest['output_sha256']['blind/test.parquet']:
        raise ValueError('Panel spectra differ from the frozen manifest')
    payload = {name: pack(raw) for name, raw in {
        'test.parquet': queries,
        'frag_rescore.py': native_fragment_source(args.source),
        'retrieval_pilot_score.py': (HERE.parent / 'four_arm/retrieval_pilot_score.py').read_bytes(),
        'rank.py': (HERE / 'rank.py').read_bytes(),
    }.items()}
    source = f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import zlib

scratch = Path('/tmp/relative-feasibility')
scratch.mkdir(exist_ok=True)
inputs = Path('/kaggle/input')
payload = {payload!r}
for name, (encoded, expected) in payload.items():
    raw = zlib.decompress(base64.b64decode(encoded))
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f'Embedded input changed: {{name}}')
    (scratch / name).write_bytes(raw)

def mounted(pattern):
    found = sorted(inputs.rglob(pattern))
    if len(found) != 1:
        raise FileNotFoundError(f'Expected one mounted {{pattern}}, got {{found}}')
    return found[0]

{METRIC_BOOTSTRAP}

subprocess.run([sys.executable, str(scratch / 'rank.py'), '--baseline', str(mounted('baseline.zip')),
                '--queries', str(scratch / 'test.parquet'), '--out', '/kaggle/working'],
               check=True)
'''
    compile(source, 'rank-feasibility', 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='rank.ipynb', language='python',
                kernel_type='notebook', is_private=True, enable_gpu=False, enable_tpu=False,
                enable_internet=False, dataset_sources=['metric/rdkit-2026-3-3-wheel'],
                kernel_sources=[args.baseline_kernel], competition_sources=[], docker_image=KAGGLE_IMAGE)
    cell = dict(cell_type='code', id='rank-feasibility', execution_count=None, metadata={}, outputs=[],
                source=source.strip().splitlines(keepends=True))
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                    cells=[cell])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'rank.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (args.out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'Blind CPU notebook created at {args.out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline-kernel', default=BASELINE_KERNEL)
    build(parser.parse_args())
