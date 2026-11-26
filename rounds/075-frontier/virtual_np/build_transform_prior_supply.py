"""Package the public-input transform prior supply as a private offline Kaggle CPU notebook."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from build import ASSETS, HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack


SLUG = 'enveda-075-transform-prior-supply'
SAMPLE_SIZE = 8_192
BANDS = 12
SEED = 20_261_125
CAP = 8


def build(out, workers):
    if not 1 <= workers <= 6:
        raise ValueError('Expected one to six Kaggle CPU workers')
    modules = ('generate.py', 'transform_prior_supply.py')
    payload = {}
    for name in modules:
        source = (HERE / name).read_bytes()
        compile(source, name, 'exec')
        payload[name] = pack(source)
    code = f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}

def mounted(name, namespace):
    path, = (p for p in inputs.rglob(name) if namespace in p.parts
             and {ASSETS.split('/')[1]!r} in p.parts)
    return path

pool = mounted('pool_meta.parquet', 'casmi26-v2-pool')
derive = mounted('derive.py', 'casmi26-v4b-models')
with tempfile.TemporaryDirectory(prefix='transform-prior-supply-') as directory:
    scratch = Path(directory)
    for name, (blob, expected) in {payload!r}.items():
        data = zlib.decompress(base64.b64decode(blob))
        if sha256(data).hexdigest() != expected:
            raise ValueError(f'Changed embedded module: {{name}}')
        (scratch / name).write_bytes(data)
    subprocess.run([sys.executable, str(scratch / 'transform_prior_supply.py'),
                    '--pool', str(pool), '--derive-source', str(derive), '--out', '/kaggle/working',
                    '--workers', {str(workers)!r}, '--sample-size', {str(SAMPLE_SIZE)!r}, '--bands', {str(BANDS)!r},
                    '--seed', {str(SEED)!r}, '--cap', {str(CAP)!r}], check=True)
'''
    compile(code, SLUG, 'exec')
    cell = dict(cell_type='code', id='public-transform-supply', execution_count=None, metadata={}, outputs=[],
                source=code.strip().splitlines(keepends=True))
    notebook = dict(nbformat=4, nbformat_minor=5, cells=[cell], metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    })
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='supply.ipynb', language='python', kernel_type='notebook',
                is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=False,
                docker_image=IMAGE, dataset_sources=[ASSETS, WHEEL], kernel_sources=[], competition_sources=[])
    out.mkdir(parents=True, exist_ok=True)
    notebook_file = out / 'supply.ipynb'
    metadata_file = out / 'kernel-metadata.json'
    notebook_file.write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    metadata_file.write_text(json.dumps(meta, indent=2) + '\n')
    return dict(slug=meta['id'], notebook_sha256=sha256(notebook_file.read_bytes()).hexdigest(),
                metadata_sha256=sha256(metadata_file.read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    print(json.dumps(build(**vars(parser.parse_args()))))
