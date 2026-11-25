"""Package the structure-only parent rebuild as a private offline Kaggle CPU notebook."""

import argparse
import json
from pathlib import Path

from build import IMAGE, METRIC_BOOTSTRAP, WHEEL, pack


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075-virtual-parents-rebuilt'
COMPETITION = 'enveda-CASMI26-molecule-id-mass-spectra'
LEGACY_POOL = 'ahmedberatozer/casmi26-v2-pool'
COCONUT_STAGING = 'jiyujia/enveda-075-coconut-parent-source/1'


def build(output, workers):
    payload = pack((HERE / 'rebuild_parents.py').read_bytes())
    code = f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import zlib

inputs = Path('/kaggle/input')

def mounted(filename, parent):
    path, = (p for p in inputs.rglob(filename) if parent.casefold() in [part.casefold() for part in p.parts])
    return path

train = mounted('train.parquet', {COMPETITION!r})
coconut = mounted('coconut-09-2024.csv.zip', 'enveda-075-coconut-parent-source')
receipt = coconut.with_name('coconut_parent_source.json')
legacy = mounted('pool_meta.parquet', 'casmi26-v2-pool')
{METRIC_BOOTSTRAP}

with TemporaryDirectory(prefix='virtual-np-parents-') as temp:
    blob, expected = {payload!r}
    source = zlib.decompress(base64.b64decode(blob))
    if sha256(source).hexdigest() != expected:
        raise ValueError('Parent-inventory generator source changed in transit.')
    program = Path(temp) / 'rebuild_parents.py'
    program.write_bytes(source)
    subprocess.run([sys.executable, str(program), '--train', str(train),
                    '--coconut-archive', str(coconut), '--coconut-receipt', str(receipt),
                    '--legacy-pool', str(legacy), '--outdir', '/kaggle/working',
                    '--workers', {str(workers)!r}], check=True)
'''
    compile(code, SLUG, 'exec')
    cell = dict(cell_type='code', id='rebuild-parent-structures', execution_count=None,
                metadata={}, outputs=[], source=code.strip().splitlines(keepends=True))
    notebook = dict(nbformat=4, nbformat_minor=5, cells=[cell], metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    metadata = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='rebuild.ipynb',
                    language='python', kernel_type='notebook', is_private=True,
                    enable_gpu=False, enable_internet=False, kernel_sources=[COCONUT_STAGING],
                    dataset_sources=[LEGACY_POOL, WHEEL], competition_sources=[COMPETITION],
                    docker_image=IMAGE)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'rebuild.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', '--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    build(**vars(parser.parse_args()))
