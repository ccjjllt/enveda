"""Build a private Kaggle CPU fit using only archived models and competition train."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075-fragment-reliability-gl'
GL = 'jiyujia/enveda-075-gl-forward-hard-pair-cpu/1'
PAIRS = 'jiyujia/enveda-075-dual-energy-pair-supply/2'
COMPETITION = 'enveda-CASMI26-molecule-id-mass-spectra'
READER_SHA = 'ad1e53bb5874e4921cf891bbd493911698829c615f4b251ddf77f93b898df6eb'


def build(out):
    common = run_path(str(HERE.parent / 'virtual_np' / 'build.py'))
    reader = (HERE / 'read_dual_training_scans.py').read_bytes()
    if sha256(reader).hexdigest() != READER_SHA:
        raise ValueError('The training scan reader differs from the archived physical comparison')
    payload = {name: common['pack'](raw) for name, raw in {
        'fit_fragment_reliability.py': (HERE / 'fit_fragment_reliability.py').read_bytes(),
        'read_dual_training_scans.py': reader,
    }.items()}
    code = f'''
import base64
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

inputs = Path('/kaggle/input')
def mounted(filename, slug):
    path, = (path for path in inputs.rglob(filename) if slug.lower() in (part.lower() for part in path.parts))
    return path

train = mounted('train.parquet', {COMPETITION!r})
receipt = mounted('dual_energy_pair_supply.json', {PAIRS.split('/')[1]!r})
pairs = receipt.with_name('dual_energy_training_pairs.jsonl')
gl = mounted('gl_forward_swap_summary.json', {GL.split('/')[1]!r}).parent
with tempfile.TemporaryDirectory(prefix='gl-fragment-reliability-') as directory:
    scratch = Path(directory)
    for filename, (encoded, digest) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(encoded))
        if sha256(raw).hexdigest() != digest:
            raise ValueError(f'Embedded implementation changed: {{filename}}')
        (scratch / filename).write_bytes(raw)
    env = {{**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}}
    subprocess.run([sys.executable, '-u', str(scratch / 'fit_fragment_reliability.py'), '--train', str(train),
                    '--pairs', str(pairs), '--receipt', str(receipt), '--gl', str(gl), '--out', '/kaggle/working'],
                   env=env, check=True)
'''
    compile(code, SLUG, 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='fragment_reliability.py', language='python',
                kernel_type='script', is_private=True, enable_gpu=False, enable_internet=False,
                dataset_sources=[], kernel_sources=[PAIRS, GL], competition_sources=[COMPETITION],
                docker_image=common['IMAGE'])
    out.mkdir(parents=True, exist_ok=True)
    (out / meta['code_file']).write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(sha256(code.encode()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    build(parser.parse_args().out)
