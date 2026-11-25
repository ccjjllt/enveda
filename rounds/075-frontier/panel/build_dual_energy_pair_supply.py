"""Package the training-only pair inventory for network-off Kaggle CPU."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path


HERE = Path(__file__).resolve().parent
COMMON = run_path(str(HERE.parent/'virtual_np'/'build.py'))
SLUG = 'enveda-075-dual-energy-pair-supply'


def build(out):
    source = HERE/'dual_energy_pair_supply.py'
    raw = source.read_bytes()
    compile(raw, str(source), 'exec')
    payload, expected = COMMON['pack'](raw)
    runtime = f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

inputs = Path('/kaggle/input')
{COMMON['METRIC_BOOTSTRAP']}
train, = (path for path in inputs.rglob('train.parquet')
          if any(part.lower() == 'enveda-casmi26-molecule-id-mass-spectra' for part in path.parts))
with tempfile.TemporaryDirectory(prefix='dual-energy-training-pairs-') as directory:
    program = Path(directory)/'dual_energy_pair_supply.py'
    raw = zlib.decompress(base64.b64decode({payload!r}))
    if sha256(raw).hexdigest() != {expected!r}:
        raise ValueError('The frozen training supply program changed')
    program.write_bytes(raw)
    subprocess.run([sys.executable, '-u', str(program), '--train', str(train), '--out', '/kaggle/working'], check=True)
'''
    compile(runtime, SLUG, 'exec')
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata={'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'}},
              cells=[dict(cell_type='code', id='dual-energy-pair-supply', metadata={}, execution_count=None,
                          outputs=[], source=runtime.strip().splitlines(keepends=True))])
    metadata = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='supply.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False, kernel_sources=[],
                    dataset_sources=[COMMON['WHEEL']], docker_image=COMMON['IMAGE'],
                    competition_sources=['enveda-CASMI26-molecule-id-mass-spectra'])
    out.mkdir(parents=True, exist_ok=True)
    book = out/metadata['code_file']
    book.write_text(json.dumps(nb, separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(sha256(book.read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    build(parser.parse_args().out)
