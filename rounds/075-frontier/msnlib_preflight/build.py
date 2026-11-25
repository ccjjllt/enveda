"""Build a CPU Kaggle notebook for public MSnLib source and acquisition evidence."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import zlib


SLUG = 'enveda-075-msnlib-preflight'
HERE = Path(__file__).resolve().parent


def build(args):
    raw = (HERE / 'audit.py').read_bytes()
    compile(raw, 'audit.py', 'exec')
    encoded = base64.b64encode(zlib.compress(raw, 9)).decode()
    expected = sha256(raw).hexdigest()
    sources = ['targetmol_pos', 'targetmol_neg']
    if args.include_nih:
        sources.extend(('nih_pos', 'nih_neg'))
    extra = [] if args.skip_raw else ['--verify-raw']
    source = f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import zlib

raw = zlib.decompress(base64.b64decode({encoded!r}))
if sha256(raw).hexdigest() != {expected!r}:
    raise ValueError('Preflight source digest differs')
path = Path('/tmp/msnlib_preflight.py')
path.write_bytes(raw)
command = [sys.executable, str(path), '--source', *{sources!r}, '--out', '/kaggle/working', *{extra!r}]
subprocess.run(command, check=True)
'''
    compile(source, 'audit-msnlib', 'exec')
    cell = dict(cell_type='code', id='audit-msnlib', execution_count=None, metadata={}, outputs=[],
                source=source.strip().splitlines(keepends=True))
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                    cells=[cell])
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='audit.ipynb', language='python',
                kernel_type='notebook', is_private=True, enable_gpu=False, enable_tpu=False,
                enable_internet=True, dataset_sources=[], competition_sources=[], kernel_sources=[])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'audit.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (args.out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'CPU public-source preflight notebook created at {args.out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--include-nih', action='store_true')
    parser.add_argument('--skip-raw', action='store_true')
    build(parser.parse_args())
