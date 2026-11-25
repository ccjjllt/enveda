"""Package a standalone Kaggle CPU evidence build from public native MSnLib spectra."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import zlib


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075-msnlib-branch-evidence'


def build(args):
    files = {name: (HERE / name).read_bytes() for name in ('audit.py', 'branch_evidence.py')}
    for name, raw in files.items():
        compile(raw, name, 'exec')
    if args.direct_proofs is not None:
        files['public_raw_scan_proofs.json'] = args.direct_proofs.read_bytes()
    payload = {name: (base64.b64encode(zlib.compress(raw, 9)).decode(), sha256(raw).hexdigest())
               for name, raw in files.items()}
    source = f'''
import base64
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
import zlib

scratch = Path('/tmp/msnlib-branch-evidence')
scratch.mkdir(exist_ok=True)
for name, (encoded, expected) in {payload!r}.items():
    raw = zlib.decompress(base64.b64decode(encoded))
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f'Staged public-source artifact changed: {{name}}')
    (scratch / name).write_bytes(raw)

env = os.environ.copy()
version = subprocess.run([sys.executable, '-c', 'from rdkit import rdBase; print(rdBase.rdkitVersion)'],
                         capture_output=True, text=True)
if version.returncode or version.stdout.strip() != '2026.03.3':
    tag = f'cp{{sys.version_info.major}}{{sys.version_info.minor}}'
    wheel, = Path('/kaggle/input').rglob(f'rdkit-2026.3.3-{{tag}}-*.whl')
    site = scratch / 'metric_site'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                    '--target', str(site), str(wheel)], check=True)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(site), env.get('PYTHONPATH')]))

command = [sys.executable, str(scratch / 'branch_evidence.py'), '--out', '/kaggle/working']
if (proofs := scratch / 'public_raw_scan_proofs.json').is_file():
    command.extend(('--direct-proofs', str(proofs)))
subprocess.run(command, env=env, check=True)
'''
    compile(source, 'msnlib-branch-evidence', 'exec')
    cell = dict(cell_type='code', id='msnlib-branch-evidence', execution_count=None, metadata={}, outputs=[],
                source=source.strip().splitlines(keepends=True))
    book = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, cells=[cell])
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='evidence.ipynb', language='python',
                kernel_type='notebook', is_private=True, enable_gpu=False, enable_tpu=False,
                enable_internet=True, dataset_sources=['metric/rdkit-2026-3-3-wheel'],
                competition_sources=[], kernel_sources=[])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'evidence.ipynb').write_text(json.dumps(book, separators=(',', ':')) + '\n')
    (args.out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'CPU public-source branch evidence notebook created at {args.out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--direct-proofs', type=Path)
    build(parser.parse_args())
