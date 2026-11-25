"""Package the private, offline CPU comparison of already sealed feasibility ranks."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from build import HERE, KAGGLE_IMAGE, METRIC_BOOTSTRAP, SLUG, pack


COMPARE_SLUG = f'{SLUG}-compare'


def build(args):
    manifest = json.loads((args.panel / 'panel_manifest.json').read_text())
    labels = (args.panel / 'private/labels.parquet').read_bytes()
    if sha256(labels).hexdigest() != manifest['output_sha256']['private/labels.parquet']:
        raise ValueError('Labels differ from their frozen private manifest')
    payload = {
        'labels.parquet': pack(labels),
        'retrieval_pilot_score.py': pack((HERE.parent / 'four_arm/retrieval_pilot_score.py').read_bytes()),
        'compare.py': pack((HERE / 'compare.py').read_bytes()),
    }
    source = f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import zlib

inputs = Path('/kaggle/input')
scratch = Path('/tmp/relative-feasibility-compare')
scratch.mkdir(exist_ok=True)
{METRIC_BOOTSTRAP}

receipt, = (path for path in inputs.rglob('receipt.json') if {SLUG!r} in path.parts)
payload = {payload!r}
for name, (blob, expected) in payload.items():
    raw = zlib.decompress(base64.b64decode(blob))
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f'Private comparison input changed: {{name}}')
    (scratch / name).write_bytes(raw)

subprocess.run([sys.executable, str(scratch / 'compare.py'), '--rank', str(receipt.parent),
                '--labels', str(scratch / 'labels.parquet'), '--out', '/kaggle/working'], check=True)
'''
    compile(source, 'compare-feasibility', 'exec')
    cell = dict(cell_type='code', id='compare-feasibility', execution_count=None, metadata={}, outputs=[],
                source=source.strip().splitlines(keepends=True))
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                    cells=[cell])
    meta = dict(id=f'jiyujia/{COMPARE_SLUG}', title=COMPARE_SLUG, code_file='compare.ipynb',
                language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                enable_tpu=False, enable_internet=False, dataset_sources=['metric/rdkit-2026-3-3-wheel'],
                kernel_sources=[args.rank_kernel], competition_sources=[], docker_image=KAGGLE_IMAGE)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'compare.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (args.out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'Private CPU comparison notebook created at {args.out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rank-kernel', default=f'jiyujia/{SLUG}/1')
    build(parser.parse_args())
