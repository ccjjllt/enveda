"""Package the label-free rebuilt parent window comparison for Kaggle CPU."""

import argparse
import json
from pathlib import Path

from build import ARCHIVED_C34_SHA, ARCHIVED_TARGETS_SHA, HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack
from build_main_window import FULL_MAIN_WINDOW_SLUG
from build_parent_shards import PREFIX, versioned
from build_parents import LEGACY_POOL


SLUG = 'enveda-075-rebuilt-parent-direct-window'
REBUILT = f'jiyujia/{PREFIX}-merge/1'
LEGACY = f'jiyujia/{FULL_MAIN_WINDOW_SLUG}/1'


def build(out, rebuilt_kernel=REBUILT):
    rebuilt_slug, legacy_slug = versioned(rebuilt_kernel), versioned(LEGACY)
    if rebuilt_slug != f'{PREFIX}-merge':
        raise ValueError('Expected a pinned parent v2 merge version')
    names = ('scan_rebuilt_parent_window.py', 'generate.py', 'rebuild_parents.py')
    payload = {name: pack((HERE / name).read_bytes()) for name in names}
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
    path, = (p for p in inputs.rglob(name) if namespace in p.parts)
    return path

rebuilt = mounted('rebuilt_parent_receipt.json', {rebuilt_slug!r})
legacy = mounted('blind_virtual_np_receipt.json', {legacy_slug!r}).parent
pool = mounted('pool_meta.parquet', {LEGACY_POOL.split('/')[1]!r})
with tempfile.TemporaryDirectory(prefix='rebuilt-parent-window-') as temp:
    scratch = Path(temp)
    for name, (blob, expected) in {payload!r}.items():
        data = zlib.decompress(base64.b64decode(blob))
        if sha256(data).hexdigest() != expected:
            raise ValueError(f'Packaged source changed: {{name}}')
        (scratch / name).write_bytes(data)
    subprocess.run([sys.executable, str(scratch / {names[0]!r}), '--rebuilt-receipt', str(rebuilt),
                    '--legacy-window', str(legacy), '--legacy-pool', str(pool), '--out', '/kaggle/working',
                    '--archived-pc-sha', {ARCHIVED_C34_SHA!r}, '--archived-targets-sha', {ARCHIVED_TARGETS_SHA!r}],
                   check=True)
'''
    compile(code, SLUG, 'exec')
    cell = dict(cell_type='code', id='rebuilt-parent-window', execution_count=None,
                metadata={}, outputs=[], source=code.strip().splitlines(keepends=True))
    notebook = dict(nbformat=4, nbformat_minor=5, cells=[cell], metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='scan.ipynb', language='python',
                kernel_type='notebook', is_private=True, enable_internet=False, enable_gpu=False,
                docker_image=IMAGE, dataset_sources=[LEGACY_POOL, WHEEL], competition_sources=[],
                kernel_sources=[rebuilt_kernel, LEGACY])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'scan.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rebuilt-kernel', default=REBUILT)
    build(**vars(parser.parse_args()))
