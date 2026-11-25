"""Build the private offline blind full-main-window proposal notebook."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from build import ARCHIVED_C34_SHA, ARCHIVED_TARGETS_SHA, ASSET_HASHES, ASSETS, HERE, IMAGE, METRIC_BOOTSTRAP, SLUG, WHEEL, pack


FULL_MAIN_WINDOW_SLUG = f'{SLUG}-full-main-window'
FULL_MAIN_SCOPE = 'full-main-window'


def build(pc, targets, out, workers):
    raw_pc, raw_targets = pc.read_bytes(), targets.read_bytes()
    if sha256(raw_pc).hexdigest() != ARCHIVED_C34_SHA or sha256(raw_targets).hexdigest() != ARCHIVED_TARGETS_SHA:
        raise ValueError('Expected the original untouched C34 archive and target receipt')
    archived = json.loads(raw_targets)
    masses = {mid: row['target'] for mid, row in archived.items()}
    if (len(masses) != 60 or any(row['error'] for row in archived.values())
            or masses != {mid: row['target'] for mid, row in json.loads(raw_pc).items()}):
        raise ValueError('Expected the same 60 valid archived C34 mass centers')
    payload = {name: pack((HERE / name).read_bytes()) for name in ('generate.py', 'scan_main_window.py')}
    payload['targets.json'] = pack(json.dumps(masses, sort_keys=True, separators=(',', ':')).encode())
    code = f'''
import base64
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

inputs = Path('/kaggle/input')
root = Path('/kaggle/working')
{METRIC_BOOTSTRAP}
pool, = (p for p in inputs.rglob('pool_meta.parquet')
         if 'casmi26-v2-pool' in p.parts and {ASSETS.split('/')[1]!r} in p.parts)
payload = {payload!r}
with tempfile.TemporaryDirectory(prefix='blind-full-main-window-') as temporary:
    scratch = Path(temporary)
    for name, (blob, expected) in payload.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Full-main-window artifact changed: {{name}}')
        (scratch / name).write_bytes(raw)
    subprocess.run([sys.executable, str(scratch / 'scan_main_window.py'), '--targets', str(scratch / 'targets.json'),
                    '--pool', str(pool), '--out', str(root), '--workers', {str(workers)!r}], check=True)

detail = root / 'receipt.json'
report = json.loads(detail.read_text())
if (report['generation_scope'] != {FULL_MAIN_SCOPE!r}
        or report['queries'] != 60 or report['target_masses_sha256'] != payload['targets.json'][1]
        or report['pool_sha256'] != {ASSET_HASHES['pool_meta.parquet']!r}):
    raise ValueError('The direct full-window scan does not match the frozen blind inputs')
sources = {{name: payload[name][1] for name in ('generate.py', 'scan_main_window.py')}}
blind = dict(generation_scope={FULL_MAIN_SCOPE!r}, published_data={ASSETS!r}, query_count=60,
             archived_c34='jiyujia/enveda-075-np-recall-base/2', archived_pc_lists_sha256={ARCHIVED_C34_SHA!r},
             target_receipt_sha256={ARCHIVED_TARGETS_SHA!r}, generator_sha256=sha256(json.dumps(sources).encode()).hexdigest(),
             source_sha256=sources, output_sha256=report['outputs'], detailed_receipt_sha256=sha256(detail.read_bytes()).hexdigest())
(root / 'blind_virtual_np_receipt.json').write_text(json.dumps(blind, indent=2) + '\\n')
'''
    compile(code, FULL_MAIN_WINDOW_SLUG, 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                    cells=[dict(cell_type='code', id='full-main-window', execution_count=None, metadata={},
                                outputs=[], source=code.strip().splitlines(keepends=True))])
    meta = dict(id=f'jiyujia/{FULL_MAIN_WINDOW_SLUG}', title=FULL_MAIN_WINDOW_SLUG, code_file='scan.ipynb',
                language='python', kernel_type='notebook', is_private=True, enable_internet=False,
                enable_gpu=False, docker_image=IMAGE, dataset_sources=[ASSETS, WHEEL],
                kernel_sources=[], competition_sources=[])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'scan.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pc', 'targets', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    build(**vars(parser.parse_args()))
