"""Package a private offline MassBank C34 COCONUT-window transfer."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack


SLUG = 'enveda-075-coconut-2026-mb110-window'
PROGRAM = 'scan_coconut_2026_mb110_window.py'


def build(stage_receipt, rebuilt_receipt, out, workers=4):
    from rebuild_parents import LEGACY_SHA256, RDKit_VERSION
    from scan_coconut_2026_window import STAGE_RECEIPT, validate_stage_receipt
    from scan_coconut_2026_mb110_window import BASE, REBUILT, STAGE

    if not 1 <= workers <= 8:
        raise ValueError('Choose one to eight workers')
    stage_data, old_data = stage_receipt.read_bytes(), rebuilt_receipt.read_bytes()
    validate_stage_receipt(json.loads(stage_data))
    merge = json.loads(old_data)
    if (merge['status'] != 'complete' or merge['version'] != 2 or merge['rdkit'] != RDKit_VERSION
            or merge['legacy_pool_sha256'] != LEGACY_SHA256
            or merge['chem_source_sha256'] != sha256((HERE / 'rebuild_parents.py').read_bytes()).hexdigest()
            or not 0 < merge['identities'] == sum(merge['structure_counts'].values())
            or set(merge['structure_counts']) - {'1', '2', '3'}):
        raise ValueError('Expected the sealed, independently normalized 2024 inventory')
    stage_sha, rebuilt_sha = sha256(stage_data).hexdigest(), sha256(old_data).hexdigest()
    modules = (PROGRAM, 'scan_coconut_2026_window.py', 'scan_rebuilt_parent_window.py', 'generate.py', 'rebuild_parents.py')
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

def mounted(name, slot):
    slug = slot.split('/')[1]
    path, = (p for p in inputs.rglob(name) if slug in p.parts)
    return path

stage = mounted({STAGE_RECEIPT!r}, {STAGE!r})
rebuilt = mounted('rebuilt_parent_receipt.json', {REBUILT!r})
pc_lists = mounted('pc_lists.json', {BASE!r})
with tempfile.TemporaryDirectory(prefix='coconut-2026-mb110-window-') as tmp:
    scratch = Path(tmp)
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Packaged source changed: {{name}}')
        (scratch / name).write_bytes(raw)
    subprocess.run([sys.executable, str(scratch / {PROGRAM!r}), '--stage-receipt', str(stage),
                    '--stage-sha', {stage_sha!r}, '--rebuilt-receipt', str(rebuilt), '--rebuilt-sha', {rebuilt_sha!r},
                    '--pc-lists', str(pc_lists), '--out', '/kaggle/working', '--workers', {str(workers)!r}], check=True)
'''
    compile(code, SLUG, 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='run_window.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, dataset_sources=[WHEEL],
                kernel_sources=[STAGE, REBUILT, BASE], competition_sources=[], docker_image=IMAGE)
    out.mkdir(exist_ok=True, parents=True)
    (out / 'run_window.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return sha256(code.encode()).hexdigest()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--stage-receipt', type=Path, required=True)
    cli.add_argument('--rebuilt-receipt', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--workers', type=int, default=4)
    print(build(**vars(cli.parse_args())))
