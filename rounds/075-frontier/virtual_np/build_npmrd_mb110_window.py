"""Package the private offline NP-MRD scan of the frozen MassBank 110 windows."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack
from build_npmrd_2025_stage import SLUG as STAGE_SLUG


SLUG = 'enveda-075-npmrd-2025-mb110-window'
PROGRAM = 'scan_npmrd_mb110_window.py'
BASE = 'jiyujia/enveda-075b-massbank-pipeline/1'
COCONUT = 'jiyujia/enveda-075-coconut-2026-mb110-window/1'
STAGE_RECEIPT = 'npmrd_2025_source_receipt.json'
COCONUT_RECEIPT = 'coconut_2026_mb110_window_receipt.json'


def build(stage_receipt, coconut_receipt, out, workers=4, stage_version=1):
    if not 1 <= workers <= 8 or stage_version < 1:
        raise ValueError('Choose one to eight workers and an explicit positive Kaggle stage version')
    stage_slot = f'jiyujia/{STAGE_SLUG}/{stage_version}'
    stage_raw, coconut_raw = stage_receipt.read_bytes(), coconut_receipt.read_bytes()
    stage, coconut = json.loads(stage_raw), json.loads(coconut_raw)
    if (stage.get('release_date') != '2025-08-01' or stage.get('gzip_integrity_checked') is not True
            or len(stage['artifacts']) != 7 or coconut.get('status') != 'complete'
            or coconut.get('scope') != 'massbank110-coconut-2026-exact-window-inventory'
            or coconut.get('archived_c34') != BASE or coconut.get('query_count') != 110):
        raise ValueError('Expected the official NP-MRD source and sealed MassBank 110 COCONUT window receipts')
    stage_sha, coconut_sha = sha256(stage_raw).hexdigest(), sha256(coconut_raw).hexdigest()
    modules = (PROGRAM, 'stage_npmrd_2025.py', 'scan_coconut_2026_mb110_window.py', 'scan_coconut_2026_window.py',
               'scan_rebuilt_parent_window.py', 'generate.py', 'rebuild_parents.py')
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

stage = mounted({STAGE_RECEIPT!r}, {stage_slot!r})
coconut = mounted({COCONUT_RECEIPT!r}, {COCONUT!r})
pc_lists = mounted('pc_lists.json', {BASE!r})
with tempfile.TemporaryDirectory(prefix='npmrd-mb110-window-') as tmp:
    scratch = Path(tmp)
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Packaged source changed: {{name}}')
        (scratch / name).write_bytes(raw)
    subprocess.run([sys.executable, str(scratch / {PROGRAM!r}), '--stage-receipt', str(stage),
                    '--stage-sha', {stage_sha!r}, '--coconut-receipt', str(coconut), '--coconut-sha', {coconut_sha!r},
                    '--pc-lists', str(pc_lists), '--out', '/kaggle/working', '--workers', {str(workers)!r},
                    '--stage-version', {str(stage_version)!r}], check=True)
'''
    compile(code, SLUG, 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='run_window.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, dataset_sources=[WHEEL],
                kernel_sources=[stage_slot, COCONUT, BASE], competition_sources=[], docker_image=IMAGE)
    out.mkdir(exist_ok=True, parents=True)
    (out / 'run_window.py').write_text(code, encoding='utf-8')
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return sha256(code.encode()).hexdigest()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('stage-receipt', 'coconut-receipt', 'out'):
        cli.add_argument('--' + name, type=Path, required=True)
    cli.add_argument('--workers', type=int, default=4)
    cli.add_argument('--stage-version', type=int, default=1)
    print(build(**vars(cli.parse_args())))
