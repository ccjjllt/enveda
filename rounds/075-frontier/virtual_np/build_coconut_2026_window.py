"""Package the blind November 2026 COCONUT exact-mass scan for Kaggle CPU."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from build import ARCHIVED_C34_SHA, ARCHIVED_TARGETS_SHA, HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack
from build_coconut_2026_stage import SLUGS as STAGE_SLUGS
from build_main_window import FULL_MAIN_WINDOW_SLUG
from build_parent_shards import versioned


SLUG = 'enveda-075-coconut-2026-window'
STAGE_SLUG = STAGE_SLUGS['lite']
STAGE = f'jiyujia/{STAGE_SLUG}/1'
LEGACY = f'jiyujia/{FULL_MAIN_WINDOW_SLUG}/1'
PROGRAM = 'scan_coconut_2026_window.py'


def build(stage_receipt, out, stage_kernel=STAGE, workers=4):
    from scan_coconut_2026_window import STAGE_RECEIPT, validate_stage_receipt

    stage_slug, legacy_slug = versioned(stage_kernel), versioned(LEGACY)
    if stage_slug != STAGE_SLUG or not 1 <= workers <= 8:
        raise ValueError('Expected a pinned COCONUT stage and one to eight workers')
    data = stage_receipt.read_bytes()
    validate_stage_receipt(json.loads(data))
    stage_sha = sha256(data).hexdigest()
    modules = (PROGRAM, 'scan_rebuilt_parent_window.py', 'generate.py', 'rebuild_parents.py')
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

def mounted(name, namespace):
    path, = (p for p in inputs.rglob(name) if namespace in p.parts)
    return path

stage = mounted({STAGE_RECEIPT!r}, {stage_slug!r})
legacy = mounted('blind_virtual_np_receipt.json', {legacy_slug!r}).parent
with tempfile.TemporaryDirectory(prefix='coconut-2026-window-') as tmp:
    scratch = Path(tmp)
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Packaged source changed: {{name}}')
        (scratch / name).write_bytes(raw)
    subprocess.run([sys.executable, str(scratch / {PROGRAM!r}),
                    '--stage-receipt', str(stage), '--stage-receipt-sha', {stage_sha!r},
                    '--legacy', str(legacy), '--out', '/kaggle/working',
                    '--archived-pc-sha', {ARCHIVED_C34_SHA!r}, '--archived-targets-sha', {ARCHIVED_TARGETS_SHA!r},
                    '--workers', {str(workers)!r}], check=True)
'''
    compile(code, SLUG, 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='run_window.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, dataset_sources=[WHEEL],
                kernel_sources=[stage_kernel, LEGACY], competition_sources=[], docker_image=IMAGE)
    out.mkdir(exist_ok=True, parents=True)
    (out / 'run_window.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return sha256(code.encode()).hexdigest()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--stage-receipt', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--stage-kernel', default=STAGE)
    cli.add_argument('--workers', type=int, default=4)
    print(build(**vars(cli.parse_args())))
