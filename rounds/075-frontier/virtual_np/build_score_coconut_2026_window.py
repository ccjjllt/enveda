"""Package the private post-seal COCONUT 2026 comparison for Kaggle CPU."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack
from build_parent_shards import versioned
from build_score_rebuilt_parent_window import ARCHIVES
from score_coconut_2026_window import SCAN_2024, SCAN_2026, SCOPE_2024, SCOPE_2026, window_seal


SLUG = 'enveda-075-coconut-2026-window-score'
FILES = ('score_coconut_2026_window.py', 'scan_coconut_2026_window.py', 'score_rebuilt_parent_window.py',
         'scan_rebuilt_parent_window.py', 'generate.py', 'rebuild_parents.py')


def build(receipt_2026, receipt_2024, out, scan_2026=SCAN_2026, scan_2024=SCAN_2024):
    if versioned(scan_2026) != versioned(SCAN_2026) or versioned(scan_2024) != versioned(SCAN_2024):
        raise ValueError('Pin both completed COCONUT window notebook versions')
    seal26, seal24 = window_seal(receipt_2026, SCOPE_2026), window_seal(receipt_2024, SCOPE_2024)
    shared = ('exact_targets_sha256', 'archived_pc_lists_sha256', 'legacy_window_receipt_sha256', 'legacy_window_sha256')
    if any(seal26[key] != seal24[key] for key in shared) or seal26['cases'].keys() != seal24['cases'].keys():
        raise ValueError('The two receipts do not describe the same 60 C34 windows')
    digest26 = sha256(receipt_2026.read_bytes()).hexdigest()
    digest24 = sha256(receipt_2024.read_bytes()).hexdigest()
    files = {}
    for name in FILES:
        raw = (HERE / name).read_bytes()
        compile(raw, name, 'exec')
        files[name] = pack(raw)
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
with tempfile.TemporaryDirectory(prefix='score-coconut-2026-window-') as temp:
    scratch = Path(temp)
    for name, (blob, expected) in {files!r}.items():
        data = zlib.decompress(base64.b64decode(blob))
        if sha256(data).hexdigest() != expected:
            raise ValueError(f'Packaged source changed: {{name}}')
        (scratch / name).write_bytes(data)
    sys.path.insert(0, temp)
    from score_coconut_2026_window import score
    score(inputs, Path('/kaggle/working'), {scan_2026!r}, {scan_2024!r}, {digest26!r}, {digest24!r})
'''
    compile(code, SLUG, 'exec')
    metadata = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='score.py', language='python', kernel_type='script',
                    is_private=True, enable_gpu=False, enable_internet=False, docker_image=IMAGE,
                    dataset_sources=[WHEEL], competition_sources=[], kernel_sources=[scan_2026, scan_2024, *ARCHIVES])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'score.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return sha256(code.encode()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt-2026', type=Path, required=True)
    parser.add_argument('--receipt-2024', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scan-2026', default=SCAN_2026)
    parser.add_argument('--scan-2024', default=SCAN_2024)
    print(build(**vars(parser.parse_args())))
