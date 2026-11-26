"""Package the private post-seal CPU coverage score without uploading answers."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack
from build_parent_shards import versioned


SLUG = 'enveda-075-rebuilt-parent-window-score'
SCAN = 'jiyujia/enveda-075-rebuilt-parent-direct-window/1'
MERGE = 'jiyujia/enveda-075-parent-v2-merge/1'
ARCHIVES = ('jiyujia/enveda-075-np-virtual-window-full-main-window/1',
            'jiyujia/enveda-075-np-recall-reseal/2', 'jiyujia/enveda-075-np-recall-compare/1',
            'jiyujia/enveda-075-np-native-index-coverage/1')
FILES = ('score_rebuilt_parent_window.py', 'scan_rebuilt_parent_window.py', 'generate.py', 'rebuild_parents.py')


def build(scan_receipt, out, scan=SCAN, merge=MERGE):
    if versioned(scan) != versioned(SCAN) or versioned(merge) != versioned(MERGE):
        raise ValueError('Pin the official direct scan and parent v2 merge')
    raw = scan_receipt.read_bytes()
    seal = json.loads(raw)
    if (seal['status'] != 'complete' or seal['scope'] != 'rebuilt-parent-direct-window'
            or seal['query_count'] != 60 or len(seal['cases']) != 60):
        raise ValueError('Expected the completed label-free 60-query scan receipt')
    digest = sha256(raw).hexdigest()
    files = {name: pack((HERE / name).read_bytes()) for name in FILES}
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
with tempfile.TemporaryDirectory(prefix='score-rebuilt-parent-window-') as temp:
    scratch = Path(temp)
    for name, (blob, expected) in {files!r}.items():
        data = zlib.decompress(base64.b64decode(blob))
        if sha256(data).hexdigest() != expected:
            raise ValueError(f'Packaged source changed: {{name}}')
        (scratch / name).write_bytes(data)
    sys.path.insert(0, temp)
    from score_rebuilt_parent_window import score
    score(inputs, Path('/kaggle/working'), {scan!r}, {merge!r}, {digest!r})
'''
    compile(code, SLUG, 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='score.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, docker_image=IMAGE,
                dataset_sources=[WHEEL], competition_sources=[], kernel_sources=[scan, merge, *ARCHIVES])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'score.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return digest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan-receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scan', default=SCAN)
    parser.add_argument('--merge', default=MERGE)
    print(build(**vars(parser.parse_args())))
