"""Build a private offline Kaggle CPU notebook from a sealed route supply receipt."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from build import IMAGE, METRIC_BOOTSTRAP, WHEEL, pack
from transform_prior_fit import DERIVE_SHA, EDGES, ENDPOINTS, POOL_SHA, RECEIPT


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075-transform-prior-fit'
SUPPLY = 'jiyujia/enveda-075-transform-prior-supply/1'


def build(receipt, out, supply):
    if not re.fullmatch(r'jiyujia/enveda-075-transform-prior-supply/[1-9][0-9]*', supply):
        raise ValueError('Supply must be a pinned jiyujia notebook output')
    raw = receipt.read_bytes()
    manifest = json.loads(raw)
    if (manifest['status'] != 'complete' or manifest['pool_sha256'] != POOL_SHA
            or manifest['chem_source_sha256'] != DERIVE_SHA or any(
                not re.fullmatch('[0-9a-f]{64}', manifest['output_sha256'][name]) for name in (ENDPOINTS, EDGES))):
        raise ValueError('Expected sealed files from the pinned public route supply')
    expected = sha256(raw).hexdigest()
    source = (HERE / 'transform_prior_fit.py').read_bytes()
    compile(source, 'transform_prior_fit.py', 'exec')
    payload, source_sha = pack(source)
    supply_slug = supply.split('/')[1]
    code = f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

inputs = Path('/kaggle/input')
receipt, = (p for p in inputs.rglob({RECEIPT!r}) if {supply_slug!r} in p.parts)
if sha256(receipt.read_bytes()).hexdigest() != {expected!r}:
    raise ValueError('Mounted route supply differs from the scored manifest')
{METRIC_BOOTSTRAP}
body = zlib.decompress(base64.b64decode({payload!r}))
if sha256(body).hexdigest() != {source_sha!r}:
    raise ValueError('Embedded fit source changed')
with tempfile.TemporaryDirectory(prefix='transform-prior-fit-') as tmp:
    script = Path(tmp) / 'transform_prior_fit.py'
    script.write_bytes(body)
    subprocess.run([sys.executable, str(script), '--supply', str(receipt.parent),
                    '--expected-sha256', {expected!r}, '--out', '/kaggle/working'], check=True)
'''
    compile(code, SLUG, 'exec')
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='run_fit.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=False, enable_internet=False, docker_image=IMAGE,
                dataset_sources=[WHEEL], kernel_sources=[supply], competition_sources=[])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'run_fit.py').write_text(code)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return dict(notebook=meta['id'], supply=supply, supply_receipt_sha256=expected,
                script_sha256=sha256(code.encode()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--supply', default=SUPPLY)
    print(json.dumps(build(**vars(parser.parse_args()))))
