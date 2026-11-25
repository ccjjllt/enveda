"""Package the pinned, single-sample Paulownia feature localization for Kaggle CPU."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import zlib


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075-flavonoid-feature-probe'


def build(out):
    payloads = {}
    for name in ('flavonoid_raw_probe.py', 'flavonoid_feature_probe.py'):
        raw = (HERE / name).read_bytes()
        compile(raw, name, 'exec')
        payloads[name] = base64.b64encode(zlib.compress(raw, 9)).decode(), sha256(raw).hexdigest()
    code = f'''
import base64
from hashlib import sha256
from pathlib import Path
from runpy import run_path
import sys
from tempfile import TemporaryDirectory
import zlib

with TemporaryDirectory(prefix='flavonoid-feature-probe-') as directory:
    for name, (encoded, digest) in {payloads!r}.items():
        raw = zlib.decompress(base64.b64decode(encoded))
        if sha256(raw).hexdigest() != digest:
            raise ValueError(f'Packed source SHA-256 changed: {{name}}')
        (Path(directory) / name).write_bytes(raw)
    sys.path.insert(0, directory)
    entry = Path(directory) / 'flavonoid_feature_probe.py'
    run_path(str(entry))['run'](Path('/kaggle/working/flavonoid_feature_probe.json'))
'''.strip() + '\n'
    compile(code, SLUG, 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                    cells=[dict(cell_type='code', id='flavonoid-feature-probe', execution_count=None,
                                metadata={}, outputs=[], source=code.splitlines(keepends=True))])
    metadata = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='feature_probe.ipynb', language='python',
                    kernel_type='notebook', is_private=True, enable_gpu=False, enable_tpu=False,
                    enable_internet=True, dataset_sources=[], kernel_sources=[], competition_sources=[])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'feature_probe.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    build(**vars(parser.parse_args()))
