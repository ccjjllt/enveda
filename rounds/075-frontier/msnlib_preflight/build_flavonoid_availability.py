"""Package a read-only Kaggle CPU availability audit for the author’s flavonoid library."""

import argparse
import ast
import base64
from hashlib import sha256
import json
from pathlib import Path
import zlib

from build_panel_availability import HERE, METRIC_BOOTSTRAP, RD_WHEEL, STRUCTURES, embedded


SLUG = 'enveda-075-flavonoid-panel-audit'
MASK_SHA = 'b43a59ffd9f5079a58ff7062dd976a92a18389c3c7c4a2748ae3941e66ca14ca'
DRIVER = '''
import json
import sys
from pathlib import Path

from flavonoid_availability import run

run(Path(sys.argv[1]), json.loads(sys.stdin.read()), Path(sys.argv[2]))
'''


def frozen_mask(package):
    cells = json.loads((package / 'availability.ipynb').read_bytes())['cells']
    targets = [node for cell in cells if cell['cell_type'] == 'code'
               for node in ast.parse(''.join(cell['source'])).body
               if isinstance(node, ast.Assign)
               and any(isinstance(key, ast.Name) and key.id == 'mask' for key in node.targets)]
    if len(targets) != 1:
        raise ValueError('Expected the TargetMol notebook with one frozen mask')
    decoder = targets[0].value
    encoded = ast.literal_eval(decoder.args[0].args[0])
    mask = zlib.decompress(base64.b64decode(encoded))
    if sha256(mask).hexdigest() != MASK_SHA:
        raise ValueError('Expected the unchanged TargetMol prior holdout mask')
    return encoded


def build(mask_package, out):
    mask = frozen_mask(mask_package)
    payload = {}
    for name in ('audit.py', 'panel_availability.py', 'flavonoid_availability.py'):
        raw = (HERE / name).read_bytes()
        compile(raw, name, 'exec')
        payload[name] = embedded(raw), sha256(raw).hexdigest()
    code = f'''
import base64
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import zlib

inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}
indexes = {{p.resolve() for p in inputs.rglob('structures.parquet')
            if 'casmi26-identity-disjoint-validation-folds' in p.parts
            and 'casmi26-natural-product-panel-benchmark' in p.parts}}
if len(indexes) != 1:
    raise ValueError('Expected the pinned competition structure index')
mask = zlib.decompress(base64.b64decode({mask!r}))
if sha256(mask).hexdigest() != {MASK_SHA!r}:
    raise ValueError('Prior holdout mask changed in transit')
with TemporaryDirectory(prefix='flavonoid-availability-') as directory:
    scratch = Path(directory)
    for name, (blob, expected) in {payload!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Availability source changed: {{name}}')
        (scratch / name).write_bytes(raw)
    driver = scratch / 'driver.py'
    driver.write_text({DRIVER!r})
    subprocess.run([sys.executable, str(driver), str(indexes.pop()),
                    '/kaggle/working/flavonoid_availability.json'], input=mask, check=True)
'''
    compile(code, SLUG, 'exec')
    compile(DRIVER, 'driver.py', 'exec')
    book = dict(nbformat=4, nbformat_minor=5,
                metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                cells=[dict(cell_type='code', id='flavonoid-availability', execution_count=None, metadata={},
                            outputs=[], source=code.strip().splitlines(keepends=True))])
    metadata = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='availability.ipynb', language='python',
                    kernel_type='notebook', is_private=True, enable_gpu=False, enable_tpu=False,
                    enable_internet=True, dataset_sources=[STRUCTURES, RD_WHEEL],
                    kernel_sources=[], competition_sources=[])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'availability.ipynb').write_text(json.dumps(book, separators=(',', ':')) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mask-package', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    build(**vars(parser.parse_args()))
