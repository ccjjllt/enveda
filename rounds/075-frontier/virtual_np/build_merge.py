"""Build an offline Kaggle notebook that seals the two independent generation packages."""

import argparse
import json
from pathlib import Path

from build import COMBINED, HERE, IMAGE, SLUG, SUPPLEMENT, pack


def build(out, original_version, supplement_version):
    payload = pack((HERE / 'merge_oxidation.py').read_bytes())
    code = f'''
import base64
from hashlib import sha256
from pathlib import Path
import runpy
import tempfile
import zlib

inputs = Path('/kaggle/input')
original, = (p.parent for p in inputs.rglob('blind_virtual_np_receipt.json') if {SLUG!r} in p.parts)
supplement, = (p.parent for p in inputs.rglob('blind_virtual_np_receipt.json') if {SUPPLEMENT!r} in p.parts)
encoded, expected = {payload!r}
contents = zlib.decompress(base64.b64decode(encoded))
if sha256(contents).hexdigest() != expected:
    raise ValueError('The approved correction merge source changed')
with tempfile.TemporaryDirectory(prefix='merge-virtual-candidates-') as directory:
    script = Path(directory) / 'merge_oxidation.py'
    script.write_bytes(contents)
    runpy.run_path(str(script))['merge'](original, supplement, Path('/kaggle/working'))
'''
    compile(code, COMBINED, 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                    cells=[dict(cell_type='code', id='merge-oxidation', execution_count=None, metadata={},
                                outputs=[], source=code.strip().splitlines(keepends=True))])
    metadata = dict(id=f'jiyujia/{COMBINED}', title=COMBINED, code_file='merge.ipynb', language='python',
                    kernel_type='notebook', is_private=True, enable_gpu=False, enable_internet=False,
                    dataset_sources=[], competition_sources=[], docker_image=IMAGE,
                    kernel_sources=[f'jiyujia/{SLUG}/{original_version}', f'jiyujia/{SUPPLEMENT}/{supplement_version}'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'merge.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--original-version', type=int, required=True)
    parser.add_argument('--supplement-version', type=int, required=True)
    build(**vars(parser.parse_args()))
