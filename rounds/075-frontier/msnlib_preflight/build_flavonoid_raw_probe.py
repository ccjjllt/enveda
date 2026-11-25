"""Package the single-file Paulownia raw acquisition metadata audit for Kaggle CPU."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import zlib

from flavonoid_raw_probe import DATASET, FILE, FILE_BYTES, MASSIVE_PROXY


HERE = Path(__file__).resolve().parent
SLUG = 'enveda-075-flavonoid-confirm-raw'
DDA_SLUG = 'enveda-075-flavonoid-hcd-raw'
FALLBACK_PATH = 'raw/manuscript/PT24-CH-fraction7_ID1.mzML.mzML'
FALLBACK = dict(relative_path=FALLBACK_PATH, bytes=122_239_367,
                download_url=f'{MASSIVE_PROXY}/{FALLBACK_PATH}', dataset=DATASET,
                selection='Smallest of the four numbered same-fraction DDA injections. The smaller unnumbered file is likely the initial MS1 inclusion-list injection described in paper section 2.4.')


def build(out, same_fraction_hcd=False):
    slug = DDA_SLUG if same_fraction_hcd else SLUG
    relative_path = FALLBACK_PATH if same_fraction_hcd else FILE
    file_bytes = FALLBACK['bytes'] if same_fraction_hcd else FILE_BYTES
    scope = ('numbered_dda_same_fraction_no_batch_or_identity_link' if same_fraction_hcd
             else 'isolated_confirmation')
    next_file = None if same_fraction_hcd else FALLBACK
    program = (HERE / 'flavonoid_raw_probe.py').read_bytes()
    compile(program, 'flavonoid_raw_probe.py', 'exec')
    payload = base64.b64encode(zlib.compress(program, 9)).decode(), sha256(program).hexdigest()
    code = f'''
import base64
from hashlib import sha256
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
import zlib

with TemporaryDirectory(prefix='flavonoid-raw-probe-') as directory:
    source = Path(directory) / 'flavonoid_raw_probe.py'
    encoded, digest = {payload!r}
    raw = zlib.decompress(base64.b64decode(encoded))
    if sha256(raw).hexdigest() != digest:
        raise ValueError('The public raw metadata probe source changed')
    source.write_bytes(raw)
    run_path(str(source))['run'](Path('/kaggle/working/flavonoid_raw_probe.json'), {next_file!r},
                                relative_path={relative_path!r}, file_bytes={file_bytes!r},
                                acquisition_scope={scope!r})
'''
    compile(code, slug, 'exec')
    book = dict(nbformat=4, nbformat_minor=5,
                metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                cells=[dict(cell_type='code', id='flavonoid-raw-probe', execution_count=None, metadata={},
                            outputs=[], source=code.strip().splitlines(keepends=True))])
    metadata = dict(id=f'jiyujia/{slug}', title=slug, code_file='raw_probe.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=False, enable_tpu=False,
                    enable_internet=True, dataset_sources=[], kernel_sources=[], competition_sources=[])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'raw_probe.ipynb').write_text(json.dumps(book, separators=(',', ':')) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--same-fraction-hcd', action='store_true')
    build(**vars(parser.parse_args()))
