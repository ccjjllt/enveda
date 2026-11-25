"""Assemble the separate, private CPU notebook for sealed MSAlign evaluation."""

import argparse
import json
from pathlib import Path
from runpy import run_path

from build import HERE, assemble


SLUG = 'enveda-075-msalign-morgan-compare'
METRIC_BOOTSTRAP = run_path(str(HERE.parent / 'virtual_np/build.py'))['METRIC_BOOTSTRAP']


def build(args):
    manifest = json.loads((args.panel / 'panel_manifest.json').read_text())
    frozen = manifest['output_sha256']
    query_payload = assemble.pack(args.panel / 'blind/test.parquet')
    label_payload = assemble.pack(args.panel / 'private/labels.parquet')
    if (query_payload[1] != frozen['blind/test.parquet']
            or label_payload[1] != frozen['private/labels.parquet']):
        raise ValueError('Panel files have changed since the blind notebook was built')
    program = {name: assemble.pack(HERE / 'scoring' / name) for name in ('score.py', 'compare.py')}
    program['test.parquet'] = query_payload
    program['exposure.json'] = assemble.pack(HERE / 'exposure.json')
    program['retrieval_pilot_score.py'] = assemble.pack(HERE.parent / 'four_arm/retrieval_pilot_score.py')

    source = f'''
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zlib
from zipfile import ZipFile

inputs = Path('/kaggle/input')
scratch = Path('/tmp/msalign-compare')
scratch.mkdir(exist_ok=True)

def mount(filename):
    found = list(inputs.rglob(filename))
    if len(found) != 1:
        raise ValueError(f'Expected one mounted {{filename}}; got {{found}}')
    return found[0]

def restore(name, payload):
    blob, expected = payload
    raw = zlib.decompress(base64.b64decode(blob))
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'Embedded file changed: {{name}}')
    path = scratch / name
    path.write_bytes(raw)
    return path

baseline_path = mount('baseline.zip')
with ZipFile(baseline_path) as baseline:
    original = json.loads(baseline.read('pilot_arm_receipt.json'))
scored = mount('msalign_receipt.json')
receipt = json.loads(scored.read_text())
if (
    receipt['uploaded_truth'] is not False
    or receipt['queries_sha256'] != {query_payload[1]!r}
    or receipt['queries_sha256'] != original['queries_sha256']
    or receipt['baseline_sha256'] != original['hashes']['submission.csv']
):
    raise ValueError('The MSAlign score receipt does not match the frozen blind baseline')
for name, key in (
    ('msalign_scores.csv', 'scores_sha256'),
    ('msalign_spectrum_scores.csv', 'spectra_scores_sha256'),
):
    with (scored.parent / name).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != receipt[key]:
            raise ValueError(f'Prediction artifact changed: {{name}}')

{METRIC_BOOTSTRAP}

for name, payload in {program!r}.items():
    restore(name, payload)
private = restore('labels.parquet', {label_payload!r})
command = [sys.executable, str(scratch / 'compare.py'),
           '--baseline', str(baseline_path), '--queries', str(scratch / 'test.parquet'),
           '--predictions', str(scored.parent), '--out', '/kaggle/working',
           '--labels', str(private), '--label-sha256', {label_payload[1]!r},
           '--exposure', str(scratch / 'exposure.json')]
subprocess.run(command, check=True)
'''
    compile(source, SLUG, 'exec')
    book = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        cells=[assemble.code_cell('compare-msalign', source)])
    meta = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='compare.ipynb',
                language='python', kernel_type='notebook', is_private=True, enable_gpu=False,
                enable_tpu=False, enable_internet=False, dataset_sources=['metric/rdkit-2026-3-3-wheel'],
                kernel_sources=[args.baseline_kernel, args.scoring_kernel], competition_sources=[])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'compare.ipynb').write_text(json.dumps(book, separators=(',', ':')) + '\n')
    (args.out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'Created separate private evaluator in {args.out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--baseline-kernel', default='jiyujia/enveda-075-np-recall-reseal/1')
    parser.add_argument('--scoring-kernel', required=True)
    build(parser.parse_args())
