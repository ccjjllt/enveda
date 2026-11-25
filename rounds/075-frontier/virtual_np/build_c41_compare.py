"""Package the private CPU scorecard for official, already-sealed C41 fusion arms."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack


BASE = 'enveda-075-np-recall-reseal'
SLUG = 'enveda-075-c41-fusion-compare'


def build(fusion, panel, original_slots_receipt, replacement_receipt, out):
    if not re.fullmatch(r'jiyujia/[a-z0-9-]+/[1-9][0-9]*', fusion):
        raise ValueError('Provide the official private Kaggle version that sealed both fusion arms')
    summary = json.loads((panel/'panel_manifest.json').read_text())['output_sha256']
    query_sha = summary['blind/test.parquet']
    seals = {}
    for arm, receipt in (('original_slots', original_slots_receipt), ('replacement', replacement_receipt)):
        raw = receipt.read_bytes()
        data = json.loads(raw)
        if (data['arm'] != arm or data['status'] != 'complete' or data['candidate_slots'] != 25
                or data['queries_sha256'] != query_sha or not data['hashes']):
            raise ValueError(f'Expected the completed blind seal for {arm}')
        seals[arm] = sha256(raw).hexdigest()

    expected = summary['private/labels.parquet']
    labels = pack((panel/'private/labels.parquet').read_bytes())
    if labels[1] != expected:
        raise ValueError('The private labels differ from the panel manifest')
    files = {
        'labels.parquet': labels,
        'compare_c41_fusion.py': pack((HERE/'compare_c41_fusion.py').read_bytes()),
        'evaluate.py': pack((HERE/'evaluate.py').read_bytes()),
        'retrieval_pilot_score.py': pack((HERE.parent/'four_arm/retrieval_pilot_score.py').read_bytes()),
    }
    fusion_slug = fusion.split('/')[1]
    runtime = f'''
import base64
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

inputs = Path('/kaggle/input')
{METRIC_BOOTSTRAP}

def mounted(name, slug):
    path, = (entry for entry in inputs.rglob(name) if slug in entry.parts)
    return path

with tempfile.TemporaryDirectory(prefix='c41-fusion-score-') as directory:
    scratch = Path(directory)
    for name, (blob, checksum) in {files!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != checksum:
            raise ValueError(f'Private scoring artifact changed: {{name}}')
        (scratch/name).write_bytes(raw)
    subprocess.run([sys.executable, str(scratch/'compare_c41_fusion.py'),
                    '--baseline-zip', str(mounted('baseline.zip', {BASE!r})),
                    '--original-slots', str(mounted('original_slots.zip', {fusion_slug!r})),
                    '--replacement', str(mounted('replacement.zip', {fusion_slug!r})),
                    '--original-slots-receipt-sha', {seals['original_slots']!r},
                    '--replacement-receipt-sha', {seals['replacement']!r},
                    '--labels', str(scratch/'labels.parquet'), '--labels-sha', {expected!r},
                    '--out', '/kaggle/working'], check=True)
'''
    compile(runtime, SLUG, 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                    cells=[dict(cell_type='code', id='c41-fusion-compare', execution_count=None, metadata={},
                                outputs=[], source=runtime.strip().splitlines(keepends=True))])
    metadata = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='compare.ipynb', language='python',
                    kernel_type='notebook', is_private=True, enable_gpu=False, enable_internet=False,
                    dataset_sources=[WHEEL], kernel_sources=[f'jiyujia/{BASE}/2', fusion],
                    competition_sources=[], docker_image=IMAGE)
    out.mkdir(parents=True, exist_ok=True)
    (out/'compare.ipynb').write_text(json.dumps(notebook, separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('panel', 'original-slots-receipt', 'replacement-receipt', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--fusion', required=True)
    build(**vars(parser.parse_args()))
