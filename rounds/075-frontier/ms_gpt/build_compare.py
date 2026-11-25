"""Pack a private CPU comparator once the MS-GPT candidate lists are sealed."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
VIRTUAL = HERE.parent / 'virtual_np'
sys.path.insert(0, str(VIRTUAL))
from build import IMAGE, METRIC_BOOTSTRAP, WHEEL, pack


GENERATOR = 'enveda-075-ms-gpt-blind-c34'
BASELINE = 'enveda-075-np-recall-reseal'
SLUG = 'enveda-075-ms-gpt-shortlist-compare'


def build(panel, receipt, version, out):
    raw = receipt.read_bytes()
    receipt_sha = sha256(raw).hexdigest()
    metadata = json.loads(raw)
    if version < 1 or (metadata['all_queries'], metadata['candidate_cap'], metadata['per_active_query_draws']) != (60, 25, 100):
        raise ValueError('Expected a sealed, versioned 60-query blind output')
    active = metadata['active_queries']
    if not 0 < active <= 60 or any(metadata['counts'][f'{arm}_draws'] != active * 100
                                   for arm in ('top1', 'top5')):
        raise ValueError('The blind receipt does not account for 100 draws per active query and arm')
    hashes = json.loads((panel / 'panel_manifest.json').read_text())['output_sha256']
    query_sha = hashes['blind/test.parquet']
    if metadata['query_sha256'] != query_sha or sha256((panel / 'blind/test.parquet').read_bytes()).hexdigest() != query_sha:
        raise ValueError('MS-GPT predictions use different spectra from the panel manifest')
    expected = hashes['private/labels.parquet']
    files = {
        'labels.parquet': pack((panel / 'private/labels.parquet').read_bytes()),
        'compare_blind.py': pack((HERE / 'compare_blind.py').read_bytes()),
        'evaluate.py': pack((VIRTUAL / 'evaluate.py').read_bytes()),
        'retrieval_pilot_score.py': pack((HERE.parent / 'four_arm/retrieval_pilot_score.py').read_bytes()),
    }
    if files['labels.parquet'][1] != expected:
        raise ValueError('Private labels changed from the panel manifest')
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
seal, = (p for p in inputs.rglob('baseline.zip') if {BASELINE!r} in p.parts)
generated, = (p.parent for p in inputs.rglob('ms_gpt_blind_receipt.json') if {GENERATOR!r} in p.parts)
with tempfile.TemporaryDirectory(prefix='ms-gpt-score-') as directory:
    scratch = Path(directory)
    for name, (blob, expected) in {files!r}.items():
        raw = zlib.decompress(base64.b64decode(blob))
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f'Private scoring input changed: {{name}}')
        (scratch / name).write_bytes(raw)
    subprocess.run([sys.executable, str(scratch / 'compare_blind.py'), '--baseline-zip', str(seal),
                    '--generated', str(generated), '--labels', str(scratch / 'labels.parquet'),
                    '--labels-sha256', {expected!r}, '--receipt-sha256', {receipt_sha!r},
                    '--out', '/kaggle/working'], check=True)
'''
    compile(runtime, SLUG, 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                    cells=[dict(cell_type='code', id='ms-gpt-compare', execution_count=None, metadata={},
                                outputs=[], source=runtime.strip().splitlines(keepends=True))])
    kernel = dict(id=f'jiyujia/{SLUG}', title=SLUG, code_file='compare.ipynb', language='python',
                  kernel_type='notebook', is_private=True, enable_gpu=False, enable_internet=False,
                  dataset_sources=[WHEEL], kernel_sources=[f'jiyujia/{BASELINE}/2', f'jiyujia/{GENERATOR}/{version}'],
                  competition_sources=[], docker_image=IMAGE)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'compare.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(kernel, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('panel', 'receipt', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--version', type=int, required=True)
    build(**vars(parser.parse_args()))
