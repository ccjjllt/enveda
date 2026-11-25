"""Build a private final comparison from a sealed fixed-budget C41 fusion."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack


HISTORICAL = 'enveda-075-np-recall-reseal'
ARMS = ('original_slots', 'native_only')


def build(panel, receipt, version, out, arm='original_slots'):
    if arm not in ARMS:
        raise ValueError('Unknown C41 arm')
    arm_slug = arm.replace('_', '-')
    fusion = f'enveda-075-c41-existing-full-{arm_slug}'
    slug = f'enveda-075-c41-{arm_slug}-final-compare'
    body = receipt.read_bytes()
    seal = json.loads(body)
    meta = json.loads((panel/'panel_manifest.json').read_text())['output_sha256']
    label_sha = meta['private/labels.parquet']
    query_sha = meta['blind/test.parquet']
    if (version < 1 or seal['arm'] != arm or seal['status'] != 'complete'
            or seal['molecules'] != 60 or seal['candidate_slots'] != 25
            or seal['queries_sha256'] != query_sha or seal['new_predictor_requests'] != 0):
        raise ValueError('Expected the complete fixed-budget fusion on the frozen panel')
    files = {
        'labels.parquet':pack((panel/'private/labels.parquet').read_bytes()),
        'score_original_slots_final.py':pack((HERE/'score_original_slots_final.py').read_bytes()),
        'evaluate.py':pack((HERE/'evaluate.py').read_bytes()),
        'retrieval_pilot_score.py':pack((HERE.parent/'four_arm/retrieval_pilot_score.py').read_bytes()),
    }
    if files['labels.parquet'][1] != label_sha:
        raise ValueError('Private labels no longer match the authorized dataset')
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
    output, = (p for p in inputs.rglob(name) if slug in p.parts)
    return output

with tempfile.TemporaryDirectory(prefix='original-slots-final-') as temp:
    root = Path(temp)
    for name, (encoded, expected) in {files!r}.items():
        value = zlib.decompress(base64.b64decode(encoded))
        if sha256(value).hexdigest() != expected:
            raise ValueError(f'Private input digest differs: {{name}}')
        (root/name).write_bytes(value)
    subprocess.run([sys.executable, str(root/'score_original_slots_final.py'),
                    '--baseline', str(mounted('baseline.zip',{HISTORICAL!r})),
                    '--fusion', str(mounted({(arm+'.zip')!r},{fusion!r})), '--arm', {arm!r},
                    '--labels', str(root/'labels.parquet'), '--seal-sha256', {sha256(body).hexdigest()!r},
                    '--label-sha256', {label_sha!r}, '--query-sha256', {query_sha!r}, '--out', '/kaggle/working'], check=True)
'''
    compile(runtime, slug, 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata={'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'}},
                    cells=[dict(cell_type='code', id='original-slots-final', metadata={}, execution_count=None,
                                outputs=[], source=runtime.strip().splitlines(keepends=True))])
    metadata = dict(id=f'jiyujia/{slug}', title=slug, code_file='compare.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=False, enable_internet=False,
                    dataset_sources=[WHEEL], docker_image=IMAGE, competition_sources=[],
                    kernel_sources=[f'jiyujia/{HISTORICAL}/2', f'jiyujia/{fusion}/{version}'])
    out.mkdir(parents=True, exist_ok=True)
    (out/'compare.ipynb').write_text(json.dumps(notebook,separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('panel','receipt','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--version',type=int,required=True)
    parser.add_argument('--arm',choices=ARMS,default='original_slots')
    build(**vars(parser.parse_args()))
