"""Build private CPU scoring only after all three final lists have been sealed."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack


BLIND = 'enveda-075-final-native-top5'
BASE = 'enveda-075-np-recall-reseal'
SLUG = 'enveda-075-final-native-top5-compare'
SEAL = 'final_native_top5_receipt.json'


def build(panel, receipt, out, version):
    raw = receipt.read_bytes()
    seal = json.loads(raw)
    meta = json.loads((panel/'panel_manifest.json').read_bytes())['output_sha256']
    query_sha, label_sha = (meta[name] for name in ('blind/test.parquet','private/labels.parquet'))
    if (version < 1 or seal['status'] != 'sealed' or seal['queries_sha256'] != query_sha
            or seal['molecules'] != 60 or seal['arm_names'] != ['historical','protected','open']
            or seal['new_predictor_requests'] != 0 or seal['actual_slots'] != 1476):
        raise ValueError('Expected the officially sealed final-list test')
    label = pack((panel/'private/labels.parquet').read_bytes())
    if label[1] != label_sha:
        raise ValueError('The exact authorized answers have changed')
    paths = {'score_final_native_top5.py':HERE/'score_final_native_top5.py',
             'evaluate.py':HERE/'evaluate.py',
             'retrieval_pilot_score.py':HERE.parent/'four_arm'/'retrieval_pilot_score.py'}
    blobs = {name:pack(path.read_bytes()) for name,path in paths.items()}
    blobs['labels.parquet'] = label
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

def mounted(slug, name):
    file, = (path for path in inputs.rglob(name) if slug in path.parts)
    return file

with tempfile.TemporaryDirectory(prefix='final-native-top5-score-') as directory:
    root = Path(directory)
    for name, (blob, expected) in {blobs!r}.items():
        value = zlib.decompress(base64.b64decode(blob))
        if sha256(value).hexdigest() != expected:
            raise ValueError(f'Private score input changed: {{name}}')
        (root/name).write_bytes(value)
    command = [sys.executable, str(root/'score_final_native_top5.py'), '--blind', str(mounted({BLIND!r}, {SEAL!r}).parent),
               '--baseline', str(mounted({BASE!r}, 'baseline.zip')), '--labels', str(root/'labels.parquet'),
               '--query-sha256', {query_sha!r}, '--label-sha256', {label_sha!r}, '--seal-sha256', {sha256(raw).hexdigest()!r},
               '--out', '/kaggle/working']
    subprocess.run(command, check=True)
'''
    compile(runtime, SLUG, 'exec')
    book = dict(nbformat=4,nbformat_minor=5, metadata={'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}},
                cells=[dict(cell_type='code',id='final-native-top5-score',metadata={},execution_count=None,outputs=[],
                            source=runtime.strip().splitlines(keepends=True))])
    metadata = dict(id=f'jiyujia/{SLUG}',title=SLUG,code_file='compare.ipynb',language='python',kernel_type='notebook',
                    is_private=True,enable_gpu=False,enable_internet=False,competition_sources=[],
                    dataset_sources=[WHEEL],docker_image=IMAGE,kernel_sources=[f'jiyujia/{BASE}/2',f'jiyujia/{BLIND}/{version}'])
    out.mkdir(parents=True,exist_ok=True)
    (out/'compare.ipynb').write_text(json.dumps(book,separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for option in ('panel','receipt','out'):
        cli.add_argument('--'+option, type=Path, required=True)
    cli.add_argument('--version', type=int, required=True)
    build(**vars(cli.parse_args()))
