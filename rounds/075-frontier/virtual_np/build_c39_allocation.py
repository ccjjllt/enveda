"""Build an offline CPU planner from completed, label-blind C39 and full-window candidate archives."""
import argparse
import json
from pathlib import Path
import re
from runpy import run_path

from allocate_c39_shadow import SHORTLISTS
from build import HERE, METRIC_BOOTSTRAP, pack


SEALED = 'jiyujia/enveda-075-np-recall-reseal/2'
IMAGE = run_path(str(HERE.parent/'feasibility'/'build.py'))['KAGGLE_IMAGE']
SLUG = 'enveda-075-c39-shadow-allocate'


def build(snapshot, out, candidate_source='unseen'):
    if not re.fullmatch(r'jiyujia/enveda-075-c39-snapshot-(cpu|t4)/[1-9][0-9]*',snapshot):
        raise ValueError('Specify the exact completed, private C39 snapshot version')
    source = {name:pack((HERE/name).read_bytes()) for name in ('allocate_c39_shadow.py','c39_request_cache.py')}
    ref, receipt_file, *_ = SHORTLISTS[candidate_source]
    setup = f"""
import base64, hashlib, subprocess, sys, tempfile, zlib
from pathlib import Path
inputs = Path('/kaggle/input')
def mounted(ref, name):
    slug = ref.split('/')[1]
    path, = (p for p in inputs.rglob(name) if slug in p.parts)
    return path
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='c39-allocation-') as temporary:
    scratch = Path(temporary)
    for name,(encoded,expected) in {source!r}.items():
        raw = zlib.decompress(base64.b64decode(encoded))
        if hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError(f'Unexpected entry program: {{name}}')
        (scratch/name).write_bytes(raw)
    sys.path.insert(0,str(scratch))
    from allocate_c39_shadow import allocate
    allocate(mounted({snapshot!r},'c39_snapshot_receipt.json').parent,
             mounted({ref!r},{receipt_file!r}).parent,
             mounted({SEALED!r},'baseline.zip'),Path('/kaggle/working'),scratch,candidate_source={candidate_source!r})
"""
    compile(setup,'c39-allocation','exec')
    cell = dict(cell_type='code',id='fixed-c39-allocation',metadata={},execution_count=None,outputs=[],
                source=setup.strip().splitlines(True))
    book = dict(nbformat=4,nbformat_minor=5,cells=[cell],
                metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    meta = dict(id='jiyujia/'+SLUG,title=SLUG,code_file='allocate.ipynb',language='python',kernel_type='notebook',
                is_private=True,enable_gpu=False,enable_internet=False,
                dataset_sources=['metric/rdkit-2026-3-3-wheel'],competition_sources=[],
                kernel_sources=[snapshot,ref,SEALED],docker_image=IMAGE)
    out.mkdir(parents=True,exist_ok=True)
    (out/'allocate.ipynb').write_text(json.dumps(book,separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--candidate-source',choices=tuple(SHORTLISTS),default='unseen')
    build(**vars(parser.parse_args()))
