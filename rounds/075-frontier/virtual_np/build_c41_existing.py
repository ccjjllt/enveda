"""Package the two existing-cache promotion arms and the frozen original-slot control in one CPU run."""
import argparse
import json
from pathlib import Path
from runpy import run_path

from build import HERE, METRIC_BOOTSTRAP, pack
from build_c39_snapshot import extract


SNAPSHOT='jiyujia/enveda-075-c39-snapshot-t4/1'
CONTROL='jiyujia/enveda-075-c41-shadow-control/1'
UNSEEN='jiyujia/enveda-075-full-main-unseen/1'
BASE='jiyujia/enveda-075-np-recall-reseal/2'
SLUG='enveda-075-c41-existing-c43'
IMAGE=run_path(str(HERE.parent/'feasibility/build.py'))['KAGGLE_IMAGE']


def build(launched,out):
    original,_,_=extract(launched)
    gate=original[43].split("print('merge stats', stats)",1)[0]
    if not gate.endswith("    rows.append((mid, ';'.join(final[:25])))\n"):
        raise ValueError('Expected the original complete C43 loop')
    files={n:(HERE/n).read_bytes() for n in ('c41_existing_study.py','c41_promote_existing.py',
                                              'allocate_c39_shadow.py','c39_request_cache.py')}
    files['original_c43.py']=gate.encode()
    for name,data in files.items():compile(data,name,'exec')
    blobs={name:pack(data) for name,data in files.items()}
    code=f"""
import base64,hashlib,subprocess,sys,tempfile,zlib
from pathlib import Path
inputs,out=Path('/kaggle/input'),Path('/kaggle/working')
def mounted(ref,name):
    slug=ref.split('/')[1]
    path, = (p for p in inputs.rglob(name) if slug in p.parts)
    return path
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='c41-scored-e1-') as tmp:
    root=Path(tmp)
    for name,(encoded,expected) in {blobs!r}.items():
        data=zlib.decompress(base64.b64decode(encoded))
        if hashlib.sha256(data).hexdigest()!=expected:
            raise ValueError(f'Changed component: {{name}}')
        (root/name).write_bytes(data)
    sys.path.insert(0,str(root))
    from c41_existing_study import run
    run(mounted({SNAPSHOT!r},'c39_snapshot_receipt.json').parent,
        mounted({CONTROL!r},'c41_study_receipt.json').parent,
        mounted({UNSEEN!r},'unseen_relist_receipt.json').parent,mounted({BASE!r},'baseline.zip'),
        (root/'original_c43.py').read_text(),out)
"""
    compile(code,'original-c41-footprint-only','exec')
    cell=dict(id='promote-already-scored',cell_type='code',metadata={},outputs=[],execution_count=None,source=code.splitlines(True))
    book=dict(nbformat=4,nbformat_minor=5,cells=[cell],metadata={'kernelspec':dict(name='python3',language='python',display_name='Python 3')})
    meta=dict(id='jiyujia/'+SLUG,title=SLUG,language='python',kernel_type='notebook',code_file='existing.ipynb',
              is_private=True,enable_gpu=False,enable_internet=False,docker_image=IMAGE,
              dataset_sources=['metric/rdkit-2026-3-3-wheel'],kernel_sources=[SNAPSHOT,CONTROL,UNSEEN,BASE],competition_sources=[])
    out.mkdir(parents=True,exist_ok=True)
    (out/'existing.ipynb').write_text(json.dumps(book,separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launched',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    build(**vars(p.parse_args()))
