"""Package a blind CPU attribution of native and shadow forward evidence on the original C41-to-C43 path."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path

from build import HERE, pack
from build_c39_snapshot import extract


SNAPSHOT='jiyujia/enveda-075-c39-snapshot-t4/1'
CONTROL='jiyujia/enveda-075-c41-shadow-control/1'
BASE='jiyujia/enveda-075-np-recall-reseal/2'
ASSET='dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets'
SLUG='enveda-075-c41-native-only-c43'
IMAGE=run_path(str(HERE.parent/'feasibility/build.py'))['KAGGLE_IMAGE']


def build(launched,out):
    inner,_,_=extract(launched)
    gate=inner[43].split("print('merge stats', stats)",1)[0]
    if not gate.endswith("    rows.append((mid, ';'.join(final[:25])))\n"):
        raise ValueError('Unexpected original C43 source')
    parts={n:(HERE/n).read_bytes() for n in ('c41_native_panel.py','native_only_c43.py')}
    parts['original_c43.py']=gate.encode()
    for name,code in parts.items():compile(code,name,'exec')
    blobs={name:pack(code) for name,code in parts.items()}
    code=f"""
import base64,hashlib,sys,tempfile,zlib
from pathlib import Path
inputs, out = Path('/kaggle/input'), Path('/kaggle/working')
def mounted(ref,name):
    slug=ref.split('/')[1]
    path, = (p for p in inputs.rglob(name) if slug in p.parts)
    return path
with tempfile.TemporaryDirectory(prefix='native-only-c43-') as tmp:
    scratch=Path(tmp)
    for name,(encoded,expected) in {blobs!r}.items():
        data=zlib.decompress(base64.b64decode(encoded))
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError(f'Changed source for {{name}}')
        (scratch/name).write_bytes(data)
    sys.path.insert(0,str(scratch))
    from native_only_c43 import run
    run(mounted({SNAPSHOT!r},'c39_snapshot_receipt.json').parent,mounted({CONTROL!r},'c41_study_receipt.json').parent,
        mounted({BASE!r},'baseline.zip'),mounted({ASSET!r},'gl_fuse.py'),(scratch/'original_c43.py').read_text(),out)
"""
    compile(code,'native-only-c43','exec')
    cell=dict(id='native-only-attribution',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))
    book=dict(nbformat=4,nbformat_minor=5,cells=[cell],
              metadata={'kernelspec':dict(name='python3',language='python',display_name='Python 3')})
    meta=dict(id='jiyujia/'+SLUG,title=SLUG,language='python',kernel_type='notebook',code_file='attribute.ipynb',
              is_private=True,enable_gpu=False,enable_internet=False,docker_image=IMAGE,
              dataset_sources=[ASSET],kernel_sources=[SNAPSHOT,CONTROL,BASE],competition_sources=[])
    out.mkdir(exist_ok=True,parents=True)
    target=out/'attribute.ipynb'
    target.write_text(json.dumps(book,separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(SLUG,target.stat().st_size,sha256(target.read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launched',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    build(**vars(p.parse_args()))
