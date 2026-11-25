"""Build two separately executed original Pablo full pipelines from sealed paired C41 results."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import zlib

from build import HERE, METRIC_BOOTSTRAP, pack
from build_c39_snapshot import CONTROL_SHA
from c41_patch_full import patch


SLUG = 'enveda-075-c41-shadow-full'
SHARED = 'jiyujia/enveda-075-np-recall-reseal/2'


def build(launched, snapshot, allocation, paired, out):
    refs=((snapshot,'enveda-075-c39-snapshot-t4'),(allocation,'enveda-075-c39-shadow-allocate'),
          (paired,'enveda-075-c41-shadow-paired'))
    if not all(re.fullmatch('jiyujia/'+name+'/[1-9][0-9]*',ref) for ref,name in refs):
        raise ValueError('Specify exact upstream versions for the same T4/670-slot study')
    meta=json.loads((launched/'kernel-metadata.json').read_bytes())
    source=(launched/meta['code_file']).read_bytes()
    if sha256(source).hexdigest()!=CONTROL_SHA:
        raise ValueError('Expected the originally launched Pablo v29 baseline/2')
    payload=patch(json.loads(source))
    for cell in payload['cells']:
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),cell.get('id','pipeline'),'exec')
    raw=json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode()
    blobs={name:pack(content) for name,content in {
        'outer.ipynb':raw,'c41_full_run.py':(HERE/'c41_full_run.py').read_bytes(),
        'retrieval_pilot_score.py':(HERE.parent/'four_arm/retrieval_pilot_score.py').read_bytes()}.items()}
    code=f"""
import base64,hashlib,subprocess,sys,tempfile,zlib
from pathlib import Path
inputs,out=Path('/kaggle/input'),Path('/kaggle/working')
def mounted(ref, name):
    slug=ref.split('/')[1]
    match, = (p for p in inputs.rglob(name) if slug in p.parts)
    return match.parent
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='c41-outer-') as tmp:
    root=Path(tmp)
    for name,(encoded,expected) in {blobs!r}.items():
        b=zlib.decompress(base64.b64decode(encoded))
        if hashlib.sha256(b).hexdigest()!=expected:
            raise ValueError(f'Unexpected full-pipeline program {{name}}')
        (root/name).write_bytes(b)
    sys.path.insert(0,str(root))
    from c41_full_run import run_full
    run_full(root/'outer.ipynb',mounted({snapshot!r},'c39_snapshot_receipt.json'),
             mounted({allocation!r},'allocation_receipt.json'),mounted({paired!r},'c41_study_receipt.json'),
             mounted({SHARED!r},'baseline.zip'),out)
"""
    compile(code,'c41-complete-pair','exec')
    cell=dict(id='two-full-arms',cell_type='code',metadata={},outputs=[],execution_count=None,source=code.splitlines(True))
    book=dict(nbformat=4,nbformat_minor=5,cells=[cell],metadata={'kernelspec':dict(name='python3',language='python',display_name='Python 3')})
    meta.update(id='jiyujia/'+SLUG,title=SLUG,language='python',kernel_type='notebook',code_file='full.ipynb',
                is_private=True,enable_gpu=True,enable_internet=False,machine_shape='NvidiaTeslaT4',
                kernel_sources=[*meta['kernel_sources'],snapshot,allocation,paired,SHARED])
    out.mkdir(parents=True,exist_ok=True)
    path=out/'full.ipynb'
    path.write_text(json.dumps(book,separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(slug=SLUG,notebook_bytes=path.stat().st_size,notebook_sha256=sha256(path.read_bytes()).hexdigest())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('launched','out'):p.add_argument('--'+n,required=True,type=Path)
    for n in ('snapshot','allocation','paired'):p.add_argument('--'+n,required=True)
    build(**vars(p.parse_args()))
