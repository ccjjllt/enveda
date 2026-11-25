"""Package a selected existing-cache C41 treatment for Kaggle T4."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from build import HERE, METRIC_BOOTSTRAP, pack
from build_c39_snapshot import CONTROL_SHA
from c41_existing_full import ARM_OPTIONS
from c41_patch_full import patch


SNAP='jiyujia/enveda-075-c39-snapshot-t4/1'
OLD='jiyujia/enveda-075-c41-shadow-control/1'
EXISTING='jiyujia/enveda-075-c41-existing-c43/1'
NATIVE_ONLY='jiyujia/enveda-075-c41-native-only-c43/1'
SHARED='jiyujia/enveda-075-np-recall-reseal/2'


def build(launched,out,arm):
    if arm not in ARM_OPTIONS:raise ValueError('Select exactly one experiment')
    meta=json.loads((launched/'kernel-metadata.json').read_bytes())
    raw=(launched/meta['code_file']).read_bytes()
    if sha256(raw).hexdigest()!=CONTROL_SHA:raise ValueError('Expected the original launched notebook /2')
    routed=patch(json.loads(raw))
    for cell in routed['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),cell.get('id','full'),'exec')
    files={name:(HERE/name).read_bytes() for name in ('c41_existing_full.py','c41_full_run.py')}
    files['retrieval_pilot_score.py']=(HERE.parent/'four_arm/retrieval_pilot_score.py').read_bytes()
    for name,data in files.items():compile(data,name,'exec')
    files['outer.ipynb']=json.dumps(routed,ensure_ascii=False,separators=(',',':')).encode()
    blobs={name:pack(data) for name,data in files.items()}
    evidence_ref, evidence_file = ((NATIVE_ONLY,'c41_native_only_receipt.json') if arm=='native_only' else
                                   (EXISTING,'c41_existing_receipt.json'))
    existing_mount = 'None' if arm=='original_slots' else f'mounted({evidence_ref!r},{evidence_file!r})'
    code=f"""
import base64,hashlib,subprocess,sys,tempfile,zlib
from pathlib import Path
inputs,out=Path('/kaggle/input'),Path('/kaggle/working')
def mounted(ref,name):
    slug=ref.split('/')[1]
    item, = (p for p in inputs.rglob(name) if slug in p.parts)
    return item.parent
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='existing-c41-outer-') as tmp:
    root=Path(tmp)
    for name,(encoded,expected) in {blobs!r}.items():
        data=zlib.decompress(base64.b64decode(encoded))
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError(f'Changed source: {{name}}')
        (root/name).write_bytes(data)
    sys.path.insert(0,str(root))
    from c41_existing_full import run
    run(root/'outer.ipynb',mounted({SNAP!r},'c39_snapshot_receipt.json'),
        mounted({OLD!r},'c41_study_receipt.json'),{existing_mount},
        mounted({SHARED!r},'baseline.zip'),out,{arm!r})
"""
    compile(code,'existing-c41-full','exec')
    slug='enveda-075-c41-existing-full-'+arm.replace('_','-')
    cell=dict(id='full-existing-pair',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))
    book=dict(nbformat=4,nbformat_minor=5,cells=[cell],metadata={'kernelspec':dict(name='python3',language='python',display_name='Python 3')})
    meta.update(id='jiyujia/'+slug,title=slug,code_file='existing_full.ipynb',kernel_type='notebook',language='python',
                enable_gpu=True,enable_internet=False,is_private=True,machine_shape='NvidiaTeslaT4',
                kernel_sources=[*meta['kernel_sources'],SNAP,OLD,*([evidence_ref] if arm!='original_slots' else []),SHARED])
    out.mkdir(parents=True,exist_ok=True)
    p=out/'existing_full.ipynb'
    p.write_text(json.dumps(book,separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(slug,p.stat().st_size,sha256(p.read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launched',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--arm',required=True,choices=ARM_OPTIONS)
    build(**vars(p.parse_args()))
