"""Package independent public-input CPU stages; no data uploads, pushes or chemistry occur during packaging."""
import argparse
import json
from pathlib import Path
import re

from build import IMAGE, METRIC_BOOTSTRAP, WHEEL, pack
from build_parents import COCONUT_STAGING, COMPETITION, LEGACY_POOL


HERE=Path(__file__).resolve().parent
PREFIX='enveda-075-parent-v2'
INDEX=f'jiyujia/{PREFIX}-index/1'
V1_SOURCE_SHA='62e8464079bad7c4f7bd8b22c35c7d618445bfa224f2bafb40423328d82084eb'


def worker_slug(lane,generation):
    return f'{PREFIX}-s{lane:02}-r{generation:02}'


def versioned(slug):
    if not re.fullmatch(r'jiyujia/[a-z0-9-]+/[1-9][0-9]*',slug):raise ValueError('Pin a private completed notebook version')
    return slug.split('/')[1]


def build(stage, out, lanes=8, lane=0, workers=4, soft_seconds=5400, generation=1, previous=None,
          index_ref=INDEX, shard_refs=None):
    if not 1<=lanes<=16 or not 0<=lane<lanes or not 1<=workers<=4 or soft_seconds<=0 or generation<1:
        raise ValueError('Unexpected lane count, lane position, worker count, generation or wall budget')
    index_slug=versioned(index_ref)
    base=(HERE/'rebuild_parents.py').read_bytes()
    v1=pack(base)
    if v1[1]!=V1_SOURCE_SHA:raise ValueError('The immutable v1 normalization/overlap implementation changed')
    data, kernels, competition=[WHEEL], [], []
    if stage=='index':
        name='parent_raw_index.py';slug=PREFIX+'-index'
        kernels=[COCONUT_STAGING];competition=[COMPETITION]
        args=f"""train=mounted({COMPETITION!r},'train.parquet')
coconut=mounted({versioned(COCONUT_STAGING)!r},'coconut-09-2024.csv.zip')
command=['--train',str(train),'--archive',str(coconut),'--source-receipt',str(coconut.with_name('coconut_parent_source.json')),
         '--out','/kaggle/working','--lanes',{str(lanes)!r}]"""
    elif stage=='worker':
        name='parent_shard_worker.py';slug=worker_slug(lane,generation);kernels=[index_ref]
        if previous:
            prev=versioned(previous)
            if prev==slug or not re.fullmatch(re.escape(f'{PREFIX}-s{lane:02}-r')+r'[0-9]+',prev):
                raise ValueError('Use an earlier distinct notebook of the same lane to resume')
            kernels.append(previous)
        args=f"""idx=mounted({index_slug!r},'parent_raw_index.json')
if json.loads(idx.read_bytes())['lanes']!={lanes}:raise ValueError('Index lane count differs')
command=['--index',str(idx),'--lane',{str(lane)!r},'--workers',{str(workers)!r},
         '--soft-seconds',{str(soft_seconds)!r},'--out','/kaggle/working']"""
        if previous:
            args+=f"\ncommand.extend(['--previous',str(mounted({prev!r},'parent_shard_receipt.json').parent)])"
    elif stage=='merge':
        name='parent_shard_merge.py';slug=PREFIX+'-merge'
        refs=shard_refs or [f'jiyujia/{worker_slug(i,1)}/1' for i in range(lanes)]
        if len(refs)!=lanes or len(set(refs))!=lanes:
            raise ValueError('Specify one completed final version for every lane')
        slugs=[versioned(r) for r in refs]
        if set(s.rsplit('-r',1)[0] for s in slugs)!={f'{PREFIX}-s{i:02}' for i in range(lanes)}:
            raise ValueError('Missing or duplicate source lane')
        kernels=[index_ref,*refs];data.append(LEGACY_POOL)
        args=f"""idx=mounted({index_slug!r},'parent_raw_index.json')
command=['--index',str(idx),'--legacy',str(mounted('casmi26-v2-pool','pool_meta.parquet')),
         '--out','/kaggle/working','--shard-dirs',*[str(mounted(s,'parent_shard_receipt.json').parent) for s in {slugs!r}]]"""
    else:raise ValueError('Expected index, worker or merge')
    code=(HERE/name).read_bytes();compile(base,'rebuild_parents.py','exec');compile(code,name,'exec')
    files={'rebuild_parents.py':v1,name:pack(code)}
    program=f"""
import base64,hashlib,json,subprocess,sys,tempfile,zlib
from pathlib import Path
inputs=Path('/kaggle/input')
def mounted(slug,name):
    p,=(p for p in inputs.rglob(name) if slug.casefold() in [s.casefold() for s in p.parts])
    return p
{METRIC_BOOTSTRAP}
{args}
with tempfile.TemporaryDirectory(prefix='parent-rebuild-') as tmp:
    root=Path(tmp)
    for file,(encoded,expected) in {files!r}.items():
        raw=zlib.decompress(base64.b64decode(encoded))
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError(f'Changed source {{file}}')
        (root/file).write_bytes(raw)
    subprocess.run([sys.executable,str(root/{name!r}),*command],check=True)
"""
    compile(program,slug,'exec')
    cell=dict(id='parent-'+stage,cell_type='code',metadata={},execution_count=None,outputs=[],source=program.strip().splitlines(True))
    book=dict(nbformat=4,nbformat_minor=5,cells=[cell],metadata={'kernelspec':dict(name='python3',language='python',display_name='Python 3')})
    meta=dict(id='jiyujia/'+slug,title=slug,code_file='parent.ipynb',language='python',kernel_type='notebook',
              is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,dataset_sources=data,
              kernel_sources=kernels,competition_sources=competition,docker_image=IMAGE)
    out.mkdir(exist_ok=True,parents=True)
    (out/'parent.ipynb').write_text(json.dumps(book,separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('index','worker','merge','all'))
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--lanes',type=int,default=8);p.add_argument('--lane',type=int,default=0)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--soft-seconds',type=int,default=5400)
    p.add_argument('--generation',type=int,default=1);p.add_argument('--previous')
    p.add_argument('--index-ref',default=INDEX);p.add_argument('--shard-refs',nargs='+')
    a=vars(p.parse_args())
    if a['stage']=='all':
        if a['previous'] or a['generation']!=1: p.error('Package continuations per specific lane')
        root=a['out']
        build(**dict(a,stage='index',out=root/'index'))
        for i in range(a['lanes']):build(**dict(a,stage='worker',lane=i,out=root/f's{i:02}'))
        build(**dict(a,stage='merge',out=root/'merge'))
    else:
        build(**a)
