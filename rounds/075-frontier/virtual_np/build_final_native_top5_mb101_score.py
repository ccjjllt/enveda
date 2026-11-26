"""Build a private CPU evaluation of the sealed 101-query MassBank 2026.03 top-five permutations."""
import argparse
import ast
from collections import Counter
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re
import tarfile

from build_final_native_top5_e180_score import ARMS, comparison, mounted, rr, totals, verified


SLUG = 'enveda-075-final-native-top5-mb101-score'
BLIND = 'enveda-075-final-native-top5-mb101'
DATA = 'enveda-075b-202603-final-ranking'
SCORED = 'enveda-075b-massbank-202603-pc50-verify'
SCORE_VERSION = 1
SEAL = 'final_native_top5_mb101_receipt.json'
ARCHIVE_SHA = '4b7284049bc80cc8b9ae94e1ff9d0cb9252acdecc266fb894ae7f61a527762d7'
BASE_SHA = '864cf9f414e4ac2d8f634aaf50c6e76ef018c337739957c9fae56b2cc2f48bc4'
OLD_RANKS_SHA = '97882577d077d5965f23b1e381c97d61a8e39a5bf1ee7e6e75d0e1d9540e6c3d'
OLD_SUMMARY_SHA = 'e6ed264976f4a5d406986a02296d986dc4a76dc2d4051f179c40589b09cc3255'
LENGTHS = {1:2,2:1,25:98}


def valid_seal(raw):
    doc = json.loads(raw)
    if (doc['status']!='sealed' or doc['molecules']!=101 or doc['candidate_budget']!=25
            or doc['new_predictor_requests']!=0 or doc['base_submission_sha256']!=BASE_SHA):
        raise ValueError('Expected the official 101-query native-only final-ranking seal')
    for arm in ARMS:
        if not re.fullmatch('[0-9a-f]{64}',doc['files'][arm+'_final.json']):
            raise ValueError(f'Invalid blind {arm} file checksum')
    return doc


def stock_list(archive):
    with tarfile.open(verified(archive,ARCHIVE_SHA),'r:gz') as stream:
        member, = [m for m in stream.getmembers() if m.name=='submission.csv' and m.isfile()]
        raw = stream.extractfile(member).read()
    if sha256(raw).hexdigest()!=BASE_SHA:
        raise ValueError('Archived final list differs from the historical score input')
    reader = csv.DictReader(StringIO(raw.decode()))
    if reader.fieldnames!=['molecule_id','smiles']:
        raise ValueError('Unexpected columns in the frozen submission')
    result = {}
    for row in reader:
        mid = row['molecule_id']
        if mid in result:
            raise ValueError('Duplicate original molecule ID')
        result[mid] = row['smiles'].split(';')
    if set(result)!={f'H{i:03}' for i in range(1,102)} or Counter(map(len,result.values()))!=LENGTHS:
        raise ValueError('Unexpected IDs or slot counts in the 101-case original final ranking')
    return result


def check_arms(original, rankings):
    if rankings['historical']!=original or any(set(lists)!=set(original) for lists in rankings.values()):
        raise ValueError('The blind original lists differ from the list used to compute the historical ranks')
    changed = {}
    for arm in ARMS[1:]:
        changed[arm] = []
        for mid,before in original.items():
            after = rankings[arm][mid]
            if (not isinstance(after,list) or len(after)!=len(before) or after[5:]!=before[5:]
                    or Counter(before[:5])!=Counter(after[:5]) or (arm=='protected' and after[:1]!=before[:1])):
                raise ValueError(f'Candidate inventory, fixed tail or protected head changed for {arm}/{mid}')
            if after!=before:
                changed[arm].append(mid)
    return changed


def score(inputs,out,seal_sha):
    import pyarrow.parquet as pq
    from original_connectivity import candidate_key_function

    file = verified(mounted(inputs,BLIND,SEAL),seal_sha)
    sealed = valid_seal(file.read_bytes())
    ranked = {}
    for arm in ARMS:
        name = arm+'_final.json'
        ranked[arm] = json.loads(verified(file.parent/name,sealed['files'][name]).read_bytes())
    original = stock_list(mounted(inputs,DATA,'final_ranking_payload.bin'))
    changed = check_arms(original,ranked)

    # The old parquet also contains a label column. Its two anonymous assessment columns suffice.
    summary = json.loads(verified(mounted(inputs,SCORED,'pc50_novel_leader_summary.json'),OLD_SUMMARY_SHA).read_bytes())
    if summary['molecules']!=101 or summary['baseline_submission_sha256']!=BASE_SHA:
        raise ValueError('Archived assessment refers to another submission')
    old_path=verified(mounted(inputs,SCORED,'pc50_novel_leader_ranks.parquet'),OLD_RANKS_SHA)
    old=pq.read_table(old_path,columns=['molecule_id','original_rank']).to_pylist()
    if len(old)!=101 or {r['molecule_id'] for r in old}!=set(original):
        raise ValueError('Archived original ranks refer to different or duplicate queries')
    identity = candidate_key_function()
    rows=[]
    for rec in sorted(old,key=lambda r:r['molecule_id']):
        mid,before=rec['molecule_id'],rec['original_rank']
        if type(before) is not int or not 0<=before<=len(original[mid]):
            raise ValueError(f'Invalid original rank in the historical assessment for {mid}')
        observed=dict.fromkeys(ARMS,0)
        if before:
            hit=identity(original[mid][before-1])
            if hit is None:
                raise ValueError(f'An original hit lacks a valid historical identity for {mid}')
            for arm in ARMS:
                observed[arm]=next((i for i,s in enumerate(ranked[arm][mid],1) if identity(s)==hit),0)
            if observed['historical']!=before or not all(observed.values()):
                raise ValueError(f'First-hit transfer disagrees with the original rank for {mid}')
        rows.append(dict(molecule_id=mid,**{a+'_rank':observed[a] for a in ARMS},
                         protected_delta_rr=rr(observed['protected'])-rr(before),open_delta_rr=rr(observed['open'])-rr(before),
                         open_vs_protected_delta_rr=rr(observed['open'])-rr(observed['protected'])))
    arms={a:totals(rows,a) for a in ARMS}
    baseline=arms['historical']
    if ((baseline['top1'],baseline['top5'],baseline['top25'])!=(68,100,101)
            or baseline['top1']!=summary['original_first'] or baseline['top25']!=summary['original_top25']
            or abs(baseline['mrr25']-summary['original_mrr'])>1e-12):
        raise ValueError('Historical original assessment failed to reproduce')
    checks={f'{a}_vs_{c}':comparison(rows,c,a) for c,a in (('historical','protected'),('historical','open'),('protected','open'))}
    out.mkdir(parents=True,exist_ok=True)
    per=out/'final_native_top5_mb101_per_query.csv'
    with per.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report=dict(molecules=len(rows),candidate_budget=25,candidate_slots=sum(map(len,original.values())),
                new_predictor_requests=0,seal_sha256=seal_sha,original_archive_sha256=ARCHIVE_SHA,
                original_submission_sha256=BASE_SHA,archived_rank_file_sha256=OLD_RANKS_SHA,
                archived_score_summary_sha256=OLD_SUMMARY_SHA,
                fixed_list_sha256={a:sealed['files'][a+'_final.json'] for a in ARMS},
                scoring='Historical RDKit 2026.03.3 de-stereochemical tautomer-normalized InChIKey14 first hit',
                original_lengths=dict(Counter(map(len,original.values()))),changed_order_ids=changed,arms=arms,
                comparisons=checks,per_query_sha256=sha256(per.read_bytes()).hexdigest())
    (out/'final_native_top5_mb101_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(arms=arms,delta_mrr25={k:v['delta_mrr25'] for k,v in checks.items()})),flush=True)


def build(seal,out):
    from build import HERE,IMAGE,METRIC_BOOTSTRAP,WHEEL,pack
    raw=seal.read_bytes()
    valid_seal(raw)
    seal_sha=sha256(raw).hexdigest()
    original=(HERE.parent/'pipeline_candidates/first_formula/score_blind_pair.py').read_text()
    fn, = (n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='candidate_key_function')
    metric='from functools import lru_cache\n'+ast.get_source_segment(original,fn)+'\n'
    compile(metric,'original_connectivity','exec')
    payload={name:pack(b) for name,b in {'original_connectivity.py':metric.encode(),
             'mb101_score.py':Path(__file__).read_bytes(),
             'build_final_native_top5_e180_score.py':(HERE/'build_final_native_top5_e180_score.py').read_bytes()}.items()}
    code=f"""
import base64,hashlib,subprocess,sys,tempfile,zlib
from pathlib import Path
inputs=Path('/kaggle/input')
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='mb101-final-score-') as tmp:
    scratch=Path(tmp)
    for name,(encoded,expected) in {payload!r}.items():
        raw=zlib.decompress(base64.b64decode(encoded))
        if hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError('Embedded score source hash mismatch')
        (scratch/name).write_bytes(raw)
    sys.path.insert(0,str(scratch))
    from mb101_score import score
    score(inputs,Path('/kaggle/working'),{seal_sha!r})
"""
    compile(code,'score-final-mb101','exec')
    cell=dict(cell_type='code',id='score-sealed',execution_count=None,metadata={},outputs=[],source=code.splitlines(True))
    book=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),cells=[cell])
    meta=dict(id='jiyujia/'+SLUG,title=SLUG,code_file='score.ipynb',language='python',kernel_type='notebook',
              is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,competition_sources=[],
              dataset_sources=[f'jiyujia/{DATA}/1',WHEEL],kernel_sources=[f'jiyujia/{BLIND}/1',f'jiyujia/{SCORED}/{SCORE_VERSION}'],
              docker_image=IMAGE)
    out.mkdir(parents=True,exist_ok=True)
    (out/'score.ipynb').write_text(json.dumps(book,separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return dict(slug=meta['id'],seal_sha256=seal_sha,notebook_sha256=sha256((out/'score.ipynb').read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    print(json.dumps(build(**vars(parser.parse_args()))))
