"""Build the independent, private CPU scorer for the sealed Enveda-180 final-top-five permutations."""
import argparse
import ast
from collections import Counter
import csv
from hashlib import sha256
import json
from math import fsum
from pathlib import Path
import re


BLIND = 'enveda-075-final-native-top5-e180'
BASE = 'enveda-075-first-formula-baseline-e180'
SCORED = 'enveda-075-e180-full-compare'
SLUG = 'enveda-075-final-native-top5-e180-score'
SEAL = 'final_native_top5_e180_receipt.json'
BASE_SHA = '2675b3cec0607a93b447c4bfbcac256812bc796cf8794b81fc8414c24fe1f55a'
OLD_RANKS_SHA = '46a7b2ff0d9a9f2778e0d099050ce36a5f69a5065c91e1c4d7d500c25d0afdf0'
ARMS = ('historical','protected','open')


def validated_seal(raw):
    seal = json.loads(raw)
    if (seal['status']!='sealed' or seal['molecules']!=121 or seal['candidate_budget']!=25
            or seal['new_predictor_requests']!=0 or seal['base_submission_sha256']!=BASE_SHA):
        raise ValueError('Expected the sealed fixed-budget 121-query study against the archived original 0.498')
    for arm in ARMS:
        if not re.fullmatch('[0-9a-f]{64}',seal['files'][arm+'_final.json']):
            raise ValueError(f'Missing or invalid sealed digest for {arm}')
    return seal


def verified(path, checksum):
    if sha256(path.read_bytes()).hexdigest()!=checksum:
        raise ValueError(f'Archived file changed: {path.name}')
    return path


def mounted(inputs, slug, name):
    path, = (p for p in inputs.rglob(name) if slug in p.parts)
    return path


def check_lists(rankings, original):
    if (len(original)!=121 or any(len(v)!=25 for v in original.values()) or rankings['historical']!=original
            or any(set(r)!=set(original) for r in rankings.values())):
        raise ValueError('Sealed historical query IDs, candidate order or candidate budget differ from the original')
    changed = {}
    for arm in ARMS[1:]:
        changed[arm] = []
        for mid,before in original.items():
            new = rankings[arm][mid]
            if (not isinstance(new,list) or len(new)!=25 or new[5:]!=before[5:]
                    or Counter(new[:5])!=Counter(before[:5]) or (arm=='protected' and new[0]!=before[0])):
                raise ValueError(f'The original candidates, protected head or positions below five changed: {arm}/{mid}')
            if new!=before:
                changed[arm].append(mid)
    return changed


def rr(rank):
    return 1./rank if rank else 0.


def totals(rows, arm):
    ranks = [row[arm+'_rank'] for row in rows]
    return dict(molecules=len(ranks),top1=ranks.count(1),top5=sum(0<r<=5 for r in ranks),
                top25=sum(r>0 for r in ranks),mrr25=fsum(map(rr,ranks))/len(ranks))


def comparison(rows, control, arm):
    ranks = [(r['molecule_id'],r[control+'_rank'],r[arm+'_rank']) for r in rows]
    result = dict(control=control,arm=arm,delta_mrr25=fsum(rr(b)-rr(a) for _,a,b in ranks)/len(ranks),
                  improved=[mid for mid,a,b in ranks if rr(b)>rr(a)],worsened=[mid for mid,a,b in ranks if rr(b)<rr(a)])
    for k in (1,5,25):
        result[f'new_top{k}']=[mid for mid,a,b in ranks if not 0<a<=k and 0<b<=k]
        result[f'lost_top{k}']=[mid for mid,a,b in ranks if 0<a<=k and not 0<b<=k]
    if any(result[key] for key in ('new_top5','lost_top5','new_top25','lost_top25')):
        raise ValueError('A first-five permutation changed top-five or top-25 identity membership')
    return result


def score(inputs, out, seal_sha):
    from original_e180_metric import candidate_key_function
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    seal_path = verified(mounted(inputs,BLIND,SEAL),seal_sha)
    seal = validated_seal(seal_path.read_bytes())
    rankings = {}
    for arm in ARMS:
        name = arm+'_final.json'
        rankings[arm] = json.loads(verified(seal_path.parent/name,seal['files'][name]).read_bytes())
    baseline = verified(mounted(inputs,BASE,'submission.csv'),BASE_SHA)
    original = {}
    with baseline.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames!=['molecule_id','smiles']:
            raise ValueError('Unexpected original submission columns')
        for row in reader:
            mid = row['molecule_id']
            if mid in original:
                raise ValueError('Duplicate historical query')
            original[mid] = row['smiles'].split(';')
    changed = check_lists(rankings,original)

    # Load historical rank indices only after all blind files pass the exact candidate checks.
    import pandas as pd
    past = verified(mounted(inputs,SCORED,'e180_pair_ranks.csv'),OLD_RANKS_SHA)
    archived = pd.read_csv(past,usecols=['molecule_id','control'],dtype={'molecule_id':str,'control':'Int64'})
    if not archived.molecule_id.is_unique or set(archived.molecule_id)!=set(original):
        raise ValueError('The historical per-query assessment has a different population')
    key = candidate_key_function()
    rows = []
    for mid,old in archived.sort_values('molecule_id').itertuples(index=False):
        before = 0 if pd.isna(old) else int(old)
        ranks = dict.fromkeys(ARMS,0)
        if before:
            if not 1<=before<=25:
                raise ValueError(f'Invalid archived original rank for {mid}')
            # Several SMILES may score as the same connectivity key. Transfer the earliest occurrence
            # under the actual historical metric rather than the slot of a single representative.
            hit = key(original[mid][before-1].strip())
            if hit is None:
                raise ValueError(f'Historical hit has no valid normalized identity: {mid}')
            for arm in ARMS:
                ranks[arm] = next((i for i,s in enumerate(rankings[arm][mid],1) if key(s.strip())==hit),0)
            if ranks['historical']!=before or not all(ranks.values()):
                raise ValueError(f'Historical first-hit rank is inconsistent with the frozen submission: {mid}')
        rows.append(dict(molecule_id=mid,**{arm+'_rank':rank for arm,rank in ranks.items()},
                         protected_delta_rr=rr(ranks['protected'])-rr(before),open_delta_rr=rr(ranks['open'])-rr(before),
                         open_vs_protected_delta_rr=rr(ranks['open'])-rr(ranks['protected'])))
    arms = {arm:totals(rows,arm) for arm in ARMS}
    old = arms['historical']
    if (old['top1'],old['top5'],old['top25'])!=(52,91,103) or abs(old['mrr25']-0.5761142047602913)>1e-12:
        raise ValueError('Original Enveda-180 baseline did not reproduce')
    comparisons = {f'{arm}_vs_{control}':comparison(rows,control,arm)
                   for control,arm in (('historical','protected'),('historical','open'),('protected','open'))}
    out.mkdir(exist_ok=True,parents=True)
    per = out/'final_native_top5_e180_per_query.csv'
    with per.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = dict(molecules=len(rows),candidate_budget=25,candidate_slots=3025,new_predictor_requests=0,
                   seal_sha256=seal_sha,original_submission_sha256=BASE_SHA,archived_ranks_sha256=OLD_RANKS_SHA,
                   fixed_list_sha256={arm:seal['files'][arm+'_final.json'] for arm in ARMS},
                   changed_order_ids=changed,arms=arms,comparisons=comparisons,
                   scoring='First historical match by RDKit 2026.03.3 tautomer-normalized connectivity InChIKey14',
                   per_query_sha256=sha256(per.read_bytes()).hexdigest())
    (out/'final_native_top5_e180_metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(arms=arms,delta_mrr25={k:v['delta_mrr25'] for k,v in comparisons.items()})),flush=True)


def build(seal, out):
    from build import HERE, IMAGE, METRIC_BOOTSTRAP, WHEEL, pack
    raw = seal.read_bytes()
    validated_seal(raw)
    seal_sha = sha256(raw).hexdigest()
    metric_program = (HERE.parent/'pipeline_candidates/first_formula/score_blind_pair.py').read_text()
    metric_node, = (n for n in ast.parse(metric_program).body if isinstance(n,ast.FunctionDef)
                    and n.name=='candidate_key_function')
    metric = 'from functools import lru_cache\n'+ast.get_source_segment(metric_program,metric_node)+'\n'
    compile(metric,'original_e180_metric','exec')
    payload = {name:pack(body) for name,body in {'original_e180_metric.py':metric.encode(),
                                                 'e180_top5_score.py':Path(__file__).read_bytes()}.items()}
    code = f"""
import base64,hashlib,subprocess,sys,tempfile,zlib
from pathlib import Path
inputs=Path('/kaggle/input')
{METRIC_BOOTSTRAP}
with tempfile.TemporaryDirectory(prefix='e180-top5-score-') as tmp:
    root=Path(tmp)
    for name,(encoded,checksum) in {payload!r}.items():
        raw=zlib.decompress(base64.b64decode(encoded))
        if hashlib.sha256(raw).hexdigest()!=checksum:
            raise ValueError('Embedded scoring source hash mismatch')
        (root/name).write_bytes(raw)
    sys.path.insert(0,str(root))
    from e180_top5_score import score
    score(inputs,Path('/kaggle/working'),{seal_sha!r})
"""
    compile(code,'score-final-native-top5-e180','exec')
    cell=dict(cell_type='code',id='score-permutation',execution_count=None,metadata={},outputs=[],source=code.splitlines(True))
    book=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':dict(name='python3',language='python',display_name='Python 3')},cells=[cell])
    meta=dict(id='jiyujia/'+SLUG,title=SLUG,code_file='score.ipynb',language='python',kernel_type='notebook',
              is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,dataset_sources=[WHEEL],
              kernel_sources=[f'jiyujia/{BLIND}/1',f'jiyujia/{BASE}/2',f'jiyujia/{SCORED}/1'],
              competition_sources=[],docker_image=IMAGE)
    out.mkdir(parents=True,exist_ok=True)
    (out/'score.ipynb').write_text(json.dumps(book,separators=(',',':'))+'\n')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return dict(slug=meta['id'],seal_sha256=seal_sha,notebook_sha256=sha256((out/'score.ipynb').read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    print(json.dumps(build(**vars(parser.parse_args()))))
