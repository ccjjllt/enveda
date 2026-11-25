"""Compare historical, native-forward-only E1 reranking, and the full original-slot inference at C43."""
from collections import Counter
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from c41_native_panel import paired_main, pinned, checksum, dump, read


ORIGINAL='jiyujia/enveda-075-np-recall-base/2'
SNAPSHOT_SHA='30ebaeda090adbabd48f0b60ef4a3a857275c12be84f0c37a58f1bd8fd6ed7dc'
OLD_SHA='9c0f5301e421b590be3e7180cf12669df6311fdc62a10d87d1c206b12f8b9cf9'
CONTROL_ZIP='7aa006825a0f8654c4239f25e89ddb490ea04575b9a195d6b744955688d5d7c5'
BASELINE_ZIP='05509b32cfb9bcbe8ab563244f91c97e89f878eba3cc40263e3f48fd4a1f372d'
QUERY_SHA='2a8bf747f50e6920fcd4f24e04302b721281f8794c947cc58fe545264b12d0d4'


def changes(before, after):
    if set(before)!=set(after):raise ValueError('Different query populations')
    for m in before:
        if len(before[m])!=len(after[m]):
            raise ValueError('Changed query candidate budget: '+m)
    return dict(changed=[m for m in before if before[m]!=after[m]],
                top1_changed=[m for m in before if before[m][:1]!=after[m][:1]],
                top5_order_changed=[m for m in before if before[m][:5]!=after[m][:5]],
                top5_members_changed=[m for m in before if set(before[m][:5])!=set(after[m][:5])])


def request_counts(native, requests, budget):
    rows, totals={},Counter()
    indexed={name:{it['mid']:it for it in lst} for name,lst in requests.items()}
    for mid,entry in native.items():
        group=entry['main']+entry['shadow_pool']
        counts=Counter(row['native_formula'] for row in group)
        active_forms={row['native_formula'] for row in group[:25] if counts[row['native_formula']]>=2}
        lookup={row[key] for row in group if row['native_formula'] in active_forms
                for key in ('smiles','native_key')}
        per_query={}
        for model in ('ice','gl'):
            for src in ('native','shadow'):
                key=model+'_'+src
                item=indexed[key].get(mid,{})
                spec=len(item.get('spectra',[]))
                candidates=item.get('cands',[])
                pairs=spec*len(candidates)
                if pairs!=budget[mid]['predictor_budget'][key+'_candidate_spectra']:
                    raise ValueError(f'Wrong original predictor budget for {mid}/{key}')
                per_query[key+'_historical_pairs']=pairs
                totals[key+'_historical_pairs']+=pairs
                if src=='shadow':
                    n=sum(s in lookup for s in candidates)*spec if entry['library_max']<.9 else 0
                    per_query[key+'_readable_top25_formula_pairs']=n
                    totals[key+'_readable_top25_formula_pairs']+=n
        if budget[mid]['predictor_budget']['gl_proxy_candidate_spectra']!=0:
            raise ValueError('Expected the original zero-proxy validation cohort')
        rows[mid]=per_query
    return dict(totals=dict(totals),cases=rows)


def run(snapshot, control, sealed, fuse, c43, out):
    out.mkdir(parents=True,exist_ok=True)
    snap=read(pinned(snapshot/'c39_snapshot_receipt.json',SNAPSHOT_SHA))
    old=read(pinned(control/'c41_study_receipt.json',OLD_SHA))
    if not snap['cache_reuse_supported'] or snap['device']!='cuda' or old['native_no_shadow_controls_exact']!=60:
        raise ValueError('Unverified T4 C39 snapshot or historical C43 routing')
    cal=read(pinned(control/'c41_calibration.json',old['calibration_sha256']))
    if not cal['same_device_t4'] or cal['max_original_scalar_error']>1e-10:
        raise ValueError('Previously checked source calibration differs')
    native=read(pinned(snapshot/'c39_native_ledgers.json',snap['hashes']['c39_native_ledgers.json']))
    with ZipFile(pinned(snapshot/'c39_original_control.zip',CONTROL_ZIP)) as z:
        def get(name):return json.loads(z.read(name))
        sources=get('four_source_lists.json')
        budget=get('pilot_forward_budget.json')
        pc=get('e004_candidate_inputs.json')['pubchem']
        native_maps={model:get(f'{model}_work/{model}_out.json') for model in ('ice','gl')}
        shadow=get('pilot_shadow_scores.json')
        req={model+'_native':get(f'{model}_work/{model}_in.json') for model in native_maps}
        req.update({model+'_shadow':get(f'pilot_{model}/{model}_in.json') for model in native_maps})
    with ZipFile(pinned(sealed,BASELINE_ZIP)) as z:
        b=json.loads(z.read('pilot_arm_receipt.json'))
        if b['queries_sha256']!=QUERY_SHA or json.loads(z.read('four_source_lists.json'))!=sources:
            raise ValueError('Historical source archive and final control disagree')
    if set(native)!=set(sources) or len(native)!=60:raise ValueError('Expected all 60 queries')
    historical={m:([r['smiles'] for r in n['main'][:25]],[r['native_key'] for r in n['main'][:25]],n['library_max'])
                for m,n in native.items()}
    if any(main[0]!=sources[m]['main'] for m,main in historical.items()):
        raise ValueError('Historical main differs from T4 C39')
    ns=dict(BASE=historical,PC=pc,te=pd.DataFrame({'molecule_id':list(native)}),np=np,pd=pd,
            LIB_TAU=.9,REL_TH=600.,SLOTS_AGG=[2,4,6,8,10],SLOTS_GENTLE=[4,8,12,16,20])
    exec(compile(c43,'public-original-c43','exec'),ns)
    historical_gate={m:s.split(';') for m,s in ns['rows']}
    if historical_gate!={m:s['prefusion'] for m,s in sources.items()}:
        raise ValueError('Historical C43 did not reproduce across all 60')
    dump(out/'historical_main.json',historical)
    dump(out/'historical_c43.json',historical_gate)
    for name in ('native_only','original_slots'):
        maps={m:dict(native=native_maps[m],shadow=shadow[m] if name=='original_slots' else {})
              for m in native_maps}
        paired_main(native,{},maps,pc,fuse,c43,sources,out,name)
    for name,key in [('original_slots_main.json','main_sha256'),('original_slots_c43.json','c43_sha256')]:
        pinned(control/name,old['original_slots'][key])
        pinned(out/name,old['original_slots'][key])
    comparisons={}
    for a,b in [('historical','native_only'),('native_only','original_slots'),('historical','original_slots')]:
        left=read(out/f'{a}_main.json');right=read(out/f'{b}_main.json')
        comparisons[f'{a}_to_{b}']=dict(main=changes({m:r[0] for m,r in left.items()},{m:r[0] for m,r in right.items()}),
                                         c43=changes(read(out/f'{a}_c43.json'),read(out/f'{b}_c43.json')))
    supply=request_counts(native,req,budget['cases'])
    dump(out/'native_only_request_evidence.json',supply)
    files={name:checksum(out/name) for name in ['native_only_request_evidence.json',
           *(f'{arm}_{stage}.json' for arm in ('historical','native_only','original_slots') for stage in ('main','c43'))]}
    seal=dict(status='c43_sealed',query_count=60,queries_sha256=QUERY_SHA,original_t4=ORIGINAL,
              historical_c43_exact=60,original_slots_c43_exact=60,original_slots_main_exact=60,
              candidate_budget=25,new_predictor_requests_in_this_job=0,
              native_only_incremental_forward_requests_vs_public_v29=0,
              original_slots_reference_shadow_archived=True,request_counts=supply['totals'],
              comparisons=comparisons,files=files)
    dump(out/'c41_native_only_receipt.json',seal)
    print('C43 changes',{k:len(v['c43']['changed']) for k,v in comparisons.items()},flush=True)
