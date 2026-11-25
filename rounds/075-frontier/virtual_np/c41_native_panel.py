"""Reproduce the native E1 scalar and compare two E1-only C41 arms before the original C43 gate."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


POOL_COUNT = 710701
PRIOR_HASH = {
    'pool_lsid.npy':'461ebd8bb1035faf019075bd122f28445a65c52b4b09a209e5d49fad1f7d71e5',
    'pool_lpmid.npy':'4cbece84f0a8ae04895df2c7184ac1e121803b9cad1a6de16333848d4834bb5b',
}
GL_FUSE_HASH = 'ea390f5f29705c015f39c509cc2b86e6a360d0bb90db61c87c507068f38ae58e'


def read(path):
    return json.loads(Path(path).read_bytes())


def dump(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False,ensure_ascii=False,indent=2)+'\n')


def checksum(path):
    from hashlib import file_digest
    with path.open('rb') as f:
        return file_digest(f,'sha256').hexdigest()


def pinned(path, wanted):
    if checksum(path)!=wanted:
        raise ValueError(f'Pinned artifact has changed: {path.name}')
    return path


def calibrate(snapshot, allocation, prior, out):
    """Recover the z score + native row-aligned pool popularity on the original T4 feature convention."""
    snap = read(snapshot/'c39_snapshot_receipt.json')
    if snap['status']!='exact' or not snap['cache_reuse_supported']:
        raise ValueError('Expected an exact C39/C41 same-panel snapshot')
    hashes = snap['hashes']
    native = read(pinned(snapshot/'c39_native_ledgers.json',hashes['c39_native_ledgers.json']))
    rows = pd.read_parquet(pinned(snapshot/'c39_full_slate.parquet',hashes['c39_full_slate.parquet']))
    allocation_seal = read(allocation/'allocation_receipt.json')
    if (allocation_seal['status']!='request_plan_only' or allocation_seal['candidate_source']!='unseen'
            or allocation_seal['candidate_archive']!='jiyujia/enveda-075-full-main-unseen/1'):
        raise ValueError('Expected the sealed, blind 670-slot E1 study')
    alloc = read(pinned(allocation/'allocations.json',allocation_seal['hashes']['allocations.json']))
    plan = read(pinned(allocation/'requests/plan.json',allocation_seal['hashes']['requests/plan.json']))
    changes = read(pinned(allocation/'requests/replacements.json',plan['files']['replacements.json']))
    populations = [np.load(pinned(prior/name,digest),allow_pickle=False).astype(np.float64) for name,digest in PRIOR_HASH.items()]
    if any(x.shape!=(POOL_COUNT,) for x in populations):
        raise ValueError('Native pool popularity row count has changed')
    population = populations[0]+populations[1]
    candidate_groups = {m:g.sort_values('raw_rank') for m,g in rows.loc[rows.first_representative].groupby('molecule_id')}
    assignments, cases, missing = {}, {}, []
    if set(native)!=set(alloc) or set(native)!=set(changes) or set(candidate_groups)-set(native):
        raise ValueError('Allocation and native rows have different queries')
    for mid, stage in native.items():
        group = candidate_groups.get(mid, rows.iloc[0:0])
        old = stage['main']+stage['shadow_pool']
        first = group.iloc[:len(stage['main'])]
        if list(first.native_unique_rank)!=list(range(1,len(first)+1)) or len(first)!=len(stage['main']):
            raise ValueError(f'Missing original pre-popularity top slate for {mid}')
        values = first.raw_ranker_score.to_numpy(np.float64)
        mean, scale = (float(np.mean(values)),float(np.std(values))+1e-9) if len(values) else (0.,1.)
        def source_score(row, deep=False):
            if not deep and len(values)<=1:
                return float(row.raw_ranker_score)
            value = (float(row.raw_ranker_score)-mean)/scale
            return value + (0.15*population[row.pool_id] if row.pool_id>=0 else 0.)
        raw = {(r.smiles,r.native_key):r for r in group.itertuples(index=False)}
        if len(raw)!=len(group):
            raise ValueError(f'Duplicate original C39 representative for {mid}')
        max_error = 0.
        for i, entry in enumerate(old):
            row = raw.get((entry['smiles'],entry['native_key']))
            if row is None:
                raise ValueError(f'Missing original E1 row for {mid}')
            # Generated entries acquire the '|gen' suffix only after the original row was selected.
            expected_form = row.formula+('|gen' if row.pool_id<0 and (len(values)>1 or i>=len(values)) else '')
            if expected_form!=entry['native_formula']:
                raise ValueError(f'E1 formula grouping differs for {mid}')
            max_error=max(max_error,abs(source_score(row,i>=len(values))-entry['source_score']))
        if max_error>1e-10:
            raise ValueError(f'Original T4 scalar normalization failed for {mid}: {max_error}')
        new = {}
        for item in alloc[mid]:
            old_s, new_s, slot = item['old_smiles'], item['new_smiles'], item['old_e1_rank']-1
            if changes[mid].get(old_s)!=new_s or not 0<=slot<len(old) or old[slot]['smiles']!=old_s:
                raise ValueError(f'Allocated E1 slot not found for {mid}')
            literal = group.loc[(group.smiles==new_s)&(group.pool_id>=0)]
            if len(literal)!=1:
                missing.append(dict(molecule_id=mid,old_e1_rank=slot+1,fpnet_a_rank=item['fpnet_a_rank'],
                                    new_smiles=new_s,eligible_literal_pool_rows=len(literal)))
                continue
            row, = literal.itertuples(index=False)
            new[slot] = dict(smiles=row.smiles,native_key=row.native_key,native_formula=row.formula,
                             source_score=float(source_score(row,True)),pool_id=int(row.pool_id))
        if len(alloc[mid])!=len(changes[mid]) or not np.isfinite(max_error):
            raise ValueError(f'Invalid allocation or score scale for {mid}')
        assignments[mid]=new
        cases[mid]=dict(main=len(stage['main']),deep=len(stage['shadow_pool']),native_entries_verified=len(old),
                        mean=mean,scale=scale,max_original_scalar_error=float(max_error),allocated=len(alloc[mid]),
                        matched_literal_pool_rows=len(new))
    cal = dict(snapshot_device=snap['device'],unmatched=missing,cases=cases,
               same_device_t4=snap['device']=='cuda' and 'T4' in str(read(snapshot/'c39_slate_verification.json')['cuda_device']),
               all_calibrated=not missing,allocated=sum(len(v) for v in alloc.values()),
               applicable=sum(len(v) for v in assignments.values()),
               max_original_scalar_error=max(v['max_original_scalar_error'] for v in cases.values()))
    dump(out/'c41_calibration.json',cal)
    return native,assignments,cal


def paired_main(native, assignments, scored, post_pc, gl_fuse_path, c43_code, canonical_archive, out, arm):
    """Return post-C41 eligible main lists. Only candidate-specific physical scores enter the scorer."""
    spec = importlib.util.spec_from_file_location('frozen_c41_gl_fuse',pinned(gl_fuse_path,GL_FUSE_HASH))
    gl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gl)
    # Build the exact v2 C43 namespace. The original cell's row builder reads precisely these fields.
    def apply_gate(base):
        n = dict(BASE=base,PC=post_pc,te=pd.DataFrame({'molecule_id':list(native)}),np=np,pd=pd,
                 LIB_TAU=.9,REL_TH=600.,SLOTS_AGG=[2,4,6,8,10],SLOTS_GENTLE=[4,8,12,16,20])
        exec(compile(c43_code,'frozen-c43-gate','exec'),n)
        return {mid:s.split(';') for mid,s in n['rows']}
    unchanged = {m:([v['smiles'] for v in n['main'][:25]],[v['native_key'] for v in n['main'][:25]],n['library_max'])
                 for m,n in native.items()}
    gate = apply_gate(unchanged)
    if any(unchanged[m][0]!=canonical_archive[m]['main'] or gate[m]!=canonical_archive[m]['prefusion'] for m in native):
        raise ValueError('The authentic no-shadow C41/C43 control differs from the original T4 archive')
    baseline = {}
    for mid, stage in native.items():
        pool = [dict(row) for row in stage['main']+stage['shadow_pool']]
        if arm=='replacement':
            for pos,v in assignments[mid].items():
                pool[pos]=v
        values = ([r['smiles'] for r in pool],[r['native_key'] for r in pool],
                  [r['source_score'] for r in pool],[r['native_formula'] for r in pool])
        branch = []
        for model in ('ice','gl'):
            branch.append({**scored[model]['native'].get(mid,{}),**scored[model]['shadow'].get(mid,{})})
        if (stage['library_max'] < .9 and pool and any(v is not None and np.isfinite(v) for b in branch for v in b.values())):
            order=gl.rerank_multi(*values,branch,[1.,1.],top_n=len(pool))
            if sorted(order)!=list(range(len(pool))):
                raise ValueError(f'Invalid C41 permutation for {mid}')
            baseline[mid]=([values[0][i] for i in order[:25]],[values[1][i] for i in order[:25]],stage['library_max'])
        else:
            baseline[mid]=unchanged[mid]
    actual_gate=apply_gate(baseline)
    dump(out/f'{arm}_main.json',baseline)
    dump(out/f'{arm}_c43.json',actual_gate)
    return dict(molecules=len(native),changed_main=sum(baseline[m][0]!=unchanged[m][0] for m in native),
                changed_c43=sum(actual_gate[m]!=gate[m] for m in native),
                main_sha256=checksum(out/f'{arm}_main.json'),c43_sha256=checksum(out/f'{arm}_c43.json'))


def fallback_slots(native, assignments, allocated_rows, old_scores, predicted, observed):
    """Return original C41 evidence for an individual slot when its new native score or physical prediction is missing."""
    effective, used, ledger = deepcopy(old_scores), {}, {}
    for mid, all_slots in allocated_rows.items():
        ledger[mid], used[mid] = [], {}
        required = [m for m in ('ice','gl') if observed[mid][m+'_shadow']['candidate_spectra']>0]
        for item in all_slots:
            position=item['old_e1_rank']-1
            new=item['new_smiles']
            old=item['old_smiles']
            planned=assignments[mid].get(position)
            missing = []
            if planned is None:
                status='old_slot_missing_native_scalar'
            else:
                if planned['smiles']!=new or not required:
                    raise ValueError(f'Claimed allocated slot differs from the attempted request for {mid}')
                for m in required:
                    row=predicted[m]['shadow'].get(mid,{})
                    if new not in row or row[new] is None or not math.isfinite(row[new]):
                        missing.append(dict(model=m,reason='absent' if new not in row else 'missing_prediction'))
                status='old_slot_missing_prediction' if missing else 'replacement_used'
                if not missing:
                    used[mid][position]=planned
                    for m in required:
                        cached=effective[m]['shadow'].setdefault(mid,{})
                        cached.pop(old,None)
                        cached[new]=predicted[m]['shadow'][mid][new]
            ledger[mid].append(dict(old_e1_rank=position+1,old_smiles=old,new_smiles=new,
                                    fpnet_a_rank=item['fpnet_a_rank'],status=status,missing=missing))
    actual=deepcopy(observed)
    for mid, fields in actual.items():
        for model in ('ice','gl'):
            for kind in ('native','shadow'):
                row=effective[model][kind].get(mid,{})
                fields[model+'_'+kind]['finite_effective']=sum(v is not None and math.isfinite(v) for v in row.values())
        fields['gl_proxy']['finite_effective']=0
    return effective,used,ledger,actual
