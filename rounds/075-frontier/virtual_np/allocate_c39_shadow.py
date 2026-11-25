"""Assign a frozen FPNet A shortlist to the original weak-library E1 shadow slots without scoring."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pyarrow.parquet as pq

from c39_request_cache import identity_function, plan_requests


FP_RECEIPT = 'f05054088941a0317cb7cc995b49f2707ec28f5180cbde1d5affa1831cd749cd'
FP_SHORTLIST = '3bbd119dac0f4302487fb9dc30e926ff8857a1e8482c86e3c6d1f0800f731a26'
UNSEEN_RECEIPT = '62a857d612fb7d04bafd8eb3ee32403f1d7bbcc95da93a42bd0f22823ffbe1de'
UNSEEN_SHORTLIST = 'eb91e2be682d8a472dc6335eb13cc65b55e8e07975cf28b540fc59f38828e584'
SHORTLISTS = {
    'unseen': ('jiyujia/enveda-075-full-main-unseen/1','unseen_relist_receipt.json',UNSEEN_RECEIPT,
               'unseen_shortlists.parquet',UNSEEN_SHORTLIST,670),
    'direct': ('jiyujia/enveda-075-np-virtual-window-full-main-fpnet-a/1','fpnet_a_score_receipt.json',FP_RECEIPT,
               'fpnet_a_ranked25.parquet',FP_SHORTLIST,1153),
}
CONTROL_ZIP = '7aa006825a0f8654c4239f25e89ddb490ea04575b9a195d6b744955688d5d7c5'
SEALED_BASE = '05509b32cfb9bcbe8ab563244f91c97e89f878eba3cc40263e3f48fd4a1f372d'
ORIGINAL = 'jiyujia/enveda-075-np-recall-base/2'
SOURCE_CHANNELS = ('main','engine2','pubchem','prefusion','fused_overflow')


def digest(path):
    from hashlib import file_digest
    with path.open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Expected the unchanged {path.name}')
    return path


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def read(path):
    return json.loads(path.read_bytes())


def read_shortlist(score_dir, original, identity, source):
    _, receipt_file, receipt_sha, shortlist_file, shortlist_sha, slots = SHORTLISTS[source]
    receipt = read(checked(score_dir/receipt_file,receipt_sha))
    path = checked(score_dir/shortlist_file, shortlist_sha)
    mids = set(original)
    if receipt['cap']!=24 or set(receipt['cases'])!=mids:
        raise ValueError('Unexpected frozen FPNet A shortlist')
    frame = pd.read_parquet(path)
    if source=='unseen':
        if (receipt['generation_scope']!='full-main-window' or receipt['fpnet_receipt_sha256']!=FP_RECEIPT
                or receipt['shortlist_sha256']!=shortlist_sha or set(receipt['excluded_sources'])!=set(('final',*SOURCE_CHANNELS))):
            raise ValueError('Expected the blind original-source exclusion and the frozen full-window FPNet scores')
        arms, selected_arm, budget_column = {'plain','unseen'}, 'unseen', 'budget'
    else:
        if receipt['proposal_artifact']!='known_main.parquet' or receipt['output_sha256'][path.name]!=shortlist_sha:
            raise ValueError('Expected native FPNet full-window shortlist')
        arms, selected_arm, budget_column = {'generated','native_c34'}, 'generated', 'candidate_budget'
    if not set(frame.arm) <= arms or set(frame.molecule_id)-mids:
        raise ValueError('Unknown FPNet A channel')
    frame = frame[frame.arm.eq(selected_arm)].sort_values(['molecule_id','rank'])
    groups = {m:list(g.itertuples(index=False)) for m,g in frame.groupby('molecule_id',sort=False)}
    shortlist = {}
    for mid in sorted(mids):
        entries = groups.get(mid,[])
        cap = receipt['cases'][mid][budget_column]
        if (not 0<=cap<=24 or [r.rank for r in entries]!=list(range(1,cap+1))
                or [(-r.fpnet_a_fz,r.smiles) for r in entries]!=sorted((-r.fpnet_a_fz,r.smiles) for r in entries)):
            raise ValueError(f'Wrong candidate budget or FPNet A ordering for {mid}')
        entered = set()
        for r in entries:
            ident = identity(r.smiles)
            if not math.isfinite(r.fpnet_a_fz) or ident is None or ident[0]!=r.score_key or ident in entered:
                raise ValueError(f'FPNet A candidate identity differs from the official metric for {mid}')
            if source=='unseen' and (r.originally_seen or ident in original[mid]):
                raise ValueError(f'Blind unseen slate contains an original-source candidate for {mid}')
            entered.add(ident)
        shortlist[mid] = entries
    if sum(map(len,shortlist.values()))!=slots or set(groups)-mids:
        raise ValueError(f'Expected the frozen {slots} {source} slots')
    return shortlist


def write_c39_lookup(raw_path, planned, rows, identity):
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    by_identity = {mid:{identity(x['new_smiles']):x for x in vals} for mid,vals in planned.items()}
    matches = defaultdict(list)
    fields = ['molecule_id','first_representative','pool_id','smiles','native_key','formula','raw_rank',
              'raw_ranker_score','native_unique_rank']
    for batch in pq.ParquetFile(raw_path).iter_batches(columns=fields,batch_size=8192):
        for raw in batch.to_pylist():
            mid = raw['molecule_id']
            if not by_identity.get(mid) or raw['pool_id']<0 or not raw['first_representative']:
                continue
            ident = identity(raw['smiles'])
            if ident in by_identity[mid]:
                matches[mid,ident].append(raw)
    for mid, entries in planned.items():
        for row in entries:
            records = matches[mid,identity(row['new_smiles'])]
            literal = [r for r in records if r['smiles']==row['new_smiles']]
            row['c39_lookup'] = ('exact_ranker_row' if len(literal)==1 else 'ambiguous_ranker_rows' if len(literal)>1
                                 else 'identity_alias_only' if records else 'outside_c39_native_slate')
            row['matching_identity_rows'] = len(records)
            row['native_formula'] = CalcMolFormula(Chem.MolFromSmiles(row['new_smiles']))
            if len(literal)==1:
                record, = literal
                row.update(c39_pool_id=record['pool_id'],c39_raw_rank=record['raw_rank'],c39_raw_score=record['raw_ranker_score'],
                           c39_native_key=record['native_key'],native_formula=record['formula'])
        if entries:
            original = rows[mid]['main']+rows[mid]['shadow_pool']
            frequencies = Counter(v['native_formula'] for v in original)
            for r in entries:
                frequencies[r['old_native_formula']]-=1
                frequencies[r['native_formula']]+=1
            for r in entries:
                r['prospective_same_formula_count'] = frequencies[r['native_formula']]


def allocate(snapshot, score_dir, sealed, out, scratch, candidate_source='unseen'):
    out.mkdir(exist_ok=True,parents=True)
    exact = read(snapshot/'c39_snapshot_receipt.json')
    if exact['original']!=ORIGINAL:
        raise ValueError('Wrong C39 parent')
    for name in ('c39_slate_verification.json','c39_full_slate.parquet','c39_native_ledgers.json',
                 'c39_request_verification.json','c39_shadow_provenance.json'):
        checked(snapshot/name,exact['hashes'][name])
    stages = read(snapshot/'c39_slate_verification.json')
    proof = read(snapshot/'c39_request_verification.json')
    if not (exact['cache_reuse_supported'] and proof['exact'] and stages['exact_original_lists']==60):
        dump(out/'allocation_receipt.json',dict(status='requires_exact_snapshot',device=exact['device'],candidate_source=candidate_source,
                                               exact_c39_lists=stages['exact_original_lists'],requests_exact=proof['exact']))
        return
    identity = identity_function()
    with ZipFile(checked(snapshot/'c39_original_control.zip',CONTROL_ZIP)) as control:
        for member in control.namelist():
            path = scratch/'original'/member
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(control.read(member))
    archive = scratch/'original'
    native = read(snapshot/'c39_native_ledgers.json')
    provenance = read(snapshot/'c39_shadow_provenance.json')
    with ZipFile(checked(sealed,SEALED_BASE)) as files:
        with files.open('submission.csv') as stream:
            final = pd.read_csv(stream,dtype=str,keep_default_na=False)
        lists = json.loads(files.read('four_source_lists.json'))
    if (not final.molecule_id.is_unique or len(native)!=60 or set(native)!=set(final.molecule_id)
            or set(native)!=set(provenance) or set(native)!=set(lists)):
        raise ValueError('Source population disagrees with original fusion')
    actual = read(archive/'four_source_lists.json')
    if actual!=lists:
        raise ValueError('The original T4 sources and sealed final archive disagree')
    final = final.set_index('molecule_id').smiles.str.split(';').to_dict()
    original_requests = defaultdict(set)
    for file in ('ice_work/ice_in.json','gl_work/gl_in.json','pilot_ice/ice_in.json','pilot_gl/gl_in.json'):
        for item in read(archive/file):
            original_requests[item['mid']].update(item['cands'])
    original = {mid:{identity(s) for s in set(final[mid])|{s for channel in SOURCE_CHANNELS for s in lists[mid][channel]}}
                for mid in native}
    shortlist = read_shortlist(score_dir,original,identity,candidate_source)
    changes, selected, cases, shortlist_audit = {}, {}, {}, []
    for mid, entries in shortlist.items():
        channels = native[mid]['main'] + native[mid]['shadow_pool']
        slots = [r for r in provenance[mid] if r['channel']=='e1']
        if slots and native[mid]['library_max']>=0.9:
            raise ValueError(f'Strong-library query has a shadow E1 slot: {mid}')
        # Displace the deepest reserve source first. Neither this assignment nor its sizing uses labels or scores from a forward predictor.
        slots.sort(key=lambda r:(-r['e1_rank'],-r['slot']))
        used = original[mid] | {identity(s) for s in original_requests[mid]|{r['smiles'] for r in channels}}
        changes[mid], selected[mid] = {}, []
        counts = Counter()
        for candidate in entries:
            ident = identity(candidate.smiles)
            if not slots:
                reason='e1_capacity_used' if selected[mid] else 'no_e1_slots'
            elif ident in used:
                reason='already_represented'
            else:
                slot=slots.pop(0)
                old=channels[slot['e1_rank']-1]
                if old['smiles']!=slot['smiles'] or slot['smiles'] in changes[mid]:
                    raise ValueError(f'Shadow slot/source lookup failed: {mid}')
                row=dict(molecule_id=mid, slot=slot['slot'], old_e1_rank=slot['e1_rank'],
                         old_smiles=slot['smiles'],old_native_formula=old['native_formula'],
                         new_smiles=candidate.smiles,fpnet_a_rank=candidate.rank,fpnet_a_fz=candidate.fpnet_a_fz)
                changes[mid][slot['smiles']]=candidate.smiles
                selected[mid].append(row)
                used.add(ident)
                reason='allocated'
            counts[reason]+=1
            shortlist_audit.append(dict(molecule_id=mid,rank=candidate.rank,smiles=candidate.smiles,decision=reason))
        cases[mid]=dict(shortlisted=len(entries), e1_capacity=len(slots)+len(selected[mid]),
                        allocated=len(selected[mid]),counts=dict(counts))
    write_c39_lookup(snapshot/'c39_full_slate.parquet',selected,native,identity)
    plan=plan_requests(archive,changes,out/'requests',snapshot=snapshot)
    rowlist=[r for m in selected for r in selected[m]]
    if plan['changed_shadow_e1_slots']!=len(rowlist):
        raise ValueError('Allocation and prediction request accounting differ')
    dump(out/'allocations.json',selected)
    dump(out/'allocation_cases.json',cases)
    pd.DataFrame(shortlist_audit).to_csv(out/'shortlist_slot_decisions.csv',index=False)
    ref, _, input_receipt_sha, _, shortlist_sha, _ = SHORTLISTS[candidate_source]
    receipt=dict(status='request_plan_only',source=ORIGINAL,source_snapshot_device=exact['device'],
                 candidate_source=candidate_source, candidate_archive=ref, shortlist_receipt_sha256=input_receipt_sha,
                 molecules=len(cases),shortlisted=sum(v['shortlisted'] for v in cases.values()),
                 e1_capacity=sum(v['e1_capacity'] for v in cases.values()),replaced=len(rowlist),
                 changed_queries=plan['changed_molecules'],c39_lookup=dict(Counter(r['c39_lookup'] for r in rowlist)),
                 prospective_same_formula_at_least_two=sum(r['prospective_same_formula_count']>=2 for r in rowlist),
                 original_final_lengths={m:len(v) for m,v in final.items()},shortlist_sha256=shortlist_sha,
                 treatment_fused=False, shadow_scores_consumed_by_original_baseline=False,
                 native_scalar_calibration_applied=False, actual_forward_success_verified=False,
                 score_handling='FPNet A selects candidates. Historical C39 raw scores are reported only for unique literal native pool rows. No new source score is assigned.',
                 hashes={name:digest(out/name) for name in ('allocations.json','allocation_cases.json','shortlist_slot_decisions.csv','requests/plan.json')})
    dump(out/'allocation_receipt.json',receipt)
    print('Planned',len(rowlist),'substitutions;',receipt['c39_lookup'],flush=True)
