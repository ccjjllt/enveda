"""Re-route already scored original E1 candidates through the original C43 using unchanged T4 requests."""
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from allocate_c39_shadow import (CONTROL_ZIP, SEALED_BASE, SOURCE_CHANNELS, UNSEEN_RECEIPT, UNSEEN_SHORTLIST,
                                checked, digest, dump, read, read_shortlist)
from c39_request_cache import REQUESTS, identity_function
from c41_promote_existing import promote


def gate(program, main, pc):
    scope=dict(BASE=main,PC=pc,te=pd.DataFrame({'molecule_id':list(main)}),np=np,pd=pd,
               LIB_TAU=.9,REL_TH=600.,SLOTS_AGG=[2,4,6,8,10],SLOTS_GENTLE=[4,8,12,16,20])
    exec(compile(program,'original-c43','exec'),scope)
    return {mid:s.split(';') for mid,s in scope['rows']}


def run(snapshot, old, unseen, sealed, gate_program, out):
    out.mkdir(parents=True,exist_ok=True)
    snap=read(snapshot/'c39_snapshot_receipt.json')
    previous=read(old/'c41_study_receipt.json')
    if (not snap['cache_reuse_supported'] or snap['device']!='cuda'
            or previous['status']!='calibrated_waiting_for_delta' or previous['native_no_shadow_controls_exact']!=60):
        dump(out/'c41_existing_receipt.json',dict(status='awaiting_verified_original_slots',
                                                  snapshot_status=snap['status'],control_status=previous['status']))
        return
    calibration=read(checked(old/'c41_calibration.json',previous['calibration_sha256']))
    if not calibration['same_device_t4'] or calibration['max_original_scalar_error']>1e-10:
        raise ValueError('Historical E1 source scalars have not been calibrated on the same T4')
    control_main=read(checked(old/'original_slots_main.json',previous['original_slots']['main_sha256']))
    control_gate=read(checked(old/'original_slots_c43.json',previous['original_slots']['c43_sha256']))
    native=read(checked(snapshot/'c39_native_ledgers.json',snap['hashes']['c39_native_ledgers.json']))
    proof=read(checked(snapshot/'c39_request_verification.json',snap['hashes']['c39_request_verification.json']))
    if not proof['exact'] or set(control_main)!=set(control_gate) or set(native)!=set(control_main) or len(native)!=60:
        raise ValueError('C39, physical requests, or the original-slot C41 control differ in scope')
    with ZipFile(checked(snapshot/'c39_original_control.zip',CONTROL_ZIP)) as archive:
        def archived(name):
            return json.loads(archive.read(name))
        source_lists=archived('four_source_lists.json')
        pc=archived('e004_candidate_inputs.json')['pubchem']
        requests={channel:archived(name) for channel,name in REQUESTS.items()}
        shadow=archived('pilot_shadow_scores.json')
        scores={m:dict(native=archived(f'{m}_work/{m}_out.json'),shadow=shadow[m]) for m in ('ice','gl')}
        budget=archived('pilot_forward_budget.json')
    with ZipFile(checked(sealed,SEALED_BASE)) as baseline:
        with baseline.open('submission.csv') as stream:
            frame=pd.read_csv(stream,dtype=str,keep_default_na=False)
        if source_lists!=json.loads(baseline.read('four_source_lists.json')):
            raise ValueError('Source candidates differ from the unmodified T4 archive')
    if len(frame)!=60 or not frame.molecule_id.is_unique or set(frame.molecule_id)!=set(native):
        raise ValueError('The final control and source archive have different queries')
    final=frame.set_index('molecule_id').smiles.str.split(';').to_dict()
    identity=identity_function()
    original={mid:{identity(s) for s in set(final[mid])|
              {s for lane in SOURCE_CHANNELS for s in source_lists[mid][lane]}} for mid in native}
    shortlist=read_shortlist(unseen,original,identity,'unseen')
    historic={m:([r['smiles'] for r in n['main'][:25]],[r['native_key'] for r in n['main'][:25]],n['library_max'])
              for m,n in native.items()}
    if (any(historic[m][0]!=source_lists[m]['main'] for m in native)
            or gate(gate_program,historic,pc)!={m:source_lists[m]['prefusion'] for m in native}
            or gate(gate_program,control_main,pc)!=control_gate):
        raise ValueError('The original C43 gate fails historical or exact original-slot parity')
    outputs={}
    files={}
    input_counts={m:budget['cases'][m]['predictor_budget'] for m in native}
    dump(out/'c41_existing_budgets.json',input_counts)
    dump(out/'original_slots_main.json',control_main)
    dump(out/'original_slots_c43.json',control_gate)
    arm_mains,arm_gates={},{}
    for arm,open_first in (('protected',False),('open',True)):
        main,audit=promote(control_main,native,shortlist,requests,scores,identity,allow_rank1=open_first)
        if set(main)!=set(control_main):raise ValueError('Unknown or missing C41 query')
        ordered=gate(gate_program,main,pc)
        changed_main=[m for m in native if main[m][0]!=control_main[m][0]]
        changed_gate=[m for m in native if ordered[m]!=control_gate[m]]
        top1=[m for m in native if ordered[m][:1]!=control_gate[m][:1]]
        top5=[m for m in native if ordered[m][:5]!=control_gate[m][:5]]
        for mid in native:
            if len(main[mid][0])!=len(control_main[mid][0]) or len(ordered[mid])!=len(control_gate[mid]):
                raise ValueError(f'Candidate budget changed for {arm}/{mid}')
            if (not open_first and main[mid][0][:1]!=control_main[mid][0][:1]) or (native[mid]['library_max']>=.9 and mid in changed_main):
                raise ValueError(f'Original protected source head or strong-library ranking changed: {arm}/{mid}')
        dump(out/f'{arm}_main.json',main)
        dump(out/f'{arm}_c43.json',ordered)
        dump(out/f'{arm}_audit.json',audit)
        for name in (f'{arm}_main.json',f'{arm}_c43.json',f'{arm}_audit.json'):
            files[name]=digest(out/name)
        outputs[arm]=dict(main_changed=changed_main,main_unchanged=len(main)-len(changed_main),
                          c43_changed=changed_gate,c43_unchanged=len(main)-len(changed_gate),c43_top1_changed=top1,
                          c43_top5_changed=top5,main_top1_changed=[m for m in native if main[m][0][:1]!=control_main[m][0][:1]])
        arm_mains[arm],arm_gates[arm]=main,ordered
    comparisons=dict(main=[m for m in native if arm_mains['open'][m][0]!=arm_mains['protected'][m][0]],
                     c43=[m for m in native if arm_gates['open'][m]!=arm_gates['protected'][m]])
    for name in ('c41_existing_budgets.json','original_slots_main.json','original_slots_c43.json'):
        files[name]=digest(out/name)
    receipt=dict(status='c43_sealed',scope='C41 source and native C43 prefusion',molecules=len(native),shortlist_slots=670,
                 original_forward_inputs_unchanged=True,new_predictor_requests=0,full_final_fusion_scored=False,
                 original_no_shadow_c43_exact=60,existing_slots_control_c43_exact=60,arms=outputs,
                 open_versus_protected=comparisons,shortlist_sha256=UNSEEN_SHORTLIST,
                 shortlist_receipt_sha256=UNSEEN_RECEIPT,baseline_zip_sha256=SEALED_BASE,
                 original_slot_receipt_sha256=digest(old/'c41_study_receipt.json'),
                 t4_snapshot_receipt_sha256=digest(snapshot/'c39_snapshot_receipt.json'),files=files)
    dump(out/'c41_existing_receipt.json',receipt)
    print('C41 existing-source changes', {arm:len(n['main_changed']) for arm,n in outputs.items()},flush=True)
