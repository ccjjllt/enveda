"""Full replay of a blind existing-cache C41 treatment with no new ICE or GL requests."""
from collections import Counter
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from c41_full_run import (BASELINE_SHA, CONTROL_ZIP, QUERY_SHA, SHARED_FIT_SHA, SHARED_MODEL_SHA,
                          SOURCE, digest, dump, pinned, read)


EXISTING_SHA = '0a22ba312d64170c5d3c6244b3885c69fc7b9411a75717a4824d84ed8199c3d7'
CONTROL_SHA = '9c0f5301e421b590be3e7180cf12669df6311fdc62a10d87d1c206b12f8b9cf9'
SNAPSHOT_SHA = '30ebaeda090adbabd48f0b60ef4a3a857275c12be84f0c37a58f1bd8fd6ed7dc'
NATIVE_ONLY_SHA = '132adbaf581f45576e86a0b1e48c78a8e2961b18c4199c035b45254e1bcc5cf5'
ARM_OPTIONS = ('original_slots','protected','open','native_only')


def check_original_budget(historical, actual):
    expected = historical['cases']
    if (set(expected)!=set(actual['cases']) or historical['shadow_input_sha256']!=actual['shadow_input_sha256']
            or historical['parameters']!=actual['parameters']):
        raise ValueError('Full run does not share the historical forward inputs')
    for mid,case in expected.items():
        for feature in ('request_sha256','predictor_budget','finite_counts'):
            if case[feature]!=actual['cases'][mid][feature]:
                raise ValueError(f'Full run differs from the real T4 {feature} for {mid}')


def checked_native_only(folder, old, sources, budgets):
    seal = pinned(folder/'c41_native_only_receipt.json', NATIVE_ONLY_SHA)
    receipt = read(seal)
    expected = dict(status='c43_sealed', query_count=60, queries_sha256=QUERY_SHA,
                    original_t4='jiyujia/enveda-075-np-recall-base/2', historical_c43_exact=60,
                    original_slots_c43_exact=60, original_slots_main_exact=60, candidate_budget=25,
                    new_predictor_requests_in_this_job=0, native_only_incremental_forward_requests_vs_public_v29=0,
                    original_slots_reference_shadow_archived=True)
    if any(receipt[key] != value for key, value in expected.items()):
        raise ValueError('Unverified native-only C43 treatment')
    names = {'native_only_request_evidence.json',
             *(f'{arm}_{stage}.json' for arm in ('historical','native_only','original_slots') for stage in ('main','c43'))}
    if set(receipt['files']) != names:
        raise ValueError('Unexpected content in the native-only C43 seal')
    evidence = {name: pinned(folder/name, receipt['files'][name]) for name in names}
    for stage in ('main','c43'):
        if receipt['files'][f'original_slots_{stage}.json'] != old['original_slots'][f'{stage}_sha256']:
            raise ValueError('The attribution used a different original-slot control')
    historical_main = read(evidence['historical_main.json'])
    historical_gate = read(evidence['historical_c43.json'])
    native_main = read(evidence['native_only_main.json'])
    native_gate = read(evidence['native_only_c43.json'])
    if any(set(arm) != set(sources) for arm in (historical_main,historical_gate,native_main,native_gate)):
        raise ValueError('Incomplete historical or native-only C43 rankings')
    for mid, row in sources.items():
        if (historical_main[mid][0] != row['main'] or historical_gate[mid] != row['prefusion']
                or len(native_main[mid][0]) != len(row['main']) or len(native_gate[mid]) != len(row['prefusion'])):
            raise ValueError(f'Native-only attribution changed sealed source coverage or slot counts: {mid}')
    requests = read(evidence['native_only_request_evidence.json'])
    if set(requests['cases']) != set(budgets):
        raise ValueError('Incomplete native-only forward-request attribution')
    totals = Counter()
    for mid, row in requests['cases'].items():
        claimed = budgets[mid]
        if (claimed['gl_proxy_candidate_spectra'] != 0
                or any(row[f'{model}_{lane}_historical_pairs'] != claimed[f'{model}_{lane}_candidate_spectra']
                       for model in ('ice','gl') for lane in ('native','shadow'))
                or any(not 0 <= row[f'{model}_shadow_readable_top25_formula_pairs'] <= row[f'{model}_shadow_historical_pairs']
                       for model in ('ice','gl'))):
            raise ValueError(f'Native-only forward-cache evidence differs from the original budget: {mid}')
        totals.update(row)
    if dict(totals) != requests['totals'] or dict(totals) != receipt['request_counts']:
        raise ValueError('Native-only request sums disagree with the sealed C43 receipt')
    return {name: evidence[name] for name in ('native_only_main.json','native_only_c43.json',
                                              'native_only_request_evidence.json') } | {seal.name: seal}


def run(outer_book, snapshot, previous, existing, shared, out, variant):
    import nbformat
    from nbclient import NotebookClient
    from retrieval_pilot_score import APPLIED, answers, fill_sparse_lists, metric_identity

    if variant not in ARM_OPTIONS:
        raise ValueError('Choose exactly one previously sealed treatment')
    old_file=pinned(previous/'c41_study_receipt.json',CONTROL_SHA)
    old=read(old_file)
    frozen=read(pinned(snapshot/'c39_snapshot_receipt.json',SNAPSHOT_SHA))
    if (old['status']!='calibrated_waiting_for_delta' or old['native_no_shadow_controls_exact']!=60
            or not frozen['cache_reuse_supported'] or frozen['device']!='cuda'):
        raise ValueError('Unverified original T4 snapshot or original-slot control')
    arms=(variant,) if variant in ('original_slots','native_only') else ('original_slots',variant)
    evidence={}
    for ext,key in [('main','main_sha256'),('c43','c43_sha256')]:
        name='original_slots_'+ext+'.json'
        evidence[name]=pinned(previous/name,old['original_slots'][key])
    if len(arms)==2:
        proof_file=pinned(existing/'c41_existing_receipt.json',EXISTING_SHA)
        proof=read(proof_file)
        if (proof['status']!='c43_sealed' or not proof['original_forward_inputs_unchanged']
                or proof['new_predictor_requests']!=0 or proof['molecules']!=60 or proof['shortlist_slots']!=670
                or proof['original_slot_receipt_sha256']!=CONTROL_SHA or proof['t4_snapshot_receipt_sha256']!=SNAPSHOT_SHA
                or proof['original_no_shadow_c43_exact']!=60 or proof['existing_slots_control_c43_exact']!=60):
            raise ValueError('Unverified existing-cache route')
        for name in ('original_slots_main.json','original_slots_c43.json'):
            pinned(existing/name,digest(evidence[name]))
        for name in ('c41_existing_budgets.json',variant+'_audit.json',variant+'_main.json',variant+'_c43.json'):
            evidence[name]=pinned(existing/name,proof['files'][name])
    route=Path('/tmp/c41_route')
    route.mkdir(exist_ok=True)
    control=Path('/tmp/c41_original_control')
    with ZipFile(pinned(snapshot/'c39_original_control.zip',CONTROL_ZIP)) as z:
        z.extractall(control)
    historical=read(control/'pilot_forward_budget.json')
    original_budgets={m:r['predictor_budget'] for m,r in historical['cases'].items()}
    if len(arms)==1:
        evidence['c41_existing_budgets.json']=route/'existing_original_budgets.json'
        dump(evidence['c41_existing_budgets.json'],original_budgets)
    if read(evidence['c41_existing_budgets.json'])!=original_budgets:
        raise ValueError('Upstream zero-new-request study differs from original per-query budgets')
    if variant=='native_only':
        evidence.update(checked_native_only(existing,old,read(control/'four_source_lists.json'),original_budgets))
    fit=read(pinned(shared/'shared_npx_fit.json',SHARED_FIT_SHA))
    for name,sha in [('shared_npx_model.pickle',SHARED_MODEL_SHA),('shared_npx_fit.json',SHARED_FIT_SHA)]:
        link=route/name
        if not link.exists():link.symlink_to(pinned(shared/name,sha))
        pinned(link,sha)
    import json
    with ZipFile(pinned(shared/'baseline.zip',BASELINE_SHA)) as z:
        receipt=json.loads(z.read('pilot_arm_receipt.json'))
        base_path=route/'historical.csv'
        base_path.write_bytes(z.read('submission.csv'))
        base=answers(base_path)
        original_ref=json.loads(z.read('four_reference_audit.json'))
        if (receipt['queries_sha256']!=QUERY_SHA or receipt['candidate_slots']!=25 or len(base)!=60
                or json.loads(z.read('four_source_lists.json'))!=read(control/'four_source_lists.json')):
            raise ValueError('Historical final rankings and retained T4 frozen source are inconsistent')
    seals={}
    with TemporaryDirectory(prefix='c41-existing-zips-') as stage_temp:
        stage=Path(stage_temp)
        for arm in arms:
            main_path,gate_path=evidence[arm+'_main.json'],evidence[arm+'_c43.json']
            main,gate=read(main_path),read(gate_path)
            if set(main)!=set(base) or set(gate)!=set(base):raise ValueError('Incomplete source arm')
            shutil.copyfile(main_path,route/'main.json')
            initial={p.name for p in out.iterdir()}
            if {'submission.csv','four_source_lists.json'} & initial:
                raise FileExistsError('Full arm requires an isolated new workspace')
            book=nbformat.read(str(outer_book),as_version=4)
            NotebookClient(book,timeout=None,kernel_name='python3',allow_errors=False,
                           resources={'metadata':{'path':str(out)}}).execute()
            fill_sparse_lists(out)
            got,sources=answers(out/'submission.csv'),read(out/'four_source_lists.json')
            if set(got)!=set(base) or set(sources)!=set(base):raise ValueError('Incomplete final arm')
            for mid,guesses in got.items():
                keys=[metric_identity(s) for s in guesses]
                if (sources[mid]['main']!=main[mid][0] or sources[mid]['prefusion']!=gate[mid]
                        or len(guesses)!=len(base[mid]) or None in keys or len(set(keys))!=len(keys)):
                    raise ValueError(f'Full arm did not preserve the route or historical slots: {arm}/{mid}')
            regenerated_budget=read(out/'pilot_forward_budget.json')
            check_original_budget(historical,regenerated_budget)
            replay=read(out/'c41_route_replay.json')
            if (not replay['native_requests_replayed'] or replay['queries']!=60
                    or replay['original_shadow_request_sha256']!=historical['shadow_input_sha256']
                    or replay['treatment_main_sha256']!=digest(main_path)):
                raise ValueError(f'Full arm consumed a different physical route: {arm}')
            audits={name:read(out/f'{name}_audit.json') for name in APPLIED}
            inner=read(out/'v43_execution_receipt.json')
            outer=read(out/'execution_receipt.json')
            device=read(out/'e004_verification.json')
            physics=read(out/'spectral_parallax_report.json')
            mass=read(out/'v8_report.json')
            if (any(v.get('status')!='applied' for v in audits.values()) or audits['npx'].get('feature_receipt')!=fit['feature_receipt']
                    or read(out/'four_reference_audit.json')!=original_ref or not outer['is_full_run']
                    or outer['processed_molecules']!=60 or not outer['candidate_promotion_fix']
                    or device['main_rdkit']!='2026.03.3' or set(device['gpu'].splitlines())!={'Tesla T4'}
                    or any(inner[s]['sha256']!=QUERY_SHA for s in ('test','processed_test'))
                    or physics['time_budget_exceeded'] is not False or physics['rows']!=60
                    or read(out/'merge_report.json')['status']!='applied' or mass['mass']['status']!='on'
                    or mass['mass_lead']['status']!='ran' or not read(out/'pilot_provenance_audit.json')['status'].startswith('disabled')):
                raise ValueError(f'Incomplete original full fusion for {arm}')
            paths={
                'c41_original_slot_receipt.json':old_file,
                'c41_existing_budgets.json':evidence['c41_existing_budgets.json'],
                'c41_routed_main.json':main_path,'c41_routed_c43.json':gate_path,
                'c41_original_forward_budget.json':control/'pilot_forward_budget.json',
            }
            if len(arms)==2:
                paths['c41_existing_receipt.json']=proof_file
                if arm==variant:paths['c41_promotion_audit.json']=evidence[variant+'_audit.json']
            if arm=='native_only':
                for name in ('c41_native_only_receipt.json','native_only_request_evidence.json'):
                    paths[name]=evidence[name]
            paths.update({name:out/name for name in ('submission.csv','submission_unpadded.csv','pilot_final_slots.json',
                          'pilot_forward_budget.json','four_source_lists.json','four_reference_audit.json','four_inner_reference.json',
                          'c41_route_replay.json','execution_receipt.json','v43_execution_receipt.json',
                          'e004_verification.json','spectral_parallax_report.json','merge_report.json','v8_report.json',
                          'apex_extlib_audit.json','pilot_provenance_audit.json',*(f'{s}_audit.json' for s in APPLIED))})
            result=dict(status='complete',arm=arm,comparison=variant+'_vs_historical' if len(arms)==1 else variant+'_vs_original_slots',
                        source=SOURCE,molecules=len(got),queries_sha256=QUERY_SHA,
                        baseline_archive_sha256=BASELINE_SHA,candidate_slots=25,slots_per_query={m:len(v) for m,v in got.items()},
                        c41_main_sha256=digest(main_path),c43_sha256=digest(gate_path),shared_npx_model_sha256=SHARED_MODEL_SHA,
                        original_slot_receipt_sha256=CONTROL_SHA,
                        new_predictor_requests=0,new_physical_requests=0,same_original_candidate_spectra=True,
                        same_original_finite_counts=True,
                        hashes={k:digest(v) for k,v in paths.items()})
            if len(arms)==2:result['c41_existing_receipt_sha256']=EXISTING_SHA
            if arm=='native_only':
                result.update(c41_native_only_receipt_sha256=NATIVE_ONLY_SHA, c41_treatment_shadow_requests=0,
                              c41_treatment_uses_native_forward_scores_only=True,
                              archived_shadow_budget_scope='original_route_replay_only')
            receipt_path=stage/f'{arm}_receipt.json'
            dump(receipt_path,result)
            target=stage/f'{arm}.zip'
            with ZipFile(target,'w',compression=ZIP_DEFLATED) as z:
                for name,path in paths.items():z.write(path,name)
                z.write(receipt_path,'c41_full_receipt.json')
            seals[arm]=digest(target)
            # Isolate the next arm without copying large transient outputs across mounts.
            with TemporaryDirectory(prefix='.completed-pipeline-',dir=out) as cleanup:
                trash=Path(cleanup)
                for path in list(out.iterdir()):
                    if path!=trash and path.name not in initial:path.rename(trash/path.name)
        for arm in seals:
            shutil.copyfile(stage/f'{arm}.zip',out/f'{arm}.zip')
            shutil.copyfile(stage/f'{arm}_receipt.json',out/f'{arm}_receipt.json')
    dump(out/'c41_full_status.json',dict(status='complete',treatment=variant,new_physical_requests=0,zip_sha256=seals))
