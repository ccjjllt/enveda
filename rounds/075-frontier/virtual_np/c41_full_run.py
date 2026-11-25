"""Run the same full 60-case outer notebook in two new kernels and seal the two ranked outputs."""
from hashlib import file_digest
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


SOURCE = 'pablodiegoo/enveda-casmi26-starter-notebook/29'
CONTROL_ZIP = '7aa006825a0f8654c4239f25e89ddb490ea04575b9a195d6b744955688d5d7c5'
BASELINE_SHA = '05509b32cfb9bcbe8ab563244f91c97e89f878eba3cc40263e3f48fd4a1f372d'
QUERY_SHA = '2a8bf747f50e6920fcd4f24e04302b721281f8794c947cc58fe545264b12d0d4'
SHARED_MODEL_SHA = '8c636d0d71ff72db7971eb6aa1a11be57b7797f85104b7b2e7f883d92f73dc64'
SHARED_FIT_SHA = '716aa01e108e54162711548289e47f4f1c05b61154c611e505f8e547b6928c9e'


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def pinned(path, expected):
    if digest(path)!=expected:
        raise ValueError(f'Incorrect archived source: {path.name}')
    return path


def read(path):
    return json.loads(path.read_bytes())


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def run_full(outer_book, snapshot, allocated, paired, shared, out):
    import nbformat
    from nbclient import NotebookClient
    from retrieval_pilot_score import APPLIED, answers, fill_sparse_lists, metric_identity

    route = Path('/tmp/c41_route')
    route.mkdir(exist_ok=True)
    control = Path('/tmp/c41_original_control')
    with ZipFile(pinned(snapshot/'c39_original_control.zip',CONTROL_ZIP)) as z:
        z.extractall(control)
    old_zip = pinned(shared/'baseline.zip',BASELINE_SHA)
    with ZipFile(old_zip) as z:
        baseline_receipt=json.loads(z.read('pilot_arm_receipt.json'))
        if baseline_receipt['queries_sha256']!=QUERY_SHA or baseline_receipt['candidate_slots']!=25:
            raise ValueError('Original sealed population differs from the blind panel')
        base_csv=route/'historical.csv'
        base_csv.write_bytes(z.read('submission.csv'))
        base=answers(base_csv)
        original_sources=json.loads(z.read('four_source_lists.json'))
        original_ref=json.loads(z.read('four_reference_audit.json'))
    if len(base)!=60 or original_sources!=read(control/'four_source_lists.json'):
        raise ValueError('The original T4 source archive and paired control differ')
    fit_path=pinned(shared/'shared_npx_fit.json',SHARED_FIT_SHA)
    fit=read(fit_path)
    for name, sha in [('shared_npx_model.pickle',SHARED_MODEL_SHA),('shared_npx_fit.json',SHARED_FIT_SHA)]:
        link=route/name
        if not link.exists():
            link.symlink_to(pinned(shared/name,sha))
        pinned(link,sha)
    allocation=read(allocated/'allocation_receipt.json')
    study=read(paired/'c41_study_receipt.json')
    if study['status']!='paired_c43_sealed':
        dump(out/'c41_full_status.json',dict(status='awaiting_paired_c43',study_status=study['status']))
        return
    delta=paired/'forward_delta'/'delta_receipt.json'
    prediction=read(pinned(delta, study['delta_receipt_sha256']))
    if (allocation['status']!='request_plan_only' or allocation['candidate_source']!='unseen'
            or study['original_allocation_plan_sha256']!=allocation['hashes']['requests/plan.json']
            or prediction['plan_sha256']!=study['plan_sha256'] or not prediction['all_deltas_observed']
            or study['native_no_shadow_controls_exact']!=60):
        raise ValueError('Both complete C41 arms need matched observed predictions')
    plan_path=pinned(paired/'attempted_requests/plan.json',prediction['plan_sha256'])
    plan=read(plan_path)
    budget_path=pinned(paired/'attempted_requests/case_budgets.json',plan['files']['case_budgets.json'])
    observed_path=pinned(paired/'c41_forward_observed.json',study['effective_observed_sha256'])
    raw_observed=read(pinned(paired/'forward_delta/delta_observed_by_case.json',prediction['files']['delta_observed_by_case.json']))
    fallbacks=pinned(paired/'c41_slot_fallbacks.json',study['slot_fallbacks_sha256'])
    expected_budgets, observed = read(budget_path), read(observed_path)
    channels={'ice_native','gl_native','ice_shadow','gl_shadow','gl_proxy'}
    if set(expected_budgets)!=set(base) or set(observed)!=set(base):
        raise ValueError('Forward budgets do not cover the original population')
    for mid in base:
        if set(observed[mid])!=channels or set(expected_budgets[mid])!=channels:
            raise ValueError(f'Missing predictor channel for {mid}')
        for name in channels:
            planned, seen = expected_budgets[mid][name], observed[mid][name]
            if (planned['candidate_spectra']!=seen['candidate_spectra'] or
                    planned['delta_candidate_spectra']!=seen['delta_candidate_spectra'] or
                    planned['finite_control']!=seen['finite_control'] or seen['finite_effective']<0
                    or any(seen[key]!=value for key,value in raw_observed[mid][name].items())):
                raise ValueError(f'Unequal actual forward budget for {mid}/{name}')
    historical_gate={m:original_sources[m]['prefusion'] for m in base}
    dump(route/'historical_c43.json',historical_gate)
    shared_files = {
        'c41_study_receipt.json':paired/'c41_study_receipt.json', 'c41_delta_receipt.json':delta,
        'c41_budget_plan.json':budget_path,'c41_forward_observed.json':observed_path,
        'c41_historical_c43.json':route/'historical_c43.json','c41_slot_fallbacks.json':fallbacks,
        'c41_calibration.json':pinned(paired/'c41_calibration.json',study['calibration_sha256']),
    }
    seals = {}
    # Each child uses the hard-coded Kaggle paths of the published implementation.
    # Move newly produced files out of these paths between arms, leaving the starting workspace intact.
    with TemporaryDirectory(prefix='paired-c41-seals-') as staging:
        staging=Path(staging)
        for arm in ('original_slots','replacement'):
            expected=study[arm]
            main_path=pinned(paired/f'{arm}_main.json',expected['main_sha256'])
            gate_path=pinned(paired/f'{arm}_c43.json',expected['c43_sha256'])
            main, gate = read(main_path), read(gate_path)
            if set(main)!=set(base) or set(gate)!=set(base):
                raise ValueError(f'Missing routed C41 or C43 rows: {arm}')
            shutil.copyfile(main_path,route/'main.json')
            initial={p.name for p in out.iterdir()}
            if 'submission.csv' in initial or 'four_source_lists.json' in initial:
                raise FileExistsError('The next full arm requires a fresh output workspace')
            book=nbformat.read(str(outer_book),as_version=4)
            client=NotebookClient(book,timeout=None,kernel_name='python3',allow_errors=False,
                                  resources={'metadata':{'path':str(out)}})
            client.execute()
            # The original fill order and official metric identity run only after NPX and E180X.
            fill_sparse_lists(out)
            got, sources=answers(out/'submission.csv'),read(out/'four_source_lists.json')
            if set(got)!=set(base) or set(sources)!=set(base):
                raise ValueError(f'Full run lost queries: {arm}')
            for mid in base:
                if (sources[mid]['main']!=main[mid][0] or sources[mid]['prefusion']!=gate[mid]
                        or len(got[mid])!=len(base[mid])
                        or len({metric_identity(s) for s in got[mid]})!=len(got[mid])
                        or any(metric_identity(s) is None for s in got[mid])):
                    raise ValueError(f'C43 routing or original final candidate budget differs for {arm}/{mid}')
            checks={name:read(out/f'{name}_audit.json') for name in APPLIED}
            if any(doc.get('status')!='applied' for doc in checks.values()):
                raise ValueError(f'Incomplete full-chain stage for {arm}')
            if (checks['npx'].get('feature_receipt')!=fit['feature_receipt']
                    or read(out/'four_reference_audit.json')!=original_ref):
                raise ValueError(f'Shared supervised calibration or reference exclusions differ for {arm}')
            evidence = read(out/'execution_receipt.json')
            version=read(out/'v43_execution_receipt.json')
            device=read(out/'e004_verification.json')
            physical=read(out/'spectral_parallax_report.json')
            mass=read(out/'v8_report.json')
            if (not evidence['is_full_run'] or evidence['processed_molecules']!=60 or not evidence['candidate_promotion_fix']
                    or device['main_rdkit']!='2026.03.3' or set(device['gpu'].splitlines())!={'Tesla T4'}
                    or any(version[s]['sha256']!=QUERY_SHA for s in ('test','processed_test'))
                    or physical['time_budget_exceeded'] is not False or physical['rows']!=60
                    or read(out/'merge_report.json')['status']!='applied'
                    or mass['mass']['status']!='on' or mass['mass_lead']['status']!='ran'
                    or not read(out/'pilot_provenance_audit.json')['status'].startswith('disabled')):
                raise ValueError(f'Original full runner did not process all 60 cases for {arm}')
            paths = {**shared_files,'c41_routed_c43.json':gate_path,'c41_routed_main.json':main_path}
            paths.update({n:out/n for n in ('submission.csv','submission_unpadded.csv','pilot_final_slots.json',
                         'four_source_lists.json','four_reference_audit.json','four_inner_reference.json',
                         'execution_receipt.json','v43_execution_receipt.json','e004_verification.json','c41_route_replay.json',
                         'spectral_parallax_report.json','merge_report.json','v8_report.json','apex_extlib_audit.json',
                         'pilot_provenance_audit.json',
                         *(f'{name}_audit.json' for name in APPLIED))})
            receipt=dict(status='complete',arm=arm,source=SOURCE,queries_sha256=QUERY_SHA,baseline_archive_sha256=BASELINE_SHA,
                         candidate_slots=25,slots_per_query={mid:len(v) for mid,v in got.items()},
                         c41_main_sha256=expected['main_sha256'],c43_sha256=expected['c43_sha256'],
                         shared_npx_model_sha256=SHARED_MODEL_SHA, plan_sha256=prediction['plan_sha256'],
                         original_allocation_plan_sha256=study['original_allocation_plan_sha256'],
                         equal_finite_counts=prediction['equal_finite_counts'],effective_replacements=study['effective'],
                         hashes={name:digest(path) for name,path in paths.items()})
            info=staging/f'{arm}_receipt.json'
            dump(info,receipt)
            zpath=staging/f'{arm}.zip'
            with ZipFile(zpath,'w',compression=ZIP_DEFLATED) as z:
                for name,path in paths.items():z.write(path,name)
                z.write(info,'c41_full_receipt.json')
            seals[arm]=digest(zpath)
            with TemporaryDirectory(prefix='finished-c41-arm-') as cleanup:
                for path in out.iterdir():
                    if path.name not in initial:
                        shutil.move(path,str(Path(cleanup)/path.name))
        # Publish both arms together after both isolated end-to-end checks succeed.
        for arm in seals:
            shutil.copyfile(staging/f'{arm}.zip',out/f'{arm}.zip')
            shutil.copyfile(staging/f'{arm}_receipt.json',out/f'{arm}_receipt.json')
    dump(out/'c41_full_status.json',dict(status='complete',zip_sha256=seals))
