"""Run the two sealed E1-shadow deltas once and reconcile their outputs with the original T4 cache."""
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys

from c39_request_cache import ARCHIVE, BUDGET_SHA, REQUESTS, checked_scores, digest, population, write_json


INVENTORY_SHA = '1b3797bb9465ef41f782a3e585bcf3d44ec7f7c925bfffc3d874973530887598'
MODEL_FILES = {'ice': ('casmi26-iceberg','ice_runner.py','fuse.py'),
               'gl': ('casmi26-glacier','gl_runner.py','gl_fuse.py')}


def read(path):
    return json.loads(path.read_bytes())


def file_digest(path):
    h = sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda:fp.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def matches_record(path,record):
    if not path.is_file() or path.stat().st_size != record['bytes']:
        return False
    if record['full']:
        return file_digest(path) == record['sha256']
    with path.open('rb') as fp:
        start = fp.read(4 << 20)
        fp.seek(-(4 << 20),2)
        return sha256(start+fp.read(4 << 20)).hexdigest() == record['sha256_head_tail']


def asset_check(inventory, pkg, model):
    slug,runner,fuse = MODEL_FILES[model]
    roots = [PurePosixPath(r['path']).parent for r in inventory
             if slug in PurePosixPath(r['path']).parts and PurePosixPath(r['path']).name == runner]
    if len(roots) != 1:
        raise ValueError(f'Original T4 inventory does not uniquely locate {slug}/{runner}')
    root, = roots
    rows = {PurePosixPath(r['path']).relative_to(root).as_posix():r for r in inventory
            if PurePosixPath(r['path']).is_relative_to(root)}
    weights = [name for name in rows if name.endswith(('.pt','.ckpt','.pth','.safetensors'))]
    if fuse not in rows or not weights:
        raise ValueError(f'Original inventory does not identify the {model} predictor files and weights')
    # The original T4 archived full digests up to 32 MiB and size plus both 4 MiB edges above that.
    for name,row in rows.items():
        if not matches_record(pkg/name,row):
            raise ValueError(f'Mounted predictor asset differs from the T4 archive: {name}')
    # The original inventory includes the full SHA of each package's own manifest.
    # Use its full weight hashes in addition to the original large-file edge hashes when available.
    manifest = read(pkg/'MANIFEST.json') if 'MANIFEST.json' in rows else {}
    full_weights = {}
    for name in weights:
        record = manifest.get('files',{}).get(name) if isinstance(manifest.get('files'),dict) else None
        expected = record.get('sha256') if isinstance(record,dict) else record
        if isinstance(expected,str) and len(expected)==64:
            full_weights[name] = file_digest(pkg/name)
            if full_weights[name] != expected:
                raise ValueError(f'Model weights differ from the originally mounted package manifest: {name}')
    return dict(archived_files=len(rows), runner_sha256=file_digest(pkg/runner), fuse_sha256=file_digest(pkg/fuse),
                fully_verified_weight_hashes=full_weights)


def checked_plan(plan, archive, expected):
    if file_digest(plan/'plan.json') != expected:
        raise ValueError('Plan does not match its sealed allocation receipt')
    meta = read(plan/'plan.json')
    if meta['control'] != ARCHIVE or meta['proxy_candidate_spectra'] != 0:
        raise ValueError('Expected the archived 60-case no-proxy control')
    for name,checksum in meta['files'].items():
        if Path(name).name != name or file_digest(plan/name) != checksum:
            raise ValueError(f'Changed or unexpected plan artifact: {name}')
    control = read(plan/'control_reference.json')
    if control['original'] != ARCHIVE or control['original_files']['pilot_forward_budget.json'] != BUDGET_SHA:
        raise ValueError('Unrecognized historical control')
    for name,checksum in control['original_files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or file_digest(archive/rel) != checksum:
            raise ValueError(f'Archived original artifact changed: {name}')
    budget = read(archive/'pilot_forward_budget.json')
    cases, changes = read(plan/'case_budgets.json'), read(plan/'replacements.json')
    if set(cases) != set(budget['cases']) or not set(changes) <= set(cases):
        raise ValueError('Different control, allocation, or planned query populations')
    shadow = read(archive/'pilot_shadow_scores.json')
    result = {}
    for model in ('ice','gl'):
        state = dict(delta=read(plan/f'{model}_delta.json'), treatment=read(plan/f'{model}_cache.json'),
                     original=dict(native=read(archive/f'{model}_work/{model}_out.json'), shadow=shadow[model]))
        delta = population(state['delta'])
        for kind in ('native','shadow'):
            channel = model+'_'+kind
            original = read(archive/REQUESTS[channel])
            treatment = read(plan/f'{channel}_treatment.json')
            if [r['mid'] for r in original] != [r['mid'] for r in treatment]:
                raise ValueError(f'{channel}: original query sequence changed')
            old, new = population(original), population(treatment)
            cached = state['treatment'][kind]
            if not set(cached) <= set(new):
                raise ValueError(f'{channel}: extra cached molecules')
            if kind == 'shadow' and not set(delta) <= set(new):
                raise ValueError(f'{channel}: delta includes an unscheduled molecule')
            for mid in cases:
                before, after = old.get(mid,{}), new.get(mid,{})
                previous, updated = before.get('cands',[]), after.get('cands',[])
                spectra = after.get('spectra',[])
                rec = cases[mid][channel]
                repl = changes.get(mid,{}) if kind=='shadow' else {}
                if (updated != [repl.get(s,s) for s in previous]
                        or digest(after) != rec['treatment_request_sha256']
                        or digest(before) != rec['original_request_sha256']
                        or digest(before) != budget['cases'][mid]['request_sha256'][channel]
                        or digest(before.get('spectra',[])) != digest(spectra)
                        or len(updated)*len(spectra) != rec['candidate_spectra']
                        or rec['candidate_spectra'] != budget['cases'][mid]['predictor_budget'][channel+'_candidate_spectra']):
                    raise ValueError(f'{channel}/{mid}: changed original slots, spectra or logical quota')
                known = cached.get(mid,{})
                if not set(known) <= set(updated) or checked_scores(known) != rec['finite_known']:
                    raise ValueError(f'{channel}/{mid}: unrecognized cached predictions')
                if kind=='native':
                    if known != state['original']['native'].get(mid,{}) or rec['pending']:
                        raise ValueError(f'{channel}/{mid}: native branch must replay exactly')
                    continue
                missing = [s for old_s,s in zip(previous,updated) if s != old_s and s not in known and spectra]
                row = delta.get(mid)
                if ((row is not None) != bool(missing) or (row and (row['cands'] != missing
                        or digest(row['spectra']) != digest(spectra))) or rec['pending'] != len(missing)
                        or rec['delta_candidate_spectra'] != len(missing)*len(spectra)):
                    raise ValueError(f'{channel}/{mid}: delta differs from the sealed pending candidates')
        result[model] = state
    return meta,cases,budget,result


def original_helper(pkg, model):
    path = pkg/MODEL_FILES[model][2]
    spec = importlib.util.spec_from_file_location(f'c41_archived_{model}_fuse',path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(pkg))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def predict(model, pkg, ice_pkg, site, items, work, expected_meta):
    helper = original_helper(pkg,model)
    work.mkdir()
    # The marker survives a runner error. A second call must not spend the same physical requests twice.
    write_json(work/'attempt.json',dict(model=model, ordered_request_sha256=digest(items)))
    options = dict(workdir=str(work), device='cuda', site=str(site), extra_args=['--any-rdkit'])
    if model=='ice':
        returned = helper.run_ice(str(pkg),items,budget_s=5400,**options)
    else:
        returned = helper.run_gl(str(pkg),items,budget_s=4000,wheels=str(ice_pkg/'wheels'),**options)
    request_file = work/f'{model}_in.json'
    if not request_file.exists() or digest(read(request_file)) != digest(items):
        raise ValueError(f'{model}: the actual submitted predictor request differs from the sealed delta')
    raw = read(work/f'{model}_out.json')
    detail = read(work/f'{model}_out.json.meta.json')
    loader = [line for line in detail.get('log_tail',[]) if 'model loaded on' in line]
    if loader and not all('numba True' in line for line in loader):
        raise ValueError(f'{model}: the native runner did not load the original numba implementation')
    if digest(raw) != digest(returned):
        raise ValueError(f'{model}: runner return differs from output artifact')
    expected_items = population(items)
    if (not isinstance(raw,dict) or not set(raw)<=set(expected_items) or
            any(not isinstance(row,dict) or not set(row)<=set(expected_items[mid]['cands']) for mid,row in raw.items())):
        raise ValueError(f'{model}: runner output contains an invalid row or a candidate outside the issued request')
    finite = sum(checked_scores(row) for row in raw.values())
    if (detail['status'] != 'ok' or not str(detail.get('device','')).startswith('cuda')
            or detail.get('rdkit') != expected_meta['rdkit'] or detail.get('torch') != expected_meta['torch']
            or detail['n_mols'] != len(items) or detail['n_cands'] != sum(len(i['cands']) for i in items)
            or detail['n_scored'] != finite
            or not 0 <= detail['n_mols_scored'] <= detail['n_mols_covered'] <= len(items)):
        raise ValueError(f'{model}: incomplete scoring or the runtime differs from its original T4 version')
    return raw,detail


def candidate_outcomes(items, scores):
    observations = {}
    for item in items:
        mid = item['mid']
        row = scores.get(mid,{})
        absent = [s for s in item['cands'] if s not in row]
        nulls = [s for s in item['cands'] if s in row and row[s] is None]
        observations[mid] = dict(submitted=len(item['cands']), finite=checked_scores(row),
                                 absent_smiles=absent, null_smiles=nulls)
    return observations


def run_delta(plan:Path, archive:Path, output:Path, *, ice_pkg:Path, gl_pkg:Path,
              preseed_site, site:Path, expected_plan_sha:str, asset_inventory:Path|None=None, execute=False):
    """The caller supplies the original C26/C41 runtime wheel selector and site preparation function."""
    plan,archive,output,site = map(Path,(plan,archive,output,site))
    pkgs = {'ice':Path(ice_pkg),'gl':Path(gl_pkg)}
    output.mkdir(parents=True,exist_ok=True)
    if (output/'prediction_started.json').exists():
        raise FileExistsError('A delta execution already started in this output directory')
    try:
        _,cases,budget,state = checked_plan(plan,archive,expected_plan_sha)
    except (KeyError,TypeError,OSError,ValueError) as exc:
        report = dict(control=ARCHIVE,plan_sha256=expected_plan_sha,executable=False,
                      blockers=[f'Could not reconcile sealed plan with original T4 inputs: {exc}'])
        write_json(output/'delta_preflight.json',report)
        return report
    pending = {m:sum(len(r['cands'])*len(r['spectra']) for r in v['delta']) for m,v in state.items()}
    assets, blockers = {}, []
    if any(pending.values()):
        if asset_inventory is None or not Path(asset_inventory).is_file() or file_digest(Path(asset_inventory)) != INVENTORY_SHA:
            blockers.append('Original T4 model asset inventory is absent or its SHA differs')
        else:
            inventory = read(Path(asset_inventory))
            for model,pkg in pkgs.items():
                if pending[model] or model=='ice':
                    try:
                        assets[model] = asset_check(inventory,pkg,model)
                    except (OSError,ValueError) as exc:
                        blockers.append(str(exc))
            selector = getattr(preseed_site,'__globals__',{}).get('simulator_rdkit_wheel')
            if not callable(selector):
                blockers.append('Expected original C41 preseed function linked to its C26 simulator wheel selector')
            else:
                try:
                    wheel = Path(selector(str(pkgs['ice'])))
                    records = [r for r in inventory if PurePosixPath(r['path']).name == wheel.name]
                    if not records or not any(matches_record(wheel,r) for r in records):
                        raise ValueError('Selected simulator RDKit wheel differs from the original T4 inventory')
                    assets['simulator_wheel'] = dict(name=wheel.name,sha256=file_digest(wheel))
                except (OSError,ValueError) as exc:
                    blockers.append(str(exc))
        if not callable(preseed_site):
            blockers.append('Original C41 simulator site preparation is missing')
        if os.environ.get('NUMBA_DISABLE_JIT','0') not in ('','0'):
            blockers.append('The runtime disables the original numba JIT')
        if execute:
            import torch
            if not torch.cuda.is_available() or 'Tesla T4' not in torch.cuda.get_device_name(0):
                blockers.append('The active device differs from the original Tesla T4')
            elif str(torch.__version__) != budget['shadow_meta']['ice']['torch']:
                blockers.append('The CUDA torch version differs from the original T4')
    preflight = dict(control=ARCHIVE, plan_sha256=expected_plan_sha, assets=assets, blockers=blockers,
                     pending_candidate_spectra=pending, hardware_checked=bool(any(pending.values()) and execute),
                     executable=not blockers)
    if execute and not blockers and any(pending.values()):
        cache_dir = Path('/kaggle/working/numba_cache')
        cache_dir.mkdir(exist_ok=True)
        os.environ.update(NUMBA_CACHE_DIR=str(cache_dir),PYTHONHASHSEED='0',PYTHONUNBUFFERED='1')
        try:
            preseed_site(str(pkgs['ice']),str(site))
        except (OSError,ValueError,RuntimeError,AssertionError) as exc:
            blockers.append(f'Original C41 simulator setup failed before any prediction: {exc}')
            preflight['executable'] = False
    write_json(output/'delta_preflight.json',preflight)
    if not execute or blockers:
        return preflight
    with (output/'prediction_started.json').open('x') as marker:
        json.dump(dict(plan_sha256=expected_plan_sha,pending_candidate_spectra=pending),marker)
    statuses, meta, files, outcomes = {}, {}, {}, {}
    for model,entry in state.items():
        items = entry['delta']
        scores = {}
        if items:
            try:
                scores,meta[model] = predict(model,pkgs[model],pkgs['ice'],site,items,
                                             output/(model+'_forward'),budget['shadow_meta'][model])
            except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
                statuses[model] = dict(complete=False,reason=str(exc))
                continue
            for mid,row in scores.items():
                entry['treatment']['shadow'][mid].update(row)
        outcomes[model] = candidate_outcomes(items,scores)
        records = list(outcomes[model].values())
        statuses[model] = dict(complete=True,delta_molecules=len(items),
                               delta_candidates=sum(r['submitted'] for r in records),
                               finite=sum(r['finite'] for r in records),
                               absent=sum(len(r['absent_smiles']) for r in records),
                               explicit_null=sum(len(r['null_smiles']) for r in records))
        for arm in ('original','treatment'):
            name = f'{model}_{arm}_scores.json'
            files[name] = write_json(output/name,entry[arm])

    rows, unequal = {}, []
    all_complete = all(v['complete'] for v in statuses.values())
    for mid,case in cases.items():
        rows[mid] = {}
        for model,run in statuses.items():
            for kind in ('native','shadow'):
                channel = model+'_'+kind
                expected = case[channel]
                observed = None
                if run['complete'] or kind=='native':
                    observed = checked_scores(state[model]['treatment'][kind].get(mid,{}))
                    if not expected['finite_known'] <= observed <= expected['finite_max']:
                        raise ValueError(f'{channel}/{mid}: finite count outside the sealed interval')
                    if observed != expected['finite_control']:
                        unequal.append([mid,channel])
                rows[mid][channel] = dict(candidate_spectra=expected['candidate_spectra'],
                                          delta_candidate_spectra=expected['delta_candidate_spectra'],
                                          observed_delta_candidate_spectra=(expected['delta_candidate_spectra']
                                                                           if run['complete'] or kind=='native' else None),
                                          finite_control=expected['finite_control'],finite_treatment=observed)
        rows[mid]['gl_proxy'] = dict(candidate_spectra=0,delta_candidate_spectra=0,observed_delta_candidate_spectra=0,
                                     finite_control=0,finite_treatment=0)
    files['delta_observed_by_case.json'] = write_json(output/'delta_observed_by_case.json',rows)
    files['delta_candidate_outcomes.json'] = write_json(output/'delta_candidate_outcomes.json',outcomes)
    missing = sum(run['absent']+run['explicit_null'] for run in statuses.values()) if all_complete else None
    result = dict(control=ARCHIVE,plan_sha256=expected_plan_sha,model_status=statuses,runner_meta=meta,
                  all_deltas_observed=all_complete, equal_finite_counts=all_complete and not unequal,
                  all_scores_finite=all_complete and missing==0, missing_delta_score_count=missing,
                  score_count_unit='One scheduled delta candidate per predictor and query; explicit null and absent entries counted separately',
                  unequal_finite_case_channels=unequal, files=files)
    write_json(output/'delta_receipt.json',result)
    return result
