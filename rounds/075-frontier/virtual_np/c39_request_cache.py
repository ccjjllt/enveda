"""Plan equal-budget C41 requests against the original 60-query T4 archive; never run predictors."""
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory


ARCHIVE = 'jiyujia/enveda-075-np-recall-base/2'
BUDGET_SHA = '0fdf62f6f401f2817de3688f1a7880f803b97bf3dc9dc7772acd4a9e6193e7c2'
REQUESTS = {'ice_native': 'ice_work/ice_in.json', 'gl_native': 'gl_work/gl_in.json',
            'ice_shadow': 'pilot_ice/ice_in.json', 'gl_shadow': 'pilot_gl/gl_in.json'}
NATIVE_SCORES = {'ice': 'ice_work/ice_out.json', 'gl': 'gl_work/gl_out.json'}
MISSING = object()
COUNT_FIELDS = ('candidate_spectra','finite_control','finite_known','finite_max','cached_null',
                'absent_unchanged','pending','delta_candidate_spectra')


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return sha256(raw.encode()).hexdigest()


def write_json(path, value):
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()+b'\n'
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


def population(items):
    by_mid = {item['mid']: item for item in items}
    if len(by_mid) != len(items):
        raise ValueError('Duplicate molecular query in a predictor input')
    for mid,item in by_mid.items():
        if (not isinstance(mid,str) or set(item) != {'mid','spectra','cands'}
                or not isinstance(item['spectra'],list) or not isinstance(item['cands'],list)
                or not all(isinstance(s,str) and s for s in item['cands'])
                or len(set(item['cands'])) != len(item['cands'])):
            raise ValueError(f'Unexpected archived predictor input for {mid}')
    return by_mid


def checked_scores(row):
    if any(v is not None and (type(v) not in (int,float) or not math.isfinite(v)) for v in row.values()):
        raise ValueError('Archived predictor returned a nonfinite or nonnumeric value')
    return sum(v is not None for v in row.values())


def identity_function():
    from rdkit import Chem, rdBase
    from rdkit.Chem.MolStandardize import rdMolStandardize
    if rdBase.rdkitVersion != '2026.03.3':
        raise ValueError('New candidate de-duplication requires the original competition metric RDKit')
    tautomer = rdMolStandardize.TautomerEnumerator()

    @lru_cache(maxsize=30000)
    def identity(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            formula = tuple(sorted(Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() != 'H').items()))
            Chem.RemoveStereochemistry(mol)
            key = Chem.MolToInchiKey(tautomer.Canonicalize(mol))[:14]
        except (ValueError,RuntimeError,TypeError):
            return None
        return (key,formula) if len(key) == 14 else None
    return identity


def validate_changes(replacements, original, sources, proof):
    identity = identity_function() if any(replacements.values()) else None
    for mid, change in replacements.items():
        if (mid not in sources or not isinstance(change,dict)
                or any(not isinstance(s,str) or not s for pair in change.items() for s in pair)):
            raise ValueError(f'Invalid E1 replacements for {mid}')
        if not change:
            continue
        if not all(proof['cases'][mid][channel]['exact'] for channel in (*REQUESTS,'gl_proxy')):
            raise ValueError(f'Unverified source slot attribution for {mid}')
        slot_list = sources[mid]
        if [r['slot'] for r in slot_list] != list(range(1,len(slot_list)+1)):
            raise ValueError(f'Shadow slot numbering changed for {mid}')
        slot_smiles = [r['smiles'] for r in slot_list]
        e1 = {r['smiles'] for r in slot_list if r['channel']=='e1'}
        if not set(change) <= e1 or any(old==new for old,new in change.items()):
            raise ValueError(f'Candidate substitution does not replace a distinct original E1 shadow slot for {mid}')
        seen_channels = 0
        retained = set()
        for channel, by_mid in original.items():
            item = by_mid.get(mid)
            if item is None:
                continue
            if channel.endswith('_shadow'):
                if item['cands'] != slot_smiles:
                    raise ValueError(f'T4 shadow order differs from source slot evidence for {mid} {channel}')
                seen_channels += 1
                retained.update(s for s in item['cands'] if s not in change)
            else:
                retained.update(item['cands'])
        if not seen_channels:
            raise ValueError(f'No original shadow predictor request for {mid}')
        occupied = {identity(s) for s in retained}
        for old,new in change.items():
            key = identity(new)
            if key is None or key in occupied or key == identity(old):
                raise ValueError(f'Invalid or duplicate replacement identity for {mid}')
            occupied.add(key)


def plan_requests(archive: Path, replacements: dict, output: Path, *, snapshot: Path | None = None):
    """Use a verified snapshot for E1 attribution; optional snapshot defaults to the archive root.

    The output directory is a new (or empty) plan directory. A populated plan is never overwritten.
    The archive and snapshot contain no input which this function reads by row of ground truth.
    """
    archive, output = Path(archive), Path(output)
    snapshot = Path(snapshot) if snapshot is not None else archive
    original_hashes = {}

    def load(root, name, hashes):
        raw = (root/name).read_bytes()
        hashes[name] = sha256(raw).hexdigest()
        return json.loads(raw)

    budget = load(archive,'pilot_forward_budget.json',original_hashes)
    if original_hashes['pilot_forward_budget.json'] != BUDGET_SHA:
        raise ValueError(f'Expected original T4 input receipts from {ARCHIVE}')
    mids = set(budget['cases'])
    original_items = {channel:load(archive,name,original_hashes) for channel,name in REQUESTS.items()}
    original = {channel:population(items) for channel,items in original_items.items()}
    original_scores = {model+'_native':load(archive,name,original_hashes) for model,name in NATIVE_SCORES.items()}
    shadow = load(archive,'pilot_shadow_scores.json',original_hashes)
    shadow_sha = digest({model:original_items[model+'_shadow'] for model in NATIVE_SCORES})
    if shadow_sha != shadow['request_sha256'] or shadow_sha != budget['shadow_input_sha256']:
        raise ValueError('Original shadow scores or joint request no longer match the original T4 budget receipt')
    for model in NATIVE_SCORES:
        original_scores[model+'_shadow'] = shadow[model]
    snapshot_hashes = {}
    proof = load(snapshot,'c39_request_verification.json',snapshot_hashes)
    sources = load(snapshot,'c39_shadow_provenance.json',snapshot_hashes)
    if (proof['original'] != ARCHIVE or snapshot_hashes['c39_shadow_provenance.json'] != proof['provenance_sha256']
            or set(sources) != mids or set(proof['cases']) != mids or not set(replacements) <= mids):
        raise ValueError('The original T4 population and source provenance disagree')

    cache = {}
    for channel, by_mid in original.items():
        model,kind = channel.split('_')
        outputs = original_scores[channel]
        if (not set(by_mid) <= mids or not set(outputs) <= set(by_mid)
                or original_hashes[REQUESTS[channel]] != proof['request_files'][channel]['file_sha256']):
            raise ValueError(f'Wrong original {channel} cache or input file')
        finite_key = 'gl_native_and_proxy' if channel=='gl_native' else channel
        for mid,case in budget['cases'].items():
            if case['predictor_budget']['gl_proxy_candidate_spectra'] or case['request_sha256']['gl_proxy'] != digest({}):
                raise ValueError('This planner is for the original no-proxy 60-case panel')
            item = by_mid.get(mid, {})
            smiles, spectra = item.get('cands',[]), item.get('spectra',[])
            scores = outputs.get(mid,{})
            if (digest(item) != case['request_sha256'][channel]
                    or len(smiles)*len(spectra) != case['predictor_budget'][channel+'_candidate_spectra']
                    or set(scores)-set(smiles) or checked_scores(scores) != case['finite_counts'][finite_key]):
                raise ValueError(f'Original {channel} scores, request or budget differs for {mid}')
            group = digest(spectra)
            for s,value in scores.items():
                key = (model,mid,group,s)
                if key in cache and cache[key] != value:
                    raise ValueError(f'Conflicting historical {model} score for identical query/spectra/SMILES: {mid}')
                cache[key] = value
    validate_changes(replacements, original, sources, proof)

    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Plan output already exists: {output}')
    with TemporaryDirectory(prefix='.c39plan-',dir=output.parent) as tmp:
        stage = Path(tmp)/'plan'
        stage.mkdir()
        artifacts = {}
        control = dict(original=ARCHIVE, original_files=original_hashes, snapshot_files=snapshot_hashes,
                       request_paths=REQUESTS, snapshot_all_requests_exact=proof['exact'])
        artifacts['control_reference.json'] = write_json(stage/'control_reference.json',control)
        empty = dict.fromkeys(COUNT_FIELDS,0)
        empty.update(spectrum_group_sha256=None,original_request_sha256=digest({}),treatment_request_sha256=digest({}))
        per_channel, by_query = {}, {mid:{name:empty.copy() for name in (*REQUESTS,'gl_proxy')} for mid in sorted(mids)}
        for model in NATIVE_SCORES:
            delta = []
            saved = {'native':{}, 'shadow':{}}
            for kind in ('native','shadow'):
                channel = model+'_'+kind
                treatment = []
                for item in original_items[channel]:
                    mid, old = item['mid'], item['cands']
                    change = replacements.get(mid,{}) if kind=='shadow' else {}
                    new = [change.get(s,s) for s in old]
                    next_item = dict(item,cands=new) if old!=new else item
                    treatment.append(next_item)
                    group = digest(item['spectra'])
                    n_spectra = len(item['spectra'])
                    scores = original_scores[channel].get(mid,{})
                    retained, pending, absent = {}, [], []
                    for a,b in zip(old,new):
                        value = scores.get(a,MISSING) if a==b else cache.get((model,mid,group,b),MISSING)
                        if value is not MISSING:
                            retained[b] = value
                        elif a!=b and n_spectra:
                            pending.append(b)
                        else:
                            absent.append(b)
                    if mid in original_scores[channel] or retained or pending:
                        saved[kind][mid] = retained
                    if pending:
                        delta.append(dict(item,cands=pending))
                    finite = checked_scores(retained)
                    counts = dict(candidate_spectra=len(new)*n_spectra, finite_control=checked_scores(scores),
                                  finite_known=finite, finite_max=finite+len(pending),
                                  cached_null=sum(v is None for v in retained.values()),
                                  absent_unchanged=len(absent), pending=len(pending),
                                  delta_candidate_spectra=len(pending)*n_spectra)
                    by_query[mid][channel] = dict(counts, spectrum_group_sha256=group,
                                                  original_request_sha256=digest(item), treatment_request_sha256=digest(next_item))
                # Full treatment is materialized. Control is referenced by immutable original artifact digest.
                name = f'{channel}_treatment.json'
                artifacts[name] = write_json(stage/name,treatment)
                per_channel[channel] = {k:sum(rows[channel][k] for rows in by_query.values()) for k in COUNT_FIELDS}
                for mid,case in budget['cases'].items():
                    if by_query[mid][channel]['candidate_spectra'] != case['predictor_budget'][channel+'_candidate_spectra']:
                        raise ValueError(f'Changed logical {channel} prediction budget for {mid}')
            for name,value in ((model+'_delta.json',delta),(model+'_cache.json',saved)):
                artifacts[name] = write_json(stage/name,value)
        artifacts['case_budgets.json'] = write_json(stage/'case_budgets.json',by_query)
        artifacts['replacements.json'] = write_json(stage/'replacements.json',replacements)
        summary = dict(control=ARCHIVE, changed_molecules=sum(bool(v) for v in replacements.values()),
                       changed_shadow_e1_slots=sum(map(len,replacements.values())), molecules=len(mids),
                       proxy_candidate_spectra=0, channels=per_channel, files=artifacts,
                       cache_key=['predictor','molecule_id','ordered_spectra_sha256','exact_smiles'],
                       post_gpu_finite_counts_verified=False)
        write_json(stage/'plan.json',summary)
        stage.replace(output)
    return summary
